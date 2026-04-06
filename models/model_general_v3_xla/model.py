import os
import shutil
import tempfile
from pathlib import Path

from src.inference.models import BaseModel
from src.utils import setup_chunklength

_XLA_CACHE_DIR     = Path(__file__).parent / 'xla_cache'
_COMPILED_SIGS_DIR = Path(__file__).parent / 'compiled_signatures'

# Set before TF is imported so XLA finds the persistent cache on first compile.
os.environ['TF_XLA_FLAGS'] = f'--tf_xla_persistent_cache_directory={str(_XLA_CACHE_DIR)}'


def _build_predict_fn(embedder_model, classifier_model, n_samples):
    """Build a jit_compile=True tf.function for a fixed sample count."""
    import tensorflow as tf

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=[n_samples], dtype=tf.float32)],
    )
    def predict(samples):
        emb = embedder_model(samples)['global_average_pooling2d']
        return classifier_model(emb)['dense']

    return predict


def _save_compiled_signatures(embedder_model, classifier_model, new_n_samples):
    """Add a new signature to the compiled_signatures SavedModel.

    Existing signatures are loaded from the current artifact and re-saved
    as-is, preserving the op names frozen into them at their original trace
    time. Only new_n_samples gets a fresh trace. This is critical: rebuilding
    an existing signature re-traces it with new process-local op counters,
    changing its HLO fingerprint and invalidating xla_cache entries for that
    shape.
    """
    import tensorflow as tf

    m = tf.Module()

    # Preserve existing signatures verbatim so their HLO fingerprints stay stable.
    if _COMPILED_SIGS_DIR.exists():
        existing = tf.saved_model.load(str(_COMPILED_SIGS_DIR))
        for attr in dir(existing):
            if attr.startswith('predict_'):
                setattr(m, attr, getattr(existing, attr))

    # Build the new signature.
    setattr(m, f'predict_{new_n_samples}', _build_predict_fn(embedder_model, classifier_model, new_n_samples))

    with tempfile.TemporaryDirectory() as tmp:
        tf.saved_model.save(m, tmp)
        if _COMPILED_SIGS_DIR.exists():
            shutil.rmtree(_COMPILED_SIGS_DIR)
        shutil.move(tmp, str(_COMPILED_SIGS_DIR))



class ModelGeneralV3XLA(BaseModel):
    modelname = "model_general_v3_xla"
    embeddername = 'yamnet_k2'
    digits_results = 2

    # Class-level cache so multiple initialize() calls within one process
    # share a single @tf.function object (and thus one XLA cache slot).
    _xla_fn_cache: dict = {}

    # Loaded compiled_signatures SavedModel; shared across instances.
    _compiled_module = None

    def initialize(self):
        os.environ['TF_XLA_FLAGS'] = (
            f'--tf_xla_persistent_cache_directory={str(_XLA_CACHE_DIR)}'
        )

        self.embedder.initialize()

        import tensorflow as tf
        from keras.layers import TFSMLayer

        dir_model = str(Path(__file__).parent)
        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')

        # Fallback: raw @tf.function (process-local; used for shapes that
        # haven't been compiled into the saved signatures yet).
        _embedder  = self.embedder.model
        _classifier = self.model
        cache_key = (id(_embedder), id(_classifier))
        if cache_key not in ModelGeneralV3XLA._xla_fn_cache:
            @tf.function(jit_compile=True)
            def _predict_xla(samples):
                emb = _embedder(samples)['global_average_pooling2d']
                return _classifier(emb)['dense']
            ModelGeneralV3XLA._xla_fn_cache[cache_key] = _predict_xla
        self._predict_xla = ModelGeneralV3XLA._xla_fn_cache[cache_key]

        # Load the compiled signatures artifact if it exists and isn't
        # already loaded in this process.
        if ModelGeneralV3XLA._compiled_module is None and _COMPILED_SIGS_DIR.exists():
            ModelGeneralV3XLA._compiled_module = tf.saved_model.load(str(_COMPILED_SIGS_DIR))

    def precompile(self, chunklength):
        """Ensure a compiled signature exists for this chunklength.

        On first call for a given chunklength, adds the new shape to the
        compiled_signatures SavedModel. Subsequent calls (same or different
        process) find the shape already present in the loaded module and skip
        immediately.

        The one-time cost is paid here, not at analysis time. After this
        returns, predict() will dispatch through the stable-named SavedModel
        function, and XLA's persistent cache will hit on every future run.

        Args:
            chunklength: Audio chunk length in seconds.
        """
        import tensorflow as tf

        samplerate    = self.embedder.samplerate
        framelength_s = self.embedder.framelength_s

        chunklength_rounded = setup_chunklength(chunklength, framelength_s, self.embedder.digits_time)
        if chunklength_rounded != chunklength:
            print(f"  precompile: {chunklength}s rounded to {chunklength_rounded}s")
        n_samples = int(chunklength_rounded * samplerate)

        mod = ModelGeneralV3XLA._compiled_module
        if mod is not None and hasattr(mod, f'predict_{n_samples}'):
            return

        print(
            f"New chunk length {chunklength_rounded}s — building compiled XLA signature "
            f"(one-time cost, will be cached for all future runs)..."
        )
        _save_compiled_signatures(self.embedder.model, self.model, n_samples)

        # Reload so predict() can use the new signature in this process too.
        ModelGeneralV3XLA._compiled_module = tf.saved_model.load(str(_COMPILED_SIGS_DIR))

        # Warm the xla_cache now. tf.saved_model.save/load only serializes the
        # graph; XLA compilation (which writes to xla_cache) happens on first
        # call. Doing it here means analysis workers start with a cache hit
        # rather than paying the compile cost on the first real chunk.
        import numpy as np
        dummy = tf.constant(np.zeros(n_samples, dtype=np.float32))
        getattr(ModelGeneralV3XLA._compiled_module, f'predict_{n_samples}')(dummy)

        print("Compiled signature saved and XLA cache warmed.")

    def predict(self, audiosamples):
        import tensorflow as tf

        if not isinstance(audiosamples, tf.Tensor):
            audiosamples = tf.constant(audiosamples)

        n = int(audiosamples.shape[0])
        mod = ModelGeneralV3XLA._compiled_module
        if mod is not None:
            fn = getattr(mod, f'predict_{n}', None)
            if fn is not None:
                return fn(audiosamples)

        # Shape not yet compiled — fall back to the process-local @tf.function.
        return self._predict_xla(audiosamples)
