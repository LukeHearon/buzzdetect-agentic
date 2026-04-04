import json
import os
from pathlib import Path

from src.inference.models import BaseModel
from src.utils import setup_chunklength

# XLA compilation cache lives alongside this model
_XLA_CACHE_DIR = Path(__file__).parent / 'xla_cache'
_COMPILED_JSON = _XLA_CACHE_DIR / 'compiled.json'

# Set before TF is imported so XLA finds the persistent cache on first compile.
# worker.py previously imported TF at module level; we removed that import so
# this module-level assignment now runs before TF initializes.
os.environ['TF_XLA_FLAGS'] = f'--tf_xla_persistent_cache_directory={str(_XLA_CACHE_DIR)}'


class ModelGeneralV3XLA(BaseModel):
    modelname = "model_general_v3_xla"
    embeddername = 'yamnet_k2'
    digits_results = 2

    def _load_compiled_sizes(self):
        """Return frozenset of sample counts that have been XLA-compiled (any device)."""
        if not _COMPILED_JSON.exists():
            return frozenset()
        with open(_COMPILED_JSON) as f:
            data = json.load(f)
        sizes = set()
        for device_sizes in data.values():
            sizes.update(device_sizes)
        return frozenset(sizes)

    def initialize(self):
        # Ensure XLA cache env var is set before any jit_compile=True call.
        os.environ['TF_XLA_FLAGS'] = (
            f'--tf_xla_persistent_cache_directory={str(_XLA_CACHE_DIR)}'
        )

        self.embedder.initialize()

        import tensorflow as tf
        from keras.layers import TFSMLayer

        dir_model = str(Path(__file__).parent)
        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')

        # Capture references for the XLA function closure
        _embedder = self.embedder.model
        _classifier = self.model

        @tf.function(jit_compile=True)
        def _predict_xla(samples):
            embeddings = _embedder(samples)['global_average_pooling2d']
            return _classifier(embeddings)['dense']

        self._predict_xla = _predict_xla

        # Standard (non-XLA) path for prewarm dummy and short last chunks
        def _predict_tf(samples):
            embeddings = _embedder(samples)['global_average_pooling2d']
            return _classifier(embeddings)['dense']

        self._predict_tf = _predict_tf

        self._compiled_sizes = self._load_compiled_sizes()

    def precompile(self, chunklength, device='GPU'):
        """Compile the XLA kernel for the given chunklength and device.

        Requires initialize() to have been called first. Skips if this
        (chunklength, device) combination is already recorded in
        xla_cache/compiled.json. Writes/updates that file on completion.

        Args:
            chunklength: Audio chunk length in seconds.
            device: 'GPU' or 'CPU'.
        """
        import numpy as np
        import tensorflow as tf

        samplerate = self.embedder.samplerate
        framelength_s = self.embedder.framelength_s

        chunklength_rounded = setup_chunklength(chunklength, framelength_s, self.embedder.digits_time)
        if chunklength_rounded != chunklength:
            print(f"precompile: chunklength {chunklength}s rounded to {chunklength_rounded}s (nearest frame boundary)")
        n_samples = int(chunklength_rounded * samplerate)

        device_key = device.upper()

        # Load existing compiled state
        if _COMPILED_JSON.exists():
            with open(_COMPILED_JSON) as f:
                data = json.load(f)
        else:
            data = {}

        if n_samples in data.get(device_key, []):
            print(f"precompile: already precompiled for chunklength {chunklength_rounded}")
            return

        device_tf = '/GPU:0' if device_key == 'GPU' else '/CPU:0'
        dummy = np.zeros(n_samples, dtype=np.float32)
        with tf.device(device_tf):
            _ = self._predict_xla(tf.constant(dummy))

        # Record to JSON
        os.makedirs(_XLA_CACHE_DIR, exist_ok=True)
        if device_key not in data:
            data[device_key] = []
        data[device_key].append(n_samples)
        with open(_COMPILED_JSON, 'w') as f:
            json.dump(data, f, indent=2)

        self._compiled_sizes = self._load_compiled_sizes()

    def predict(self, audiosamples):
        import tensorflow as tf

        if not isinstance(audiosamples, tf.Tensor):
            audiosamples = tf.constant(audiosamples)

        if audiosamples.shape[0] not in self._compiled_sizes:
            return self._predict_tf(audiosamples)

        result = self._predict_xla(audiosamples)
        return result
