import json
import os
from pathlib import Path

from src.inference.models import BaseModel
from src.utils import setup_chunklength

# XLA compilation cache lives alongside this model
_XLA_CACHE_DIR = Path(__file__).parent / 'xla_cache'

# Set before TF is imported so XLA finds the persistent cache on first compile.
# worker.py previously imported TF at module level; we removed that import so
# this module-level assignment now runs before TF initializes.
os.environ['TF_XLA_FLAGS'] = f'--tf_xla_persistent_cache_directory={str(_XLA_CACHE_DIR)}'


class ModelGeneralV3XLA(BaseModel):
    modelname = "model_general_v3_xla"
    embeddername = 'yamnet_k2'
    digits_results = 2
    _xla_fn_cache: dict = {}

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

        # Reuse a single @tf.function object across all initialize() calls so that
        # all workers share one XLA cache slot per input shape rather than each
        # creating their own (which would generate distinct cache keys and trigger
        # redundant recompilations).
        _embedder = self.embedder.model
        _classifier = self.model

        cache_key = (id(_embedder), id(_classifier))
        if cache_key not in ModelGeneralV3XLA._xla_fn_cache:
            @tf.function(jit_compile=True)
            def _predict_xla(samples):
                embeddings = _embedder(samples)['global_average_pooling2d']
                return _classifier(embeddings)['dense']

            ModelGeneralV3XLA._xla_fn_cache[cache_key] = _predict_xla

        self._predict_xla = ModelGeneralV3XLA._xla_fn_cache[cache_key]

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
            print(f"precompile: chunklength {chunklength}s rounded to {chunklength_rounded}s (nearest whole frame length)")
        n_samples = int(chunklength_rounded * samplerate)
        dummy = np.zeros(n_samples, dtype=np.float32)
        self._predict_xla(tf.constant(dummy))

        # device_key = device.upper()
        #
        # device_tf = '/GPU:0' if device_key == 'GPU' else '/CPU:0'
        # with tf.device(device_tf):
        #     _ = self._predict_xla(tf.constant(dummy))

    def predict(self, audiosamples):
        import tensorflow as tf

        if not isinstance(audiosamples, tf.Tensor):
            audiosamples = tf.constant(audiosamples)

        return self._predict_xla(audiosamples)
