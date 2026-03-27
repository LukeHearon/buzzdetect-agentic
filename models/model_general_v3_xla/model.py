import os
from pathlib import Path

import src.config as cfg
from src.inference.models import BaseModel

# XLA compilation cache lives alongside this model
_XLA_CACHE_DIR = str(Path(__file__).parent / 'xla_cache')

# Set before TF is imported so XLA finds the persistent cache on first compile.
# worker.py previously imported TF at module level; we removed that import so
# this module-level assignment now runs before TF initializes.
os.environ['TF_XLA_FLAGS'] = f'--tf_xla_persistent_cache_directory={_XLA_CACHE_DIR}'


class ModelGeneralV3XLA(BaseModel):
    modelname = "model_general_v3_xla"
    embeddername = 'yamnet_k2'
    digits_results = 2

    # Only use XLA for exact precompiled sizes. Anything else (partial last
    # chunks, prewarm dummy) goes to the standard TF path to avoid triggering
    # slow XLA recompilations during the eval run.
    # Precompiled: 200s=3194880, 600s=9600000, 1200s=19200000 samples
    _XLA_PRECOMPILED_SIZES = frozenset([3194880, 9600000, 19200000])

    def initialize(self):
        # Point XLA to the pre-compiled cache directory.
        # Must happen before the first jit_compile=True call.
        os.environ['TF_XLA_FLAGS'] = (
            f'--tf_xla_persistent_cache_directory={_XLA_CACHE_DIR}'
        )

        self.embedder.initialize()

        import tensorflow as tf
        from keras.layers import TFSMLayer

        dir_model = os.path.abspath(
            os.path.join(cfg.DIR_MODELS, 'model_general_v3')
        )
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

    def predict(self, audiosamples):
        import tensorflow as tf

        if not isinstance(audiosamples, tf.Tensor):
            audiosamples = tf.constant(audiosamples)

        if audiosamples.shape[0] not in self._XLA_PRECOMPILED_SIZES:
            return self._predict_tf(audiosamples)

        result = self._predict_xla(audiosamples)
        # Force GPU sync: prevents pipeline starvation and write-thread overhead
        # that made 09_xla_per_worker SLOWER (+7.4%)
        return result.numpy()
