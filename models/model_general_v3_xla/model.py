from pathlib import Path

from src.inference.models import BaseModel
from src.utils import setup_chunklength


class ModelGeneralV3XLA(BaseModel):
    modelname = "model_general_v3_xla"
    embeddername = 'yamnet_k2'
    digits_results = 2

    # Class-level cache so multiple initialize() calls within one process
    # share a single @tf.function object (and thus one XLA compilation slot).
    _xla_fn_cache: dict = {}

    def initialize(self):
        self.embedder.initialize()

        import tensorflow as tf
        from keras.layers import TFSMLayer

        dir_model = str(Path(__file__).parent)
        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')

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

    def precompile(self, chunklength):
        """Warm the XLA kernel for this chunklength (in-process, in-memory only).

        Runs a dummy inference so XLA compiles before workers start.
        No disk cache is written; compilation is per-process.
        """
        import numpy as np
        import tensorflow as tf

        samplerate = self.embedder.samplerate
        framelength_s = self.embedder.framelength_s

        chunklength_rounded = setup_chunklength(chunklength, framelength_s, self.embedder.digits_time)
        n_samples = int(chunklength_rounded * samplerate)
        self._predict_xla(tf.constant(np.zeros(n_samples, dtype=np.float32)))

    def predict(self, audiosamples):
        import tensorflow as tf

        if not isinstance(audiosamples, tf.Tensor):
            audiosamples = tf.constant(audiosamples)

        return self._predict_xla(audiosamples)
