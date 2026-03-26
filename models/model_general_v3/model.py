import os

import numpy as np

import src.config as cfg
from src.inference.models import BaseModel


class ModelGeneralV3(BaseModel):
    modelname = "model_general_v3"
    embeddername = 'yamnet_k2'
    digits_results = 2

    def initialize(self):
        # Note: embedder is NOT initialized here — its class-level attributes
        # (samplerate, framelength_s, etc.) are sufficient. Inference is fully
        # handled by the combined ONNX session below, avoiding TF GPU init.

        import onnxruntime as ort
        onnx_path = os.path.abspath(os.path.join(cfg.DIR_MODELS, self.modelname, 'model_combined.onnx'))

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Prefer CUDA, fall back to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, audiosamples):
        """
        Generate predictions for audio data using combined ONNX model.

        Args:
            audiosamples: numpy array of audio samples at self.embedder.samplerate

        Returns:
            numpy array of class predictions
        """
        if not isinstance(audiosamples, np.ndarray):
            audiosamples = np.asarray(audiosamples, dtype=np.float32)
        elif audiosamples.dtype != np.float32:
            audiosamples = audiosamples.astype(np.float32)

        return self._session.run([self._output_name], {self._input_name: audiosamples})[0]
