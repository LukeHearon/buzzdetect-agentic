import os

import src.config as cfg
from src.inference.models import BaseModel


class ModelGeneralV3ORT(BaseModel):
    """model_general_v3 using ONNX Runtime with CUDAExecutionProvider for the classifier."""

    modelname = "model_general_v3_ort"
    embeddername = "yamnet_k2_ort"
    digits_results = 2

    def initialize(self):
        import onnxruntime as ort

        self.embedder.initialize()

        onnx_path = os.path.join(cfg.DIR_MODELS, 'model_general_v3', 'model_general_v3.onnx')
        self.model = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self._input_name = self.model.get_inputs()[0].name
        self._output_name = self.model.get_outputs()[0].name

    def predict(self, audiosamples):
        embeddings = self.embedder.embed(audiosamples)
        results = self.model.run([self._output_name], {self._input_name: embeddings})[0]
        return results
