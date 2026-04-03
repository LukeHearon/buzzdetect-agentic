# 07_onnx_gpu

Replaced both TF inference models (YAMNet embedder + classifier) with ONNX Runtime
sessions using `CUDAExecutionProvider`. Created new `yamnet_k2_ort` embedder and
`model_general_v3_ort` model variants. ONNX models exported with tf2onnx (opset 17).

Hypothesis: ONNX Runtime's CUDA EP would run the models faster than TF's native GPU
runtime, similar to how ORT often wins on inference benchmarks.
