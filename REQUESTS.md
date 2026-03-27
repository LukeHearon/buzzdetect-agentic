# Dependency Requests

Dependencies that would unlock specific optimizations, in priority order.

---

## 1. TensorFlow built with TensorRT support (highest priority)

**What:** A TF build where `tf.python.compiler.tensorrt` actually works — i.e., `TrtGraphConverterV2` doesn't raise `RuntimeError: Tensorflow has not been built with TensorRT support`.

**Why:** TF-TRT rewrites the SavedModel graph with fused TRT kernels. This is the only approach that optimizes the existing TF models *at the graph level* without switching runtimes. It's the highest-potential inference speedup available for these models on a CUDA GPU.

**How to get it:** Either install a pre-built wheel like `tensorflow-gpu` that includes TRT support, or rebuild TF with `-Dtensorrt=true`. Alternatively, a Docker image like `nvcr.io/nvidia/tensorflow` ships with this already compiled in.

**Note:** The `tensorrt` Python package (10.16.0.72) is now installed in the venv. The missing piece is TF's own compiled-in TRT integration.

---

## 2. `cupy` (lower priority)

**What:** `pip install cupy-cuda12x`

**Why:** Would allow GPU-accelerated resampling and audio preprocessing — keeping data on the GPU between the audio I/O and inference stages, avoiding CPU↔GPU transfers. Only worth trying if inference gets faster first (audio I/O is not the current bottleneck).

---

## 3. `torch` + `torch2trt` (alternative TRT path, low priority)

**What:** PyTorch + the `torch2trt` converter.

**Why:** An alternative route to TRT-optimized inference that doesn't require a special TF build — convert the model weights to PyTorch, then to TRT engines via `torch2trt`. High complexity and risk of result mismatch. Only worth trying if TF-TRT remains unavailable.

---

## Already installed / not needed

- `tensorrt` Python package — installed (10.16.0.72) but currently unused without TF-TRT support
- `onnxruntime` with CUDA EP — tested in 07_onnx_gpu, was slower than TF native
- `tf2onnx` — available, used for 07_onnx_gpu model conversion
