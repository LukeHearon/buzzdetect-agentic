# Dependency Requests

Dependencies that would unlock specific optimizations, in priority order.

---

## ~~1. TensorFlow built with TensorRT support~~ — ruled out

TF-TRT requires a per-GPU-architecture TF build and produces non-portable TRT engines.
This project must be deployable without per-GPU builds. **Do not pursue.**

---

## 1. `cupy` (lower priority)

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
