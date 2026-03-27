# Next Steps after 07_onnx_gpu

## Result
SLOWER (+15.8%). ONNX Runtime CUDA EP was 40.6% slower on inference vs TF native GPU.

## Why it was slower
TF 2.x with SavedModels runs on GPU using XLA-compiled kernels and cuDNN — already
highly optimized for the specific ops in YAMNet (STFT, MelSpectrogram, depthwise conv).
ORT CUDA EP can't match this for models originally trained and optimized in TF.
Additionally, there's overhead converting between TF and ORT memory representations.

## What to try next

### TF-TRT (highest potential, blocked by missing libnvinfer)
TF was not compiled with TensorRT support (`RuntimeError: Tensorflow has not been built
with TensorRT support`). TensorRT pip install failed due to SSL cert errors in the
sandbox network. If TRT becomes available, this is the highest-priority optimization
— it rewrites the TF graph with fused TRT kernels at the SavedModel level.

### TF graph optimization
- `tf.function` with `jit_compile=True` (XLA JIT) on the inference call — may already
  be happening but worth checking
- `@tf.function` tracing with concrete input shapes to eliminate dynamic dispatch overhead

### Model quantization via TF Lite
Convert to TFLite with GPU delegate — TFLite GPU delegate often outperforms full TF on
inference-only workloads, especially on lower-end GPUs like GTX 1650.

### Batching strategy
Profile whether the GPU is actually saturated (GPU util %) during inference. If utilization
is below ~80%, there may be room to increase batch size or overlap inference with I/O.

### Pruning/distillation
If the model can be made smaller (fewer parameters) without affecting results, inference
gets faster. This is a bigger project but potentially high-yield.

## What to avoid
- ONNX Runtime with any EP — TF native is faster for these models
- More audio I/O optimization — GPU is the bottleneck, not I/O
