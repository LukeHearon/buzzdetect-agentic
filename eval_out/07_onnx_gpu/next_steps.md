# Next Steps after 07_onnx_gpu

## Result
SLOWER (+15.8%). ONNX Runtime CUDA EP was 40.6% slower on inference vs TF native GPU.

## Why it was slower
TF 2.x with SavedModels runs on GPU using XLA-compiled kernels and cuDNN — already
highly optimized for the specific ops in YAMNet (STFT, MelSpectrogram, depthwise conv).
ORT CUDA EP can't match this for models originally trained and optimized in TF.
Additionally, there's overhead converting between TF and ORT memory representations.

## What to try next

### TF-TRT (highest potential — needs TF built with TRT support)
TF was not compiled with TensorRT support (`RuntimeError: Tensorflow has not been built
with TensorRT support`). The `tensorrt` Python package (10.16.0.72) IS now installed,
but TF-TRT also requires TF itself to be compiled with `-Dtensorrt=true`. A pre-built
`tensorflow-gpu` wheel or an NVIDIA Docker image (e.g. `nvcr.io/nvidia/tensorflow`) would
unblock this. See REQUESTS.md. This is the highest-priority next step if the dependency
can be satisfied — it rewrites the TF graph with fused TRT kernels at the SavedModel level.

### XLA JIT compilation
`tf.config.optimizer.set_jit(True)` enables XLA JIT globally — it recompiles TF ops into
fused XLA kernels at runtime. Quick to try (one line in worker.py), no new dependencies.
XLA JIT is NOT currently enabled (confirmed by `tf.config.optimizer.get_jit()` returning
empty). Worth trying as experiment 08.

### TFLite with GPU delegate
Convert models to TFLite format and run via the TFLite GPU delegate. TFLite GPU delegate
is often faster than full TF on lower-end GPUs for inference-only workloads. Requires
`tflite-runtime` or a TF build with TFLite GPU delegate support. See REQUESTS.md.

### Batching strategy
Profile whether the GPU is actually saturated (GPU util %) during inference. `nvidia-smi dmon`
during a run would show this. If utilization is below ~80%, larger chunk sizes (already swept
by auto-tune up to 1200s) or overlapping inference with I/O could help.

### Pruning/distillation
If the model can be made smaller (fewer parameters) without affecting results, inference
gets faster. This is a bigger project but potentially high-yield.

## What to avoid
- ONNX Runtime with any EP — TF native is faster for these models
- More audio I/O optimization — GPU is the bottleneck, not I/O
