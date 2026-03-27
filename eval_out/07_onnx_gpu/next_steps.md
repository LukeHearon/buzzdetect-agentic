# Next Steps after 07_onnx_gpu

## Result
SLOWER (+15.8%). ORT CUDA EP was 40.6% slower on inference vs TF native GPU.

## Correction: 01_onnx was already GPU
The 05_soxr_only next_steps.md incorrectly described 01_onnx as "ONNX CPU." It was
not — 01_onnx used `model_general_v3_onnx` with `CUDAExecutionProvider` and `gpu: true`,
running a single combined ONNX model (YAMNet + classifier fused) on GPU. It was +11.5%
SLOWER than baseline. 07_onnx_gpu replicated that approach but *worse* — two separate
ORT sessions with an extra numpy handoff between them added overhead.

## Why ORT is slower than TF native on these models
TF 2.x with SavedModels uses XLA-compiled kernels and cuDNN ops already tuned for
YAMNet's STFT/MelSpectrogram/depthwise-conv graph. ORT CUDA EP does not match this
for models that were trained and shaped around TF's graph execution. This holds whether
the ONNX model is split (07_onnx_gpu) or fused (01_onnx).


## What to try next

### ~~TF-TRT~~ — ruled out
TF-TRT requires TF compiled with TensorRT support, and TRT engines are GPU-architecture-
specific. This project needs to be portable and deployable without per-GPU builds.
TF-TRT is untenable. Do not pursue.

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
