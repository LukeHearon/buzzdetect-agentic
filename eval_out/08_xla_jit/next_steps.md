# Next Steps after 08_xla_jit

## Result
SLOWER (+22.9%). XLA JIT made inference 22.7% slower. Worse than ONNX.

## Why XLA JIT failed
XLA JIT must recompile TF subgraphs into fused kernels at runtime. For
TFSMLayer/SavedModel graphs (YAMNet's STFT + MelSpec + depthwise conv),
this recompilation overhead dominates any kernel fusion gain. TF's default
XLA-compiled kernels (already used for GPU ops) are already near-optimal
for this architecture. Explicit JIT adds cost without benefit here.

## What to rule out
- tf.config.optimizer.set_jit(True) — makes inference ~23% slower
- ONNX Runtime (any EP) — already ruled out in 07_onnx_gpu
- TF-TRT — ruled out (requires per-GPU builds)

## What to try next

### GPU utilization / batching investigation
The next_steps from 07 suggested profiling actual GPU utilization with
`nvidia-smi dmon`. If GPU is below ~80% util, there may be room to improve
throughput. The auto-tune only sweeps chunk sizes up to 1200s — if VRAM
allows larger chunks without OOM, that could help. Worth checking.

Run `nvidia-smi dmon -s u` during an eval to see GPU utilization %.

### Concurrent inference overlapping with I/O
Currently a single GPU worker serializes inference. If I/O (read/resample)
is slow relative to inference, overlapping them could help. Check the
profiling data: if `read` + `resample` is a larger fraction of `overall`
than `inference`, this is worth pursuing.

### Model quantization (INT8/FP16 via TF)
06_fp16 was NEUTRAL (+2.2%). FP16 via direct cast did not help much.
True TF mixed-precision training/inference with `tf.keras.mixed_precision`
policy might behave differently from a simple cast.

### TFLite with GPU delegate
Convert YAMNet and classifier to TFLite format; run via TFLite GPU delegate.
TFLite GPU delegate is different from full TF GPU — uses OpenCL/Vulkan for
GPU acceleration and can be faster for inference-only workloads. Requires
`ai-edge-litert` or similar package. Worth exploring if GPU util is the limit.

### Pruning/distillation
If the model can be made smaller without affecting output, inference is faster.
Bigger project but potentially high-yield.

## Key insight
Both XLA JIT and ONNX made inference slower. TF native with SavedModel/TFSMLayer
is already highly optimized for YAMNet on GPU. The bar for beating it is high.
Focus on architectural changes (pipeline parallelism, batching) rather than
kernel-level compilation tricks.
