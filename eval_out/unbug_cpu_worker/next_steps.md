# Next Steps

## What we learned

**Bug fix shifts the bottleneck to GPU inference.** With the CPU analyzer bug fixed, only the GPU runs. Per-chunk inference dropped from 0.101s (more_streamers, GIL-contended) to 0.067s (-12% vs baseline 0.077s). The GPU runs cleaner without a CPU thread competing.

**Wall-clock inference is now 44.1s** (654 chunks × 0.067s / 1 analyzer). Streaming is 24.4s (6 streamers). GPU is the bottleneck.

**Overall is ~54s in both more_streamers and unbug_cpu_worker** — two different paths to the same result. The CPU-in-baseline was inadvertently helping by providing a second analyzer, masking the streaming bottleneck. With the fix, the GPU is the sole bottleneck.

**To go faster from here, we must speed up GPU inference.**

## What to try next

1. **Speed up per-chunk GPU inference.** At 0.067s/chunk × 654 chunks = 44.1s wall-clock, this is the sole bottleneck. Options:
   - **ONNX conversion** (explicitly allowed in CLAUDE.md): convert yamnet_k2 + dense model to ONNX and run with `onnxruntime-gpu`. ONNX Runtime typically has lower overhead than TF SavedModel inference and better CUDA kernel fusion. Could give 20-40% speedup on inference.
   - **XLA JIT compilation**: wrap `predict()` in `tf.function(jit_compile=True)` — compiles the TF graph with XLA for GPU. Could give 10-20% speedup with zero result change.
   - **TensorRT**: convert to TensorRT for maximum GPU throughput on NVIDIA hardware. Complex but high potential.

2. **Add a second GPU analyzer back, intentionally.** With the bug fixed, `analyzers_cpu=1, gpu=True` would give 1 CPU + 1 GPU — which is what the baseline accidentally had but now controlled. Per the more_streamers analysis, this halved inference wall-clock time. Whether the CPU helps depends on whether it actually runs faster than it waits. Worth testing.

3. **Multiprocessing streamers.** Still valid: moving streamers to processes would eliminate GIL entirely. With GPU as the bottleneck now, this matters less immediately, but if GPU inference is sped up and streaming becomes a bottleneck again, this is the next lever.

4. **Larger n_streamers is irrelevant now** — 6 streamers at 26.8 chunks/s already outpaces 1 GPU analyzer at 14.9 chunks/s. No benefit to more streamers unless GPU gets faster.
