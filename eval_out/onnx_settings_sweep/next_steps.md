# Next Steps

## What we learned

**ONNX inference is not faster than TF** in this GPU context. Raw inference per chunk:
- TF (unbug_cpu_worker): 0.067s/chunk
- ONNX: 0.070s/chunk (slightly worse)

The expected GIL-release benefit of ONNX was measured in isolation (ONNX 7% faster under
heavy thread contention), but didn't materialize in the actual eval because real streamers
do significant I/O (which releases GIL naturally).

**The real win was eliminating the TF tensor `.numpy()` call in the write path.** The old
code had `a_chunk.results.numpy()` inside write_io/formatting. For TF tensors, this forces
a GPU→CPU synchronization per chunk. Over 654 chunks, this was ~7-8s overhead.

With ONNX (or with the np.asarray fix), results are already on CPU as numpy arrays, so no
transfer needed.

**Note on sweep tests**: The 5 onnx_sweep_* tests in eval_out/ were run before removing the
`embedder.initialize()` call. Those results (55-56s) are NOT representative of the current
code; they show an intermediate state with extra overhead.

## Current bottlenecks

- Inference: 0.070s/chunk × 654 = 45.8s (81% of overall 53s)
- GPU remains the bottleneck; streamers outpace it at ~29 chunks/s vs GPU's ~14 chunks/s

## What to try next

1. **Fix write path for TF models too**: The `np.asarray()` fix in write/worker.py is
   already in place. But we could also try calling `.numpy()` INSIDE the inference worker
   (before putting on q_write) to get the GPU sync done during inference time instead of
   write time — potentially allowing the GPU to start the next chunk earlier.

2. **Multiprocessing streamers**: Still the highest-potential optimization. GIL contention
   adds ~0.003s/chunk to ONNX inference (measured at 0.067s without streamers, 0.070s with).
   Moving streamers to processes would let inference run at full speed (~0.040s/chunk without
   GIL, from TF benchmark). Projected: 0.040 × 654 = 26s inference → ~35-37s overall.
   Key challenge: Coordinator uses threading.Queue (not multiprocessing-safe), Profiler has
   threading.Lock. Would need StreamerProxy class + multiprocessing.Queue for q_stream/q_analyze.

3. **Return numpy directly from TF models**: Instead of using ONNX, we could modify the
   existing TFSMLayer code to call `.numpy()` in `model.predict()` and return numpy arrays.
   This would give the same write-path benefit as ONNX without the inference overhead.
   Combined with XLA (if compile overhead is amortized), this could be significant.

4. **Reduce per-chunk overhead**: At 0.070s/chunk × 654 chunks, there's still room to
   optimize. Look at ONNX IOBinding with persistent GPU buffers to reduce CPU-GPU transfer.

5. **XLA with persistent cache**: XLA compile overhead was ~11s first run, ~1.1s from cache.
   The cache DID NOT load correctly in fresh processes (1.1s was in-process reuse, not real
   disk cache). XLA is not viable unless the cache loading issue is resolved.
