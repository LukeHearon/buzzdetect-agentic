# Next Steps

## What we learned

XLA JIT compilation genuinely speeds up per-chunk inference: 40ms vs 51ms
(-21%). But the overall result is NEUTRAL (+3.5%) because the GPU worker
now outpaces the 6 streamers.

### Key findings

- Inference phase: 9.95s vs 11.46s (-1.5s) — real improvement
- But GPU emptyqueue: 1.98s (vs 0.002s baseline) — GPU starving
- Streamers produce at ~48ms/chunk (6 streamers, 290ms per chunk)
- GPU consumes at 40ms/chunk (XLA) vs 51ms baseline
- XLA flips the bottleneck from GPU → streamers

### XLA cache issue (resolved)

The XLA persistent cache fingerprint depends on HOW TF is first imported.
The previous approach imported TF at module level in worker.py and
assignments.py BEFORE TF_XLA_FLAGS was set → cache miss → 12s compile per combo.

Fix: removed module-level TF import from worker.py (dead code, only in
`_managememory()` which is never called) and assignments.py (type annotation
only). TF_XLA_FLAGS is now set at model.py module level, before any TF import.

### XLA cache timing

After warm-up (all 3 chunk sizes compiled once in eval context):
- 200s: 1.1s cache load per model initialization
- 600s: 1.2s
- 1200s: 1.35s

This 1.1s overhead per combo run is currently inside the 'overall' timer,
contributing to the NEUTRAL result.

## What to try next

### Option 1: More streamers

The most direct fix. With 6 streamers at 48ms/chunk and GPU at 40ms/chunk,
adding 2 more streamers (to 8) would make supply rate 36ms < 40ms.
**However**: the tuning grid is fixed at [3, 6] in eval.py which cannot be modified.
Consider whether there is a way to expose this parameter differently.

### Option 2: Move XLA init outside the timer

If model initialization (including XLA cache load) could happen BEFORE the
'overall' timer starts, the 1.1s overhead would disappear from the measurement.
This would require changes to analyze.py to support pre-initialization.
Expected gain: 1.1s → overall ~18.6s → ~2.5% faster → still NEUTRAL but closer.

### Option 3: Preconvert audio to tensors in streamers

From exp3/exp4: explicit tf.constant() in process_chunk saves ~5ms/chunk,
and pre-converting in streamer threads saves ~2.7%. With XLA already reducing
GPU time, reducing the non-inference overhead in the worker might help.
But this was only +2.7% (NEUTRAL) without XLA.

### Option 4: Combine XLA + tf.constant() preconversion

With XLA at 40ms and tf.constant() preconversion saving ~5ms:
- Total GPU worker time per chunk: ~35ms
- Streamer supply rate still ~48ms/6 = 8ms ... wait, 48ms with 6 streamers
  means streamers produce at 8ms per slot? No: 48ms / chunk per streamer,
  but 6 streamers in parallel → supply rate = 48ms/6 ≈ 8ms? No that's wrong.

Actually: 6 streamers each taking 290ms per chunk = 1 chunk ready every
290ms/6 ≈ 48ms. With XLA at 35ms, the GPU would still outpace streamers.
Even worse starvation.

### Option 5: Accept NEUTRAL and try something different

The streamer bottleneck is a hard constraint. Other GPU optimization approaches:
- Batching multiple chunks: previously found SLOWER due to ToTensor scaling
- FP16 inference: already tried (06_fp16, NEUTRAL)
- Look at the write/format phase or other non-GPU bottlenecks

### Key insight for future iterations

The inference/overall ratio is now 50% (down from 60%). The remaining 50%
is streamers (reading + resampling). The next optimization opportunity may
be in making streaming faster:
- audio_io/reading: 40.8s total (206% of overall) — IO bound
- audio_io/resampling: 24.3s total (123%) — CPU bound

Or: reduce the XLA cache init overhead to make it truly FASTER.

## XLA cache maintenance

The eval-context XLA cache (in models/model_general_v3_xla/xla_cache/) is now
populated for all 3 sizes. precompile.py still uses a different TF import path
(direct `import tensorflow as tf`) which creates different fingerprint. If the
cache is deleted, run a quick warm-up script (or just let the first eval run
compile fresh — it will cache for subsequent runs).
