# GPU Utilization Profiling Scratch

## Context
- Baseline best: 05_soxr_only @ 19.077s overall, 200s_6str settings
- GPU (GTX 1650) is ~10% SM average during inference (80% peak during bursts)
- Inference is wall-clock bottleneck: 60% of overall
- `audio_io/fullqueue` = 64% confirms GPU is slower than I/O

## Key Findings

### Timing (exp1b, corrected)
- Standard `model.predict(numpy)`: **~49ms per 200s chunk** (synchronous, TF eager blocks)
- Pre-built tensor `model.predict(tensor)`: **~39ms per chunk** — 10ms faster (saves TFSMLayer numpy-handling overhead)
- Back-to-back dispatch does NOT pipeline: N chunks take N × 49ms (no GPU async overlap with standard TF)

### XLA JIT — targeted vs global (exp5)
- `@tf.function(jit_compile=True)` with fixed input shape: **40.2ms per chunk** (microbenchmark)
- BUT: XLA dispatches asynchronously! `predict_xla()` returns in ~27ms, GPU finishes async in ~40ms
- This means the GPU worker's hot loop runs at ~27ms/chunk vs 49ms baseline — nearly 2x faster dispatch
- Full pipeline test: **6.8% faster overall** (inference phase: 11.96s → 6.04s)
- Only works if XLA is pre-compiled before the 'overall' timer starts

### XLA compilation overhead (exp5, exp6)
- First compile (no cache): **12.2 seconds** — kills any benefit inside the eval timer
- Disk cache (`TF_XLA_FLAGS=--tf_xla_persistent_cache_directory`): **1.1s to load** (10x faster)
- Cache survives across Python processes ✓
- Cache is per input shape: need to pre-compile 200s, 600s, 1200s separately

### Streamer tensor pre-loading (exp3, exp4)
- Explicit `tf.constant()` in GPU worker before predict: **5% faster** in pipeline
- Pre-converting in streamer threads: **2.7% faster** overall (absorbed into queue-wait dead time)
- Only ~2.7% because GIL contention limits true parallelism of tf.constant() across threads

### Batch accumulation (exp1, exp1b)
- Concatenating N×200s chunks into one model call: **SLOWER** (non-linear ToTensor scaling)
- XLA also doesn't benefit from batch accumulation

## Implementation Plan for eval

### Model: `model_general_v3_xla`
1. Sets `TF_XLA_FLAGS` to persistent cache dir in `_managememory()`
2. Creates `@tf.function(jit_compile=True)` in `initialize()`
3. Routes small inputs (prewarm dummy) to standard TF, full chunks to XLA
4. Pre-compiled cache for 200s, 600s, 1200s stored in model dir

### Why this beats global XLA (08_xla_jit was +22.9% SLOWER)
- Global JIT: TFSMLayer traces dynamically-shaped graphs → constant recompilation overhead
- Targeted JIT: one function, fixed shape, compiled once → no retrace overhead
- Disk cache: each new model instance loads in 1.1s instead of 12s

### Expected result
- Per chunk: 49ms → 27ms dispatch (GPU finishes 40ms async)
- First chunk per combo: 1.1s cache load (vs 0ms normally)
- 222 chunks at 200s: 1.1 + 221 × 27ms = 7.1s inference (vs 11.46s)
- Overall estimate: ~17.7s (vs 19.077s) = ~7% faster → FASTER verdict
