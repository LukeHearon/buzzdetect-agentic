# Next Steps

## What we learned

Two-stage streaming is SLOWER (+10.7%) and the dynamics are **scale-independent** —
the same bottleneck structure applies at 3,000 hours as on the test set.

### Key numbers (200s, 6 streamers)
- overall: 20.5s (vs 18.5s baseline)
- read: 197ms/chunk (was 164ms — IO got SLOWER per chunk)
- fullqueue: 83.4s total (was 13.0s — 6.4× increase in blocking)
- emptyqueue: 0.004s (GPU barely ever starving — excellent, but didn't help)

### Why it failed

1. **Thread contention slows IO**: 12 active threads (6 IO + 6 convert) vs 6 single
   threads causes more GIL/memory-bus contention. Per-chunk read time regressed
   164ms → 197ms. The IO threads are actually slower, not faster.

2. **Backpressure collapses the overlap**: When GPU consumption rate limits q_analyze,
   convert threads block on q_analyze → local_q fills → IO thread blocks anyway.
   The intended IO/convert overlap disappears exactly when the system is under load.

3. **Not a scale issue**: The per-chunk IO regression and backpressure collapse are
   structural. On 3,000 hours you'd have 12 threads competing for the same duration,
   same slowdown ratios.

### What the profile reveals about the real bottleneck

With prepfunc in streamers (13_prepare_samples_base), we're at near-balance:
- GPU consuming at ~48ms/chunk
- 6 streamers supplying at ~48ms/chunk
- Any approach that adds overhead to the streaming path will push us into SLOWER

The system is tightly balanced. Small per-chunk regressions matter a lot.

## What to try next

### Reduce per-chunk streaming cost directly

The real opportunity is making each chunk faster, not adding parallelism:

1. **soxr quality reduction**: HQ → MQ or LQ resampling. The next_steps from 13 flagged
   this. Resample is 123ms/chunk (133% of overall across 6 streamers). MQ might cut
   this significantly. Risk: accuracy change — need to verify results check passes.

2. **soundfile → alternative reader**: sf.SoundFile read is ~197ms/chunk. Could a
   faster audio decoder (e.g., audioread with ffmpeg backend, or pydub) help? Profile
   shows read is 213% of overall.

3. **Lower resample rate**: If the model accepts a slightly different input rate and
   the audio is already close, we might avoid full resample. Probably not viable
   without accuracy impact.

### Look outside the streaming pipeline

Streaming is 213% (read) + 133% (resample) of overall = effectively capped. The GPU
is well-fed (emptyqueue ~0). We might be near the practical minimum for this hardware
config with the current model. Consider:

- Different model format (ONNX was +15% worse in 01_onnx, but that used CPU)
- Investigate whether the model itself could be made faster (quantization, pruning)
- Accept 18.5s as near-optimal and document the hardware ceiling

### Most promising: soxr MQ quality

Try `quality='MQ'` in the soxr.resample call. It's a single-line change, easy to
revert if results fail. Prior tests haven't tried this. Resample is the second-largest
cost and MQ could be 2–3× faster than HQ.
