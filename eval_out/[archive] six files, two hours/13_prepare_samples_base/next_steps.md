# Next Steps

## What we learned

Moving `tf.constant()` conversion from the GPU worker into the streamer threads
is a real win: **FASTER, new best at 18.5s** (~3% faster than previous best of 19.1s,
~3.6% faster than 01_baseline).

### Key numbers (200s, 6 streamers)
- overall: 18.5s (new best)
- inference: 10.6s, 47.8ms/chunk mean — GPU is very fast
- inference/emptyqueue: 0.11s — GPU is barely starving (excellent!)
- audio_io/fullqueue: 13.0s total — streamers are now the clear bottleneck
- read: 36.5s (197% of overall), resample: 25.0s (135%) across 6 streamers

### GPU is outpacing streaming

With tensor pre-conversion in the streamer, the GPU worker gets pre-built tensors
and burns through chunks fast. But the streamers are heavily loaded:
- Each streamer does: read + resample + tf.constant()
- 6 streamers × ~290ms per chunk = supply rate of ~48ms/chunk
- GPU consuming at ~48ms/chunk: they're matched, but only barely

The `fullqueue` total (13s, 70% of overall!) shows streamers frequently blocking
on a full queue — i.e., they're saturated, not the GPU.

### XLA with prepfunc (13_prepare_samples_xla)
- XLA overall: 19.7s — SLOWER than base with prepfunc
- XLA inference: 9.0s (faster per-chunk), but emptyqueue: 1.1s (GPU starving)
- XLA flips bottleneck back to streamers; base model is now the better option

## Concern: prepfunc may be serializing IO and conversion

Currently each streamer does read → resample → tf.constant() sequentially.
The `tf.constant()` call copies a large numpy array into TF memory — for a 200s
chunk at 16kHz that's ~12MB. This CPU copy takes a few ms but happens while
the streamer is NOT doing IO or resampling.

**Potential optimization**: split each streamer into two stages with a local
buffer queue:
- Stage 1 thread: read + resample (IO-bound)
- Stage 2 thread: tf.constant() conversion (CPU memory-bound)

This would allow IO to continue in parallel with conversion, potentially
keeping the queue fuller.

## Other directions

### Reduce per-chunk streaming time
- Read + resample dominate (197% + 135% of overall across 6 streamers)
- soxr quality could be reduced (HQ → MQ or LQ) at potential accuracy cost
- Prefetching / async IO might help the read phase

### Accept current state and look elsewhere
- write_io is very small (7% formatting + 5% write)
- The remaining opportunity is almost entirely in the streaming pipeline
- Could try increasing effective streamer count via the two-stage approach above
