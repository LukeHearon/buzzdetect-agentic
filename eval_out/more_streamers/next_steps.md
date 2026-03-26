# Next Steps

## What we learned

**Bottleneck was streaming, not inference.** The baseline had 4 streamers producing at ~17.9 chunks/s while 2 analyzers (1 CPU + 1 GPU — see note below) consumed at ~26 chunks/s. This created 491 buffer bottleneck events with analyzers idle ~38% of the time.

Increasing to 6 streamers (1 per audio file) brought production to ~26.8 chunks/s, matching consumption and yielding a 16% speedup.

**Surprising: per-chunk times INCREASED with more streamers.** Inference went from 0.077s to 0.101s/chunk (+32%), and audio I/O from 0.224s to 0.346s/chunk (+54%). This is GIL contention — more Python threads slowing each other. Despite this, overall was faster because analyzers no longer wait idle.

**Important architectural finding:** The code in `_launch_analyzers()` uses `for a in range(self.coordinator.analyzers_total)` to create CPU analyzers, where `analyzers_total = analyzers_cpu + analyzer_gpu`. With `analyzers_cpu=0, gpu=True`, `analyzers_total=1`, so 1 CPU analyzer is ALWAYS created even when `analyzers_cpu=0`. This means the baseline actually runs 1 CPU + 1 GPU analyzer (2 total). This is not intuitive from the settings.

## What to try next

1. **Tune the GIL contention problem.** The 32% slowdown in per-chunk inference from GIL pressure is significant. Options:
   - Try n_streamers=5 instead of 6: slightly less GIL pressure, might give better per-chunk times while still eliminating most bottlenecks
   - Use `multiprocessing` instead of `threading` for streamers: true parallelism, avoids GIL entirely. This is a larger code change but could give very large speedups.

2. **Multiprocessing for streamers** is probably the highest-potential next step. The streamers do heavy CPU work (MP3 decode + resample) that would benefit from true parallelism. Moving streamer workers from threads to processes would eliminate GIL contention between streamers and the inference threads. Expected benefit: per-chunk I/O times return to ~0.224s instead of 0.346s, AND we can have 6 truly parallel streamers.

3. **Tune per-chunk inference speed.** At 0.101s/chunk for 654 chunks with 2 analyzers, inference is now ~33s wall-clock. Options:
   - XLA/tf.function(jit_compile=True) wrapping of the predict() call
   - ONNX conversion (allowed per CLAUDE.md)
   - Prewarm with full-size chunks to avoid first-chunk retrace

4. **Note on n_streamers vs n_files:** With 6 audio files, n_streamers > 6 gives no extra active workers (extra streamers just get None from q_stream immediately). n_streamers=6 is optimal for this eval. Future evaluations with more files would benefit from more streamers.
