Ran with the CPU-analyzer bug fix applied (analyzers_cpu=0 now correctly spawns 0 CPU workers) alongside n_streamers=6, stream_buffer_depth=6 from more_streamers.

Result is nearly identical to more_streamers (~54s), but the dynamics are different:
- Only 1 GPU analyzer runs (no spurious CPU worker)
- Per-chunk inference is faster (0.067s vs 0.101s in more_streamers) because there is no CPU thread competing for GPU/GIL
- Wall-clock inference is higher (44.1s for 1 analyzer vs ~33s for 2) but prewarm is faster (only 1 model loaded)
- GPU is now clearly the bottleneck; 6 streamers easily feed it
