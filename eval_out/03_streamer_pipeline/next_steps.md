# Next Steps

## What we learned

**MP3 decoding is CPU-bound.** The core mistake: assuming soundfile.read() was I/O-bound
so overlapping it with soxr resampling would help. It isn't — libsndfile decodes MP3 on
CPU. Adding 6 reader threads just created 12 competing CPU-intensive threads that starved
the ONNX inference session, causing a 5.6x inference slowdown.

**soxr direct vs librosa: modest gain, not a game-changer.**
Scratch benchmark (`bench_read.py`) showed:
- After librosa prewarm: librosa ≈ 0.130s/chunk, soxr ≈ 0.096s/chunk (1.35x faster)
- Wall-clock for 6 concurrent streamers: soxr is only 1.13x faster
- Threading added overhead and provided no benefit over sequential soxr

**librosa cold-start is real but already handled.** First call to librosa.resample
takes ~1.85s (filter caching). The existing `_prewarm_resample()` in worker.py
correctly covers this before timed analysis.

**Streamers are not the bottleneck.** Baseline has 10 starvation events totaling ~4.4s,
but inference (73% of wall time) is the real constraint. Faster streaming doesn't help
if the queue is already keeping the inferer fed between stalls.

## What to avoid

- **Any approach that adds CPU-intensive threads during inference.** Even "background"
  threads doing MP3 decode or resampling will compete with the ONNX session on a CPU-only
  machine (no GPU).
- **Overcomplicating the streamer.** The current sequential read → resample → queue is
  close to optimal given the constraints. The GIL is not the bottleneck here.

## Promising next directions

- **Batch inference** (the real bottleneck). The ONNX session processes one chunk at a
  time. If multiple frames within a chunk can be batched, or if the session's internal
  threading can be tuned (`intra_op_num_threads`), there may be gains.
- **ONNX session thread tuning.** Current sessions use default thread counts. Explicitly
  setting `intra_op_num_threads=N` for the CPU provider (with N tuned to leave headroom
  for streamers) may improve throughput. Try N = 4–6 on the 8-core machine.
- **Larger chunks.** Inference has per-call overhead. Fewer, larger chunks = less
  overhead. The auto-tune sweeps 200/600/1200s — but ONNX session options are fixed
  regardless of chunk size. Tuning session options for large-chunk runs may help.
- **Avoid streamer optimizations entirely** until inference is addressed. Any streamer
  speedup is wasted while inference is the limiting step.
