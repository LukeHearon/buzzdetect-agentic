# Next Steps

## What we learned

**Multiprocessing streamers are unambiguously slower.** The +7.5% overall penalty
comes from pickling numpy arrays across the process boundary. Key data points:

- **Pickling cost is chunk-size proportional.** 200s chunks (12.8 MB @ 16 kHz)
  showed +19.2% read overhead. 600s → +201%, 1200s → +451%. This confirms the
  user's prior expectation.

- **Threads already release the GIL for I/O.** `soundfile` and `librosa`/`scipy`
  are C extensions that release the GIL during their hot loops. Threading is
  effectively equivalent to multiprocessing for this workload — without the IPC cost.

- **Inference is the real bottleneck** (73% of overall). Streamers are not
  limiting throughput; the GPU is. Any approach that targets streamer speed
  without touching inference will have diminishing returns.

## What to avoid

- **Multiprocessing for streamers.** The GIL is not the bottleneck here.
  Even shared-memory MP (avoiding pickling) would likely add process-management
  overhead without meaningful gains, since threads already work well for I/O.
- **Large chunks with this architecture.** 600s+ amplifies IPC costs if any
  cross-process transfer is involved.

## Promising next directions

- **Batch inference.** Inference is 73% of overall time. The ONNX session
  processes one chunk at a time. If multiple chunks can be batched into a single
  `session.run()` call, GPU utilization may improve and overall time could drop.
  Worth profiling GPU utilization (nvidia-smi) during a run to see if the GPU
  is actually saturated or if there's feed latency between chunks.

- **Streamer-side resampling optimization.** Resampling is 114% of overall
  (parallelized). `librosa.resample` uses soxr by default; verify this and
  confirm no fallback to slower backends. Could also try `soxr` directly.

- **Inter-chunk overlap/pipelining.** Currently inference blocks while the next
  chunk is being read. Investigate whether q_analyze is ever empty during
  inference (buffer starvation), which would indicate streamer throughput limits.

- **ONNX session parallelism settings.** The current session uses default thread
  counts. Try setting `sess_options.intra_op_num_threads` and
  `inter_op_num_threads` explicitly for CPU fallback paths.
