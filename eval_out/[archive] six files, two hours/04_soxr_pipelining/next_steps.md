# Next Steps after 04_soxr_pipelining

## What happened
The reader thread approach backfired badly. Adding a per-streamer reader thread to pipeline I/O+resampling caused:
- Read time: +129.9% (much slower)
- Queue streamer wait: +266.4%
- Overall: +6.5% (SLOWER)

The soxr resampling itself was -12.4% faster, but the threading overhead swamped the gain.

## Why the reader thread hurt
Python's GIL means adding more threads doesn't help CPU-bound work. With 6 streamers each spawning a reader thread, we have 12 threads competing for the GIL. The ThreadQueue synchronization between reader and main thread adds overhead without meaningful parallelism benefit.

The soundfile library may release the GIL during C-level I/O, but the overhead of thread coordination outweighed any overlap benefit.

## What to try next
- Soxr replacement alone (no reader thread) was tested as 05_soxr_only and showed NEUTRAL (-0.5%). Small win but below threshold.
- To get more significant gains, look at the inference pipeline — it's 60% of total time.
- Consider reducing the number of concurrent streamers to reduce queue contention (already auto-tuned, so unlikely).
- Look into GPU utilization during inference: is the GPU idle while waiting for the next chunk? Profiling GPU util could reveal headroom.
- Explore whether the inference model itself can be optimized (quantization, TensorRT, etc.)
