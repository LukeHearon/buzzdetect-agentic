# Next Steps after 05_soxr_only

## Result
NEUTRAL (-0.5%). Soxr resampling was -8.6% faster than librosa, but resampling is only ~25% of wall-clock time, so the total impact was minimal. The soxr change is kept (it's a clean improvement).

## Where time goes (baseline)
- inference: 60.4% of overall wall-clock
- audio_io/reading: largest cumulative time but parallelized across 6 streamers
- audio_io/resampling: ~12.5% wall-clock contribution (cumulative 28s / 6 streamers ≈ 4.7s / 19s)
- audio_io/fullqueue: streamers spend ~90% of cumulative time waiting to enqueue → GPU is the bottleneck

## Key insight: GPU is the bottleneck
The fullqueue wait time (90% of overall in cumulative terms) shows streamers are already faster than inference. Speeding up audio I/O won't help much — the GPU bottleneck dominates. Audio optimization has diminishing returns until inference gets faster.

## Recommended next directions
1. **Inference optimization** — this is where 60% of time goes. Options:
   - TensorRT conversion of the model (dramatic GPU speedup possible)
   - FP16 inference (GPU supports it, may give 2x speedup with minimal accuracy loss)
   - Batch size tuning: are we batching optimally for the GPU?
2. **Model inspection** — check what batch size the model runs with and whether increasing it would help GPU utilization
3. **Profile GPU util** — run `nvidia-smi dmon` during a run to see if GPU is actually saturated or idling

## What to avoid
- More audio I/O threading (04_soxr_pipelining showed this is counterproductive)
- ONNX on GPU (01_onnx was +11.5% SLOWER — used CUDAExecutionProvider with a combined YAMNet+classifier ONNX model; TF native GPU is faster. Not a CPU vs GPU issue.)
