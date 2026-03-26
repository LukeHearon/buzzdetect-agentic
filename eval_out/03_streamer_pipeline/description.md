# 03_streamer_pipeline

Two changes to the audio streamers:
1. Replaced `librosa.resample` with `soxr.resample` directly (identical results, fewer Python layers)
2. Added a per-streamer reader thread to overlap MP3 decoding (I/O) with resampling (CPU)

## Result

**SLOWER +336.9%** (97.4s vs 22.3s baseline)

Inference time exploded from 0.074s/chunk to 0.414s/chunk. The 12 extra threads
(6 readers + 6 resamplers) running during inference saturated the 8-core CPU and
starved the ONNX session.

The threading idea was based on a false premise — MP3 decoding via libsndfile is
CPU-bound, not I/O-bound. There was nothing to overlap.
