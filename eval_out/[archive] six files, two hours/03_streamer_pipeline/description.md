# 03_streamer_pipeline

Two changes to the audio streamers:
1. Replaced `librosa.resample` with `soxr.resample` directly (identical results, fewer Python layers)
2. Added a per-streamer reader thread to overlap MP3 decoding (I/O) with resampling (CPU)

