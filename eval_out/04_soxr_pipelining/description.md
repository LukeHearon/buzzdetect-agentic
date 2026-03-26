# 04_soxr_pipelining

Two changes to audio streamers:
1. Replaced `librosa.resample` with `soxr.resample` directly (same quality level: HQ/soxr_hq)
2. Added a per-streamer reader thread to overlap MP3 decoding (I/O) with resampling (CPU)
