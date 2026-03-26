# 05_soxr_only

Single change: replaced `librosa.resample` with `soxr.resample` directly.
- Quality: `HQ` (equivalent to librosa's default `soxr_hq`) — results identical
- Fewer Python layers than going through librosa's wrapper
