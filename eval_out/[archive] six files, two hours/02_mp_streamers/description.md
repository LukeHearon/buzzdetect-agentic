# 02_mp_streamers

Replaced thread-based streamers with multiprocessing.Process workers to test
whether bypassing the GIL reduces audio I/O overhead.

## Architecture change

- Streamer threads → `multiprocessing.Process` (fork mode, Linux default)
- Cross-process chunk transfer via `multiprocessing.Queue` (pickle-based IPC)
- Bridge thread: reads from MP queue → forwards to thread queue for inference worker
- Profile data collected locally per-process, merged to main profiler after join

## Result

**SLOWER +7.5%** (best combo: 200s_6str, 23.97s vs baseline 22.30s)

- read: +19.2%, resample: +9.3%, inference: +2.6%
- Larger chunks amplify pickling cost dramatically (1200s: read +451%, resample +525%)
