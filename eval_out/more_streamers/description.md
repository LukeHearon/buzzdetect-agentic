Increased n_streamers from 4 to 6 and stream_buffer_depth from 4 to 6.

The baseline had 4 streamers producing at ~17.9 chunks/s, while 2 analyzers (1 CPU + 1 GPU, due to the for-loop logic in _launch_analyzers) consumed at ~26 chunks/s. This caused 491 buffer bottleneck events in the baseline, with analyzers idle ~0.1s per chunk (~38% of the time).

With 6 streamers (1 per audio file), production rate rises to ~26.8 chunks/s, matching consumption. This nearly eliminates analyzer idle time.
