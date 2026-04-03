# 13_prepare_samples_base

Added `prepfunc` to `BaseEmbedder` — an optional callable set by `initialize_prepfunc()` before worker threads start. `WorkerStreamer` calls it on resampled numpy samples before enqueuing, converting them to `tf.constant` tensors in the streamer thread rather than inside the GPU worker's inference phase.

For `yamnet_k2`, `prepfunc = tf.constant` (set after TF is imported).

Result: FASTER vs baseline, FASTER vs all previous tests (new best at 18.5s).
