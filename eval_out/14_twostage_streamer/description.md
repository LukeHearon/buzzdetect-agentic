# 14_twostage_streamer

Split each streamer into two concurrent stages with a local buffer queue (maxsize=2):
- Stage 1 (IO thread): read + resample → local_q
- Stage 2 (main streamer thread): prepfunc (tf.constant) + enqueue → q_analyze

The intent was to let IO continue in parallel with the tf.constant() memory copy,
keeping q_analyze fuller and reducing GPU starvation.

**Verdict: SLOWER (+10.7%)**
