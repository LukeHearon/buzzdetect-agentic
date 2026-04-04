## What we learned

Padding the last chunk to a fixed size gives a -5.6% inference improvement by routing all chunks through the XLA kernel. On short eval audio the read/queue overhead makes overall look +9% worse, but on real long-form audio (hundreds to thousands of hours) the inference gain dominates strongly.

The bottleneck profile shows:
- `audio_io/reading` at 250% of overall (summed across 8 streamers) — streamers are I/O bound
- `audio_io/fullqueue` at 90% — streamers frequently block waiting to enqueue; analyzer can't drain the queue fast enough

## What to try next

**Reduce streamer/inference imbalance.** The queue is persistently full, meaning the GPU analyzer is the bottleneck, not I/O. Options:
1. **Batch inference**: accumulate N chunks in the analyzer before calling predict, so the GPU processes a larger tensor per kernel launch. This amortizes XLA kernel launch overhead and could improve GPU utilization significantly.
2. **Profile GPU utilization directly** (e.g. `nvidia-smi dmon`) during a run to confirm the GPU is under-utilized between kernel launches.
3. **Increase queue depth or streamers** — but the queue is already full (fullqueue time is high), so more streamers won't help unless inference speeds up.

Batching is the most promising next step given the current bottleneck profile.
