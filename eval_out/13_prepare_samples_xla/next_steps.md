# Next Steps (XLA variant)

See 13_prepare_samples_base/next_steps.md for the primary recommendations.

## XLA summary

XLA with prepfunc = 19.7s, worse than base+prepfunc = 18.5s.

XLA per-chunk inference is faster (~215ms/chunk vs 48ms for base at 1200s chunks),
but emptyqueue = 1.1s shows the GPU is outpacing 6 streamers again once the
tensor conversion load is removed from the GPU worker. XLA's advantage disappears
at the 6-streamer ceiling.

XLA is not worth pursuing further unless the streamer count can exceed 6.
