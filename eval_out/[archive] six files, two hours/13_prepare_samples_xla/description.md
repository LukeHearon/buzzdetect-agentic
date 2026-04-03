# 13_prepare_samples_xla

Same prepfunc optimization as 13_prepare_samples_base, run against model_general_v3_xla.

XLA is no longer the best option: base model with prepfunc (18.5s) beats XLA with prepfunc (19.7s). XLA overhead from cache load and emptyqueue starvation outweighs its per-chunk inference savings at this streamer count.
