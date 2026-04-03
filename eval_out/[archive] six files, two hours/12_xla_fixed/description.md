# 12_xla_fixed

XLA JIT compilation via `@tf.function(jit_compile=True)` with persistent disk cache,
fixing all pipeline issues from 09/10/11.

## Changes

- New model `model_general_v3_xla` with XLA-compiled predict path
- Fixed TF_XLA_FLAGS timing: set at model.py module level, before TF imports
- Fixed worker.py: removed module-level `import tensorflow as tf` (was causing
  TF to initialize before TF_XLA_FLAGS could be set → cache miss → 12s compile)
- Fixed assignments.py: removed module-level `import tensorflow as tf`
  (type annotation `results: tf.Tensor` changed to `results: object`)
- Fixed prediction routing: only exact precompiled sizes (3194880, 9600000,
  19200000 samples) use XLA; all other sizes (partial last chunks) use plain TF
  to avoid per-chunk cache miss compilations
- Pre-populated XLA cache for all 3 sizes in the correct eval context

## Result

Best combo: 200s_6str at 19.74s (+3.5% vs 19.08s baseline)
Inference: 9.95s vs 11.46s baseline (-13.2%) — XLA is genuinely faster
Overall: NEUTRAL due to pipeline starvation (GPU faster than 6 streamers)
