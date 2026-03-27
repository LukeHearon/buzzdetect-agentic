# 08_xla_jit

Added `tf.config.optimizer.set_jit(True)` in `WorkerInferer.run()` before
model initialization to enable XLA JIT compilation globally for all TF ops.

One-line change; no new dependencies.
