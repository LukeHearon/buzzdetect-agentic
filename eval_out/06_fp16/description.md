# 06_fp16

Enabled TensorFlow mixed precision FP16 globally (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`) in the GPU branch of `WorkerInferer._managememory()`, before model initialization.

Hypothesis: GTX 1650 (Turing, compute 7.5) tensor cores accelerate FP16 math, reducing inference time.
