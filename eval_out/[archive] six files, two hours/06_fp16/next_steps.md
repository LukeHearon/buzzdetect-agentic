# Next Steps after 06_fp16

## Result
NEUTRAL (+2.2%). FP16 mixed precision was slightly *slower*, not faster. Inference phase was +1.5% slower.

## Why it didn't work
`tf.keras.mixed_precision.set_global_policy('mixed_float16')` sets a Keras policy but TFSMLayer-loaded SavedModels don't re-trace their computation graph to use FP16 kernels. The policy affects variable dtype but the underlying CUDA kernels still run in FP32. The GTX 1650 tensor cores are not utilized this way.

## What to try next
1. **TF-TRT (TensorRT)**: The proper way to get FP16 on this GPU. `TrtGraphConverterV2` with `precision_mode='FP16'` rewrites the op graph to use TRT FP16 kernels. This is separate from Keras mixed precision — it works at the SavedModel/graph level, not the Keras layer level. Expected to be a meaningful speedup.

2. **TF-TRT FP32**: Even without FP16, TRT kernel fusion and optimization might speed up the graph.

3. **Input casting**: Manually cast embeddings to float16 before passing to the classifier (and ensure the classifier weights are float16). More invasive but could work.

## What to avoid
- Keras `mixed_float16` policy alone — doesn't actually trigger FP16 compute in TFSMLayer models
