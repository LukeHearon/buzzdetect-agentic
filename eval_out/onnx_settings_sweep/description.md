# ONNX Conversion + Settings Sweep

## What changed

1. **Converted both models to ONNX**: yamnet_k2 (yamnet_wholehop.onnx) and model_general_v3
   (model.onnx, model_combined.onnx). Used tf2onnx with opset 17.

2. **Switched inference to combined ONNX model**: A single ONNX session that takes raw
   audio samples and returns class predictions (yamnet + dense layers merged). This eliminates
   the intermediate CPU round-trip between the two models.

3. **Removed unnecessary TF model initialization**: The original ONNX model erroneously called
   `self.embedder.initialize()` which loaded the YAMNet TFSMLayer into TF/GPU (wasted ~2s).
   Removed since the combined ONNX handles the full pipeline.

4. **Fixed write worker**: Changed `a_chunk.results.numpy()` to `np.asarray(a_chunk.results)`.
   With TF tensors, `.numpy()` forces a GPU→CPU sync + transfer. With ONNX numpy output,
   this was already resolved, but the fix makes the code robust to both.

5. **Settings tuning**: Ran a programmatic sweep over chunklength × n_streamers × buffer_depth.
   Best: chunklength=200, n_streamers=6, stream_buffer_depth=6.

## Key insight

The biggest win came from ONNX returning numpy arrays directly instead of TF tensors. The old
code called `.numpy()` in the write path which forced implicit GPU→CPU synchronization per chunk.
ONNX output eliminates this entirely, saving ~7-8s across 654 chunks.

## Settings used

chunklength=200, n_streamers=6, stream_buffer_depth=6, gpu=True, analyzers_cpu=0
