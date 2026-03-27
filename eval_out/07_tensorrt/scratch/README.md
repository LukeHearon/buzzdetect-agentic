# TF-TRT Conversion Scripts

## Overview

`convert_trt.py` — converts any TensorFlow SavedModel to a TensorRT-optimized
SavedModel. The output is a drop-in replacement loadable with `TFSMLayer` or
`tf.saved_model.load` — no TRT-specific loading code required.

## How to use

Run from the **workspace root** with the project venv:

```bash
# Convert YAMNet embedder (wholehop variant) to TRT FP32
.venv/bin/python eval_out/07_tensorrt/scratch/convert_trt.py \
    --input  embedders/yamnet_k2/models/yamnet_wholehop \
    --output embedders/yamnet_k2/models/yamnet_wholehop_trt_fp32 \
    --precision FP32 \
    --input-shape "1,15360" \
    --build

# Convert YAMNet embedder (halfhop variant) to TRT FP32
.venv/bin/python eval_out/07_tensorrt/scratch/convert_trt.py \
    --input  embedders/yamnet_k2/models/yamnet_halfhop \
    --output embedders/yamnet_k2/models/yamnet_halfhop_trt_fp32 \
    --precision FP32 \
    --input-shape "1,15360" \
    --build

# Convert classifier to TRT FP32
# Input shape is (batch, 1024) embeddings; use a representative batch size
.venv/bin/python eval_out/07_tensorrt/scratch/convert_trt.py \
    --input  models/model_general_v3 \
    --output models/model_general_v3_trt_fp32 \
    --precision FP32 \
    --input-shape "208,1024" \
    --build
```

For FP16 variants, replace `FP32` with `FP16` and change output dir accordingly.

## Precision modes

| Mode  | Notes |
|-------|-------|
| FP32  | Safe default; same accuracy as original |
| FP16  | ~2× faster on tensor-core GPUs; tiny accuracy drop usually acceptable |
| INT8  | Fastest; requires calibration data (not implemented here) |

## Input shape guidance

- YAMNet: `1,15360` — one frame of 0.96 s at 16 kHz (or use larger batch for chunk-at-once processing)
- Classifier: `N,1024` where N = frames per chunk (e.g. 208 for a 200 s chunk with wholehop)

Using a representative batch size for `--build` gives TRT the best optimization opportunity.

## What gets created

A new SavedModel directory. The original is left untouched.
The new directory is a complete SavedModel and can be loaded identically
to the original — just point `dir_model` at the new path.

## Adapting for a new model

1. Identify the model's input shape (use the `inspect_model.py` helper below or
   check the embedder/model `embed()`/`predict()` call).
2. Run `convert_trt.py` with appropriate `--input-shape`.
3. Point the embedder/model's `initialize()` at the new directory.
