"""
TF-TRT SavedModel Conversion Script
=====================================
Converts a TensorFlow SavedModel to a TensorRT-optimized SavedModel.
The output is a standard SavedModel that can be loaded with TFSMLayer or
tf.saved_model.load — no TRT-specific loading code needed.

Usage:
    python convert_trt.py \
        --input  embedders/yamnet_k2/models/yamnet_wholehop \
        --output embedders/yamnet_k2/models/yamnet_wholehop_trt_fp32 \
        --precision FP32 \
        --input-shape "1,15360"

    python convert_trt.py \
        --input  models/model_general_v3 \
        --output models/model_general_v3_trt_fp32 \
        --precision FP32 \
        --input-shape "1,1024"

Precision modes: FP32, FP16, INT8 (INT8 requires calibration data).

Run from the workspace root with the project venv active:
    .venv/bin/python eval_out/07_tensorrt/scratch/convert_trt.py ...

After conversion, warm up the TRT engines by calling build_trt.py
(or pass --build here) so that the first real inference call isn't slow.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_shape(shape_str: str):
    """Parse a comma-separated shape string like '1,15360' into a list of ints."""
    return [int(x) for x in shape_str.split(",")]


def convert(input_dir: str, output_dir: str, precision: str, input_shape: list[int], build: bool):
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    print(f"Converting: {input_dir}")
    print(f"       To: {output_dir}")
    print(f"Precision: {precision}")
    print(f"    Shape: {input_shape}")

    params = trt.TrtConversionParams(precision_mode=precision)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_dir,
        conversion_params=params,
    )

    print("Converting graph...")
    converter.convert()

    if build:
        print("Building TRT engines with representative input...")
        dummy = np.zeros(input_shape, dtype=np.float32)

        def input_fn():
            yield (tf.constant(dummy),)

        converter.build(input_fn=input_fn)
        print("TRT engines built.")
    else:
        print("Skipping build step (TRT engines will be built on first inference).")

    print(f"Saving to {output_dir} ...")
    converter.save(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert a SavedModel to TF-TRT")
    parser.add_argument("--input", required=True, help="Path to input SavedModel directory")
    parser.add_argument("--output", required=True, help="Path to output SavedModel directory")
    parser.add_argument(
        "--precision",
        default="FP32",
        choices=["FP32", "FP16", "INT8"],
        help="TRT precision mode (default: FP32)",
    )
    parser.add_argument(
        "--input-shape",
        required=True,
        help="Comma-separated input shape for engine building, e.g. '1,15360'",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Pre-build TRT engines (recommended; avoids cold-start on first inference)",
    )
    args = parser.parse_args()

    shape = parse_shape(args.input_shape)
    convert(args.input, args.output, args.precision, shape, args.build)


if __name__ == "__main__":
    main()
