"""
TF → ONNX Conversion Script
============================
Converts a TensorFlow SavedModel to ONNX format.
The output can then be used with ONNX Runtime (CUDAExecutionProvider,
TensorrtExecutionProvider, or CPUExecutionProvider).

Requirements: tf2onnx  (.venv/bin/python -m pip install tf2onnx)

Usage (from workspace root):
    # Convert YAMNet embedder models
    .venv/bin/python eval_out/07_tensorrt/scratch/convert_onnx.py \
        --input embedders/yamnet_k2/models/yamnet_wholehop \
        --output embedders/yamnet_k2/models/yamnet_wholehop.onnx

    .venv/bin/python eval_out/07_tensorrt/scratch/convert_onnx.py \
        --input embedders/yamnet_k2/models/yamnet_halfhop \
        --output embedders/yamnet_k2/models/yamnet_halfhop.onnx

    # Convert classifier
    .venv/bin/python eval_out/07_tensorrt/scratch/convert_onnx.py \
        --input models/model_general_v3 \
        --output models/model_general_v3/model_general_v3.onnx

After conversion, verify with verify_onnx.py.
"""

import argparse
import subprocess
import sys


def convert(input_dir: str, output_path: str, opset: int):
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", input_dir,
        "--output", output_path,
        "--opset", str(opset),
        "--signature_def", "serving_default",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Conversion failed.")
        sys.exit(result.returncode)
    print(f"Saved ONNX model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a SavedModel to ONNX")
    parser.add_argument("--input", required=True, help="Path to input SavedModel directory")
    parser.add_argument("--output", required=True, help="Path for output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()
    convert(args.input, args.output, args.opset)


if __name__ == "__main__":
    main()
