"""
ONNX Model Verification Script
================================
Loads an ONNX model with a given execution provider and runs a dummy inference
to confirm the model works and is using the intended hardware.

Usage (from workspace root):
    # Verify YAMNet with CUDA EP
    .venv/bin/python eval_out/07_tensorrt/scratch/verify_onnx.py \
        --model embedders/yamnet_k2/models/yamnet_wholehop.onnx \
        --input-shape 3200000 \
        --provider CUDA

    # Verify classifier with CUDA EP (input = N embeddings of dim 1024)
    .venv/bin/python eval_out/07_tensorrt/scratch/verify_onnx.py \
        --model models/model_general_v3/model_general_v3.onnx \
        --input-shape 209,1024 \
        --provider CUDA
"""

import argparse
import numpy as np


PROVIDER_MAP = {
    "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "TRT": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "CPU": ["CPUExecutionProvider"],
}


def verify(model_path: str, input_shape: list[int], provider: str):
    import onnxruntime as ort

    providers = PROVIDER_MAP[provider]
    sess = ort.InferenceSession(model_path, providers=providers)

    print(f"Model: {model_path}")
    print(f"Requested provider: {provider}")
    print(f"Active providers: {sess.get_providers()}")

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"Inputs:  {[(i.name, i.shape, i.type) for i in inputs]}")
    print(f"Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

    dummy = np.zeros(input_shape, dtype=np.float32)
    result = sess.run(None, {inputs[0].name: dummy})
    print(f"Output shapes: {[r.shape for r in result]}")
    print("OK")


def main():
    parser = argparse.ArgumentParser(description="Verify an ONNX model")
    parser.add_argument("--model", required=True, help="Path to .onnx model file")
    parser.add_argument(
        "--input-shape",
        required=True,
        help="Comma-separated input shape, e.g. '3200000' or '209,1024'",
    )
    parser.add_argument(
        "--provider",
        default="CUDA",
        choices=["CUDA", "TRT", "CPU"],
        help="Execution provider (default: CUDA)",
    )
    args = parser.parse_args()
    shape = [int(x) for x in args.input_shape.split(",")]
    verify(args.model, shape, args.provider)


if __name__ == "__main__":
    main()
