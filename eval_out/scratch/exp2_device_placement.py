"""
Experiment 2: Check which device (CPU vs GPU) each model op actually runs on.

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp2_device_placement.py
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # verbose TF logs to see device placement

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"GPUs found: {gpus}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.inference.models import load_model

MODEL = 'model_general_v3'
model = load_model(MODEL, framehop_prop=1.0, initialize=False)
model.initialize()

n_samples = int(200 * 16000)
dummy = np.random.randn(n_samples).astype(np.float32)

# Force TF to log device placement
tf.debugging.set_log_device_placement(True)

print("\n=== Running inference with device placement logging ===")
emb = model.embedder.model(dummy)['global_average_pooling2d']
print(f"\nEmbedder output device: {emb.device}")
print(f"Embedder output shape: {emb.shape}")

result = model.model(emb)['dense']
print(f"\nClassifier output device: {result.device}")
print(f"Classifier output shape: {result.shape}")

# Also check with explicit GPU tensor input
print("\n\n=== Forcing input onto GPU first ===")
with tf.device('/GPU:0'):
    tensor_gpu = tf.constant(dummy)
print(f"Input tensor device: {tensor_gpu.device}")

emb_gpu = model.embedder.model(tensor_gpu)['global_average_pooling2d']
print(f"Embedder output device (GPU input): {emb_gpu.device}")

tf.debugging.set_log_device_placement(False)

# Benchmark CPU vs GPU explicitly
print("\n\n=== Explicit CPU vs GPU timing (10 runs each) ===")

# Baseline (current behavior — let TF decide)
times_auto = []
for _ in range(13):
    t0 = time.perf_counter()
    emb = model.embedder.model(dummy)['global_average_pooling2d']
    _ = emb.numpy()
    t1 = time.perf_counter()
    times_auto.append(t1 - t0)
print(f"Auto placement (current): {np.mean(times_auto[3:])*1000:.1f}ms ± {np.std(times_auto[3:])*1000:.1f}ms")

# Force GPU input
times_gpu_in = []
for _ in range(13):
    t0 = time.perf_counter()
    with tf.device('/GPU:0'):
        tensor_in = tf.constant(dummy)
    emb = model.embedder.model(tensor_in)['global_average_pooling2d']
    _ = emb.numpy()
    t1 = time.perf_counter()
    times_gpu_in.append(t1 - t0)
print(f"GPU input forced:          {np.mean(times_gpu_in[3:])*1000:.1f}ms ± {np.std(times_gpu_in[3:])*1000:.1f}ms")

# Force CPU
with tf.device('/CPU:0'):
    times_cpu = []
    for _ in range(13):
        t0 = time.perf_counter()
        emb = model.embedder.model(dummy)['global_average_pooling2d']
        _ = emb.numpy()
        t1 = time.perf_counter()
        times_cpu.append(t1 - t0)
print(f"CPU forced:                {np.mean(times_cpu[3:])*1000:.1f}ms ± {np.std(times_cpu[3:])*1000:.1f}ms")
