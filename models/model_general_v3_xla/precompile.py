"""
Pre-compile XLA kernels for model_general_v3_xla.

Run once before eval:
    .venv/bin/python models/model_general_v3_xla/precompile.py

Compiles @tf.function(jit_compile=True) for all chunk sizes in the eval
sweep (200s, 600s, 1200s at 16kHz with 0.96s frame rounding) and saves
compiled kernels to xla_cache/. Subsequent loads take ~1s instead of ~12s.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_DIR = str(Path(__file__).parent / 'xla_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TF_XLA_FLAGS'] = f'--tf_xla_persistent_cache_directory={CACHE_DIR}'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.inference.models import load_model

print("Loading model...")
model = load_model('model_general_v3_xla', framehop_prop=1.0, initialize=False)
model.initialize()

# Chunk sizes from eval.py settings grid: [200, 600, 1200] seconds
# Each rounded to nearest 0.96s frame boundary
EVAL_CHUNK_SIZES_S = [200, 600, 1200]
SAMPLERATE = 16000
FRAMELENGTH_S = 0.96

print(f"\nPre-compiling XLA for {len(EVAL_CHUNK_SIZES_S)} chunk sizes...")
print(f"Cache dir: {CACHE_DIR}\n")

for chunk_s in EVAL_CHUNK_SIZES_S:
    n_frames = round(chunk_s / FRAMELENGTH_S)
    n_samples = int(n_frames * FRAMELENGTH_S * SAMPLERATE)
    dummy = np.random.randn(n_samples).astype(np.float32)

    print(f"  {chunk_s}s → {n_samples} samples... ", end='', flush=True)
    t0 = time.perf_counter()
    _ = model.predict(dummy)
    t1 = time.perf_counter()
    print(f"{t1-t0:.1f}s")

cache_files = list(Path(CACHE_DIR).glob('*'))
print(f"\nDone. {len(cache_files)} cache files written to {CACHE_DIR}")
print("Run eval.py with --model model_general_v3_xla")
