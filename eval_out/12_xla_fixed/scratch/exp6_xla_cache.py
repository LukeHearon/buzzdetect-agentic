"""
Experiment 6: XLA persistent disk cache.

XLA can cache compiled kernels to disk and reload them on subsequent runs.
If loading from cache is fast (<1s), we can:
  1. Pre-compile all chunk sizes once (outside eval timer)
  2. Load from cache during prewarm (inside timer, but fast)

Tests:
  A) How long does cache LOAD take vs compile?
  B) Does the cache survive across Python processes?
  C) What is the realistic overhead per analyze() call?

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp6_xla_cache.py
"""
import os, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_DIR = '/tmp/xla_cache_buzzdetect'
os.makedirs(CACHE_DIR, exist_ok=True)

# XLA_FLAGS must be set before TF/XLA initializes
os.environ['XLA_FLAGS'] = f'--xla_persistent_cache_dir={CACHE_DIR} --xla_persistent_cache_min_entry_size=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.inference.models import load_model

MODEL = 'model_general_v3'
model = load_model(MODEL, framehop_prop=1.0, initialize=False)
model.initialize()

embedder_fn = model.embedder.model
classifier_fn = model.model

@tf.function(jit_compile=True)
def predict_xla(samples):
    embeddings = embedder_fn(samples)['global_average_pooling2d']
    return classifier_fn(embeddings)['dense']

# Chunk sizes used in eval sweep
chunk_sizes = {
    200:  int(round(200  / 0.96) * 0.96 * 16000),   # 3,194,880
    600:  int(round(600  / 0.96) * 0.96 * 16000),   # 9,600,000
    1200: int(round(1200 / 0.96) * 0.96 * 16000),   # 19,199,040
}

print(f"XLA cache dir: {CACHE_DIR}")
print(f"Cache files before: {list(Path(CACHE_DIR).glob('*'))}\n")

print("=== Phase A: Compile + cache (first call per shape) ===")
for chunk_s, n_samples in chunk_sizes.items():
    dummy = np.random.randn(n_samples).astype(np.float32)
    t0 = time.perf_counter()
    _ = predict_xla(dummy).numpy()
    t1 = time.perf_counter()
    print(f"  {chunk_s}s ({n_samples} samples): {(t1-t0):.2f}s (compile+run)")

cache_files = list(Path(CACHE_DIR).glob('*'))
print(f"\nCache files after compile: {len(cache_files)} files")

print("\n=== Phase B: Warm cache (same process, second call) ===")
for chunk_s, n_samples in chunk_sizes.items():
    dummy = np.random.randn(n_samples).astype(np.float32)
    t0 = time.perf_counter()
    _ = predict_xla(dummy).numpy()
    t1 = time.perf_counter()
    print(f"  {chunk_s}s: {(t1-t0)*1000:.1f}ms (cached in memory)")

print("\n=== Phase C: Cross-process cache load ===")
print("Spawning new Python process to measure cold-cache-load time...")

script = f"""
import os, sys, time
os.environ['XLA_FLAGS'] = '--xla_persistent_cache_dir={CACHE_DIR} --xla_persistent_cache_min_entry_size=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
sys.path.insert(0, '.')
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from src.inference.models import load_model
model = load_model('model_general_v3', framehop_prop=1.0, initialize=False)
model.initialize()
embedder_fn = model.embedder.model
classifier_fn = model.model

@tf.function(jit_compile=True)
def predict_xla(samples):
    embeddings = embedder_fn(samples)['global_average_pooling2d']
    return classifier_fn(embeddings)['dense']

chunk_sizes = {{
    200:  {chunk_sizes[200]},
    600:  {chunk_sizes[600]},
    1200: {chunk_sizes[1200]},
}}
for chunk_s, n_samples in chunk_sizes.items():
    dummy = np.random.randn(n_samples).astype(np.float32)
    t0 = time.perf_counter()
    _ = predict_xla(dummy).numpy()
    t1 = time.perf_counter()
    print(f"  {{chunk_s}}s: {{(t1-t0):.3f}}s (cross-process cache load)", flush=True)
"""

result = subprocess.run(
    ['.venv/bin/python', '-c', script],
    capture_output=True, text=True
)
print(result.stdout if result.stdout else f"  Error: {result.stderr[-300:]}")

print("\n=== Summary ===")
print("If cross-process cache load is < 2s per shape:")
print("  - Pre-compile all 3 chunk sizes ONCE (36s, one-time)")
print("  - Modify _prewarm_model() to load from cache for all chunk sizes")
print("  - Prewarm overhead: 3 × cache_load_time")
print("  - Inference speedup: ~9.6ms/chunk × 222 = 2.1s saved")
print("  - Net gain if cache_load < 0.7s per shape: FASTER overall")
