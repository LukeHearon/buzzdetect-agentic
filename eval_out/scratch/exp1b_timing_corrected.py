"""
Experiment 1b: Corrected timing breakdown — no forced intermediate syncs.

The real pipeline never calls .numpy() between embedder and classifier.
GPU ops are queued asynchronously; sync only happens when results are consumed.

This measures:
  A) Time to dispatch all GPU ops (mostly Python/TF overhead)
  B) Time until GPU finishes (final .numpy() sync)
  C) Whether explicit tf.constant() before predict() actually helps

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp1b_timing_corrected.py
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.inference.models import load_model

MODEL = 'model_general_v3'
model = load_model(MODEL, framehop_prop=1.0, initialize=False)
model.initialize()

# prewarm
dummy_frame = np.zeros(int(0.96 * 16000), dtype=np.float32)
model.predict(dummy_frame)
model.predict(dummy_frame)

REPS = 15
n_samples = int(200 * 16000)
dummy = np.random.randn(n_samples).astype(np.float32)
dummy_tensor = tf.constant(dummy)   # pre-built GPU tensor for comparison

# ── A: Full predict() with numpy input (current real behavior) ────────────────
# No intermediate sync — mirrors real process_chunk() exactly.
times_dispatch = []   # time until predict() returns (GPU ops queued)
times_total = []      # time until results synced to CPU (full GPU compute)

for _ in range(REPS + 3):
    t0 = time.perf_counter()
    results = model.predict(dummy)         # queues GPU ops, may return before done
    t1 = time.perf_counter()
    _ = results.numpy()                    # forces GPU sync
    t2 = time.perf_counter()
    times_dispatch.append((t1 - t0) * 1000)
    times_total.append((t2 - t0) * 1000)

times_dispatch = times_dispatch[3:]
times_total = times_total[3:]

print("=== numpy input (current) ===")
print(f"  Dispatch (predict() returns): {np.mean(times_dispatch):.1f}ms ± {np.std(times_dispatch):.1f}ms")
print(f"  Total (incl GPU sync):        {np.mean(times_total):.1f}ms ± {np.std(times_total):.1f}ms")
print(f"  GPU async time (total-dispatch): {np.mean(times_total)-np.mean(times_dispatch):.1f}ms")

# ── B: Pre-built tensor input ─────────────────────────────────────────────────
times_dispatch_t = []
times_total_t = []

for _ in range(REPS + 3):
    t0 = time.perf_counter()
    results = model.predict(dummy_tensor)   # tensor already on GPU
    t1 = time.perf_counter()
    _ = results.numpy()
    t2 = time.perf_counter()
    times_dispatch_t.append((t1 - t0) * 1000)
    times_total_t.append((t2 - t0) * 1000)

times_dispatch_t = times_dispatch_t[3:]
times_total_t = times_total_t[3:]

print("\n=== pre-built tensor input ===")
print(f"  Dispatch (predict() returns): {np.mean(times_dispatch_t):.1f}ms ± {np.std(times_dispatch_t):.1f}ms")
print(f"  Total (incl GPU sync):        {np.mean(times_total_t):.1f}ms ± {np.std(times_total_t):.1f}ms")
print(f"  GPU async time (total-dispatch): {np.mean(times_total_t)-np.mean(times_dispatch_t):.1f}ms")

# ── C: What does process_chunk()'s profiler actually measure? ─────────────────
# process_chunk does: with profiler.phase('inference'): results = model.predict(samples)
# The profiler measures ONLY the dispatch time — GPU may still be running after!
# The real GPU cost is deferred to when the writer calls .numpy() on results.
print("\n=== What the 'inference' profiler phase captures ===")
print(f"  process_chunk() profiler measures: dispatch time only ≈ {np.mean(times_dispatch):.1f}ms per chunk")
print(f"  Actual GPU compute finishes later: +{np.mean(times_total)-np.mean(times_dispatch):.1f}ms async")
print(f"  Total per-chunk GPU cost: {np.mean(times_total):.1f}ms")

# ── D: Back-to-back dispatch — does GPU pipeline across chunks? ───────────────
# Queue ops for N chunks rapidly. If GPU pipelines, total/N should be < single-call total.
print("\n=== Back-to-back dispatch (N chunks, one final sync) ===")
print(f"  {'N':>4} {'dispatch_ms':>12} {'total_ms':>10} {'ms/chunk':>10} {'vs 1-chunk':>12}")
single_total = np.mean(times_total)

for n in [1, 2, 4, 8]:
    chunks = [np.random.randn(n_samples).astype(np.float32) for _ in range(n)]
    results_list = []
    # warmup
    for c in chunks:
        r = model.predict(c)
        results_list.append(r)
    _ = [r.numpy() for r in results_list]

    run_dispatches = []
    run_totals = []
    for _ in range(8):
        results_list = []
        t0 = time.perf_counter()
        for c in chunks:
            results_list.append(model.predict(c))
        t1 = time.perf_counter()
        for r in results_list:
            _ = r.numpy()
        t2 = time.perf_counter()
        run_dispatches.append((t1 - t0) * 1000)
        run_totals.append((t2 - t0) * 1000)

    d = np.mean(run_dispatches)
    tot = np.mean(run_totals)
    ratio = (tot / n) / single_total
    print(f"  {n:>4} {d:>11.1f}ms {tot:>9.1f}ms {tot/n:>9.1f}ms {ratio:>11.2f}x")
