"""
Experiment 1: Fine-grained timing of inference phases.

Runs the real analyze() pipeline but patches model.predict() to emit
per-phase timing data. Shows how time is distributed between:
  - numpy → tf.constant (CPU tensor creation)
  - embedder forward pass (STFT + MelSpec + MobileNet)
  - classifier forward pass (dense layers)
  - GPU sync / tf.numpy() extraction

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp1_timing_breakdown.py
"""
import os, sys, time, csv, subprocess, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Setup GPU ────────────────────────────────────────────────────────────────
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Import after TF config ────────────────────────────────────────────────────
from src.inference.models import load_model

MODEL = 'model_general_v3'
EMBEDDER_PATH = 'embedders/yamnet_k2/models/yamnet_wholehop'

model = load_model(MODEL, framehop_prop=1.0, initialize=False)
model.initialize()

# ── Prewarm ───────────────────────────────────────────────────────────────────
n_samples_frame = int(0.96 * 16000)
dummy_frame = np.zeros(n_samples_frame, dtype=np.float32)
model.predict(dummy_frame)  # trigger JIT compilation

# ── Instrumented predict ───────────────────────────────────────────────────────
def timed_predict(audiosamples, records):
    t0 = time.perf_counter()
    tensor_in = tf.constant(audiosamples)
    t1 = time.perf_counter()

    embeddings = model.embedder.model(tensor_in)['global_average_pooling2d']
    t2 = time.perf_counter()

    # Force GPU sync by calling .numpy()
    emb_np = embeddings.numpy()
    t3 = time.perf_counter()

    results = model.model(embeddings)['dense']
    t4 = time.perf_counter()

    _ = results.numpy()
    t5 = time.perf_counter()

    records.append({
        'n_samples': len(audiosamples),
        'n_frames': emb_np.shape[0],
        'ms_to_tensor': (t1 - t0) * 1000,
        'ms_embedder': (t2 - t1) * 1000,
        'ms_emb_sync': (t3 - t2) * 1000,
        'ms_classifier': (t4 - t3) * 1000,
        'ms_cls_sync': (t5 - t4) * 1000,
        'ms_total': (t5 - t0) * 1000,
    })

# ── Run on realistic chunk sizes ──────────────────────────────────────────────
chunk_sizes_s = [200, 400, 600, 1200]
REPS = 10

print(f"\n{'Chunk':>8} {'Frames':>7} {'ToTensor':>10} {'Embed':>10} {'EmbSync':>10} {'Classify':>10} {'ClsSync':>10} {'Total':>10}")
print("-" * 90)

for chunk_s in chunk_sizes_s:
    n_samp = int(chunk_s * 16000)
    dummy = np.random.randn(n_samp).astype(np.float32)

    # warmup
    for _ in range(3):
        r = []
        timed_predict(dummy, r)

    records = []
    for _ in range(REPS):
        timed_predict(dummy, records)

    means = {k: np.mean([r[k] for r in records]) for k in records[0] if k.startswith('ms')}
    n_frames = records[0]['n_frames']

    print(f"{chunk_s:>7}s {n_frames:>7} "
          f"{means['ms_to_tensor']:>9.2f}ms "
          f"{means['ms_embedder']:>9.2f}ms "
          f"{means['ms_emb_sync']:>9.2f}ms "
          f"{means['ms_classifier']:>9.2f}ms "
          f"{means['ms_cls_sync']:>9.2f}ms "
          f"{means['ms_total']:>9.2f}ms")

# ── Batch accumulation experiment ─────────────────────────────────────────────
print("\n\n=== Batch accumulation: feeding N×200s concatenated audio ===")
print(f"\n{'N chunks':>10} {'Total_s':>10} {'N_frames':>10} {'Total_ms':>10} {'ms/chunk':>10} {'speedup':>10}")
print("-" * 65)

base_chunk_s = 200
base_n_samp = int(base_chunk_s * 16000)
base_dummy = np.random.randn(base_n_samp).astype(np.float32)
base_records = []
for _ in range(REPS + 3):
    timed_predict(base_dummy, base_records)
base_ms = np.mean([r['ms_total'] for r in base_records[3:]])
print(f"{'1':>10} {base_chunk_s:>9}s {base_records[3]['n_frames']:>10} {base_ms:>9.2f}ms {'':>10} {'1.00x':>10}")

for n_chunks in [2, 4, 8]:
    total_s = n_chunks * base_chunk_s
    n_samp = int(total_s * 16000)
    dummy = np.random.randn(n_samp).astype(np.float32)

    recs = []
    for _ in range(3):
        timed_predict(dummy, recs)
    recs = []
    for _ in range(REPS):
        timed_predict(dummy, recs)

    total_ms = np.mean([r['ms_total'] for r in recs])
    ms_per_chunk = total_ms / n_chunks
    speedup = base_ms / ms_per_chunk

    print(f"{n_chunks:>10} {total_s:>9}s {recs[0]['n_frames']:>10} {total_ms:>9.2f}ms {ms_per_chunk:>9.2f}ms {speedup:>9.2f}x")

print("\nDone. Higher speedup = batching helps.")
