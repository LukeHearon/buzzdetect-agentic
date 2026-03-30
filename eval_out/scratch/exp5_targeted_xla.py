"""
Experiment 5: Targeted XLA JIT with concrete input shape.

08_xla_jit tested tf.config.optimizer.set_jit(True) GLOBALLY, which caused
recompilation overhead for TFSMLayer's dynamic-shape graphs → 22.9% SLOWER.

This test uses @tf.function(jit_compile=True) with a FIXED input shape
(200s = 3,194,880 samples after frame-rounding at 16kHz/0.96s frame).
XLA compiles once for this exact shape → no retracing, kernel fusion.

Handles shorter last chunks via a separate non-JIT path.

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp5_targeted_xla.py
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
from src.analyze import analyze

MODEL = 'model_general_v3'

# ── Figure out exact chunk size for 200s at 16kHz with 0.96s frames ───────────
# analyze.py rounds chunklength to nearest frame: round(200/0.96)*0.96 = 208*0.96 = 199.68s
# n_samples = int(199.68 * 16000) = 3,194,880
CHUNK_S = round(200 / 0.96) * 0.96     # = 199.68s
N_SAMPLES_CHUNK = int(CHUNK_S * 16000)  # = 3,194,880

print(f"Chunk size: {CHUNK_S}s = {N_SAMPLES_CHUNK} samples")

# ── Build targeted XLA-compiled predict function ───────────────────────────────
model = load_model(MODEL, framehop_prop=1.0, initialize=False)
model.initialize()

embedder_fn  = model.embedder.model
classifier_fn = model.model

# XLA-compiled version for the standard chunk size
@tf.function(
    jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(N_SAMPLES_CHUNK,), dtype=tf.float32)]
)
def predict_xla(samples):
    embeddings = embedder_fn(samples)['global_average_pooling2d']
    return classifier_fn(embeddings)['dense']

# Non-JIT fallback for shorter last chunks
@tf.function(
    input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)]
)
def predict_fallback(samples):
    embeddings = embedder_fn(samples)['global_average_pooling2d']
    return classifier_fn(embeddings)['dense']

# ── Prewarm both ───────────────────────────────────────────────────────────────
dummy_std = np.random.randn(N_SAMPLES_CHUNK).astype(np.float32)
dummy_short = np.random.randn(N_SAMPLES_CHUNK // 2).astype(np.float32)

print("Compiling XLA function (first call may take a moment)...")
t0 = time.perf_counter()
_ = predict_xla(dummy_std).numpy()
t1 = time.perf_counter()
print(f"XLA compilation: {(t1-t0)*1000:.0f}ms (one-time cost)")

_ = predict_fallback(dummy_short).numpy()
print("Fallback compiled.")

# ── Benchmark: XLA vs baseline predict() ──────────────────────────────────────
REPS = 20

# Baseline
baseline_times = []
for _ in range(REPS + 3):
    t0 = time.perf_counter()
    r = model.predict(dummy_std)
    _ = r.numpy()
    baseline_times.append((time.perf_counter() - t0) * 1000)
baseline_times = baseline_times[3:]

# XLA
xla_times = []
for _ in range(REPS + 3):
    t0 = time.perf_counter()
    r = predict_xla(dummy_std)
    _ = r.numpy()
    xla_times.append((time.perf_counter() - t0) * 1000)
xla_times = xla_times[3:]

print(f"\n=== Per-chunk timing (200s = {N_SAMPLES_CHUNK} samples) ===")
print(f"  Baseline model.predict(): {np.mean(baseline_times):.1f}ms ± {np.std(baseline_times):.1f}ms")
print(f"  XLA-compiled predict:     {np.mean(xla_times):.1f}ms ± {np.std(xla_times):.1f}ms")
print(f"  Speedup: {np.mean(baseline_times)/np.mean(xla_times):.2f}x")

# ── Full pipeline test via monkey-patch ───────────────────────────────────────
import shutil, csv
import src.inference.worker as infer_mod

_orig_process_chunk = infer_mod.WorkerInferer.process_chunk

# We build the XLA predict at module level so the patch can use it
_predict_xla = predict_xla
_predict_fallback = predict_fallback
_n_samples_chunk = N_SAMPLES_CHUNK

def xla_process_chunk(self, a_chunk):
    with self.coordinator.profiler.phase('inference'):
        samples = a_chunk.samples
        if not isinstance(samples, tf.Tensor):
            samples = tf.constant(samples)
        # Route to XLA path for standard chunk size, fallback otherwise
        if samples.shape[0] == _n_samples_chunk:
            a_chunk.results = _predict_xla(samples)
        else:
            a_chunk.results = _predict_fallback(samples)
    self.coordinator.q_write.put(a_chunk)
    self.report_rate(a_chunk)

def read_phase(path, phase):
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row['phase'] == phase:
                    return float(row['total_s'])
    except:
        return None

AUDIO_DIR = 'audio_eval'
RUNS = 3

print(f"\n=== Full pipeline test (real analyze(), {RUNS} runs each) ===")

print("\nBaseline:")
bl_overall, bl_infer = [], []
for i in range(RUNS):
    out = f'/tmp/exp5_baseline_{i}'
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.perf_counter()
    analyze(modelname=MODEL, classes_out='all', chunklength=200,
            analyzer_gpu=True, analyzers_cpu=0, n_streamers=6,
            dir_audio=AUDIO_DIR, dir_out=out,
            verbosity_print='WARNING', verbosity_log='WARNING',
            profile=True, profile_path=f'/tmp/exp5_baseline_{i}.csv',
            silent_profile=True)
    t1 = time.perf_counter()
    ov = read_phase(f'/tmp/exp5_baseline_{i}.csv', 'overall')
    inf = read_phase(f'/tmp/exp5_baseline_{i}.csv', 'inference')
    bl_overall.append(ov); bl_infer.append(inf)
    print(f"  Run {i+1}: {t1-t0:.2f}s  overall={ov:.3f}s  inference={inf:.3f}s")

print("\nXLA-patched:")
xla_overall, xla_infer = [], []
for i in range(RUNS):
    infer_mod.WorkerInferer.process_chunk = xla_process_chunk
    out = f'/tmp/exp5_xla_{i}'
    shutil.rmtree(out, ignore_errors=True)
    try:
        t0 = time.perf_counter()
        analyze(modelname=MODEL, classes_out='all', chunklength=200,
                analyzer_gpu=True, analyzers_cpu=0, n_streamers=6,
                dir_audio=AUDIO_DIR, dir_out=out,
                verbosity_print='WARNING', verbosity_log='WARNING',
                profile=True, profile_path=f'/tmp/exp5_xla_{i}.csv',
                silent_profile=True)
        t1 = time.perf_counter()
    finally:
        infer_mod.WorkerInferer.process_chunk = _orig_process_chunk
    ov = read_phase(f'/tmp/exp5_xla_{i}.csv', 'overall')
    inf = read_phase(f'/tmp/exp5_xla_{i}.csv', 'inference')
    xla_overall.append(ov); xla_infer.append(inf)
    print(f"  Run {i+1}: {t1-t0:.2f}s  overall={ov:.3f}s  inference={inf:.3f}s")

print(f"\nSummary:")
print(f"  Overall:   baseline={np.mean(bl_overall):.3f}s  xla={np.mean(xla_overall):.3f}s  delta={np.mean(xla_overall)-np.mean(bl_overall):+.3f}s ({(np.mean(xla_overall)-np.mean(bl_overall))/np.mean(bl_overall)*100:+.1f}%)")
print(f"  Inference: baseline={np.mean(bl_infer):.3f}s   xla={np.mean(xla_infer):.3f}s   delta={np.mean(xla_infer)-np.mean(bl_infer):+.3f}s")
