"""
Experiment 3: Pre-convert numpy→tensor in streamer threads vs. current approach.

Tests whether moving tf.constant() out of the GPU worker's hot path
(into the streamers, which run concurrently) reduces overall inference time.

Uses the REAL analyze() pipeline via monkey-patching, with audio_eval audio.

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp3_tensor_preload.py
"""
import os, sys, shutil, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.analyze import analyze

MODEL = 'model_general_v3'
AUDIO_DIR = 'audio_eval'
RUNS = 3  # average over N runs for stability

# ─── Baseline: current behavior ──────────────────────────────────────────────
def run_baseline(run_idx):
    out_dir = f'/tmp/exp3_baseline_{run_idx}'
    shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    analyze(
        modelname=MODEL,
        classes_out='all',
        chunklength=200,
        analyzer_gpu=True,
        analyzers_cpu=0,
        n_streamers=6,
        dir_audio=AUDIO_DIR,
        dir_out=out_dir,
        verbosity_print='WARNING',
        verbosity_log='WARNING',
        profile=True,
        profile_path=f'/tmp/exp3_baseline_{run_idx}.csv',
        silent_profile=True,
    )
    t1 = time.perf_counter()
    return t1 - t0

# ─── Patched: pre-convert numpy→tensor in streamer ────────────────────────────
# We monkey-patch AssignChunk to carry a tensor, WorkerStreamer to pre-convert,
# and WorkerInferer.process_chunk to use the tensor if available.

import src.pipeline.assignments as assignments_mod
import src.stream.worker as stream_mod
import src.inference.worker as infer_mod

# Save originals
_orig_AssignChunk_init = None  # AssignChunk is a dataclass, patch differently
_orig_streamer_add_to_queue = None
_orig_process_chunk = infer_mod.WorkerInferer.process_chunk

# Patch WorkerStreamer: after building a_chunk with samples, pre-convert to tensor
_orig_streamer_call = stream_mod.WorkerStreamer.__call__

# Simpler approach: patch WorkerInferer.process_chunk to convert BEFORE calling predict
# This doesn't move work to streamer threads but eliminates the implicit conversion
# inside TFSMLayer by passing a pre-built tensor.
# (Pre-built tensor means TFSMLayer skips its internal numpy-check path)

def patched_process_chunk(self, a_chunk):
    # Pre-convert numpy samples → GPU tensor in the GPU worker
    # (same thread, but explicit — tests if explicit pre-conversion is faster)
    with self.coordinator.profiler.phase('inference'):
        tensor_samples = tf.constant(a_chunk.samples)
        a_chunk.results = self.model.predict(tensor_samples)
    self.coordinator.q_write.put(a_chunk)
    self.report_rate(a_chunk)

# We also need model.predict() to accept tensors cleanly.
# model_general_v3/model.py: predict calls embedder.embed(audiosamples)
# embedder.embed calls self.model(audiosamples)['global_average_pooling2d']
# TFSMLayer(tensor) works fine.

def run_explicit_tensor(run_idx):
    """Same as baseline but explicit tf.constant() before model.predict()"""
    # Patch
    infer_mod.WorkerInferer.process_chunk = patched_process_chunk

    out_dir = f'/tmp/exp3_explicit_tensor_{run_idx}'
    shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    analyze(
        modelname=MODEL,
        classes_out='all',
        chunklength=200,
        analyzer_gpu=True,
        analyzers_cpu=0,
        n_streamers=6,
        dir_audio=AUDIO_DIR,
        dir_out=out_dir,
        verbosity_print='WARNING',
        verbosity_log='WARNING',
        profile=True,
        profile_path=f'/tmp/exp3_explicit_tensor_{run_idx}.csv',
        silent_profile=True,
    )
    t1 = time.perf_counter()

    # Restore
    infer_mod.WorkerInferer.process_chunk = _orig_process_chunk
    return t1 - t0

# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Exp 3: numpy vs. pre-built tensor input to GPU worker")
print("=" * 60)
print(f"\nBaseline (current — numpy to model.predict):")
baseline_times = []
for i in range(RUNS):
    t = run_baseline(i)
    baseline_times.append(t)
    print(f"  Run {i+1}: {t:.3f}s")
print(f"  Mean: {np.mean(baseline_times):.3f}s ± {np.std(baseline_times):.3f}s")

print(f"\nExplicit tf.constant() in process_chunk before predict:")
tensor_times = []
for i in range(RUNS):
    t = run_explicit_tensor(i)
    tensor_times.append(t)
    print(f"  Run {i+1}: {t:.3f}s")
print(f"  Mean: {np.mean(tensor_times):.3f}s ± {np.std(tensor_times):.3f}s")

delta_pct = (np.mean(tensor_times) - np.mean(baseline_times)) / np.mean(baseline_times) * 100
print(f"\nDelta: {delta_pct:+.1f}% ({'faster' if delta_pct < 0 else 'slower'})")

# ─── Read profiles to check inference-specific time ───────────────────────────
import csv

def read_inference_time(path):
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row['phase'] == 'inference':
                    return float(row['total_s'])
    except:
        return None

print("\nInference phase times (from profiler):")
for i in range(RUNS):
    bl = read_inference_time(f'/tmp/exp3_baseline_{i}.csv')
    et = read_inference_time(f'/tmp/exp3_explicit_tensor_{i}.csv')
    print(f"  Run {i+1}: baseline={bl:.3f}s  explicit_tensor={et:.3f}s  delta={et-bl:+.3f}s")
