"""
Experiment 4: Pre-convert numpy→tensor in streamer threads (real pipeline).

Streamers spend ~55ms/chunk waiting for queue space (audio_io/fullqueue).
The tf.constant() conversion (~4.55ms) fits inside this dead time for free.
GPU worker then receives pre-built tensors, saving ~9.8ms/chunk from its hot path.

Monkey-patches WorkerStreamer.stream_to_queue to call tf.constant() after
resampling, before enqueuing. WorkerInferer.process_chunk passes the tensor
directly to model.predict() when available.

Run from /workspace:
    .venv/bin/python eval_out/scratch/exp4_streamer_tensor.py
"""
import os, sys, shutil, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from src.analyze import analyze
import src.stream.worker as stream_mod
import src.inference.worker as infer_mod
from src.pipeline.assignments import AssignChunk

MODEL = 'model_general_v3'
AUDIO_DIR = 'audio_eval'
RUNS = 3

# ── Patch: streamer converts numpy → tensor after resampling ──────────────────
_orig_stream_to_queue = stream_mod.WorkerStreamer.stream_to_queue

def patched_stream_to_queue(self, a_file):
    import soxr, soundfile as sf
    from queue import Full
    from src import config as cfg
    from src.stream.audio import mark_eof

    track = sf.SoundFile(a_file.path_audio)
    samplerate_native = track.samplerate

    def queue_chunk(chunk, track, samplerate_native):
        sample_from = int(chunk[0] * samplerate_native)
        sample_to = int(chunk[1] * samplerate_native)
        read_size = sample_to - sample_from

        with self.coordinator.profiler.phase('audio_io/reading'):
            track.seek(sample_from)
            samples = track.read(read_size, dtype=np.float32)
            if track.channels > 1:
                samples = np.mean(samples, axis=1)
            n_samples = len(samples)
            if n_samples < read_size:
                self.handle_bad_read(track, a_file)
                chunk = (chunk[0], round(chunk[0] + (n_samples / track.samplerate), 1))
                abort_stream = True
            else:
                abort_stream = False

        with self.coordinator.profiler.phase('audio_io/resampling'):
            samples = soxr.resample(samples, samplerate_native, self.resample_rate, quality='HQ')

        # Pre-convert to tensor here — runs concurrently with other streamers,
        # absorbed into queue-wait time (audio_io/fullqueue).
        tensor_samples = tf.constant(samples)

        a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=tensor_samples)
        t_wait_start = time.perf_counter()
        while not self.coordinator.event_exitanalysis.is_set():
            try:
                self.coordinator.q_analyze.put(a_chunk, timeout=3)
                wait_s = time.perf_counter() - t_wait_start
                if not self._enqueue_skipped_first:
                    self._enqueue_skipped_first = True
                else:
                    self.coordinator.profiler.record('audio_io/fullqueue', wait_s)
                break
            except Full:
                continue
        return abort_stream

    for chunk in a_file.chunklist:
        abort_stream = queue_chunk(chunk, track, samplerate_native)
        if abort_stream:
            return True
        if self.coordinator.event_exitanalysis.is_set():
            self.log("exit event set, terminating", 'DEBUG')
            return False
    return True


# ── Baseline ──────────────────────────────────────────────────────────────────
def run_baseline(run_idx):
    out_dir = f'/tmp/exp4_baseline_{run_idx}'
    shutil.rmtree(out_dir, ignore_errors=True)
    analyze(
        modelname=MODEL, classes_out='all', chunklength=200,
        analyzer_gpu=True, analyzers_cpu=0, n_streamers=6,
        dir_audio=AUDIO_DIR, dir_out=out_dir,
        verbosity_print='WARNING', verbosity_log='WARNING',
        profile=True, profile_path=f'/tmp/exp4_baseline_{run_idx}.csv',
        silent_profile=True,
    )

def run_patched(run_idx):
    stream_mod.WorkerStreamer.stream_to_queue = patched_stream_to_queue
    out_dir = f'/tmp/exp4_patched_{run_idx}'
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        analyze(
            modelname=MODEL, classes_out='all', chunklength=200,
            analyzer_gpu=True, analyzers_cpu=0, n_streamers=6,
            dir_audio=AUDIO_DIR, dir_out=out_dir,
            verbosity_print='WARNING', verbosity_log='WARNING',
            profile=True, profile_path=f'/tmp/exp4_patched_{run_idx}.csv',
            silent_profile=True,
        )
    finally:
        stream_mod.WorkerStreamer.stream_to_queue = _orig_stream_to_queue

# ── Read overall time from profile ────────────────────────────────────────────
import csv

def read_phase(path, phase):
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row['phase'] == phase:
                    return float(row['total_s'])
    except:
        return None

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Exp 4: streamer tensor pre-conversion vs baseline")
print("=" * 60)

print("\nBaseline (numpy in queue):")
for i in range(RUNS):
    t0 = time.perf_counter()
    run_baseline(i)
    t1 = time.perf_counter()
    overall = read_phase(f'/tmp/exp4_baseline_{i}.csv', 'overall')
    infer   = read_phase(f'/tmp/exp4_baseline_{i}.csv', 'inference')
    print(f"  Run {i+1}: wall={t1-t0:.2f}s  overall={overall:.3f}s  inference={infer:.3f}s")

print("\nPatched (tensor in queue, converted in streamer):")
for i in range(RUNS):
    t0 = time.perf_counter()
    run_patched(i)
    t1 = time.perf_counter()
    overall = read_phase(f'/tmp/exp4_patched_{i}.csv', 'overall')
    infer   = read_phase(f'/tmp/exp4_patched_{i}.csv', 'inference')
    print(f"  Run {i+1}: wall={t1-t0:.2f}s  overall={overall:.3f}s  inference={infer:.3f}s")

# Summary
bl_overall = [read_phase(f'/tmp/exp4_baseline_{i}.csv', 'overall') for i in range(RUNS)]
pt_overall = [read_phase(f'/tmp/exp4_patched_{i}.csv',  'overall') for i in range(RUNS)]
bl_infer   = [read_phase(f'/tmp/exp4_baseline_{i}.csv', 'inference') for i in range(RUNS)]
pt_infer   = [read_phase(f'/tmp/exp4_patched_{i}.csv',  'inference') for i in range(RUNS)]

print(f"\nSummary:")
print(f"  Overall:   baseline={np.mean(bl_overall):.3f}s  patched={np.mean(pt_overall):.3f}s  delta={np.mean(pt_overall)-np.mean(bl_overall):+.3f}s ({(np.mean(pt_overall)-np.mean(bl_overall))/np.mean(bl_overall)*100:+.1f}%)")
print(f"  Inference: baseline={np.mean(bl_infer):.3f}s   patched={np.mean(pt_infer):.3f}s   delta={np.mean(pt_infer)-np.mean(bl_infer):+.3f}s")
