"""
Benchmark: original sequential read vs threaded read/resample pipeline.
Simulates 6 concurrent streamers on the real eval audio.
"""
import os
import threading
import time
import queue

import librosa
import numpy as np
import soundfile as sf
import soxr

AUDIO_DIR = "audio_eval/files"
CHUNK_S = 200
N_STREAMERS = 6
N_CHUNKS = 6  # one chunk per file, first chunk only

files = []
for d in os.listdir(AUDIO_DIR):
    dpath = os.path.join(AUDIO_DIR, d)
    if os.path.isdir(dpath):
        for f in os.listdir(dpath):
            if f.endswith(".mp3"):
                files.append(os.path.join(dpath, f))
files = files[:N_STREAMERS]
print(f"Using {len(files)} files")

TARGET_SR = 16000


def read_one_chunk_sequential_librosa(path):
    """Original approach: read + librosa resample, sequential."""
    track = sf.SoundFile(path)
    sr = track.samplerate
    n = int(CHUNK_S * sr)
    t0 = time.perf_counter()
    samples = track.read(n, dtype=np.float32)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    samples = librosa.resample(y=samples, orig_sr=sr, target_sr=TARGET_SR)
    t_resample = time.perf_counter() - t0
    return t_read, t_resample


def read_one_chunk_sequential_soxr(path):
    """soxr direct, sequential."""
    track = sf.SoundFile(path)
    sr = track.samplerate
    n = int(CHUNK_S * sr)
    t0 = time.perf_counter()
    samples = track.read(n, dtype=np.float32)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    samples = soxr.resample(samples, sr, TARGET_SR, quality='HQ')
    t_resample = time.perf_counter() - t0
    return t_read, t_resample


def read_one_chunk_threaded_soxr(path):
    """Threaded: reader thread reads, main thread resamples."""
    track = sf.SoundFile(path)
    sr = track.samplerate
    n = int(CHUNK_S * sr)
    raw_q = queue.Queue(maxsize=2)

    def reader():
        t0 = time.perf_counter()
        samples = track.read(n, dtype=np.float32)
        t_read = time.perf_counter() - t0
        raw_q.put((samples, t_read))
        raw_q.put(None)

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    item = raw_q.get()
    raw_q.get()  # sentinel
    rt.join()
    samples, t_read = item

    t0 = time.perf_counter()
    samples = soxr.resample(samples, sr, TARGET_SR, quality='HQ')
    t_resample = time.perf_counter() - t0
    return t_read, t_resample


def run_concurrent(fn, files, label):
    results = [None] * len(files)

    def worker(i):
        results[i] = fn(files[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(files))]
    t_wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_wall0

    reads = [r[0] for r in results]
    resamples = [r[1] for r in results]
    print(f"\n{label} ({len(files)} concurrent streamers):")
    print(f"  Wall time:        {wall:.3f}s")
    print(f"  Read   mean/max:  {sum(reads)/len(reads):.3f}s / {max(reads):.3f}s")
    print(f"  Resamp mean/max:  {sum(resamples)/len(resamples):.3f}s / {max(resamples):.3f}s")
    return wall


# Warmup
print("Warming up...")
for f in files:
    read_one_chunk_sequential_soxr(f)

print("\n=== Concurrent streamer benchmark (6 threads) ===")
for _ in range(3):
    w1 = run_concurrent(read_one_chunk_sequential_librosa, files, "Sequential + librosa (original)")
    w2 = run_concurrent(read_one_chunk_sequential_soxr, files, "Sequential + soxr direct")
    w3 = run_concurrent(read_one_chunk_threaded_soxr, files, "Threaded read + soxr resample")
    print(f"\nSpeedup soxr vs librosa: {w1/w2:.2f}x")
    print(f"Speedup threaded vs librosa: {w1/w3:.2f}x")
    print("---")
