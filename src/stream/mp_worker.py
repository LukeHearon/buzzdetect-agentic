"""Multiprocessing-based streamer worker.

Provides a top-level picklable function for use with multiprocessing.Process.
Timing data is collected locally per-process and returned via q_profile at exit
as a list of (phase_name, duration_s) tuples, to be merged into the main Profiler.
"""
import time

import librosa
import numpy as np
import soundfile as sf

from src import config as cfg
from src.pipeline.assignments import AssignFile, AssignChunk


def _handle_bad_read(track, a_file, n_samples, read_size):
    """Truncate chunk duration to match actual readable samples."""
    final_frame = track.tell()
    final_second = final_frame / track.samplerate
    chunk_start = a_file.chunklist[0][0] if a_file.chunklist else 0  # unused; caller provides
    return final_second


def run_mp_streamer(id_streamer, resample_rate, q_stream, q_analyze, q_profile, event_exit):
    """Entry point for a multiprocessing streamer process.

    Reads audio files chunk by chunk, resamples, and enqueues AssignChunk objects
    into q_analyze (a multiprocessing.Queue). Profiling data is returned via
    q_profile as a list of (phase, duration_s) tuples when the process finishes.

    Note: each AssignChunk is pickled when placed in q_analyze.  For a 200-second
    chunk at 16 kHz the numpy array is ~12.8 MB, so pickling is the dominant IPC
    cost.  This is the mechanism under test for the 02_mp_streamers evaluation.
    """
    profile_data = []

    # Prewarm librosa (lazy backend import + filter computation)
    dummy = np.zeros(44100, dtype=np.float32)
    librosa.resample(y=dummy, orig_sr=44100, target_sr=resample_rate)

    while True:
        a_file = q_stream.get()
        if a_file is None:
            break

        track = sf.SoundFile(a_file.path_audio)
        samplerate_native = track.samplerate
        abort_file = False

        for chunk in a_file.chunklist:
            if event_exit.is_set():
                abort_file = True
                break

            sample_from = int(chunk[0] * samplerate_native)
            sample_to = int(chunk[1] * samplerate_native)
            read_size = sample_to - sample_from

            # --- Read ---
            t0 = time.perf_counter()
            track.seek(sample_from)
            samples = track.read(read_size, dtype=np.float32)
            if track.channels > 1:
                samples = np.mean(samples, axis=1)
            n_samples = len(samples)
            profile_data.append(('audio_io/reading', time.perf_counter() - t0))

            if n_samples < read_size:
                # Corrupt/truncated audio — adjust chunk end and skip remaining
                chunk = (chunk[0], round(chunk[0] + (n_samples / samplerate_native), 1))
                abort_file = True

            # --- Resample ---
            t0 = time.perf_counter()
            samples = librosa.resample(y=samples, orig_sr=samplerate_native, target_sr=resample_rate)
            profile_data.append(('audio_io/resampling', time.perf_counter() - t0))

            # --- Enqueue chunk (pickle happens here) ---
            a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=samples)
            while not event_exit.is_set():
                try:
                    q_analyze.put(a_chunk, timeout=3)
                    break
                except Exception:
                    continue

            if abort_file:
                break

        track.close()
        if event_exit.is_set():
            break

    # Return collected profiling data to main process
    q_profile.put(profile_data)
