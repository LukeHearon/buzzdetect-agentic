import queue
import threading
import time
from queue import Full

import numpy as np
import soundfile as sf
import soxr

from src import config as cfg
from src.pipeline.assignments import AssignFile, AssignChunk, AssignLog
from src.pipeline.coordination import Coordinator
from src.stream.audio import mark_eof


class WorkerStreamer:
    def __init__(self,
                 id_streamer,
                 resample_rate: float,
                 coordinator: Coordinator,
                 prepfunc=None, ):

        self.id_streamer = id_streamer
        self.resample_rate = resample_rate
        self.coordinator = coordinator
        self.prepfunc = prepfunc
        self._enqueue_skipped_first = False

    def __call__(self):
        self.run()

    def log(self, msg, level_str):
        self.coordinator.q_log.put(AssignLog(message=f'streamer {self.id_streamer}: {msg}', level_str=level_str))

    def handle_bad_read(self, track: sf.SoundFile, a_file: AssignFile):
        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.

        final_frame = track.tell()
        mark_eof(path_audio=a_file.path_audio, final_frame=final_frame)

        final_second = final_frame/track.samplerate

        msg = f"Unreadable audio at {round(final_second, 1)}s out of {round(a_file.duration_audio, 1)}s for {a_file.shortpath_audio}."
        if 1 - (final_second / a_file.duration_audio) > cfg.BAD_READ_ALLOWANCE:
            # if we get a bad read in the middle of a file, this deserves a warning.
            level = 'WARNING'
            msg += '\nAborting early due to corrupt audio data.'
        else:
            # but bad reads at the ends of files are almost guaranteed when the batteries run out
            level = 'DEBUG'
            msg += '\nBad audio is near file end, results should be mostly unaffected.'

        self.log(msg, level)

    def _run_io_stage(self, a_file: AssignFile, local_q: queue.Queue):
        """Stage 1: read + resample. Puts (chunk, samples, abort_file) tuples into local_q,
        then a None sentinel when done."""
        track = sf.SoundFile(a_file.path_audio)
        samplerate_native = track.samplerate

        for chunk in a_file.chunklist:
            if self.coordinator.event_exitanalysis.is_set():
                break

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
                    chunk = (chunk[0], round(chunk[0] + (n_samples/track.samplerate), 1))
                    abort_file = True
                else:
                    abort_file = False

            with self.coordinator.profiler.phase('audio_io/resampling'):
                samples = soxr.resample(samples, samplerate_native, self.resample_rate, quality='HQ')

            # Enqueue to local buffer; loop with timeout to respect exit event
            while not self.coordinator.event_exitanalysis.is_set():
                try:
                    local_q.put((chunk, samples, abort_file), timeout=0.5)
                    break
                except queue.Full:
                    continue

            if abort_file or self.coordinator.event_exitanalysis.is_set():
                break

        # Signal completion. Loop handles the case where local_q is full after
        # the convert stage has exited (exit event): drain one item to make room.
        while True:
            try:
                local_q.put(None, timeout=0.5)
                break
            except queue.Full:
                try:
                    local_q.get_nowait()
                except queue.Empty:
                    pass

    def _run_convert_stage(self, a_file: AssignFile, local_q: queue.Queue):
        """Stage 2: apply prepfunc + enqueue to q_analyze. Returns when it sees the None sentinel
        or when the exit event is set."""
        while True:
            if self.coordinator.event_exitanalysis.is_set():
                return

            try:
                item = local_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                return

            chunk, samples, abort_file = item

            if self.prepfunc is not None:
                samples = self.prepfunc(samples)

            a_chunk = AssignChunk(file=a_file, chunk=chunk, samples=samples)
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

            if abort_file:
                # IO stage broke after this chunk; only None remains in local_q.
                # Drain it so io_thread can finish, then stop processing this file.
                while True:
                    try:
                        remaining = local_q.get(timeout=0.5)
                        if remaining is None:
                            break
                    except queue.Empty:
                        if self.coordinator.event_exitanalysis.is_set():
                            break
                return

    def stream_to_queue(self, a_file: AssignFile):
        local_q = queue.Queue(maxsize=2)

        io_thread = threading.Thread(
            target=self._run_io_stage,
            args=(a_file, local_q),
            daemon=True,
        )
        io_thread.start()

        # This thread runs the convert stage while the io_thread runs Stage 1.
        self._run_convert_stage(a_file, local_q)

        # If convert exited early (exit event), drain local_q so io_thread's
        # put() can unblock and finish.
        while True:
            try:
                local_q.get_nowait()
            except queue.Empty:
                break

        io_thread.join()

        if self.coordinator.event_exitanalysis.is_set():
            return False
        return True

    def _prewarm_resample(self):
        """Trigger soxr's filter computation before timed analysis."""
        dummy = np.zeros(44100, dtype=np.float32)  # 1s at a common source rate
        soxr.resample(dummy, 44100, self.resample_rate, quality='HQ')

    def run(self):
        self.log('launching', 'INFO')
        self._prewarm_resample()
        a_stream = self.coordinator.q_stream.get()
        while a_stream is not None:
            self.log(f"buffering {a_stream.shortpath_audio}", 'INFO')
            keep_streaming = self.stream_to_queue(a_stream)
            if not keep_streaming:
                break

            a_stream = self.coordinator.q_stream.get()

        self.log("terminating", 'INFO')
