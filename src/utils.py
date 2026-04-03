import csv
import os
import re
import statistics
import threading
import time
from datetime import datetime


class Timer:
    def __init__(self):
        self.time_start = datetime.now()
        self.time_end = datetime.now()

    def stop(self):
        self.time_end = datetime.now()

    def restart(self):
        self.time_start = datetime.now()

    def get_current(self):
        return datetime.now() - self.time_start

    def get_total(self, decimals=2):
        time_total = self.time_end - self.time_start
        total_formatted = time_total.total_seconds().__round__(decimals)

        return total_formatted


class Profiler:
    """Thread-safe phase profiler. Accumulates named timing measurements across threads.

    Supports subphases via slash notation (e.g. ``'audio_io/reading'``). Subphases
    are displayed indented under their parent in the summary and sorted to appear
    immediately after the parent phase.

    Usage::

        profiler = Profiler(enabled=True)

        with profiler.phase('audio_io/reading'):
            samples = track.read(...)
        with profiler.phase('audio_io/resampling'):
            samples = librosa.resample(...)

        print(profiler.summary())
        profiler.save_csv('/path/to/output/profile.csv')
    """

    _FIELDS = ['phase', 'n', 'total_s', 'mean_s', 'median_s', 'sd_s', 'pct_of_overall']

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._phases: dict[str, list[float]] = {}

    def record(self, phase: str, duration_s: float):
        """Record a timing measurement (seconds) for a named phase."""
        if not self.enabled:
            return
        with self._lock:
            if phase not in self._phases:
                self._phases[phase] = []
            self._phases[phase].append(duration_s)

    class _PhaseContext:
        __slots__ = ('_profiler', '_phase', '_t0')

        def __init__(self, profiler, phase):
            self._profiler = profiler
            self._phase = phase
            self._t0 = None

        def __enter__(self):
            if self._profiler.enabled:
                self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            if self._profiler.enabled and self._t0 is not None:
                self._profiler.record(self._phase, time.perf_counter() - self._t0)

    def phase(self, name: str) -> '_PhaseContext':
        """Context manager that records elapsed time for the named phase."""
        return self._PhaseContext(self, name)

    def _build_rows(self) -> list[dict]:
        """Compute per-phase statistics. Caller must hold self._lock."""
        overall_total = sum(self._phases.get('overall', [0.0]))
        rows = []
        for ph, durations in self._phases.items():
            n = len(durations)
            total = sum(durations)
            mean = statistics.mean(durations)
            median = statistics.median(durations)
            sd = statistics.stdev(durations) if n >= 2 else 0.0
            pct = (total / overall_total * 100) if overall_total > 0 and ph != 'overall' else None
            rows.append({
                'phase': ph,
                'n': n,
                'total_s': total,
                'mean_s': mean,
                'median_s': median,
                'sd_s': sd,
                'pct_of_overall': pct,
            })
        # Sort: 'overall' first, then top-level phases by descending total,
        # with subphases (slash notation) grouped immediately after their parent.
        phase_totals = {r['phase']: r['total_s'] for r in rows}

        def _sort_key(r):
            ph = r['phase']
            if ph == 'overall':
                return (0, 0.0, '', 0, 0.0)
            if '/' in ph:
                parent = ph.split('/')[0]
                parent_total = phase_totals.get(parent, 0.0)
                return (1, -parent_total, parent, 1, -r['total_s'])
            return (1, -r['total_s'], ph, 0, 0.0)

        rows.sort(key=_sort_key)
        return rows

    def summary(self) -> str:
        """Return a formatted console table of per-phase statistics."""
        with self._lock:
            if not self._phases:
                return "Profiler: no data recorded."
            rows = self._build_rows()

        phase_names = {r['phase'] for r in rows}
        lines = ["\n=== PROFILING SUMMARY ==="]
        for r in rows:
            pct_str = f"  ({r['pct_of_overall']:.1f}%)" if r['pct_of_overall'] is not None else ""
            if '/' in r['phase'] and r['phase'].split('/')[0] in phase_names:
                label = '  ' + r['phase'].split('/', 1)[1]
                width = 20
            else:
                label = r['phase']
                width = 22
            if r['n'] == 1:
                lines.append(
                    f"  {label:<{width}} {r['total_s']:>8.3f}s{pct_str}"
                )
            else:
                lines.append(
                    f"  {label:<{width}} {r['total_s']:>8.3f}s  n={r['n']:<6}"
                    f"  mean={r['mean_s']:.4f}s  median={r['median_s']:.4f}s  sd={r['sd_s']:.4f}s{pct_str}"
                )
        lines.append("=" * 45)
        return "\n".join(lines)

    def save_csv(self, path: str):
        """Write per-phase statistics to a CSV file at the given path."""
        with self._lock:
            if not self._phases:
                return
            rows = self._build_rows()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    'phase': r['phase'],
                    'n': r['n'],
                    'total_s': f"{r['total_s']:.6f}",
                    'mean_s': f"{r['mean_s']:.6f}",
                    'median_s': f"{r['median_s']:.6f}",
                    'sd_s': f"{r['sd_s']:.6f}",
                    'pct_of_overall': f"{r['pct_of_overall']:.2f}" if r['pct_of_overall'] is not None else '',
                })


def setup_chunklength(chunklength, framelength_s, digits_time):
    """Round chunklength to the nearest whole number of frames."""
    chunklength = round(chunklength / framelength_s) * framelength_s
    chunklength = round(chunklength, digits_time)
    if chunklength < framelength_s:
        chunklength = framelength_s
    return chunklength


def search_dir(dir_in, extensions=None):
    if extensions is not None and not (extensions.__class__ is list and extensions[0].__class__ is str):
        raise ValueError("input extensions should be None, or list of strings")

    paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            paths.append(os.path.join(root, file))

    if extensions is None:
        return paths

    # convert extensions into regex, if they aren't already
    for i, extension in enumerate(extensions):
        if extension[-1] != "$":
            extension = extension + "$"

        extension = extension.lower()
        extensions[i] = extension

    paths = [p for p in paths if True in [bool(re.search(e, p.lower())) for e in extensions]]
    return paths


def build_ident(path, root_dir, tag=None):
    ident = re.sub(root_dir, '', path)
    ident = os.path.splitext(ident)[0]

    if tag is not None:
        ident = re.sub(re.escape(tag), '', ident)

    ident = re.sub('^/', '', ident)

    return ident
