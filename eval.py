import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

# Reduce TensorFlow and other verbose logs where possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from src.analyze import analyze
except Exception as e:
    print(f"Failed to import analyze from src.analyze: {e}", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "model_general_v3"
BASELINE_PROFILE = Path("eval_out/baseline/profile.csv")

# Grid swept on every run (baseline and non-baseline). All combinations are
# tried; the best overall runtime is promoted to the top-level output directory.
# Do not modify — this is managed infrastructure, not an agent tuning target.
TUNE_GRID = {
    "chunklength":         [150, 200, 300],
    "n_streamers":         [4, 6, 8],
    "stream_buffer_depth": [4, 6],
}


def _combo_label(chunklength, n_streamers, stream_buffer_depth):
    cl = int(chunklength) if chunklength == int(chunklength) else chunklength
    return f"{cl}s_{n_streamers}str_buf{stream_buffer_depth}"


def _read_overall_time(profile_path: Path):
    """Return the overall mean_s from a profile CSV, or None if unavailable."""
    if not profile_path.exists():
        return None
    with profile_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] == "overall":
                return float(row["mean_s"])
    return None


def _compare_profiles(current_path: Path, baseline_path: Path):
    """Compare two profile CSVs. Returns (rows, overall_delta_s, overall_delta_pct)."""
    def load(p):
        with p.open(newline="") as f:
            return {row["phase"]: row for row in csv.DictReader(f)}

    baseline = load(baseline_path)
    current = load(current_path)

    rows = []
    overall_delta_s = None
    overall_delta_pct = None

    for phase, b_row in baseline.items():
        if phase not in current:
            continue
        b_s = float(b_row["mean_s"])
        c_s = float(current[phase]["mean_s"])
        delta_s = c_s - b_s
        delta_pct = (delta_s / b_s * 100) if b_s != 0 else 0.0
        row = {
            "phase": phase,
            "baseline_s": b_s,
            "current_s": c_s,
            "delta_s": delta_s,
            "delta_pct": delta_pct,
        }
        rows.append(row)
        if phase == "overall":
            overall_delta_s = delta_s
            overall_delta_pct = delta_pct

    return rows, overall_delta_s, overall_delta_pct


def _print_comparison(rows, overall_delta_s, overall_delta_pct):
    print("\n=== BASELINE COMPARISON ===")
    header = f"{'phase':<30} {'baseline_mean_s':>15} {'current_mean_s':>15} {'delta_s':>10} {'delta_%':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['phase']:<30} {row['baseline_s']:>15.3f} {row['current_s']:>15.3f}"
            f" {row['delta_s']:>+10.3f} {row['delta_pct']:>+8.1f}%"
        )
    if overall_delta_s is not None:
        if overall_delta_pct is not None and overall_delta_pct < -5.0:
            direction = "FASTER"
        elif overall_delta_pct is not None and overall_delta_pct > 5.0:
            direction = "SLOWER"
        else:
            direction = "NEUTRAL"
        print(f"\nVERDICT: {direction}  ({overall_delta_s:+.3f}s mean vs baseline overall)")


def _compare_results(current_files_dir: Path, baseline_files_dir: Path):
    """Compare first 15 rows of each results CSV against baseline.

    Returns a list of mismatch descriptions (empty = all match).
    """
    baseline_csvs = sorted(baseline_files_dir.rglob("*_buzzdetect.csv"))
    if not baseline_csvs:
        return []

    mismatches = []
    for b_csv in baseline_csvs:
        rel = b_csv.relative_to(baseline_files_dir)
        c_csv = current_files_dir / rel
        if not c_csv.exists():
            mismatches.append(f"  Missing result file: {rel}")
            continue

        def read_rows(p):
            with p.open(newline="") as f:
                reader = csv.DictReader(f)
                return [row for _, row in zip(range(15), reader)]

        b_rows = read_rows(b_csv)
        c_rows = read_rows(c_csv)

        for i, (b, c) in enumerate(zip(b_rows, c_rows)):
            if b["start"] != c["start"] or b["activation_ins_buzz"] != c["activation_ins_buzz"]:
                mismatches.append(
                    f"  {rel} row {i+1}: baseline=({b['start']}, {b['activation_ins_buzz']})"
                    f" current=({c['start']}, {c['activation_ins_buzz']})"
                )

        if len(c_rows) < len(b_rows):
            mismatches.append(
                f"  {rel}: baseline has {len(b_rows)} rows but current has only {len(c_rows)} (checked up to 15)"
            )

    return mismatches


def _run_single(dir_out: Path, model: str, chunklength: float, n_streamers: int,
                stream_buffer_depth: int, analyzers_cpu: int, gpu: bool,
                dir_audio: Path, label: str = "") -> dict:
    """Run one analysis pass into dir_out. Returns result dict with keys:
    success, results_bad, verdict, comparison (or None).
    """
    os.makedirs(dir_out, exist_ok=True)

    settings = {
        "model": model,
        "chunklength": chunklength,
        "n_streamers": n_streamers,
        "stream_buffer_depth": stream_buffer_depth,
        "analyzers_cpu": analyzers_cpu,
        "gpu": gpu,
    }
    with (dir_out / "settings.json").open("w") as f:
        json.dump(settings, f, indent=2)

    profile_path = str(dir_out / "profile.csv")
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}chunk={chunklength}s  streamers={n_streamers}  buf={stream_buffer_depth}"
          f"  cpu={analyzers_cpu}  gpu={gpu}  model={model}")

    try:
        analyze(
            modelname=model,
            classes_out=["ins_buzz"],
            framehop_prop=1,
            chunklength=chunklength,
            analyzers_cpu=analyzers_cpu,
            analyzer_gpu=gpu,
            n_streamers=n_streamers,
            stream_buffer_depth=stream_buffer_depth,
            dir_audio=str(dir_audio),
            dir_out=str(dir_out),
            verbosity_print='WARNING',
            log_progress=False,
            event_stopanalysis=None,
            profile=True,
            profile_path=profile_path,
        )
        success = True
    except Exception as e:
        print(f"{prefix}Analysis failed: {e}", file=sys.stderr)
        return {"success": False, "results_bad": False, "verdict": None, "comparison": None}

    # Results correctness check (skipped when no baseline files exist yet)
    baseline_files_dir = Path("eval_out/baseline/files")
    current_files_dir = dir_out / "files"
    results_bad = False
    if baseline_files_dir.exists():
        mismatches = _compare_results(current_files_dir, baseline_files_dir)
        if mismatches:
            print(f"\n{prefix}=== RESULTS MISMATCH ===", file=sys.stderr)
            for m in mismatches:
                print(m, file=sys.stderr)
            results_bad = True
        else:
            print(f"{prefix}Results check passed.")
        if not results_bad and current_files_dir.exists():
            shutil.rmtree(current_files_dir)

    # Profile comparison against baseline (skipped when baseline doesn't exist yet)
    verdict = None
    comparison = None
    current_profile = dir_out / "profile.csv"
    if BASELINE_PROFILE.exists() and current_profile.exists():
        rows, overall_delta_s, overall_delta_pct = _compare_profiles(current_profile, BASELINE_PROFILE)
        _print_comparison(rows, overall_delta_s, overall_delta_pct)

        if overall_delta_s is not None:
            if overall_delta_pct < -5.0:
                verdict = "FASTER"
            elif overall_delta_pct > 5.0:
                verdict = "SLOWER"
            else:
                verdict = "NEUTRAL"
        else:
            verdict = "UNKNOWN"

        comparison = {
            "overall_mean_delta_s": overall_delta_s,
            "overall_mean_delta_pct": overall_delta_pct,
            "verdict": verdict,
            "phases": rows,
        }
        with (dir_out / "comparison.json").open("w") as f:
            json.dump(comparison, f, indent=2)

    verdict_str = "BADRESULTS" if results_bad else verdict
    if verdict_str is not None:
        (dir_out / "verdict.txt").write_text(verdict_str)

    return {
        "success": success,
        "results_bad": results_bad,
        "verdict": verdict_str,
        "comparison": comparison,
    }


def _run_sweep(dir_out: Path, model: str, analyzers_cpu: int, gpu: bool,
               dir_audio: Path, is_baseline: bool) -> int:
    """Sweep TUNE_GRID, promote winner to dir_out. Returns exit code."""
    dir_tuning = dir_out / "tuning"
    os.makedirs(dir_tuning, exist_ok=True)

    keys = list(TUNE_GRID.keys())
    combos = list(itertools.product(*TUNE_GRID.values()))
    total = len(combos)
    print(f"\nSweeping {total} settings combinations.")

    combo_results = []
    for i, vals in enumerate(combos, 1):
        combo = dict(zip(keys, vals))
        label = _combo_label(combo["chunklength"], combo["n_streamers"], combo["stream_buffer_depth"])
        combo_dir = dir_tuning / label
        print(f"\n── Combo {i}/{total}: {label} ──")
        result = _run_single(
            dir_out=combo_dir,
            model=model,
            dir_audio=dir_audio,
            analyzers_cpu=analyzers_cpu,
            gpu=gpu,
            **combo,
        )
        entry = {
            "name": label,
            "settings": {**combo, "analyzers_cpu": analyzers_cpu, "gpu": gpu, "model": model},
            "success": result["success"],
            "results_bad": result.get("results_bad", False),
        }
        if is_baseline:
            entry["overall_s"] = _read_overall_time(combo_dir / "profile.csv")
        else:
            entry["verdict"] = result["verdict"]
            entry["overall_delta_pct"] = (result["comparison"] or {}).get("overall_mean_delta_pct")
        combo_results.append(entry)

    # Select winner
    if is_baseline:
        eligible = [r for r in combo_results if r["success"] and r.get("overall_s") is not None]
        eligible.sort(key=lambda r: r["overall_s"])
        rank_key = "overall_s"
    else:
        eligible = [
            r for r in combo_results
            if r["success"] and not r["results_bad"]
            and r.get("overall_delta_pct") is not None
            and r.get("verdict") not in (None, "BADRESULTS")
        ]
        eligible.sort(key=lambda r: r["overall_delta_pct"])
        rank_key = "overall_delta_pct"

    for rank, r in enumerate(eligible, 1):
        r["rank"] = rank
    for r in combo_results:
        if "rank" not in r:
            r["rank"] = None

    tuning_results = {
        "best_combo": eligible[0]["name"] if eligible else None,
        "combos": sorted(combo_results, key=lambda r: (r["rank"] is None, r["rank"] or 0)),
    }
    with (dir_out / "tuning_results.json").open("w") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"\nTuning results saved to: {dir_out / 'tuning_results.json'}")

    if not eligible:
        print("\nNo eligible results to promote (all BADRESULTS or failed).", file=sys.stderr)
        return 1

    winner = eligible[0]
    winner_dir = dir_tuning / winner["name"]
    if is_baseline:
        print(f"\nBest combo: {winner['name']}  ({winner['overall_s']:.3f}s overall)")
    else:
        print(f"\nBest combo: {winner['name']}  ({winner['overall_delta_pct']:+.1f}%  {winner['verdict']})")
    print(f"Promoting to {dir_out}/")

    for fname in ("settings.json", "comparison.json", "verdict.txt", "profile.csv"):
        src = winner_dir / fname
        if src.exists():
            shutil.copy2(src, dir_out / fname)

    # Print final summary for the winner
    if not is_baseline:
        winner_comparison_path = dir_out / "comparison.json"
        if winner_comparison_path.exists():
            with winner_comparison_path.open() as f:
                comp = json.load(f)
            _print_comparison(
                comp.get("phases", []),
                comp.get("overall_mean_delta_s"),
                comp.get("overall_mean_delta_pct"),
            )
        verdict_path = dir_out / "verdict.txt"
        if verdict_path.exists():
            print(f"\nFinal verdict: {verdict_path.read_text().strip()}")

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate buzzdetect analysis on a test dataset (audio_eval). "
            "Every run sweeps a settings grid and promotes the best result."
        )
    )
    parser.add_argument(
        "--test-name",
        required=True,
        help=(
            "Label for this run. Creates eval_out/<test-name>/ with results. "
            "Use 'baseline' to establish a reference run."
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Name of the model directory under models/ to use for inference. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--analyzers-cpu",
        type=int,
        default=0,
        help=(
            "Number of parallel CPU inference workers. "
            "Applied to every combo in the sweep. Default: 0"
        ),
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch a GPU inference worker for every combo in the sweep. "
            "Use --no-gpu for CPU-only (requires --analyzers-cpu > 0). Default: True"
        ),
    )

    args = parser.parse_args(argv)
    test_name = args.test_name

    dir_audio = Path("audio_eval").resolve()
    if not dir_audio.exists() or not dir_audio.is_dir():
        print(f"Audio directory not found or not a directory: {dir_audio}", file=sys.stderr)
        return 2

    is_baseline = test_name == "baseline"

    if not is_baseline and not BASELINE_PROFILE.exists():
        print(
            "Error: no baseline found. Run with --test-name baseline first.",
            file=sys.stderr,
        )
        return 2

    dir_out = Path("eval_out", test_name).resolve()
    if is_baseline and dir_out.exists():
        print("Note: overwriting existing baseline.")
        shutil.rmtree(dir_out)
    os.makedirs(dir_out, exist_ok=True)

    return _run_sweep(
        dir_out=dir_out,
        model=args.model,
        analyzers_cpu=args.analyzers_cpu,
        gpu=args.gpu,
        dir_audio=dir_audio,
        is_baseline=is_baseline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
