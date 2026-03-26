import argparse
import csv
import json
import os
import shutil
import sys
import time
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate buzzdetect analysis on a test dataset (audio_eval) using GPU and record wall-clock time."
    )

    parser.add_argument(
        "--test-name",
        required=True,
        help=(
            "Label for this run. Creates eval_out/<test-name>/ and saves settings.json, "
            "profile.csv, and comparison.json there. Use 'baseline' to record a reference run."
        ),
    )
    parser.add_argument(
        "--n-streamers",
        type=int,
        default=4,
        help=(
            "Number of parallel audio-reader threads (WorkerStreamer). Each thread reads a "
            "chunk from disk, resamples it, and pushes it into the shared q_analyze queue that "
            "feeds the inference worker(s). Increase if the GPU sits idle waiting for audio. Default: 4"
        ),
    )
    parser.add_argument(
        "--stream-buffer-depth",
        type=int,
        default=4,
        help=(
            "Max number of resampled audio chunks held in q_analyze (the queue between streamers "
            "and the inference worker). Each streamer also holds one chunk while waiting to enqueue, "
            "so peak RAM usage ≈ (n_streamers + stream_buffer_depth) × chunklength seconds of audio. Default: 4"
        ),
    )
    parser.add_argument(
        "--chunklength",
        type=float,
        default=200.0,
        help=(
            "Length (seconds) of each audio chunk passed through q_analyze to the inference "
            "worker. Larger values reduce queue overhead but increase per-chunk RAM. Default: 200.0"
        ),
    )
    parser.add_argument(
        "--analyzers-cpu",
        type=int,
        default=0,
        help=(
            "Number of parallel CPU inference workers (WorkerInferer with processor='CPU'). "
            "Each runs the TensorFlow model on CPU independently. Normally 0 when --gpu is used. Default: 0"
        ),
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch a single GPU inference worker (WorkerInferer with processor='GPU'). "
            "Use --no-gpu to run CPU-only (requires --analyzers-cpu > 0 to do any inference). Default: True"
        ),
    )

    args = parser.parse_args(argv)

    if args.chunklength > 1200:
        print(
            f"Error: --chunklength {args.chunklength} exceeds the limit of 1200 seconds. "
            "This limit is enforced because larger values may exceed GPU VRAM.",
            file=sys.stderr,
        )
        return 2

    dir_audio = Path("audio_eval").resolve()
    if not dir_audio.exists() or not dir_audio.is_dir():
        print(f"Audio directory not found or not a directory: {dir_audio}", file=sys.stderr)
        return 2

    test_name = args.test_name
    if test_name != "baseline" and not BASELINE_PROFILE.exists():
        print(
            "Error: no baseline found. Run with --test-name baseline first to establish a reference.",
            file=sys.stderr,
        )
        return 2

    dir_out = Path("eval_out", test_name).resolve()
    if test_name == "baseline" and BASELINE_PROFILE.exists():
        print("Note: overwriting existing baseline.")
        shutil.rmtree(dir_out)
    os.makedirs(dir_out, exist_ok=True)

    argval = {
        "test_name": test_name,
        "model": MODEL_NAME,
        "chunklength": args.chunklength,
        "n_streamers": args.n_streamers,
        "stream_buffer_depth": args.stream_buffer_depth,
        "analyzers_cpu": args.analyzers_cpu,
        "gpu": args.gpu,
    }
    print("Eval args: " + ", ".join(f"{k}={v}" for k, v in argval.items()))

    settings_path = dir_out / "settings.json"
    with settings_path.open("w") as f:
        json.dump(argval, f, indent=2)
    print(f"Settings saved to: {settings_path}")

    profile_path = str(dir_out / "profile.csv")

    try:
        analyze(
            modelname=MODEL_NAME,
            classes_out=["ins_buzz"],
            framehop_prop=1,
            chunklength=args.chunklength,
            analyzers_cpu=args.analyzers_cpu,
            analyzer_gpu=args.gpu,
            n_streamers=args.n_streamers,
            stream_buffer_depth=args.stream_buffer_depth,
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
        print(f"Analysis failed: {e}", file=sys.stderr)
        success = False

    if not success:
        return 1

    # Results correctness check (skipped when running as baseline)
    baseline_files_dir = Path("eval_out/baseline/files")
    current_files_dir = dir_out / "files"
    results_bad = False
    if baseline_files_dir.exists() and test_name != "baseline":
        mismatches = _compare_results(current_files_dir, baseline_files_dir)
        if mismatches:
            print("\n=== RESULTS MISMATCH (first 15 rows) ===", file=sys.stderr)
            for m in mismatches:
                print(m, file=sys.stderr)
            print(
                "\nERROR: Result CSVs differ from baseline. Fix the regression before proceeding.",
                file=sys.stderr,
            )
            results_bad = True
        else:
            print("Results check passed: first 15 rows of all CSVs match baseline.")

        if not results_bad and current_files_dir.exists():
            shutil.rmtree(current_files_dir)
            print(f"Cleaned up results files: {current_files_dir}")

    # Baseline comparison (skipped when running as baseline or baseline doesn't exist yet)
    verdict = None
    current_profile = dir_out / "profile.csv"
    if BASELINE_PROFILE.exists() and current_profile.exists() and test_name != "baseline":
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
        comparison_path = dir_out / "comparison.json"
        with comparison_path.open("w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {comparison_path}")

    # Write verdict to file
    if test_name != "baseline" and (verdict is not None or results_bad):
        verdict_str = "BADRESULTS" if results_bad else verdict
        verdict_path = dir_out / "verdict.txt"
        verdict_path.write_text(verdict_str)
        print(f"Verdict: {verdict_str} (written to {verdict_path})")

    if results_bad:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
