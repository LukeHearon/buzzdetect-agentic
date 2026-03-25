import argparse
import csv
import json
import os
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
        b_s = float(b_row["total_s"])
        c_s = float(current[phase]["total_s"])
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
    header = f"{'phase':<30} {'baseline_s':>12} {'current_s':>12} {'delta_s':>10} {'delta_%':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['phase']:<30} {row['baseline_s']:>12.3f} {row['current_s']:>12.3f}"
            f" {row['delta_s']:>+10.3f} {row['delta_pct']:>+8.1f}%"
        )
    if overall_delta_s is not None:
        direction = "FASTER" if overall_delta_s < 0 else ("SLOWER" if overall_delta_s > 0 else "SAME")
        print(f"\nVERDICT: {direction}  ({overall_delta_s:+.3f}s vs baseline overall)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate buzzdetect analysis on a test dataset (audio_eval) using GPU and record wall-clock time."
    )

    parser.add_argument(
        "--test-name",
        default=None,
        help="Optional name of the test; defaults to the name of the audio directory.",
    )
    parser.add_argument(
        "--n-streamers",
        type=int,
        default=4,
        help="Number of streamer workers to feed the GPU. Default: 4",
    )
    parser.add_argument(
        "--stream-buffer-depth",
        type=int,
        default=4,
        help="Depth of the streaming buffer. Default: 4",
    )
    parser.add_argument(
        "--framehop-prop",
        type=float,
        default=1.0,
        help="Frame hop proportion for analysis. Default: 1.0",
    )
    parser.add_argument(
        "--chunklength",
        type=float,
        default=200.0,
        help="Chunk length (seconds) for analysis. Default: 200.0",
    )
    parser.add_argument(
        "--analyzers-cpu",
        type=int,
        default=0,
        help="Number of parallel CPU analyzer workers. Default: 0",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU analyzer (default: True). Use --no-gpu to disable.",
    )

    args = parser.parse_args(argv)

    dir_audio = Path("audio_eval").resolve()
    if not dir_audio.exists() or not dir_audio.is_dir():
        print(f"Audio directory not found or not a directory: {dir_audio}", file=sys.stderr)
        return 2

    test_name = args.test_name or dir_audio.name
    dir_out = Path("eval_out", test_name).resolve()
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

    profile_path = str(dir_out / "profile.csv")

    try:
        analyze(
            modelname=MODEL_NAME,
            classes_out=["ins_buzz"],
            precision=None,
            framehop_prop=1,
            chunklength=args.chunklength,
            analyzers_cpu=args.analyzers_cpu,
            analyzer_gpu=args.gpu,
            n_streamers=args.n_streamers,
            stream_buffer_depth=args.stream_buffer_depth,
            dir_audio=str(dir_audio),
            dir_out=str(dir_out),
            verbosity_print='WARNING',
            verbosity_log='WARNING',
            log_progress=False,
            q_gui=None,
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

    # Baseline comparison (skipped when running as baseline or baseline doesn't exist yet)
    current_profile = dir_out / "profile.csv"
    if BASELINE_PROFILE.exists() and current_profile.exists() and test_name != "baseline":
        rows, overall_delta_s, overall_delta_pct = _compare_profiles(current_profile, BASELINE_PROFILE)
        _print_comparison(rows, overall_delta_s, overall_delta_pct)

        if overall_delta_s is not None:
            threshold_pct = 1.0
            if overall_delta_pct < -threshold_pct:
                verdict = "faster"
            elif overall_delta_pct > threshold_pct:
                verdict = "slower"
            else:
                verdict = "same"
        else:
            verdict = "unknown"

        comparison = {
            "overall_delta_s": overall_delta_s,
            "overall_delta_pct": overall_delta_pct,
            "verdict": verdict,
            "phases": rows,
        }
        comparison_path = dir_out / "comparison.json"
        with comparison_path.open("w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {comparison_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
