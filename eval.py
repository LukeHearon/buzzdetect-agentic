import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Reduce TensorFlow and other verbose logs where possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from src.analyze import analyze
except Exception as e:
    print(f"Failed to import analyze from src.analyze: {e}", file=sys.stderr)
    sys.exit(1)


def _append_result(row_path: Path, row: dict, fieldnames: list[str]) -> None:
    row_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not row_path.exists() or row_path.stat().st_size == 0
    with row_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate buzzdetect analysis on a test dataset (audio_eval) using GPU and record wall-clock time."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use (must correspond to a model available to buzzdetect).",
    )
    parser.add_argument(
        "--test-name",
        default=None,
        help="Optional name of the test; defaults to the name of the audio directory.",
    )
    parser.add_argument(
        "--results-file",
        default="eval_results.csv",
        help="CSV file to append results to. Default: eval_results.csv",
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
    parser.add_argument(
        "--verbosity",
        default="ERROR",
        choices=["PROGRESS", "INFO", "DEBUG", "WARNING", "ERROR"],
        help="Verbosity level for console output/logs. Default: ERROR (quiet).",
    )

    args = parser.parse_args(argv)

    audio_dir = Path("audio_eval").resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Audio directory not found or not a directory: {audio_dir}", file=sys.stderr)
        return 2

    test_name = args.test_name or audio_dir.name
    results_path = Path(args.results_file).resolve()

    # Hard-coded output directory for analysis results (cleaned up after run)
    dir_out = Path("eval_out").resolve()
    os.makedirs(dir_out, exist_ok=True)

    # Ensure minimal console/log noise unless explicitly requested
    verbosity_print = args.verbosity
    verbosity_log = args.verbosity

    # Log selected options (exclude hard-coded paths/values)
    argval = {
        "model": args.model,
        "test_name": test_name,
        "framehop_prop": args.framehop_prop,
        "chunklength": args.chunklength,
        "n_streamers": args.n_streamers,
        "stream_buffer_depth": args.stream_buffer_depth,
        "analyzers_cpu": args.analyzers_cpu,
        "gpu": args.gpu,
        "verbosity": args.verbosity,
    }
    print("Eval args: " + ", ".join(f"{k}={v}" for k, v in argval.items()))

    # Time the end-to-end analyze() call
    start = time.perf_counter()
    try:
        analyze(
            modelname=args.model,
            classes_out="ins_buzz",
            precision=None,
            framehop_prop=args.framehop_prop,
            chunklength=args.chunklength,
            analyzers_cpu=args.analyzers_cpu,
            analyzer_gpu=args.gpu,
            n_streamers=args.n_streamers,
            stream_buffer_depth=args.stream_buffer_depth,
            dir_audio=str(audio_dir),
            dir_out=str(dir_out),
            verbosity_print=verbosity_print,
            verbosity_log=verbosity_log,
            log_progress=False,
            q_gui=None,
            event_stopanalysis=None,
        )
        success = True
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        success = False
    finally:
        duration_s = time.perf_counter() - start

    # Append results
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_name": test_name,
        "model": args.model,
        "duration_seconds": f"{duration_s:.3f}",
        "analyzers_cpu": args.analyzers_cpu,
        "gpu": args.gpu,
        "success": success,
    }
    fields = ["timestamp", "test_name", "model", "duration_seconds", "analyzers_cpu", "gpu", "success"]
    _append_result(results_path, row, fields)

    # Clean up outputs immediately (dir_out is hard-coded for eval)
    try:
        if dir_out.exists():
            shutil.rmtree(dir_out, ignore_errors=True)
    except Exception as e:
        # Do not fail the evaluation if cleanup has issues
        print(f"Warning: failed to clean outputs: {e}", file=sys.stderr)

    # Exit status: 0 on success, 1 on analysis failure, 2 on input error
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())