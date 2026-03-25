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
    results_path = Path(args.results_file).resolve()

    dir_out = Path("eval_out", test_name).resolve()
    os.makedirs(dir_out, exist_ok=True)

    # Log selected options (exclude hard-coded paths/values)
    argval = {
        "test_name": test_name,
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
        )
        success = True

    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        success = False

    # Exit status: 0 on success, 1 on analysis failure, 2 on input error
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())