"""
Grid search across model, streamer count, chunk length, and buffer depth.
Run from the repo root:
    .venv/bin/python eval_out/01_baseline/grid_search.py

Results are written to eval_out/01_baseline/tuning/<combo_name>/
Each combo dir ends up with profile.csv, *.log, and settings.json — no bulk result files.

Multiple grids are defined in GRIDS below and run back-to-back.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_utils import cleanup_combo, ensure_xla_precompiled, run_combo, read_overall_time

# ---------------------------------------------------------------------------
# Grids — define one dict per search; all will be run in sequence.
# Each dict must have: label, models, chunk_lengths, n_streamers, n_gpu,
# buffer_depths.
# ---------------------------------------------------------------------------
GRIDS = [
    # Grid 1: sweep chunk lengths, fixed streamers/depth
    {
        "label":         "g1_sweep_chunklength",
        "models":        ["model_general_v3_xla"],
        "chunk_lengths": [100, 200, 300, 600],
        "n_streamers":   [8],
        "n_gpu":         [1],
        "buffer_depths": [16],
    },
    # Grid 2: sweep streamer count, fixed chunk length/depth
    {
        "label":         "g2_sweep_streamers",
        "models":        ["model_general_v3_xla"],
        "chunk_lengths": [200],
        "n_streamers":   [4, 8, 16, 32, 64],
        "n_gpu":         [1],
        "buffer_depths": [16],
    },
    # Grid 3: sweep buffer depth, fixed streamers/chunk length
    {
        "label":         "g3_sweep_depth",
        "models":        ["model_general_v3_xla"],
        "chunk_lengths": [200],
        "n_streamers":   [8],
        "n_gpu":         [1],
        "buffer_depths": [1, 3, 8, 16, 32, 64, 128],
    },
    # Grid 4: sweep buffer depth and chunk length at higher streamer count
    {
        "label":         "g4_sweep_depth_chunklength",
        "models":        ["model_general_v3_xla"],
        "chunk_lengths": [200, 300, 600],
        "n_streamers":   [32],
        "n_gpu":         [1],
        "buffer_depths": [8, 16, 32, 64],
    },
]

# Fixed settings
AUDIO_DIR     = "/media/server storage/experiments/Luke - Wooster Strawberry"
CLASSES_OUT   = ["ins_buzz"]
FRAMEHOP_PROP = 1.0

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE  = REPO_ROOT / "eval_out" / "01_baseline" / "tuning"

# ---------------------------------------------------------------------------


def combo_name(model: str, chunklength: int, n_streamers: int, n_gpu: int, buffer_depth: int) -> str:
    short = model.replace("model_general_", "")   # v3  or  v3_xla
    return f"{short}_{chunklength}s_{n_streamers}str_{n_gpu}gpu_depth{buffer_depth}"


def _run_combo(model: str, chunklength: int, n_streamers: int, n_gpu: int, buffer_depth: int) -> bool:
    name    = combo_name(model, chunklength, n_streamers, n_gpu, buffer_depth)
    out_dir = OUT_BASE / name

    if out_dir.exists():
        print(f"  SKIP: {name} (output dir already exists)")
        return True

    print(f"\n{'='*70}")
    print(f"  COMBO: {name}")
    print(f"{'='*70}")

    result = run_combo(
        out_dir=out_dir,
        model=model,
        chunklength=chunklength,
        n_streamers=n_streamers,
        buffer_depth=buffer_depth,
        analyzers_gpu=n_gpu,
        analyzers_cpu=0,
        audio_dir=AUDIO_DIR,
        classes_out=CLASSES_OUT,
        framehop_prop=FRAMEHOP_PROP,
    )

    cleanup_combo(out_dir)

    tag = "OK" if result["success"] else ("OOM" if result["oom"] else f"EXIT")
    overall = read_overall_time(out_dir / "profile.csv")
    time_str = f"{overall:.1f}s overall" if overall is not None else f"{result['elapsed']:.1f}s elapsed"
    print(f"\n  [{tag}] {name}  {time_str}\n")

    return result["success"]


def run_grid(grid: dict) -> dict[str, str]:
    label  = grid["label"]
    combos = list(itertools.product(
        grid["models"],
        grid["chunk_lengths"],
        grid["n_streamers"],
        grid["n_gpu"],
        grid["buffer_depths"],
    ))
    total = len(combos)

    print(f"\n{'#'*70}")
    print(f"  GRID: {label}  ({total} combos)")
    print(f"{'#'*70}")

    for model in grid["models"]:
        ensure_xla_precompiled(model, grid["chunk_lengths"])

    results = {}
    for i, (model, chunklength, n_streamers, n_gpu, buffer_depth) in enumerate(combos, 1):
        print(f"[{i}/{total}]", end=" ")
        ok = _run_combo(model, chunklength, n_streamers, n_gpu, buffer_depth)
        results[combo_name(model, chunklength, n_streamers, n_gpu, buffer_depth)] = "ok" if ok else "failed"

    return results


def main():
    all_results: dict[str, str] = {}
    for grid in GRIDS:
        all_results.update(run_grid(grid))

    print("\n" + "="*70)
    print("ALL GRIDS COMPLETE")
    print("="*70)
    for name, status in all_results.items():
        mark = "✓" if status == "ok" else "✗"
        print(f"  {mark} {name}")

    failed = [k for k, v in all_results.items() if v == "failed"]
    if failed:
        print(f"\n{len(failed)} combo(s) failed. Check logs in {OUT_BASE.relative_to(REPO_ROOT)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(all_results)} combos completed successfully.")


if __name__ == "__main__":
    main()
