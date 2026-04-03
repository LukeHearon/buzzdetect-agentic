import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import eval_utils

baseline_files_dir = Path("baseline_results")

# Grid swept on every run. All combinations are tried; the best overall runtime
# is promoted to the top-level output directory.
# Do not modify — this is managed infrastructure, not an agent tuning target.
TUNE_GRID = {
    "chunklength":         [200, 600, 1200],
    "n_streamers":         [3, 6]
}


def _combo_label(chunklength, n_streamers):
    return f"{int(chunklength)}s_{n_streamers}str"


# Phases shown as columns in the rankings table. Edit to add/remove/reorder.
# Each entry is (profile_phase_name, column_header).
DISPLAY_PHASES = [
    ("overall",                              "overall"),
    ("inference",                            "inference"),
    ("audio_io/reading",                     "read"),
    ("audio_io/resampling",                  "resample"),
    ("write_io/formatting",                  "format"),
    ("write_io/writing",                     "write"),
    ("audio_io/fullqueue",                   "q_streamer"),
    ("inference/emptyqueue",                 "q_analyzer"),
]


def compare_tests(dir: Path, reference: str | None = None, n: int = 3) -> list[dict]:
    """Read profile.csv from each subdirectory of dir.

    All % differences are computed against the best performer (lowest overall_s).
    If a reference test is given, it is pinned to the top of the table (always first,
    separated by a divider) and its deltas are shown vs. the best non-reference entry
    (so if reference IS the best, it compares to next-best; if not, it compares to best).
    The returned entries include compare_delta_pct/compare_name on the reference entry.

    Returns all entries as a sorted list of dicts:
      {name, overall_s, delta_s, delta_pct, phase_pcts}
    """
    entries = []
    for sub in sorted(dir.iterdir()):
        if not sub.is_dir():
            continue
        phases = eval_utils.read_profile(sub / "profile.csv")
        if "overall" in phases:
            entries.append({"name": sub.name, "overall_s": phases["overall"], "phases": phases})

    if not entries:
        return entries

    entries.sort(key=lambda e: e["overall_s"])
    best = entries[0]
    best_s = best["overall_s"]

    if reference is not None:
        ref = next((e for e in entries if e["name"] == reference), None)
        if ref is None:
            print(f"Warning: reference test '{reference}' not found in {dir}")
    else:
        ref = None

    # All deltas vs. best performer
    for e in entries:
        e["delta_s"] = e["overall_s"] - best_s
        e["delta_pct"] = (e["delta_s"] / best_s * 100) if best_s else 0.0
        e["phase_pcts"] = {
            phase: ((e["phases"].get(phase, 0) - best["phases"].get(phase, 0))
                    / best["phases"][phase] * 100)
            if best["phases"].get(phase) else 0.0
            for phase, _ in DISPLAY_PHASES
        }

    # Reference row compares vs best non-reference entry (next-best if ref is best, else best)
    if ref is not None:
        others = [e for e in entries if e["name"] != ref["name"]]
        ref_cmp = others[0] if others else None
        if ref_cmp is not None:
            ref_cmp_s = ref_cmp["overall_s"]
            ref["compare_delta_pct"] = ((ref["overall_s"] - ref_cmp_s) / ref_cmp_s * 100) if ref_cmp_s else 0.0
            ref["compare_name"] = ref_cmp["name"]
            ref["compare_phase_pcts"] = {
                phase: ((ref["phases"].get(phase, 0) - ref_cmp["phases"].get(phase, 0))
                        / ref_cmp["phases"][phase] * 100)
                if ref_cmp["phases"].get(phase) else 0.0
                for phase, _ in DISPLAY_PHASES
            }
        else:
            ref["compare_delta_pct"] = 0.0
            ref["compare_name"] = None
            ref["compare_phase_pcts"] = {phase: 0.0 for phase, _ in DISPLAY_PHASES}

    # Top n excludes reference (it's shown separately above the divider)
    top_n = [e for e in entries if ref is None or e["name"] != ref["name"]][:n]

    col_w = 10
    name_w = 35
    headers = [label for _, label in DISPLAY_PHASES]
    header = f"{'test':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers)
    title = f"RANKINGS for {reference}" if reference else f"RANKINGS (best: {best['name']}, {best_s:.3f}s)"
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))

    if ref is not None:
        row = f"{ref['name']:<{name_w}}"
        if ref["compare_name"] is not None:
            for phase, _ in DISPLAY_PHASES:
                pct = ref["compare_phase_pcts"].get(phase, 0.0)
                row += f"{pct:>+9.1f}%"
            row += f"  (vs {ref['compare_name']})"
        else:
            for _ in DISPLAY_PHASES:
                row += f"{'—':>{col_w}}"
        print(row)
        print("-" * len(header))

    for e in top_n:
        row = f"{e['name']:<{name_w}}"
        for phase, _ in DISPLAY_PHASES:
            pct = e["phase_pcts"].get(phase, 0.0)
            row += f"{pct:>+9.1f}%" if e["name"] != best["name"] else f"{'—':>{col_w}}"
        print(row)
    return entries


def _check_results_match(current_files_dir: Path):
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


def _run_sweep(dir_out: Path, model: str, analyzers_cpu: int, gpu: bool,
               dir_audio: Path, test_name: str) -> int:
    """Sweep TUNE_GRID, promote winner to dir_out. Returns exit code."""
    dir_tuning = dir_out / "tuning"
    os.makedirs(dir_tuning, exist_ok=True)

    analyzers_gpu = 1 if gpu else 0
    eval_utils.ensure_xla_precompiled(model, TUNE_GRID["chunklength"], device="GPU" if gpu else "CPU")

    keys = list(TUNE_GRID.keys())
    combos = list(itertools.product(*TUNE_GRID.values()))
    total = len(combos)
    print(f"\nSweeping {total} settings combinations.")

    combo_results = []
    for i, vals in enumerate(combos, 1):
        combo = dict(zip(keys, vals))
        label = _combo_label(combo["chunklength"], combo["n_streamers"])
        combo_dir = dir_tuning / label
        print(f"\n── Combo {i}/{total}: {label} ──")
        print(f"chunk={combo['chunklength']}s  streamers={combo['n_streamers']}  "
              f"cpu={analyzers_cpu}  gpu={gpu}  model={model}")

        result = eval_utils.run_combo(
            out_dir=combo_dir,
            model=model,
            chunklength=combo["chunklength"],
            n_streamers=combo["n_streamers"],
            buffer_depth=6,
            analyzers_gpu=analyzers_gpu,
            analyzers_cpu=analyzers_cpu,
            audio_dir=dir_audio,
            classes_out=["ins_buzz"],
            framehop_prop=1.0,
            verbosity_print="WARNING",
            log_progress=False,
        )

        results_bad = False
        if result["success"]:
            current_files_dir = combo_dir / "files"
            if baseline_files_dir.exists():
                mismatches = _check_results_match(current_files_dir)
                if mismatches:
                    print(f"\n[{label}] === RESULTS MISMATCH ===", file=sys.stderr)
                    for m in mismatches:
                        print(m, file=sys.stderr)
                    results_bad = True
                else:
                    print(f"[{label}] Results check passed.")
            else:
                shutil.move(str(current_files_dir), str(baseline_files_dir))

        eval_utils.cleanup_combo(combo_dir)

        entry = {
            "name": label,
            "settings": {**combo, "analyzers_cpu": analyzers_cpu, "gpu": gpu, "model": model},
            "success": result["success"],
            "results_bad": results_bad,
            "overall_s": eval_utils.read_overall_time(combo_dir / "profile.csv"),
        }
        combo_results.append(entry)

    # Select winner by lowest overall_s
    eligible = [
        r for r in combo_results
        if r["success"] and not r["results_bad"] and r.get("overall_s") is not None
    ]
    eligible.sort(key=lambda r: r["overall_s"])

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
    print(f"\nBest combo: {winner['name']}  ({winner['overall_s']:.3f}s overall)")
    print(f"Promoting to {dir_out}/")

    for fname in ("settings.json", "profile.csv"):
        src = winner_dir / fname
        if src.exists():
            shutil.copy2(src, dir_out / fname)

    # Print cross-run rankings (current test vs all others)
    # NOTE: skipping auto-tune rankings, unnecessarily verbose.
    eval_out = Path("eval_out")
    if eval_out.exists():
        all_results = compare_tests(eval_out, reference=test_name)
        my_result = next((r for r in all_results if r["name"] == test_name), None)
        if my_result and "delta_pct" in my_result:
            dpct = my_result.get("compare_delta_pct", my_result["delta_pct"])
            cmp_name = my_result.get("compare_name") or (all_results[0]["name"] if all_results else "?")
            if dpct < -5.0:
                verdict = "FASTER"
            elif dpct > 5.0:
                verdict = "SLOWER"
            else:
                verdict = "NEUTRAL"
            print(f"\nFinal verdict for {test_name}: {verdict}  ({dpct:+.1f}% vs {cmp_name})")

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
        help="Label for this run. Creates eval_out/<test-name>/ with results. Must start with the next number in sequence ",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Name of the model directory under models/ to use for inference.",
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

    dir_out = Path("eval_out", test_name).resolve()
    os.makedirs(dir_out, exist_ok=True)

    return _run_sweep(
        dir_out=dir_out,
        model=args.model,
        analyzers_cpu=args.analyzers_cpu,
        gpu=args.gpu,
        dir_audio=dir_audio,
        test_name=test_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
