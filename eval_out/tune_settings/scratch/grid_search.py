"""
Grid search over model × chunklength × n_streamers.
Runs buzzdetect_cli.py for each combo and collects profiling data.
"""

import subprocess
import csv
import shutil
import itertools
from pathlib import Path
from datetime import datetime

# Grid parameters
MODELS        = ["model_general_v3_xla"]
CHUNK_LENGTHS = [200, 600, 1200]
N_STREAMERS   = [2, 4, 6]

DIR_AUDIO = "audio_eval/files"
VENV_PY   = ".venv/bin/python"
SCRIPT    = "buzzdetect_cli.py"

SCRATCH      = Path("eval_out/13_tune_settings/scratch/grid_runs")
RESULTS_FILE = Path("eval_out/13_tune_settings/scratch/grid_results.csv")

# If a run folder with a profile CSV already exists: True = skip, False = overwrite
SKIP_EXISTING = True

SCRATCH.mkdir(parents=True, exist_ok=True)

FIELDS = ["name", "model", "chunklength", "n_streamers", "success",
          "overall_s", "inference_s", "emptyqueue_s", "n_chunks", "median_inference_s"]

results_file = open(RESULTS_FILE, "w", newline="")
writer = csv.DictWriter(results_file, fieldnames=FIELDS, extrasaction="ignore")
writer.writeheader()

combos = list(itertools.product(MODELS, CHUNK_LENGTHS, N_STREAMERS))
total = len(combos)
print(f"Total combos: {total}")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

for i, (model, cl, ns) in enumerate(combos, 1):
    name = f"{model}_cl{cl}_ns{ns}"
    dir_out = SCRATCH / name

    cmd = [
        VENV_PY, SCRIPT,
        "--modelname", model,
        "--dir_audio", DIR_AUDIO,
        "--dir_out", str(dir_out),
        "--analyzer_gpu", "true",
        "--analyzers_cpu", "0",
        "--chunklength", str(cl),
        "--n_streamers", str(ns),
        "--profile", "true",
        "--verbosity_print", "WARNING",
    ]

    existing_profiles = list(dir_out.glob("*_profile.csv")) if dir_out.exists() else []

    if existing_profiles and SKIP_EXISTING:
        print(f"[{i:3d}/{total}] {name} ...  SKIPPED (existing profile)")
        continue
    elif existing_profiles and not SKIP_EXISTING:
        shutil.rmtree(dir_out)

    print(f"[{i:3d}/{total}] {name} ...", end="", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Find the profile CSV written to dir_out
    csvs = list(dir_out.glob("*_profile.csv"))
    if not csvs:
        print(f"  NO PROFILE (rc={proc.returncode})")
        if proc.stderr:
            print("  STDERR:", proc.stderr[:300])
        writer.writerow({"name": name, "model": model, "chunklength": cl,
                          "n_streamers": ns, "success": False})
        results_file.flush()
    else:
        with open(csvs[0]) as f:
            reader = csv.DictReader(f)
            phases = {row["phase"]: row for row in reader}

        overall    = float(phases["overall"]["total_s"])
        inference  = float(phases.get("inference", {}).get("total_s", 0) or 0)
        emptyq     = float(phases.get("inference/emptyqueue", {}).get("total_s", 0) or 0)
        n_chunks   = int(phases.get("inference", {}).get("n", 0) or 0)
        median_inf = float(phases.get("inference", {}).get("median_s", 0) or 0)

        print(f"  overall={overall:.2f}s  inf={inference:.2f}s  emptyq={emptyq:.2f}s  chunks={n_chunks}")

        writer.writerow({
            "name": name, "model": model, "chunklength": cl, "n_streamers": ns,
            "success": True, "overall_s": overall, "inference_s": inference,
            "emptyqueue_s": emptyq, "n_chunks": n_chunks, "median_inference_s": median_inf,
        })
        results_file.flush()


results_file.close()
print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")
print(f"Results saved to: {RESULTS_FILE}")
