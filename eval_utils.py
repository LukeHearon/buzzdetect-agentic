"""
Shared utilities for eval.py and eval_out/01_baseline/grid_search.py.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON    = str(REPO_ROOT / ".venv" / "bin" / "python")
CLI       = str(REPO_ROOT / "buzzdetect_cli.py")


def read_profile(profile_path: Path) -> dict[str, float]:
    """Return {phase: mean_s} for all phases in a profile CSV."""
    if not profile_path.exists():
        return {}
    with profile_path.open(newline="") as f:
        return {row["phase"]: float(row["mean_s"]) for row in csv.DictReader(f)}


def read_overall_time(profile_path: Path):
    """Return overall mean_s from profile.csv, or None if unavailable."""
    return read_profile(profile_path).get("overall")


def ensure_xla_precompiled(model_name: str, chunklengths: list, device: str = "GPU"):
    """Precompile XLA for all chunklengths if model is model_general_v3_xla."""
    if model_name != "model_general_v3_xla":
        return
    from src.inference.models import load_model
    print(f"\nChecking XLA precompilation for {len(chunklengths)} chunklength(s) on {device}...")
    model = load_model(model_name, framehop_prop=1.0, initialize=True)
    for cl in chunklengths:
        print(f"  Precompiling chunklength={cl}s on {device}...", end=" ", flush=True)
        model.precompile(cl, device=device)
        print("done")


def stderr_has_oom(text: str) -> bool:
    t = text.lower()
    return "out of memory" in t or "resource exhausted" in t or "oom" in t


def cleanup_combo(out_dir: Path) -> None:
    """Promote profile and log from files/ to out_dir; delete result CSVs; remove empty dirs.

    End state: out_dir contains profile.csv, *.log, settings.json — no bulk result files.
    """
    files_dir = out_dir / "files"
    if files_dir.exists():
        for f in files_dir.iterdir():
            if not f.is_file():
                continue
            if f.name.endswith("_profile.csv") or f.name == "profile.csv":
                dest = out_dir / "profile.csv"
                if not dest.exists():
                    f.rename(dest)
            elif f.suffix == ".log":
                dest = out_dir / f.name
                if not dest.exists():
                    f.rename(dest)

    for pattern in ("*_buzzdetect.csv", "*_buzzpart.csv"):
        for f in out_dir.rglob(pattern):
            f.unlink()

    for d in sorted(out_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def run_combo(
    out_dir: Path,
    model: str,
    chunklength: int | float,
    n_streamers: int,
    buffer_depth: int,
    analyzers_gpu: int,
    analyzers_cpu: int,
    audio_dir,
    classes_out: list[str],
    framehop_prop: float = 1.0,
    verbosity_print: str = "PROGRESS",
    verbosity_log: str = "DEBUG",
    log_progress: bool = True,
) -> dict:
    """Run one combo via buzzdetect_cli.py subprocess. Returns {success, oom, elapsed}.

    Writes settings.json to out_dir. Does NOT clean up — caller is responsible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_out = out_dir / "files"

    settings = {
        "modelname":           model,
        "chunklength":         chunklength,
        "n_streamers":         n_streamers,
        "stream_buffer_depth": buffer_depth,
        "analyzers_gpu":       analyzers_gpu,
        "analyzers_cpu":       analyzers_cpu,
        "classes_out":         classes_out,
        "framehop_prop":       framehop_prop,
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    cmd = [
        PYTHON, CLI,
        "--modelname",           model,
        "--dir_audio",           str(audio_dir),
        "--dir_out",             str(analysis_out),
        "--chunklength",         str(chunklength),
        "--n_streamers",         str(n_streamers),
        "--stream_buffer_depth", str(buffer_depth),
        "--analyzers_gpu",       str(analyzers_gpu),
        "--analyzers_cpu",       str(analyzers_cpu),
        "--classes_out",         *classes_out,
        "--framehop_prop",       str(framehop_prop),
        "--verbosity_print",     verbosity_print,
        "--verbosity_log",       verbosity_log,
        "--log_progress",        str(log_progress).lower(),
        "--profile",             "true",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0

    if proc.stderr:
        sys.stderr.write(proc.stderr)

    oom     = proc.returncode != 0 and stderr_has_oom(proc.stderr or "")
    success = proc.returncode == 0

    if not success:
        tag = "OOM" if oom else f"EXIT {proc.returncode}"
        (out_dir / "ERROR").write_text(f"{tag}\n{proc.stderr or ''}")

    return {"success": success, "oom": oom, "elapsed": elapsed}
