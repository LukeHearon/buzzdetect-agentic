"""
Settings sweep for ONNX inference.

Tests different combinations of:
- chunklength
- n_streamers
- stream_buffer_depth

Runs a quick benchmark for inference speed at each chunk size,
then runs mini-evals for the most promising settings.
"""

import subprocess
import sys
import json
import time
import numpy as np
from pathlib import Path

PYTHON = sys.executable
EVAL_SCRIPT = str(Path(__file__).parents[3] / "eval.py")

# ─── Part 1: Inference speed vs chunk size ───────────────────────────────────

def benchmark_onnx_inference():
    """Benchmark ONNX inference time per chunk at various sizes."""
    print("\n=== ONNX Inference Speed vs Chunk Size ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(Path(__file__).parents[3] / "models/model_general_v3/model_combined.onnx"),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    samplerate = 16000
    chunk_sizes_s = [50, 100, 150, 200, 250, 300, 400]
    results = {}

    for chunk_s in chunk_sizes_s:
        audio = np.random.randn(int(chunk_s * samplerate)).astype(np.float32)
        # Warmup
        for _ in range(3):
            sess.run(['dense'], {'input_1': audio})
        # Benchmark
        N = 8
        t0 = time.perf_counter()
        for _ in range(N):
            sess.run(['dense'], {'input_1': audio})
        t = (time.perf_counter() - t0) / N
        n_total_chunks = 129600 / chunk_s  # for 6 × 21600s files
        est_inference_total = n_total_chunks * t
        results[chunk_s] = {'per_chunk': t, 'est_total': est_inference_total}
        print(f"  chunk={chunk_s:4d}s: {t:.4f}s/chunk → est {est_inference_total:.1f}s inference total")

    best_chunk = min(results, key=lambda k: results[k]['est_total'])
    print(f"\nBest chunk size: {best_chunk}s (est. {results[best_chunk]['est_total']:.1f}s inference)")
    return best_chunk, results


# ─── Part 2: Quick streamer throughput check ──────────────────────────────────

def benchmark_streaming_throughput():
    """Estimate whether N streamers can keep up with ONNX GPU demand."""
    print("\n=== Streaming Throughput Check ===")
    import librosa
    import soundfile as sf
    from pathlib import Path

    # Use first audio file
    audio_files = list(Path("audio_eval").rglob("*.mp3"))
    if not audio_files:
        print("No audio files found, skipping streaming benchmark")
        return 6

    test_file = audio_files[0]
    chunk_s = 199.68  # rounded chunk length
    samplerate_target = 16000

    with sf.SoundFile(str(test_file)) as track:
        sr_native = track.samplerate
        read_size = int(chunk_s * sr_native)
        t0 = time.perf_counter()
        samples = track.read(read_size, dtype=np.float32)
        read_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    librosa.resample(samples, orig_sr=sr_native, target_sr=samplerate_target)
    resample_time = time.perf_counter() - t0

    total_per_chunk = read_time + resample_time
    print(f"  Read: {read_time:.3f}s, Resample: {resample_time:.3f}s, Total: {total_per_chunk:.3f}s per chunk")

    # ONNX inference demand (from bench above, ~0.067s/chunk)
    onnx_rate = 1 / 0.067
    required_streamers = onnx_rate / (1 / total_per_chunk)
    print(f"  ONNX processes at {onnx_rate:.1f} chunks/s")
    print(f"  Each streamer produces {1/total_per_chunk:.1f} chunks/s")
    print(f"  Required streamers: {required_streamers:.1f}")

    for n in [4, 6, 8]:
        throughput = n / total_per_chunk
        surplus = throughput - onnx_rate
        print(f"  n_streamers={n}: {throughput:.1f} chunks/s ({'surplus' if surplus > 0 else 'deficit'}: {abs(surplus):.1f})")

    # Recommend
    recommended = max(4, min(8, int(np.ceil(required_streamers)) + 1))
    print(f"\n  Recommended n_streamers: {recommended}")
    return recommended


# ─── Part 3: Mini eval runs ───────────────────────────────────────────────────

def run_eval(test_name, chunklength, n_streamers, buffer_depth, timeout=600):
    """Run eval.py with given settings. Returns comparison dict or None on failure."""
    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--test-name", test_name,
        "--chunklength", str(chunklength),
        "--n-streamers", str(n_streamers),
        "--stream-buffer-depth", str(buffer_depth),
    ]
    print(f"\nRunning: {test_name}  chunk={chunklength} streamers={n_streamers} buf={buffer_depth}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr[-500:])
        comp_path = Path(f"eval_out/{test_name}/comparison.json")
        if comp_path.exists():
            with open(comp_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parents[3])  # go to workspace root

    # Warmup librosa
    import librosa
    librosa.resample(np.zeros(1000, dtype=np.float32), orig_sr=44100, target_sr=16000)

    best_chunk, inference_results = benchmark_onnx_inference()
    n_streamers_rec = benchmark_streaming_throughput()

    # Save benchmark results
    bench_out = {
        "inference_by_chunk": {str(k): v for k, v in inference_results.items()},
        "best_chunk_s": best_chunk,
        "recommended_n_streamers": n_streamers_rec,
    }
    out_path = Path(__file__).parent / "bench_results.json"
    with open(out_path, "w") as f:
        json.dump(bench_out, f, indent=2)
    print(f"\nBenchmark results saved to {out_path}")

    print("\n=== Running parameter sweep ===")

    # Test a focused set of promising settings
    sweep_configs = [
        # (chunklength, n_streamers, buffer_depth, label_suffix)
        (200, 6, 4,  "200s_6str_buf4"),   # current default with ONNX
        (200, 6, 6,  "200s_6str_buf6"),   # more buffer
        (200, 8, 6,  "200s_8str_buf6"),   # more streamers
        (150, 6, 4,  "150s_6str_buf4"),   # smaller chunks
        (300, 6, 4,  "300s_6str_buf4"),   # larger chunks
    ]

    sweep_results = []
    for chunklength, n_str, buf, suffix in sweep_configs:
        test_name = f"onnx_sweep_{suffix}"
        comp = run_eval(test_name, chunklength, n_str, buf)
        if comp:
            overall = comp.get("overall_mean_delta_pct", None)
            sweep_results.append({
                "name": test_name,
                "chunklength": chunklength,
                "n_streamers": n_str,
                "buffer_depth": buf,
                "overall_delta_pct": overall,
                "verdict": comp.get("verdict"),
            })
            print(f"  {suffix}: {overall:+.1f}% ({comp.get('verdict')})")

    # Sort by performance
    sweep_results.sort(key=lambda x: x["overall_delta_pct"] or 0)
    best = sweep_results[0] if sweep_results else None

    sweep_path = Path(__file__).parent / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    if best:
        print(f"\nBest settings: chunk={best['chunklength']}s, "
              f"streamers={best['n_streamers']}, buf={best['buffer_depth']} → "
              f"{best['overall_delta_pct']:+.1f}% ({best['verdict']})")
