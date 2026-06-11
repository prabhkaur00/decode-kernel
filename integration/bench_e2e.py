"""
End-to-end SGLang throughput benchmark.

Launches an SGLang server with the custom split-KV backend on Llama 3.2 1B,
runs SGLang's own benchmark script with a fixed prompt mix, and compares
results against the stock FlashInfer backend on the same workload.

Results are written to results/sglang/.

Usage:
    python integration/bench_e2e.py             # runs both backends
    python integration/bench_e2e.py --backend split_kv  # one backend only
    python integration/bench_e2e.py --backend flashinfer

Prerequisites:
    See integration/README.md for the sgl_kernel source build instructions.
    SGLang server must NOT be running before invoking this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SERVER_URL  = "http://localhost:30000"
MODEL_ID    = "meta-llama/Llama-3.2-1B"
SERVER_PORT = 30000
RESULTS_DIR = Path("results/sglang")

# Benchmark parameters
NUM_PROMPTS   = 200
INPUT_LEN     = 512
OUTPUT_LEN    = 128


def start_server(backend: str, split_kv: int = 8) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    if backend == "split_kv":
        env["SGLANG_ATTENTION_BACKEND"] = "split_kv_triton"
        env["SPLIT_KV"] = str(split_kv)
    # backend == "flashinfer" uses sglang defaults

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",          MODEL_ID,
        "--port",                str(SERVER_PORT),
        "--tp",                  "1",
        "--dtype",               "float16",
        "--mem-fraction-static", "0.80",
        "--disable-radix-cache",
    ]
    print(f"  Starting SGLang server (backend={backend}) ...")
    proc = subprocess.Popen(cmd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(proc: subprocess.Popen, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline().decode("utf-8", errors="replace")
        if "ready" in line.lower() or "running" in line.lower():
            return True
        if proc.poll() is not None:
            print("  Server process exited unexpectedly")
            return False
    return False


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_sglang_bench(output_file: Path) -> dict:
    """Invokes SGLang's built-in benchmark_serving.py script."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend",       "sglang",
        "--base-url",      SERVER_URL,
        "--model",         MODEL_ID,
        "--num-prompts",   str(NUM_PROMPTS),
        "--input-len",     str(INPUT_LEN),
        "--output-len",    str(OUTPUT_LEN),
        "--output-file",   str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Benchmark failed:\n{result.stderr}")
        return {}
    # Parse JSON output if available
    if output_file.exists():
        with open(output_file) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {}


def print_comparison(results: dict):
    if not results:
        return
    print("\n── Results ──────────────────────────────────────────────────────")
    headers = ["backend", "throughput_tok_s", "ttft_ms_mean", "tbt_ms_mean"]
    fmt = "{:<20} {:>18} {:>14} {:>13}"
    print(fmt.format(*headers))
    print("─" * 68)
    for backend, r in results.items():
        if not r:
            continue
        print(fmt.format(
            backend,
            f"{r.get('output_throughput', 0):.1f}",
            f"{r.get('mean_ttft_ms', 0):.2f}",
            f"{r.get('mean_tpot_ms', 0):.2f}",
        ))


def run_benchmark(backends: list[str]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Backend: {backend}")
        print('='*60)

        proc = start_server(backend)
        ok = wait_for_server(proc)
        if not ok:
            print(f"  Server failed to start for backend={backend}")
            stop_server(proc)
            continue

        out_file = RESULTS_DIR / f"bench_{backend}.json"
        print(f"  Running benchmark ({NUM_PROMPTS} prompts) ...")
        r = run_sglang_bench(out_file)
        results[backend] = r

        stop_server(proc)
        print(f"  Results saved to {out_file}")
        time.sleep(3)   # allow port to free up

    print_comparison(results)

    summary_path = RESULTS_DIR / "comparison.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison written to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["split_kv", "flashinfer", "both"],
                        default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.backend == "both":
        backends = ["flashinfer", "split_kv"]
    else:
        backends = [args.backend]
    run_benchmark(backends)
