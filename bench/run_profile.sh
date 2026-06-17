#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --cuda-memory-usage=true \
    -o results/naive_nvtx \
    python bench/profile_naive_nvtx.py
nsys stats results/naive_nvtx.nsys-rep