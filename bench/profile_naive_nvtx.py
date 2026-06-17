"""
Standalone driver for nsys profiling of the decode attention CUDA kernels.

Run with:
  nsys profile --trace=cuda,nvtx --force-overwrite=true \
      -o results/naive_nvtx python bench/profile_naive_nvtx.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.cpp_extension import load

# ── repo paths ────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
SRC  = REPO / "src"
sys.path.insert(0, str(SRC))                 # so `import layout` works
from layout import synthesize                # noqa: E402

# ── build the extension (JIT) ─────────────────────────────────────────────
ext = load(
    name="attention_ext",
    sources=[str(SRC / "attention_ext.cu")],
    extra_include_paths=[str(SRC)],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

# ── workload config ───────────────────────────────────────────────────────
BATCH, CTX, HQ, HKV, HD, PAGE = 4, 2048, 8, 2, 128, 16
SPLIT_KV = 4
WARMUP_ITERS = 5
PROFILE_ITERS = 20

q, kv_data, kv_indptr, kv_indices, kv_last, layout = synthesize(
    batch=BATCH, context_length=CTX,
    num_q_heads=HQ, num_kv_heads=HKV,
    head_dim=HD, page_size=PAGE,
    dtype=torch.float16, device="cuda",
)
layout.validate(kv_data, kv_indptr, kv_indices, kv_last)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Q shape: {tuple(q.shape)}, kv_data shape: {tuple(kv_data.shape)}")

# ── warmup (labelled so you can ignore it on the timeline) ────────────────
nvtx.range_push("warmup")
for _ in range(WARMUP_ITERS):
    _ = ext.decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last)
    _ = ext.decode_attention_split_kv(q, kv_data, kv_indptr, kv_indices, kv_last, SPLIT_KV)
torch.cuda.synchronize()
nvtx.range_pop()

# ── profiled iterations ───────────────────────────────────────────────────
nvtx.range_push("naive_loop")
for i in range(PROFILE_ITERS):
    nvtx.range_push(f"naive_iter_{i}")
    out_naive = ext.decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last)
    torch.cuda.synchronize()
    nvtx.range_pop()
nvtx.range_pop()

nvtx.range_push("split_kv_loop")
for i in range(PROFILE_ITERS):
    nvtx.range_push(f"splitkv_iter_{i}")
    out_split = ext.decode_attention_split_kv(q, kv_data, kv_indptr, kv_indices, kv_last, SPLIT_KV)
    torch.cuda.synchronize()
    nvtx.range_pop()
nvtx.range_pop()

# ── sanity check ──────────────────────────────────────────────────────────
max_diff = (out_naive.float() - out_split.float()).abs().max().item()
print(f"max abs diff naive vs split_kv: {max_diff:.5f}")
print("done")