# Split-KV Decode Attention — Triton & CUDA

A decode-time attention kernel that partitions the KV cache across multiple
CTAs (split-KV), implemented in both Triton and CUDA C++, benchmarked against
FlashInfer's paged attention.  Layout is fully compatible with FlashInfer's
NHD paged KV format.

```
src/
  layout.py           paged KV layout helpers
  reference.py        fp32 CPU reference (correctness oracle)
  kernel_naive.py     Triton — single CTA per (batch, q_head)
  kernel_split_kv.py  Triton — split-KV two-pass
  attention.py        unified dispatch (Triton or CUDA via ATTN_BACKEND)
  cuda/
    decode_attn.cuh   warp reduce + shared memory layout
    attention_ext.cu  CUDA kernels + PyTorch bindings
    build.py          JIT compilation via torch.utils.cpp_extension
bench/
  microbench.py       latency + bandwidth sweep → CSV
  plot.py             generate plots from CSV
  profile.py          NCU profiling configs
  e2e_smoke.py        Llama 3.2 1B end-to-end check
writeup/
  REPORT.md           write-up template (fill in after running benchmarks)
  triton_primer.md    Triton concepts from scratch
  cuda_primer.md      CUDA concepts from scratch
notebook.ipynb        Colab notebook (runs everything top-to-bottom)
```

---

## Requirements

- Python 3.10+
- CUDA 12.1 (or 11.8 — setup.sh adapts automatically)
- An NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
- PyTorch 2.3.0

For the SGLang integration (`integration/`) you need compute capability ≥ 8.0
(Ampere).

---

## Setup

### Option A — Colab (recommended)

Open `notebook.ipynb` in Google Colab.  Section 0 auto-detects your CUDA and
PyTorch versions, installs the matching FlashInfer wheel, and verifies the
environment.  All subsequent sections run end-to-end without any manual steps.

### Option B — local machine

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd triton-flashinfer

# 2. Install PyTorch first (if not already installed)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Run the setup script (detects CUDA version, installs FlashInfer + deps)
bash env/setup.sh

# 4. Verify everything imported correctly
python env/verify.py
```

### Option C — pip only (known CUDA 12.1 + torch 2.3)

```bash
pip install -r env/requirements.txt \
  --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
```

---

## Correctness checks

Run from the repo root with `src/` on the Python path:

```bash
cd triton-flashinfer
PYTHONPATH=src python - <<'EOF'
from layout import synthesize
from reference import reference_attention_paged
from kernel_naive import decode_attention_naive
from kernel_split_kv import decode_attention_split_kv
import torch

q, kv, indptr, indices, last_len, _ = synthesize(
    batch=2, context_length=512,
    num_q_heads=32, num_kv_heads=8, head_dim=128,
)

ref = reference_attention_paged(q, kv, indptr, indices, last_len)

out_naive = decode_attention_naive(q, kv, indptr, indices, last_len)
out_split = decode_attention_split_kv(q, kv, indptr, indices, last_len, split_kv=4)

print("naive  max_err:", (out_naive.float() - ref.float()).abs().max().item())
print("split4 max_err:", (out_split.float() - ref.float()).abs().max().item())
EOF
```

Expected output: both errors below `1e-2`.

---

## Benchmarks

```bash
# Full sweep (~20 min on A100)
PYTHONPATH=src python bench/microbench.py --out results/bench.csv

# Quick sweep for CI (~2 min)
PYTHONPATH=src python bench/microbench.py --quick --out results/bench_quick.csv

# Generate all plots from a completed sweep
PYTHONPATH=src python bench/plot.py --csv results/bench.csv --out results/
```

Plots are written to `results/` as PNG files:
- `latency_vs_split_kv.png`
- `latency_vs_ctx.png`
- `bw_pct.png`
- `roofline.png`
- `speedup_heatmap.png`
- `gap_to_flashinfer.png`

---

## Switching backends (Triton vs CUDA)

The CUDA extension is JIT-compiled on first use (~30–60 s, cached to
`~/.cache/torch_extensions/` afterwards).

```bash
# Via environment variable
ATTN_BACKEND=cuda PYTHONPATH=src python your_script.py

# Via Python API
from attention import set_backend, decode_attention
set_backend("cuda")
out = decode_attention(q, kv, indptr, indices, last_len, split_kv=4)

# Per-call override
out = decode_attention(q, kv, indptr, indices, last_len, split_kv=4, backend="triton")
```

To compare both backends on the same input:

```python
from attention import compare_backends
results = compare_backends(q, kv, indptr, indices, last_len, split_kv=4)
print(results)
# {'split_kv': 4, 'triton_max_err': ..., 'cuda_max_err': ..., 'triton_vs_cuda_max_err': ...}
```

---

## NCU profiling

```bash
PYTHONPATH=src python bench/profile.py
```

Runs three configs (short context / small batch, long context / optimal split,
long context / over-split) and prints roofline-relevant metrics.  Requires
`ncu` on `PATH`; falls back to proxy metrics if not found.

---

## End-to-end smoke test (Llama 3.2 1B)

Downloads Llama 3.2 1B from Hugging Face (~2.5 GB) and verifies top-5 token
agreement between the reference and the split-KV kernel.

```bash
PYTHONPATH=src python bench/e2e_smoke.py
```

Requires a Hugging Face account and `HF_TOKEN` set in the environment, or use
the Colab notebook which handles authentication automatically.

---

## SGLang integration (optional, Ampere only)

See [integration/README.md](integration/README.md) for instructions on
building `sgl_kernel` from source and running the split-KV backend inside a
live SGLang server.

---

## Study materials

| File | What it covers |
|------|---------------|
| [writeup/triton_primer.md](writeup/triton_primer.md) | GPU hardware → Triton programming model → reading every line of the kernels |
| [writeup/cuda_primer.md](writeup/cuda_primer.md) | CUDA thread hierarchy, shared memory, warp shuffles, coalescing → reading `attention_ext.cu` |
| [writeup/REPORT.md](writeup/REPORT.md) | Write-up template — fill in after running benchmarks |
| [src/layout.md](src/layout.md) | Exact tensor shapes and strides for the FlashInfer NHD paged KV layout |
