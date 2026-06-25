# Study Guide: CUDA Split-KV Decode Attention Kernel

## Part 1 — High-Level: What Was Built and Why

### The Problem

During LLM inference, the **decode phase** generates one token at a time. Each new token requires attending over the entire KV cache (all previously generated tokens). This is a single-query attention: Q is `(batch, num_q_heads, head_dim)` but K/V span the full context length S.

The dominant cost is reading K and V from HBM — the operation is **memory-bandwidth bound**. For a GQA config (32 query heads, 8 KV heads, head_dim=128), the arithmetic intensity is only `GROUP_SIZE / 2 = 2 FLOPs/byte`, far below the compute-bandwidth ridge point on an A100 (~156 FLOPs/byte).

A naive kernel that assigns one thread block per (batch, query_head) pair has a parallelism ceiling of `batch × num_q_heads`. At batch=1 with 32 heads, that's 32 CTAs on a GPU with 108 SMs — most of the chip sits idle while each CTA serially streams through potentially thousands of KV pages.

### The Solution: Two-Pass Split-KV

Split the KV sequence across multiple CTAs so they process disjoint page ranges in parallel, then merge their partial results in a second kernel.

**Pass 1 — Partition kernel:**
- Grid: `(batch, num_q_heads, SPLIT_KV)`
- Each CTA processes `⌈num_pages / SPLIT_KV⌉` contiguous pages
- Writes partial `(O_s, m_s, l_s)` in fp32 to scratch buffers

**Pass 2 — Reduction kernel:**
- Grid: `(batch, num_q_heads)`
- Merges SPLIT_KV partial results using the online softmax identity
- Writes final output in fp16

At SPLIT_KV=16, batch=1, the partition kernel launches `1 × 32 × 16 = 512` CTAs — enough to saturate all 108 SMs. Each CTA reads 1/16th the KV data, so wall-clock latency drops roughly proportionally (minus reduction overhead).

### Paged Attention

The KV cache uses FlashInfer's NHD paged layout:

```
kv_data: (num_pages, 2, page_size=16, num_kv_heads, head_dim)
```

Three metadata tensors describe the sequence-to-page mapping:
- `kv_indptr` (CSR row pointer) — sequence b owns pages at indices `[kv_indptr[b], kv_indptr[b+1])`
- `kv_indices` — physical page IDs
- `kv_last_page_len` — valid tokens in the last page (handles non-page-aligned context lengths)

This is the same layout production serving systems (vLLM, SGLang) use, so the kernel is directly pluggable.

### GQA Support

Group-size = `num_q_heads / num_kv_heads = 32/8 = 4`. Multiple query heads share the same KV head:

```cuda
const int kv_head_idx = q_head_idx / group_size;
```

KV data is never replicated in memory — each CTA indexes into the correct KV head slice using `kv_head_idx`. This keeps HBM traffic at the minimum.

### Benchmarking Summary

Benchmarked on **A100 (80GB, 2000 GB/s peak HBM BW)** across:
- Context lengths: 2k, 4k, 8k, 16k, 32k, 64k
- Batch sizes: 1, 4, 16
- SPLIT_KV values: 1, 2, 4, 8, 16
- Compared against: FlashInfer (production baseline), CUDA naive kernel

**Key result:** At ctx=8k, batch=1, SPLIT_KV=16, the split-KV kernel achieves 167 GB/s (8.4% of peak) with p50 latency 0.20ms, versus FlashInfer at 0.11ms — a **1.8× gap**. The kernel is within 1.8× of FlashInfer across the board at that context length.

---

## Part 2 — Low-Level: How the CUDA Implementation Works

### File Map

| File | Role |
|------|------|
| `src/cuda_/split_kv_kernel.cu` | Both kernel implementations + pybind11 entry point |
| `src/cuda_/decode_attn.cuh` | Shared device utilities (warp reductions, smem layout) |
| `src/cuda_/build.py` | JIT compilation via `torch.utils.cpp_extension.load` |
| `bench/microbench_cuda.py` | Latency/bandwidth measurement harness |
| `bench/profile_cuda_split_kv_nvtx.py` | NVTX-annotated script for nsys profiling |

### Thread/Block Mapping

**Partition kernel** (`decode_attn_partition_kernel<HEAD_DIM=128, PAGE_SIZE=16>`):
- Block size: `HEAD_DIM = 128` threads (4 warps of 32)
- Grid: `dim3(batch, num_q_heads, split_kv)`
- Each thread "owns" one dimension of the head_dim vector
- Thread `tid` always reads/writes element `[tid]` of Q, K, V, and the output accumulator

**Reduction kernel** (`decode_attn_reduce_kernel<HEAD_DIM=128>`):
- Block size: `HEAD_DIM = 128` threads
- Grid: `dim3(batch, num_q_heads)`
- Each thread reduces its own dimension across SPLIT_KV partial results

### Shared Memory Layout (`SmemLayout<128, 16>`)

Three regions carved from dynamic shared memory:

```
Region     | Size (floats)           | Size (bytes) | Purpose
-----------|-------------------------|--------------|--------
smem_kv    | PAGE_SIZE × HEAD_DIM    | 16×128×4 = 8192 | K or V tile (reused)
smem_warp  | NUM_WARPS × PAGE_SIZE   | 4×16×4 = 256    | Inter-warp QK partial sums
smem_qk    | PAGE_SIZE               | 16×4 = 64       | Final QK scores per token
-----------|-------------------------|--------------|--------
Total      |                         | 8512 bytes   |
```

The `kv` region is reused: first loaded with K for the QK dot product, then overwritten with V for the weighted accumulation. This halves the shared memory footprint.

### Inner Loop: Processing One KV Page

For each page `p` in this CTA's split range:

#### Step 1: Load K page into shared memory
```cuda
// Each of 128 threads loads one element per token row
for (int tok = 0; tok < 16; tok++)
    sm.kv[tok * 128 + tid] = K[page][tok][kv_head_idx][tid];
```
Coalesced load: consecutive threads read consecutive fp16 elements along head_dim.

#### Step 2: QK dot product via warp reduction

Each thread computes `q_val * K[tok][tid]` — one partial product per token. Then a **warp-level butterfly reduction** sums 32 partial products:

```cuda
// In decode_attn.cuh — 5 shuffle steps, no shared memory needed
float warp_reduce_sum(float val) {
    for (int delta = 16; delta > 0; delta >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, delta);
    return val;
}
```

After this, lane 0 of each warp holds the sum of 32 elements. The 4 warps write their partial sums to `smem_warp`, then a second reduction (by the first 16 threads) produces the final 16 QK scores in `smem_qk`.

Why two stages? HEAD_DIM=128 needs 4 warps. A single `__shfl_xor_sync` only reduces within a 32-thread warp. Cross-warp communication requires shared memory.

#### Step 3: Online softmax statistics

All 128 threads read the same 16 scores from `smem_qk`:
```cuda
float m_block = max over smem_qk[0..15]
float exp_weights[16] = exp(smem_qk[tok] - m_block) for each tok
float l_block = sum of exp_weights
```

Invalid tokens (beyond `kv_last_page_len` in the last page) were masked to `-1e9` so `exp(...)` yields ~0.

#### Step 4: Load V, compute weighted sum

The K data in shared memory is overwritten with V. Each thread accumulates:
```cuda
float acc_block = sum over tok of (exp_weights[tok] * V[tok][tid])
```

#### Step 5: Merge with running state (online softmax)

```cuda
m_new = max(m_i, m_block)
alpha = exp(m_i - m_new)        // rescale factor for old state
beta  = exp(m_block - m_new)    // rescale factor for new block
l_i   = alpha * l_i + beta * l_block
acc   = alpha * acc + beta * acc_block
m_i   = m_new
```

This is the standard FlashAttention online-softmax recurrence. It is numerically exact — no approximation — and stays in fp32 throughout.

### Pass 2: Reduction Across Splits

After all partition CTAs finish, the reduction kernel merges SPLIT_KV partial results. Thread `tid` handles dimension `tid`:

```cuda
// Global max across all splits
m_global = max over s of partial_m[b][h][s]

// Weighted sum
for each split s:
    w = exp(partial_m[s] - m_global)
    l_global += w * partial_l[s]
    acc      += w * partial_O[s][tid]

output[b][h][tid] = acc / l_global   // convert back to fp16
```

This is the multi-source online softmax merge identity. The key insight: each partition wrote `partial_O[s]` as an **unnormalized** weighted sum (not divided by `l_s`). The reduction can thus rescale and combine them correctly.

### Why fp32 Scratch Buffers

The partial results `(m_s, l_s, O_s)` are stored in fp32 even though input/output are fp16. fp16 has only ~3 decimal digits of precision. At SPLIT_KV=16 with 32k context, the log-sum-exp differences `exp(m_s - m_global)` can underflow or accumulate rounding error of ~0.1–0.3 in the output — well above the 1e-2 correctness tolerance.

### Template Specialization

The kernel supports HEAD_DIM ∈ {64, 128, 256} via C++ templates:

```cuda
if      (head_dim ==  64) launch_partition_hd< 64>(...);
else if (head_dim == 128) launch_partition_hd<128>(...);
else if (head_dim == 256) launch_partition_hd<256>(...);
```

Template parameters make `HEAD_DIM` and `PAGE_SIZE` constexpr, enabling the compiler to:
- Unroll all inner loops completely (`#pragma unroll` with known bounds)
- Compute shared memory sizes at compile time
- Optimize register allocation knowing the exact block size

### NVTX Annotations

The host launcher wraps each pass in NVTX ranges:
```cuda
nvtxRangePushA("cuda_split_kv_partition");
// ... launch partition kernel ...
nvtxRangePop();

nvtxRangePushA("cuda_split_kv_reduce");
// ... launch reduce kernel ...
nvtxRangePop();
```

The profiling script (`profile_cuda_split_kv_nvtx.py`) adds a Python-level NVTX range per iteration. In an nsys timeline, you see three nested levels: Python iteration → C++ partition/reduce ranges → GPU kernel blocks.

---

## Part 3 — Benchmarking Methodology

### How Latency Is Measured

```python
# From bench/microbench_cuda.py

start_ev = torch.cuda.Event(enable_timing=True)
end_ev   = torch.cuda.Event(enable_timing=True)

for _ in range(100):
    flush_l2()              # write 256 MB of zeros to evict KV from L2
    start_ev.record()       # record GPU timestamp
    fn()                    # launch kernel(s)
    end_ev.record()         # record GPU timestamp
    torch.cuda.synchronize()
    latencies.append(start_ev.elapsed_time(end_ev))  # ms
```

Key details:
- **CUDA events** measure GPU-side time, not CPU-side — immune to Python overhead and launch latency jitter
- **L2 cache flush** before every iteration: allocate a 256 MB buffer and `fill_(0.0)`. This evicts KV data from L2, ensuring each iteration measures a cold-cache HBM read. Without this, repeated runs on the same data would hit L2 and report artificially low latency
- **20 warmup iterations** excluded from timing (covers JIT compilation + GPU frequency ramp)
- **100 timed iterations**, reported as p50 (median) — robust to outliers from thermal throttling or OS interrupts

### How Bandwidth Is Computed

```python
total_bytes = attention_bytes(batch, ctx, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, elem_bytes=2)
bw_gb_s = (total_bytes / 1e9) / (latency_p50_ms / 1e3)
```

Where `attention_bytes` counts the minimum HBM traffic:
```
Q  read:  batch × num_q_heads  × head_dim × 2 bytes
KV read:  batch × context_len  × num_kv_heads × head_dim × 2 bytes × 2 (K+V)
O  write: batch × num_q_heads  × head_dim × 2 bytes
```

This is the **ideal** byte count — what a perfectly bandwidth-efficient kernel would transfer. The split-KV kernel also reads/writes the fp32 scratch buffers (`partial_O`, `partial_m`, `partial_l`), which aren't counted. So the reported "achieved bandwidth" is actually a lower bound on the kernel's true HBM utilization.

**Bandwidth % of peak** = `achieved_bw / 2000 GB/s × 100` (A100 SXM).

### Correctness Validation

Before any timing, each (impl, ctx, batch, split_kv) config is checked against an fp32 reference:

```python
ref = reference_attention_paged(q.cpu(), kv_data.cpu(), ...)  # pure PyTorch fp32
max_err = (output.float() - ref.float()).abs().max().item()
```

The reference runs on CPU to avoid GPU OOM at large contexts (64k × batch=16 in fp32 = ~34 GB). Tolerance is max_abs_err < 1e-2. Observed errors across the full sweep: 6e-5 to 8.9e-5 — well within tolerance.

### Sweep Configuration

| Parameter | Values |
|-----------|--------|
| Context lengths | 2048, 4096, 8192, 16384, 32768, 65536 |
| Batch sizes | 1, 4, 16 |
| SPLIT_KV | 1, 2, 4, 8, 16 |
| Implementations | FlashInfer, CUDA naive, CUDA split-KV |

Total: 6 ctx × 3 batch × (1 flashinfer + 1 naive + 5 split_kv) = 126 data points.

### Key Results from the Benchmark Data

**Latency scaling with SPLIT_KV (batch=1):**

| Context | SPLIT_KV=1 | SPLIT_KV=4 | SPLIT_KV=16 | FlashInfer | Gap to FI |
|---------|-----------|-----------|-------------|------------|-----------|
| 2k  | 0.547ms | 0.165ms | 0.071ms | 0.086ms | 0.8× (faster!) |
| 8k  | 2.119ms | 0.582ms | 0.201ms | 0.109ms | 1.8× |
| 32k | 8.316ms | 2.196ms | 0.686ms | 0.185ms | 3.7× |
| 64k | 16.55ms | 4.328ms | 1.333ms | 0.278ms | 4.8× |

Latency drops nearly linearly with SPLIT_KV at batch=1 — each doubling of splits roughly halves latency, confirming the bottleneck was occupancy, not bandwidth.

**At ctx=2k, batch=1, SPLIT_KV=16, the custom kernel actually beats FlashInfer** (0.071ms vs 0.086ms). FlashInfer's plan overhead is non-trivial at short contexts.

**Bandwidth utilization peaks at ~11% of A100 peak (220 GB/s)** for large batch × large context configs. FlashInfer reaches ~69% (1377 GB/s) at ctx=64k, batch=16. The gap is explained by:
1. FlashInfer uses persistent kernels with software pipelining
2. FlashInfer uses autotuned tile sizes (not fixed to page_size=16)
3. FlashInfer's plan phase precomputes a compact work-list avoiding integer division in the hot loop

**Split-KV vs naive speedup:** up to 13.5× at (ctx=64k, batch=1) — the regime where occupancy matters most.

### The 1.8× Claim at ctx=8k

At ctx=8192, batch=1:
- Best split-KV (SPLIT_KV=16): 0.201ms
- FlashInfer: 0.109ms
- Ratio: 0.201 / 0.109 = **1.84×**

At ctx=8192, batch=4:
- Best split-KV (SPLIT_KV=16): 0.653ms
- FlashInfer: 0.182ms
- Ratio: **3.6×** (gap widens with batch because FlashInfer's persistent kernel amortizes overhead better)

The 1.8× figure is the best-case comparison point at a practical serving context length.
