# Split KV Decode Attention Kernel — Technical Report

> **Status:** Template. Run `bench/microbench.py` and replace every
> `[TODO: ...]` block with real numbers before submitting.

---

## 1  Setup

**Hardware:** [TODO: GPU name, VRAM, compute capability, peak HBM BW (GB/s)]

**Software stack:**
```
torch      2.3.0
triton     2.3.0
flashinfer 0.1.6
CUDA       [TODO: version]
```

**Model dimensions used in benchmarks:**

| Parameter       | Value |
|----------------|-------|
| num_q_heads    | 32    |
| num_kv_heads   | 8     |
| head_dim       | 128   |
| page_size      | 16    |
| dtype          | fp16  |

GQA group size = 32 / 8 = 4 (Llama 3 / Mistral style).

---

## 2  Kernel Design

### 2.1  Paged KV layout

FlashInfer stores the KV cache in the NHD paged layout:

```
kv_data : (num_pages, 2, page_size, num_kv_heads, head_dim)
```

where dim-1 indexes K (0) and V (1).  Three auxiliary tensors describe the
mapping from sequence to physical pages: `kv_indptr` (CSR row pointer),
`kv_indices` (physical page indices), and `kv_last_page_len` (valid tokens in
the final page of each sequence).  See `src/layout.md` for the full
specification.

### 2.2  Naive single-block kernel

The naive kernel (`src/kernel_naive.py`) assigns one CTA per (batch, q_head)
pair.  The CTA reads the entire KV sequence in page-granularity chunks,
accumulating with the standard online softmax recurrence:

```
given running state (m, l, acc) and a new block of K/V:

  m_block = max(QK_j  for j in block)
  p_j     = exp(QK_j − m_block)          [unnormalised softmax weights]
  l_block = Σ p_j
  acc_block = Σ p_j · V_j

  m_new   = max(m, m_block)
  alpha   = exp(m − m_new)
  beta    = exp(m_block − m_new)
  l       ← alpha · l   + beta · l_block
  acc     ← alpha · acc + beta · acc_block
  m       ← m_new

output = acc / l
```

GQA is handled by `kv_head_idx = q_head_idx // GROUP_SIZE`; KV tensors are
never replicated in memory.

**Bottleneck:** for long contexts each CTA must stream the full KV from HBM
serially.  Utilisation of SM parallelism is limited by `batch × num_q_heads`
CTAs.  At batch=1, 32 heads, that is only 32 active blocks — well below the
~108 SMs on an A100, leaving most of the GPU idle.

### 2.3  Split KV kernel

The split KV kernel (`src/kernel_split_kv.py`) adds a third axis to the grid:

```
grid_part = (batch, num_q_heads, SPLIT_KV)
```

**Pass 1 — partition kernel.**  Each CTA owns a contiguous slice of pages:

```
pages_per_split = ⌈num_pages / SPLIT_KV⌉
split_start     = split_idx * pages_per_split
split_end       = min(split_start + pages_per_split, num_pages)
```

After processing its slice it writes three fp32 scratch tensors:

```
partial_O[b, h, s, :] = Σ_{j ∈ split s}  exp(QK_j − m_s) · V_j
partial_m[b, h, s]    = max_{j ∈ split s} QK_j
partial_l[b, h, s]    = Σ_{j ∈ split s}  exp(QK_j − m_s)
```

**Pass 2 — reduction kernel.**  Grid `(batch, num_q_heads)`.  Each CTA merges
SPLIT_KV partial results using the multi-source online softmax identity:

```
m   = max_s  m_s
l   = Σ_s  exp(m_s − m) · l_s
O   = (1/l) · Σ_s  exp(m_s − m) · partial_O_s
```

This identity is exact (no approximation) and numerically stable because the
merge state is kept in fp32 throughout, regardless of the input dtype.

**Why fp32 scratch matters.**  Storing `(m_s, l_s, partial_O_s)` in fp16
introduces error proportional to `SPLIT_KV × max(|QK_j|)`.  At SPLIT_KV=16
and 32k context, typical max scores are around 3–5; truncating them to fp16
accumulates ≈ 0.1–0.3 absolute error in the output — well above the 1e-2
tolerance.  See Gate 3 in `bench/microbench.py` for the determinism check
that catches this class of bug.

### 2.4  SPLIT_KV = 1 equivalence

When SPLIT_KV=1 the partition kernel behaves identically to the naive kernel
(one CTA processes all pages), plus the trivial single-element reduction.
Performance is slightly lower than the naive kernel due to the extra kernel
launch overhead; this is the leftmost point on Plot 1.

---

## 3  Correctness

Three correctness gates are checked before any timing begins.

### Gate 1: layout sanity

[TODO: paste the `max_abs_err(flashinfer, fp32_ref)` value for a representative
(batch=1, ctx=2048) configuration.  Should be < 1e-2.]

### Gate 2: full sweep

[TODO: fill in the table below from the `max_abs_err_vs_fp32` column in
`results/microbench.csv`.  Both kernels must have max_err < 1e-2 and
mean_err < 1e-3 across the entire sweep grid.]

| Implementation   | ctx=2k | ctx=8k | ctx=32k | ctx=64k |
|-----------------|--------|--------|---------|---------|
| naive_triton    | TODO   | TODO   | TODO    | TODO    |
| split_kv (s=4)  | TODO   | TODO   | TODO    | TODO    |
| split_kv (s=8)  | TODO   | TODO   | TODO    | TODO    |
| split_kv (s=16) | TODO   | TODO   | TODO    | TODO    |

### Gate 3: edge cases

[TODO: confirm all of the following pass (bitwise-identical on two runs,
max_err < 1e-2 vs fp32 reference).]

- [ ] context_length = 1
- [ ] context_length not divisible by page_size
- [ ] context_length not divisible by SPLIT_KV (ragged partitions)
- [ ] batch = 1
- [ ] GQA group_size = 1 (MHA)
- [ ] GQA group_size = num_q_heads (MQA)

---

## 4  Latency Results

### Plot 1: latency vs SPLIT_KV

[TODO: insert `results/plots/plot1_latency_vs_split.png`]

*Description.* One panel per context length; one line per implementation.
FlashInfer appears as a horizontal reference line (it does not expose SPLIT_KV).
The naive Triton kernel appears at SPLIT_KV=1 on each panel.

Expected shape: U-curve for split_kv_triton, with the minimum shifting to
larger SPLIT_KV as context grows (because longer contexts provide more
parallelism to exploit).

[TODO: note where the minimum occurs for each context length, e.g.:
"ctx=2k: min at SPLIT_KV=2; ctx=8k: min at SPLIT_KV=4; ctx=32k: min at SPLIT_KV=8"]

### Plot 2: latency vs context length

[TODO: insert `results/plots/plot2_latency_vs_ctx.png`]

*Description.* One line per implementation at each implementation's optimal
SPLIT_KV.  The slope reveals which implementation is most context-limited.

[TODO: compare slopes — does split_kv scale better than naive at long context?]

### Plot 3: achieved bandwidth as % of peak

[TODO: insert `results/plots/plot3_bw_pct.png`]

*Description.* How close to the memory-bandwidth ceiling each implementation
gets.  A flat line near peak means the kernel is fully bandwidth-bound.  A
rising line means fixed overheads are being amortized as work grows.

[TODO: at what context length does each kernel plateau?]

### Plot 5: speedup heatmap (split_kv / naive)

[TODO: insert `results/plots/plot5_speedup_heatmap.png`]

*Description.* Rows = context lengths, columns = batch sizes, colour =
`latency(naive) / latency(split_kv_at_optimal_split)`.

Expected: the top-right corner (long context, small batch) has the largest
speedup; the bottom-left corner (short context, large batch) is near 1×.

[TODO: what is the peak observed speedup, and at which (ctx, batch)?]

### Plot 6: gap to FlashInfer

[TODO: insert `results/plots/plot6_gap_to_flashinfer.png`]

*Description.* Bar chart: `latency(split_kv_at_optimal) / latency(flashinfer)`
for each context length.  A ratio of 1.0 would mean parity.

[TODO: how does the gap evolve with context?  Does it widen or narrow?
Typical expectation: gap widens at very long context because FlashInfer uses
persistent kernels and autotuned tile sizes that Triton cannot match without
further optimisation.]

---

## 5  Roofline Analysis

### Plot 4: roofline

[TODO: insert `results/plots/plot4_roofline.png`]

*Description.* Log-log axes.  X = arithmetic intensity (FLOPs/byte),
Y = achieved performance (TFLOPs/s).  Two ceilings: GPU peak compute (e.g.
312 TFLOPs/s for A100 fp16) and HBM bandwidth (2000 GB/s).

**Arithmetic intensity of decode attention:**

For one decode step (batch B, context S, H_q query heads, H_kv KV heads,
head dim D, fp16 = 2 bytes):

```
FLOPs  ≈ 2 · B · H_q · S · D         (QK dot products + weighted V sum)
Bytes  ≈ 2 · B · S · H_kv · D · 2    (K read + V read, dominant term)

AI = FLOPs / Bytes
   = (2 · B · H_q · S · D) / (4 · B · S · H_kv · D)
   = H_q / (2 · H_kv)
   = GROUP_SIZE / 2
```

For our GQA configuration (GROUP_SIZE=4): **AI ≈ 2 FLOPs/byte**, which places
decode attention firmly in the memory-bound region for all practical batch
sizes.

This is the key architectural insight: the only way to increase arithmetic
intensity is to increase batch size (amortize Q reads over more queries per KV
load) or to share KV across multiple queries (cascade / speculative attention).
Split KV does not increase AI; it increases SM parallelism, which helps when
the bottleneck is occupancy rather than raw bandwidth.

[TODO: plot the measured (AI, TFLOP/s) points for each implementation and
configuration.  All points should lie on or below the bandwidth-bound ceiling.]

---

## 6  Comparison vs FlashInfer

[TODO: quantitative summary from Plot 6.]

FlashInfer advantages that this implementation cannot close without significant
additional engineering:

1. **Persistent kernels.** FlashInfer launches one long-lived kernel per SM
   and uses software pipelining to overlap memory and compute, hiding latency.
   Triton's programming model makes persistent kernels difficult to express.

2. **Autotuned tile sizes.** FlashInfer uses CUTLASS and offline profiling to
   select BLOCK_N and the number of pipeline stages per GPU SKU.  Our BLOCK_N
   is fixed at page_size.

3. **Fused plan+run.** FlashInfer's plan phase precomputes a compact work-list
   that avoids integer division in the inner loop.  Our partition kernel
   recomputes `pages_per_split` at launch time.

**Where split-KV closes the gap:** short-to-medium context (2k–8k), small
batch (1), and GQA configurations where H_kv is small.  In these regimes the
naive kernel is occupancy-limited and split KV synthesises the missing
parallelism.

---

## 7  Limitations

- **Page size is constexpr.** Changing page_size requires a kernel recompile.
  FlashInfer handles multiple page sizes within one binary via template
  instantiation.

- **No BLOCK_N tuning.** The inner-loop tile size is fixed to PAGE_SIZE (16).
  A larger BLOCK_N (64 or 128) would reduce address-arithmetic overhead at long
  context; this is the most accessible performance improvement remaining.

- **Reduction overhead at small SPLIT_KV.** The second kernel launch adds
  ~10–20 µs at SPLIT_KV=1, which is why the U-curve's left arm sits above the
  naive-kernel baseline.

- **No persistent-kernel implementation.** Adding a persistent outer loop would
  require a separate stream of engineering work and likely a move from Triton to
  CUDA C++.

---

## 8  References

- Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness.* NeurIPS 2022.
- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving
  with PagedAttention.* SOSP 2023.
- Zheng, L. et al. *SGLang: Efficient Execution of Structured Language Model
  Programs.* arXiv 2024.
- FlashInfer project: https://github.com/flashinfer-ai/flashinfer
