# Triton Primer — From Zero to This Repo

A bottom-up introduction for someone who knows PyTorch but has never written a
GPU kernel.  By the end you should be able to read every line in
`src/kernel_naive.py` and `src/kernel_split_kv.py` without confusion.

---

## 0  Why does Triton exist?

PyTorch operations are fast because they call hand-tuned CUDA kernels written
by NVIDIA engineers.  The problem is that custom operations — anything not
already in PyTorch — require writing CUDA C++, which is notoriously tedious.
You must manage thread indexing, shared memory, vectorized loads, and occupancy
tuning all at once.

Triton is a Python DSL (embedded in Python via a JIT decorator) that lets you
write kernels at the **block** level instead of the **thread** level.  You say
"this CTA (cooperative thread array, i.e., block of threads) handles tile
`[i*BLOCK:(i+1)*BLOCK]` of the input" and Triton figures out the per-thread
indexing, vectorized memory instructions, and register allocation.

Result: 80% of CUDA performance in 20% of the code, with full Python
interoperability.

---

## 1  The GPU execution model in one page

```
GPU
├── SM 0  (Streaming Multiprocessor)
│   ├── warp 0  (32 threads that execute in lockstep)
│   ├── warp 1
│   └── ...    (up to 32 warps = 1024 threads per SM on Ampere)
├── SM 1
└── ...  (108 SMs on A100)
```

**Key hardware facts:**

| Resource       | A100 (SXM4) | T4      |
|---------------|-------------|---------|
| SMs           | 108         | 40      |
| VRAM (HBM)    | 80 GB       | 16 GB   |
| HBM bandwidth | 2000 GB/s   | 300 GB/s|
| L2 cache      | 40 MB       | 4 MB    |
| Shared memory / SM | 164 KB | 64 KB  |

**Memory hierarchy (slow → fast, large → small):**

```
HBM (VRAM)          ← all torch tensors live here
   ↕  ~2000 GB/s
L2 cache (40 MB)    ← automatic, not programmer-controlled
   ↕  much faster
Shared memory (SRAM, 164 KB/SM) ← explicit in CUDA; Triton manages it
   ↕  ~20 TB/s effective
Registers           ← per-thread; fastest
```

The critical insight for attention: **decode attention is memory-bound**, not
compute-bound.  The GPU spends most of its time waiting for data to arrive from
HBM, not doing arithmetic.  This is why bandwidth utilisation (GB/s) is the
primary performance metric here, not TFLOP/s.

---

## 2  The Triton programming model

### 2.1  Programs, not threads

In CUDA you write a kernel function that runs once per **thread**, and you use
`threadIdx.x`, `blockIdx.x`, etc. to figure out which piece of work that thread
owns.

In Triton you write a kernel function that runs once per **program** (one CTA),
and you use `tl.program_id(axis)` to figure out which tile of work that program
owns.  Within the program you operate on **blocks of values**, not scalars.

```python
# CUDA style (one thread handles one element)
def kernel(A, B, C, N):
    i = blockIdx.x * blockDim.x + threadIdx.x
    if i < N:
        C[i] = A[i] + B[i]

# Triton style (one program handles BLOCK_N elements)
@triton.jit
def kernel(A, B, C, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)                # which tile am I?
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # element indices
    mask = offs < N
    a = tl.load(A + offs, mask=mask)      # load a vector
    b = tl.load(B + offs, mask=mask)
    tl.store(C + offs, a + b, mask=mask)  # store a vector
```

The grid — how many programs launch — is specified in Python:
```python
grid = (triton.cdiv(N, BLOCK_N),)   # ceil(N / BLOCK_N) programs
kernel[grid](A, B, C, N, BLOCK_N=128)
```

### 2.2  `tl.constexpr`

Any parameter annotated `tl.constexpr` is a **compile-time constant**.  Triton
generates a separate compiled binary for each unique combination of constexpr
values.

This matters for two reasons:

1. `tl.arange(0, BLOCK_N)` requires `BLOCK_N` to be constexpr — the array
   shape must be known at compile time.
2. `for s in tl.static_range(SPLIT_KV)` unrolls at compile time; without
   constexpr this would be a slow dynamic loop.

Rule of thumb: block sizes and loop trip counts that are tuning parameters
should be constexpr.  Shapes that vary across different dataset configurations
(batch, context length) should be runtime values.

### 2.3  `tl.arange`, shapes, and broadcasting

`tl.arange(0, N)` creates a 1-D block of `N` integers `[0, 1, ..., N-1]`.
`N` must be a power of 2 and constexpr.

2-D blocks use NumPy-style broadcasting:
```python
offs_row = tl.arange(0, ROWS)   # shape [ROWS]
offs_col = tl.arange(0, COLS)   # shape [COLS]

# 2-D pointer grid: shape [ROWS, COLS]
ptrs = base + offs_row[:, None] * stride_row + offs_col[None, :] * stride_col
data = tl.load(ptrs)             # shape [ROWS, COLS]
```

This pattern is used in both kernels to load the K and V blocks:
```python
# From kernel_naive.py:
k_ptrs = (k_base
          + offs_ps[:, None] * stride_kvs   # page-positions: [PAGE_SIZE, 1]
          + offs_d[None, :]  * stride_kvd)  # head-dim:       [1, BLOCK_D]
# Result: k_ptrs has shape [PAGE_SIZE, BLOCK_D]
k = tl.load(k_ptrs, mask=tok_mask[:, None] & d_mask[None, :], other=0.0)
```

### 2.4  Masking

Triton arrays are always power-of-2 sized, but real data often isn't.  The
`mask` argument to `tl.load` / `tl.store` handles out-of-bounds positions:

```python
offs = tl.arange(0, 128)        # BLOCK_D = 128 (next power of 2 ≥ head_dim)
mask = offs < head_dim           # e.g., head_dim=100 → mask is True for 0..99
q = tl.load(q_ptr + offs, mask=mask, other=0.0)  # pad with 0 beyond head_dim
```

Two masks appear in the attention kernels:
- `d_mask = offs_d < head_dim` — pads BLOCK_D to a power of 2
- `tok_mask = offs_ps < valid_toks` — masks the last (partially-filled) page

### 2.5  Reductions

`tl.sum(x, axis=0)` / `tl.max(x, axis=0)` reduce over the first axis.

```python
qk = tl.sum(q[None, :] * k, axis=1)  # shape [PAGE_SIZE, BLOCK_D] → [PAGE_SIZE]
```

This is the QK dot product: for each of the `PAGE_SIZE` key vectors, compute
the dot product with the single query vector `q`.

### 2.6  `tl.static_range` vs `range`

```python
# Dynamic loop — loop count known only at runtime
for p in range(0, my_num_pages):   # Triton emits a loop instruction
    ...

# Static loop — loop count is constexpr, fully unrolled at compile time
for s in tl.static_range(SPLIT_KV):  # Triton emits SPLIT_KV copies of the body
    ...
```

The reduction kernel uses `tl.static_range(SPLIT_KV)` because SPLIT_KV is
small (1–16) and constexpr; unrolling avoids branch overhead and lets the
compiler do scalar replacement.

---

## 3  Attention — the algorithm

Standard scaled dot-product attention for one query token against a context of
`S` key/value pairs:

```
scores[i] = dot(Q, K[i]) / sqrt(D)        for i in 0..S-1
weights    = softmax(scores)               weights[i] = exp(scores[i]) / Σ exp(scores[j])
output     = Σ_i  weights[i] * V[i]
```

For decode (autoregressive generation), `Q` has sequence-length 1 (one new
token) and `K`, `V` hold the entire context so far.

**The problem with naïve softmax:** you need all scores before you can compute
the denominator `Σ exp(scores[j])`.  If the context is long (32k tokens), all
scores don't fit in registers simultaneously.

### 3.1  Online softmax (the key algorithmic trick)

Process the context in blocks.  Maintain a running state `(m, l, acc)`:

| Symbol | Meaning |
|--------|---------|
| `m`    | running maximum score seen so far |
| `l`    | running sum of `exp(score - m)` |
| `acc`  | running sum of `exp(score - m) * V` (unnormalised output) |

When a new block of scores arrives:

```
m_block = max(new_scores)
p_block = exp(new_scores - m_block)    # [BLOCK_N] weights, relative to m_block
l_block = sum(p_block)
acc_block = p_block @ V_block          # [D] partial output

# Merge into running state:
m_new   = max(m, m_block)
alpha   = exp(m - m_new)               # rescale old state
beta    = exp(m_block - m_new)         # rescale new block
l       = alpha * l   + beta * l_block
acc     = alpha * acc + beta * acc_block
m       = m_new

# After all blocks: output = acc / l
```

**Why this is numerically stable:** every `exp` is computed relative to a local
maximum, so the argument to `exp` is always ≤ 0.  No overflow.

This recurrence is the core of FlashAttention and every modern attention kernel.
It appears verbatim in both `kernel_naive.py` and `kernel_split_kv.py`.

### 3.2  GQA (Grouped Query Attention)

In standard MHA (multi-head attention), each query head has its own K/V head.
In GQA, `G` query heads share one K/V head:

```
kv_head_idx = q_head_idx // GROUP_SIZE
```

This reduces the KV cache by factor `G` (crucial for long contexts).  In Llama
3 and Mistral: `num_q_heads=32`, `num_kv_heads=8`, `GROUP_SIZE=4`.

The kernels handle GQA by computing `kv_head_idx` at the start and always
indexing into the KV cache with the reduced index.  No data replication.

---

## 4  Paged KV cache

Instead of one large contiguous tensor per sequence, the KV cache is divided
into fixed-size **pages** (default: 16 tokens/page).  Pages are referenced
through an indirection table, allowing:
- Variable-length sequences without padding
- Memory sharing across sequences (used by PagedAttention / vLLM)

**The four tensors** (fully documented in `src/layout.md`):

```
kv_data      : (num_pages, 2, page_size, num_kv_heads, head_dim)  fp16
kv_indptr    : (batch + 1,)   int32    — CSR row pointer
kv_indices   : (total_pages,) int32    — physical page index per slot
kv_last_page_len: (batch,)    int32    — valid tokens in last page
```

To iterate over sequence `b`'s KV:
```python
for ptr in range(kv_indptr[b], kv_indptr[b+1]):
    page = kv_indices[ptr]
    is_last = (ptr == kv_indptr[b+1] - 1)
    valid = kv_last_page_len[b] if is_last else page_size
    # K tokens: kv_data[page, 0, :valid, kv_head, :]
    # V tokens: kv_data[page, 1, :valid, kv_head, :]
```

This is exactly the loop structure in the Triton kernel, translated into pointer
arithmetic.

---

## 5  Reading `kernel_naive.py` line by line

```python
@triton.jit
def _naive_decode_kernel(...):
    batch_idx  = tl.program_id(0)   # which sequence
    q_head_idx = tl.program_id(1)   # which query head
    kv_head_idx = q_head_idx // GROUP_SIZE   # GQA: shared KV head
```
Grid is `(batch, num_q_heads)` — one program per (sequence, query-head) pair.

```python
    offs_d = tl.arange(0, BLOCK_D)         # [BLOCK_D] index vector
    d_mask = offs_d < head_dim              # mask padding

    q_base = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
    q = q * scale                           # scale = 1/sqrt(D)
```
Load the query vector for this (batch, head).  Promote to fp32 immediately —
all arithmetic is fp32 even when the input is fp16.

```python
    start_ptr = tl.load(KV_indptr_ptr + batch_idx)
    end_ptr   = tl.load(KV_indptr_ptr + batch_idx + 1)
    last_len  = tl.load(KV_lastlen_ptr + batch_idx)
    num_pages = end_ptr - start_ptr
```
Look up this sequence's page range.  Three scalar loads from global memory.

```python
    m_i = -1e9;  l_i = 0.0;  acc = tl.zeros([BLOCK_D], dtype=tl.float32)
```
Initialise online softmax state.  `acc` is a [BLOCK_D] vector register.

```python
    for p in range(0, num_pages):
        page_idx   = tl.load(KV_indices_ptr + start_ptr + p)
        valid_toks = tl.where(p == num_pages - 1, last_len, PAGE_SIZE)
        tok_mask   = offs_ps < valid_toks
```
Dynamic loop over pages.  Load the physical page index.  Compute mask for last
page (which may be partially filled).

```python
        k_base = KV_ptr + page_idx * stride_kvp + 0 * stride_kvr + kv_head_idx * stride_kvh
        k_ptrs = k_base + offs_ps[:, None] * stride_kvs + offs_d[None, :] * stride_kvd
        k = tl.load(k_ptrs, mask=tok_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
```
2-D pointer grid: `[PAGE_SIZE, BLOCK_D]`.  This loads one page of K vectors
from HBM in one call.

```python
        qk = tl.sum(q[None, :] * k, axis=1)   # [PAGE_SIZE]
        qk = tl.where(tok_mask, qk, -1e9)
        m_block = tl.max(qk, axis=0)
        p_exp   = tl.exp(qk - m_block)
        l_block = tl.sum(p_exp, axis=0)
```
QK dot products, then block-level softmax stats.  Masked-out positions get
-1e9 so `exp(-1e9 - m) ≈ 0`.

```python
        # ... load V, compute acc_block = p_exp @ V ...
        m_new = tl.maximum(m_i, m_block)
        l_i   = exp(m_i - m_new) * l_i  + exp(m_block - m_new) * l_block
        acc   = exp(m_i - m_new) * acc  + exp(m_block - m_new) * acc_block
        m_i   = m_new
```
Online softmax merge.  Rescales both old and new state to a common baseline
`m_new`, then adds them.

```python
    acc = acc / l_i
    tl.store(o_base + offs_d * stride_od, acc, mask=d_mask)
```
Normalise and write.  Triton casts fp32 → fp16 automatically at the store site.

---

## 6  Why the naive kernel is slow at long context

Grid size = `batch × num_q_heads`.  For batch=1, 32 heads: **32 programs total**.

An A100 has 108 SMs.  With 32 programs, 76 SMs are idle — 70% of the GPU is
doing nothing.

Each active CTA must stream the entire KV sequence serially.  At 32k context
with 8 KV heads and head_dim=128, each CTA reads:

```
32768 tokens × 128 dims × 2 bytes × 2 (K+V) = 16 MB
```

This takes 8 µs at 2 TB/s.  32 CTAs running in parallel means ~8 µs total,
but 76 SMs are wasted.  If we could spread the work across all 108 SMs, we'd
get a ~3× speedup — that's exactly what split KV achieves.

---

## 7  The split KV idea

Expand the grid with a third axis: `(batch, num_q_heads, SPLIT_KV)`.

Each CTA now handles `1/SPLIT_KV` of the pages.  With SPLIT_KV=8:
- Programs: 32 × 8 = 256 → all 108 SMs busy
- Each CTA reads 1/8 of the KV → takes 1/8 of the serial time

**The catch:** you now have partial softmax statistics from SPLIT_KV
independent CTAs.  You can't just add partial outputs — the softmax
normalisation denominators are different.

**The fix:** store the merge state `(m_i, l_i, partial_O_i)` for each partition
and combine them in a second kernel pass using the multi-source online softmax
identity:

```
m_global = max_i  m_i
l_global = Σ_i  exp(m_i − m_global) · l_i
O        = (1/l_global) · Σ_i  exp(m_i − m_global) · partial_O_i
```

This is mathematically identical to running the single-pass kernel — just
rearranged so that independent partitions can run in parallel.

**Why fp32 scratch buffers?** `m_i` is a score (could be ±10), and `l_i` can
be large (sum over thousands of exp values).  Truncating these to fp16
introduces errors that grow linearly with SPLIT_KV.  At SPLIT_KV=16, the error
often exceeds the 1e-2 tolerance.  fp32 scratch costs 4× the memory of fp16
but it's a tiny buffer: `batch × H_q × SPLIT_KV × (D + 2)` ≈ a few MB.

---

## 8  The U-shaped latency curve

Why does the latency curve for split KV look like a U shape as SPLIT_KV
increases?

```
Latency
  |  \          /
  |   \        /
  |    \      /
  |     \____/
  |
  +──────────────── SPLIT_KV
      1  2  4  8  16
```

- **Left arm (small SPLIT_KV):** too few programs, SMs underutilised.  The
  kernel is occupancy-limited.  Adding more splits helps.

- **Minimum:** optimal split.  Work is spread evenly, overheads are small.

- **Right arm (large SPLIT_KV):** each partition is tiny (few pages per CTA),
  so the kernel launch overhead and the second reduction pass dominate.  The
  per-CTA work is too small to amortize the fixed costs.

The minimum shifts right as context grows (longer context → more pages →
more work per CTA → can afford more splits before the right arm rises).

---

## 9  Things Triton does NOT do (vs CUDA)

| Feature | CUDA C++ | Triton |
|---------|----------|--------|
| Explicit shared memory | Yes | Managed automatically |
| Warp-level intrinsics (`__shfl_sync`) | Yes | No |
| Persistent kernels (one kernel loop over many tiles) | Yes | No (each launch exits) |
| Arbitrary data structures | Yes | Flat tensors only |
| Mixed precision in registers | Full control | Limited |
| Tuning via template meta-programming | Yes | Via `tl.constexpr` params |

FlashInfer's performance advantage over this repo's Triton kernels is largely
because it uses CUDA C++ with explicit shared memory pipelining and persistent
kernel design — both of which are hard or impossible to express in Triton.

---

## 10  Minimal working example

Run this to confirm Triton is installed and your GPU is accessible:

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(X_ptr, Y_ptr, Z_ptr, N: tl.constexpr):
    i = tl.program_id(0)
    offs = i * N + tl.arange(0, N)
    x = tl.load(X_ptr + offs)
    y = tl.load(Y_ptr + offs)
    tl.store(Z_ptr + offs, x + y)

N = 1024
x = torch.ones(N, device='cuda', dtype=torch.float32)
y = torch.ones(N, device='cuda', dtype=torch.float32)
z = torch.empty_like(x)

add_kernel[(N // 32,)](x, y, z, N=32)  # 32 programs, each handles 32 elements
assert z.allclose(torch.full((N,), 2.0, device='cuda'))
print("Triton works!")
```

Once this passes, you are ready to read the kernels in `src/`.

---

## 11  Suggested reading order for this repo

| Step | File | What you learn |
|------|------|----------------|
| 1 | `src/layout.md` | Exact tensor shapes and strides |
| 2 | `src/layout.py` | How to build a paged KV block table |
| 3 | `src/reference.py` | The pure PyTorch fp32 ground truth |
| 4 | `src/kernel_naive.py` | One-CTA-per-head Triton kernel |
| 5 | `src/kernel_split_kv.py` | Two-pass split KV kernel |
| 6 | `bench/microbench.py` | How to time GPU kernels correctly |
| 7 | `writeup/REPORT.md` | Interpretation of results (fill in after running) |

The notebook (`notebook.ipynb`) walks through steps 1–6 interactively on Colab.

---

## Glossary

| Term | Meaning |
|------|---------|
| CTA | Cooperative Thread Array — a block of up to 1024 threads that run on one SM |
| SM | Streaming Multiprocessor — one GPU "core" |
| warp | 32 threads that execute in lockstep (SIMT) |
| HBM | High Bandwidth Memory — the VRAM chips attached to the GPU |
| SRAM | On-chip memory (registers + shared memory); orders of magnitude faster than HBM |
| GQA | Grouped Query Attention — H_q query heads share H_kv < H_q KV heads |
| MHA | Multi-Head Attention — special case of GQA with group_size = 1 |
| MQA | Multi-Query Attention — special case of GQA with H_kv = 1 |
| online softmax | Algorithm to compute softmax incrementally without storing all scores |
| constexpr | A Triton annotation meaning the value is a compile-time constant |
| roofline | A performance model showing whether a kernel is compute- or bandwidth-bound |
| arithmetic intensity | FLOPs / bytes transferred — determines where a kernel sits on the roofline |
