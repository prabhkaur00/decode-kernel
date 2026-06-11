# KV Layout and Tensor Boundary Specification

This document specifies the exact shape, stride, dtype, and semantics of every
tensor that crosses the kernel boundary.  It is derived from FlashInfer source
(`flashinfer/page.py`, `flashinfer/decode.py`) and the FlashInfer documentation
for the `BatchDecodeWithPagedKVCacheWrapper` API.

---

## Coordinate system

All indices are zero-based.  The abbreviations used throughout:

| Symbol | Meaning |
|--------|---------|
| B      | batch size |
| H_q    | number of query heads |
| H_kv   | number of KV heads (H_q / H_kv = GQA group size G) |
| D      | head dimension |
| P      | page size (tokens per page, power of 2, default 16) |
| N_p    | total pages allocated across all sequences |
| S      | context length (tokens) per sequence (uniform in our sweeps) |
| K_p    | pages per sequence = ⌈S / P⌉ |

---

## Tensor: Q (query)

```
shape  : (B, H_q, D)
strides: (H_q * D,  D,  1)
dtype  : fp16 (or bf16)
device : CUDA
```

Decode attention produces exactly one output token per batch entry, so the
sequence dimension is 1 and is contracted away.

---

## Tensor: kv_data (paged KV cache)

```
shape  : (N_p, 2, P, H_kv, D)
strides: (2*P*H_kv*D,  P*H_kv*D,  H_kv*D,  D,  1)   [NHD layout, contiguous]
dtype  : fp16 (or bf16, must match Q)
device : CUDA
```

Dimension semantics:

| Dim | Size | Meaning |
|-----|------|---------|
| 0   | N_p  | Physical page index |
| 1   | 2    | 0 = Key, 1 = Value |
| 2   | P    | Position within page (0 = oldest token in page) |
| 3   | H_kv | KV head index |
| 4   | D    | Head dimension |

`N_p` is the number of pages actually needed: `B * ⌈S / P⌉` in the uniform
sweep.  Pages are not shared across sequences (no aliasing in our harness).

To access `K[page=k, pos=p, head=h, dim=d]`:

```
ptr = kv_data_ptr + k*(2*P*H_kv*D) + 0*(P*H_kv*D) + p*(H_kv*D) + h*D + d
```

And equivalently for V with the `0` replaced by `1`.

---

## Tensor: kv_indptr

```
shape  : (B + 1,)
strides: (1,)
dtype  : int32
device : CUDA
```

CSR-style row pointer into `kv_indices`.  The physical pages of sequence `b`
are `kv_indices[kv_indptr[b] : kv_indptr[b+1]]`.

For a uniform context sweep:

```
kv_indptr[b] = b * K_p
kv_indptr[B] = B * K_p
```

---

## Tensor: kv_indices

```
shape  : (B * K_p,)   [uniform context; ragged otherwise]
strides: (1,)
dtype  : int32
device : CUDA
```

Maps logical page slot to physical page index into `kv_data`.  For the uniform
sweep without aliasing: `kv_indices[i] = i`.

---

## Tensor: kv_last_page_len

```
shape  : (B,)
strides: (1,)
dtype  : int32
device : CUDA
```

Number of valid tokens in the final page of each sequence.  If `S` is an exact
multiple of `P`, this equals `P`.  All prior pages are fully packed (P tokens).

```
kv_last_page_len[b] = S - (K_p - 1) * P
                     = S mod P   (if S mod P != 0)
                     = P         (if S mod P == 0)
```

---

## Tensor: output O

```
shape  : (B, H_q, D)
strides: (H_q * D,  D,  1)
dtype  : fp16 (or bf16, same as Q)
device : CUDA
```

Result of the attention computation.  GQA head broadcast is performed inside the
kernel; the output has H_q channels, not H_kv.

---

## Scratch buffers (split KV kernel only)

These tensors are allocated in fp32 regardless of the input dtype.  Storing the
merge state in fp16 is the most common source of numerical divergence when the
number of splits is large.

| Tensor      | Shape                    | Dtype  | Semantics |
|-------------|--------------------------|--------|-----------|
| partial_O   | (B, H_q, SPLIT_KV, D)   | fp32   | Unnormalised partial attention output per partition |
| partial_m   | (B, H_q, SPLIT_KV)      | fp32   | Row-max of QK scores within partition |
| partial_l   | (B, H_q, SPLIT_KV)      | fp32   | Sum of exp(score − m) within partition |

The unnormalised partial output for partition `s` is:

```
partial_O[b, h, s, :] = Σ_{j ∈ partition s}  exp(QK_j − m_s) · V_j
```

where `m_s = max_{j ∈ partition s} QK_j`.  The final output after the reduction
pass is:

```
m_global = max_s  m_s
l_global = Σ_s  exp(m_s − m_global) · partial_l[b, h, s]
O[b, h, :] = (1 / l_global) · Σ_s  exp(m_s − m_global) · partial_O[b, h, s, :]
```

---

## Invariants checked at runtime

1. `kv_data.dim() == 5` and `kv_data.shape[1] == 2`
2. `kv_indptr[-1] == kv_indices.shape[0]`
3. `(kv_last_page_len > 0).all()` and `(kv_last_page_len <= page_size).all()`
4. `q.shape[0] == kv_indptr.shape[0] - 1`
5. `q.shape[1] % kv_data.shape[3] == 0`  (H_q divisible by H_kv)
6. `q.dtype == kv_data.dtype`
