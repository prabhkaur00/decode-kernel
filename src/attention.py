"""
Unified decode attention dispatch.

Selects between the Triton and CUDA kernel backends.

Backend selection (precedence: call-site arg > set_backend() > env var):
    export ATTN_BACKEND=cuda    # or "triton" (default)
    from attention import set_backend
    set_backend("cuda")
    out = decode_attention(q, kv_data, ..., split_kv=4)

Or per-call:
    out = decode_attention(q, kv_data, ..., backend="cuda")

Both backends implement exactly the same algorithm and must produce outputs
within the tolerance defined in the correctness gates (max_abs_err < 1e-2
vs the fp32 reference).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# ── Global backend setting ────────────────────────────────────────────────

_BACKEND: str = os.environ.get("ATTN_BACKEND", "triton").lower()


def get_backend() -> str:
    """Returns the current global backend ('triton' or 'cuda')."""
    return _BACKEND


def set_backend(name: str) -> None:
    """
    Sets the global backend for all subsequent decode_attention calls.

    Args:
        name: 'triton' or 'cuda'
    """
    global _BACKEND
    name = name.lower()
    if name not in ("triton", "cuda"):
        raise ValueError(f"backend must be 'triton' or 'cuda', got {name!r}")
    _BACKEND = name
    print(f"[attention] Backend set to '{name}'")


# ── Main API ──────────────────────────────────────────────────────────────

def decode_attention(
    q: torch.Tensor,                      # (B, H_q, D)
    kv_data: torch.Tensor,                # (N_p, 2, P, H_kv, D)
    kv_indptr: torch.Tensor,              # (B+1,) int32
    kv_indices: torch.Tensor,             # (total_pages,) int32
    kv_last_page_len: torch.Tensor,       # (B,) int32
    split_kv: int = 1,
    backend: Optional[str] = None,
    _scratch: Optional[dict] = None,      # Triton only: pre-allocated scratch
) -> torch.Tensor:
    """
    Decode attention over a paged KV cache.

    Args:
        q                : query tensor (batch, num_q_heads, head_dim), fp16
        kv_data          : paged KV cache in FlashInfer NHD layout, fp16
        kv_indptr        : CSR row pointer into kv_indices
        kv_indices       : physical page indices per slot
        kv_last_page_len : valid tokens in the last page of each sequence
        split_kv         : KV partitions per (batch, q_head).
                           1 → naive single-CTA mode.
                           >1 → split-KV two-pass mode.
        backend          : 'triton' or 'cuda'.  Defaults to global backend.
        _scratch         : optional dict for Triton scratch buffer reuse.

    Returns:
        output tensor (batch, num_q_heads, head_dim), same dtype as q.
    """
    effective_backend = (backend or _BACKEND).lower()

    if effective_backend == "triton":
        return _triton(q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                       split_kv, _scratch)
    elif effective_backend == "cuda":
        return _cuda(q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                     split_kv)
    else:
        raise ValueError(
            f"Unknown backend {effective_backend!r}.  Must be 'triton' or 'cuda'."
        )


# ── Backend implementations ───────────────────────────────────────────────

def _triton(q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv, _scratch):
    if split_kv == 1:
        from kernel_naive import decode_attention_naive
        return decode_attention_naive(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
    else:
        from kernel_split_kv import decode_attention_split_kv
        return decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv=split_kv, _scratch=_scratch,
        )


def _cuda(q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv):
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location(
        "_cuda_build",
        Path(__file__).resolve().parent / "cuda" / "build.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ext = _mod.get_cuda_ext()
    if split_kv == 1:
        return ext.decode_attention_naive(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len
        )
    else:
        return ext.decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len, split_kv
        )


# ── Convenience: compare both backends ───────────────────────────────────

def compare_backends(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    split_kv: int = 4,
) -> dict:
    """
    Runs both backends on the same inputs and returns a comparison dict.
    Useful for quick sanity checks on new hardware.
    """
    from reference import reference_attention_paged

    ref = reference_attention_paged(
        q, kv_data, kv_indptr, kv_indices, kv_last_page_len
    ).to(q.device)

    out_triton = _triton(q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                         split_kv, None)
    out_cuda   = _cuda(q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
                       split_kv)

    def stats(name, out):
        err = (out.float() - ref.float()).abs()
        return {
            f"{name}_max_err":  err.max().item(),
            f"{name}_mean_err": err.mean().item(),
        }

    result = {"split_kv": split_kv}
    result.update(stats("triton", out_triton))
    result.update(stats("cuda",   out_cuda))
    cross_err = (out_triton.float() - out_cuda.float()).abs()
    result["triton_vs_cuda_max_err"] = cross_err.max().item()
    return result
