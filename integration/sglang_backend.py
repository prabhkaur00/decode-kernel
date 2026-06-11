"""
SGLang attention backend backed by the split-KV Triton kernel.

Implements SGLang's `AttentionBackend` interface (sglang >= 0.2.x).
The paged KV cache that SGLang passes in uses the FlashInfer NHD layout by
default, so our kernel is layout-compatible with no data copies.

Registration:
    Set SGLANG_ATTENTION_BACKEND=split_kv_triton before launching the server,
    or call `register_backend()` programmatically before importing sglang.

Part 2 requirements (see integration/README.md):
  - Ampere GPU (sm_80+)
  - sglang >= 0.2.13
  - sgl_kernel built from source for sm_80 (see README)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_split_kv import decode_attention_split_kv

# Default split — tuned empirically on A100 for 32k context
DEFAULT_SPLIT_KV = int(os.environ.get("SPLIT_KV", "8"))


# ── SGLang interface ───────────────────────────────────────────────────────
# SGLang's AttentionBackend is imported lazily so this file can be imported
# even without sglang installed (e.g., during unit testing).

def _import_sglang_base():
    try:
        from sglang.srt.layers.attention import AttentionBackend
        return AttentionBackend
    except ImportError:
        # Stub for development / testing without sglang installed
        class AttentionBackend:
            def forward(self, *a, **kw): ...
        return AttentionBackend


class SplitKVTritonBackend(_import_sglang_base()):
    """
    SGLang attention backend that routes decode steps to the split-KV Triton
    kernel and falls back to FlashInfer for prefill.

    The split size can be tuned via the SPLIT_KV environment variable or
    by passing split_kv to the constructor.
    """

    def __init__(self, split_kv: int = DEFAULT_SPLIT_KV):
        self.split_kv = split_kv
        self._scratch: dict = {}  # reused scratch buffers
        self._fi_wrapper = None   # FlashInfer wrapper for prefill

    # ── Prefill (standard FlashInfer) ──────────────────────────────────

    def _prefill_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Delegate prefill to FlashInfer's optimised implementation."""
        try:
            import flashinfer
            # Use FlashInfer's varlen prefill API
            return flashinfer.single_prefill_with_kv_cache(q, k, v, **kwargs)
        except Exception:
            # Fallback: standard sdpa
            return torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            ).squeeze(0)

    # ── Decode (split-KV Triton) ────────────────────────────────────────

    def forward(
        self,
        q: torch.Tensor,            # (batch, num_q_heads, head_dim) for decode
        kv_data: torch.Tensor,      # (num_pages, 2, page_size, num_kv_heads, head_dim)
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        is_prefill: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if is_prefill:
            return self._prefill_forward(q, kv_data, kv_indptr, **kwargs)

        return decode_attention_split_kv(
            q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
            split_kv=self.split_kv,
            _scratch=self._scratch,
        )

    # ── SGLang hook ────────────────────────────────────────────────────

    @classmethod
    def name(cls) -> str:
        return "split_kv_triton"


# ── Registration ───────────────────────────────────────────────────────────

def register_backend():
    """
    Register SplitKVTritonBackend with SGLang's backend registry.
    Call before importing sglang.server or launching the server subprocess.
    """
    try:
        from sglang.srt.layers.attention import register_attention_backend
        register_attention_backend("split_kv_triton", SplitKVTritonBackend)
        print("[split_kv_triton] Backend registered with SGLang.")
    except ImportError:
        print("[split_kv_triton] WARNING: sglang not installed; "
              "backend registration skipped.")
    except AttributeError:
        # Older sglang versions use a different registry mechanism
        try:
            import sglang.srt.layers.attention as attn_mod
            attn_mod._BACKEND_REGISTRY["split_kv_triton"] = SplitKVTritonBackend
            print("[split_kv_triton] Backend injected into registry (legacy path).")
        except Exception as e:
            print(f"[split_kv_triton] Registration failed: {e}")


# ── Entry point for environment-variable–based auto-registration ───────────

if os.environ.get("SGLANG_ATTENTION_BACKEND") == "split_kv_triton":
    register_backend()
