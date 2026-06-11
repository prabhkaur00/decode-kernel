"""
JIT-compiles the CUDA extension using torch.utils.cpp_extension.load.

The first call triggers nvcc compilation (~30–60 s).
Subsequent calls in the same process return the cached module instantly.
The compiled binary is cached on disk at ~/.cache/torch_extensions/.

Usage:
    from cuda.build import get_cuda_ext
    ext = get_cuda_ext()
    out = ext.decode_attention_naive(q, kv_data, ...)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ext = None   # module-level singleton

def get_cuda_ext(verbose: bool = False):
    """
    Returns the compiled CUDA extension module (singleton).

    Args:
        verbose: if True, print nvcc output during compilation.
    """
    global _ext
    if _ext is not None:
        return _ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA extension requires a GPU.  "
            "Use ATTN_BACKEND=triton as a CPU fallback."
        )

    from torch.utils.cpp_extension import load

    src_dir = Path(__file__).resolve().parent

    props = torch.cuda.get_device_properties(0)
    sm = f"{props.major}{props.minor}"   # e.g. "80" for A100

    _ext = load(
        name="attention_cuda",
        sources=[str(src_dir / "attention_ext.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
            "-std=c++17",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _ext


if __name__ == "__main__":
    print("Building CUDA extension …")
    ext = get_cuda_ext(verbose=True)
    print(f"Loaded: {ext}")
    print("Functions:", [f for f in dir(ext) if not f.startswith("_")])
