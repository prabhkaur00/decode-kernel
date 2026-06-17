"""
JIT-compiles the CUDA extensions using torch.utils.cpp_extension.load.

Three entry points:
  get_naive_ext()     — naive_kernel.cu only
  get_split_kv_ext()  — split_kv_kernel.cu only (links nvToolsExt for NVTX3)
  get_cuda_ext()      — original combined attention_ext.cu (backward compat)

First call triggers nvcc compilation (~30–60 s); subsequent calls in the
same process return the cached module instantly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_naive_ext    = None
_split_kv_ext = None
_combined_ext = None


def _sm_flag() -> str:
    import torch
    props = torch.cuda.get_device_properties(0)
    return f"-gencode=arch=compute_{props.major}{props.minor},code=sm_{props.major}{props.minor}"


def get_naive_ext(verbose: bool = False):
    """Returns the compiled naive-kernel extension (singleton)."""
    global _naive_ext
    if _naive_ext is not None:
        return _naive_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _naive_ext = load(
        name="naive_attention_cuda",
        sources=[str(src_dir / "naive_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _naive_ext


def get_split_kv_ext(verbose: bool = False):
    """Returns the compiled split-KV-kernel extension (singleton).

    Links nvToolsExt so the C++ NVTX3 markers inside split_kv_kernel.cu
    are active. nsys will show cuda_split_kv_partition and
    cuda_split_kv_reduce as separate CPU-side ranges.
    """
    global _split_kv_ext
    if _split_kv_ext is not None:
        return _split_kv_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_ext = load(
        name="split_kv_attention_cuda",
        sources=[str(src_dir / "split_kv_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_ext


def get_cuda_ext(verbose: bool = False):
    """Returns the original combined extension (backward compat)."""
    global _combined_ext
    if _combined_ext is not None:
        return _combined_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent
    props = torch.cuda.get_device_properties(0)
    sm = f"{props.major}{props.minor}"

    _combined_ext = load(
        name="attention_cuda",
        sources=[str(src_dir / "attention_ext.cu")],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
            "-std=c++17",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _combined_ext


if __name__ == "__main__":
    print("Building naive kernel …")
    ext = get_naive_ext(verbose=True)
    print(f"naive: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV kernel …")
    ext = get_split_kv_ext(verbose=True)
    print(f"split_kv: {[f for f in dir(ext) if not f.startswith('_')]}")
