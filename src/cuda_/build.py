"""
JIT-compiles the CUDA extensions using torch.utils.cpp_extension.load.

Entry points:
  get_naive_ext()             — naive_kernel.cu only
  get_split_kv_ext()          — split_kv_kernelv1.cu (v1, q-head-centric grid)
  get_split_kv_v2_ext()       — split_kv_kernelv2.cu (v2, kv-head-centric grid)
  get_split_kv_v2_5_ext()     — split_kv_kernelv2.5.cu (v2 + K/V both resident, QK scores in shared mem)
  get_split_kv_pipelined_ext() — split_kv_kernelv3.cu (cp.async double-buffered)
  get_split_kv_v3_5_ext()     — split_kv_kernelv3_5.cu (v3 pipelining + v2 group fusion)
  get_split_kv_v4_ext()       — split_kv_kernelv4.cu (pipelined + reduced register pressure)
  get_cuda_ext()              — original combined attention_ext.cu (backward compat)

First call triggers nvcc compilation (~30–60 s); subsequent calls in the
same process return the cached module instantly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_naive_ext              = None
_split_kv_ext           = None
_split_kv_v2_ext        = None
_split_kv_v2_5_ext      = None
_split_kv_pipelined_ext = None
_split_kv_v3_5_ext      = None
_split_kv_v4_ext        = None
_combined_ext           = None


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

    Links nvToolsExt so the C++ NVTX3 markers inside split_kv_kernelv1.cu
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
        sources=[str(src_dir / "split_kv_kernelv1.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_ext


def get_split_kv_v2_ext(verbose: bool = False):
    """Returns the compiled split-KV v2 kernel extension (KV-head-centric grid)."""
    global _split_kv_v2_ext
    if _split_kv_v2_ext is not None:
        return _split_kv_v2_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_v2_ext = load(
        name="split_kv_v2_attention_cuda",
        sources=[str(src_dir / "split_kv_kernelv2.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_v2_ext


def get_split_kv_v2_5_ext(verbose: bool = False):
    """Returns the compiled split-KV v2.5 kernel extension (singleton).

    v2.5 = v2's KV-head-centric grid/reuse, restructured so K and V tiles
    are both resident in shared memory at once and each q_head's
    score -> softmax -> accumulate is fused before moving to the next
    q_head. QK scores for the in-flight q_head live in a small shared
    buffer instead of a per-thread qk_scores[GROUP_SIZE][PAGE_SIZE]
    register array, trading a little shared memory for reduced register
    pressure.
    """
    global _split_kv_v2_5_ext
    if _split_kv_v2_5_ext is not None:
        return _split_kv_v2_5_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_v2_5_ext = load(
        name="split_kv_v2_5_attention_cuda",
        sources=[str(src_dir / "split_kv_kernelv2.5.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_v2_5_ext


def get_split_kv_pipelined_ext(verbose: bool = False):
    """Returns the compiled split-KV pipelined-kernel extension (singleton).

    Uses cp.async (LDGSTS), which requires SM 80+ (Ampere or newer).
    """
    global _split_kv_pipelined_ext
    if _split_kv_pipelined_ext is not None:
        return _split_kv_pipelined_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        raise RuntimeError(
            f"split_kv_kernelv3.cu requires SM 80+ (Ampere+) for "
            f"cp.async; detected sm_{props.major}{props.minor}."
        )

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_pipelined_ext = load(
        name="split_kv_pipelined_attention_cuda",
        sources=[str(src_dir / "split_kv_kernelv3.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_pipelined_ext


def get_split_kv_v3_5_ext(verbose: bool = False):
    """Returns the compiled split-KV v3.5 kernel extension (singleton).

    v3.5 = v3's software-pipelined (cp.async double-buffered) K/V loads
    combined with v2's KV-head-centric group fusion: one CTA per kv_head,
    looping over GROUP_SIZE q_heads and reusing the pipelined K/V stage
    buffers across all of them. Requires SM 80+ (Ampere+) for cp.async,
    same as the pipelined kernel.
    """
    global _split_kv_v3_5_ext
    if _split_kv_v3_5_ext is not None:
        return _split_kv_v3_5_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        raise RuntimeError(
            f"split_kv_kernelv3_5.cu requires SM 80+ (Ampere+) for "
            f"cp.async; detected sm_{props.major}{props.minor}."
        )

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_v3_5_ext = load(
        name="split_kv_v3_5_attention_cuda",
        sources=[str(src_dir / "split_kv_kernelv3_5.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_v3_5_ext


def get_split_kv_v4_ext(verbose: bool = False):
    """Returns the compiled split-KV v4 kernel extension (singleton).

    v4 = pipelined (cp.async double-buffered) baseline with the per-thread
    parts[PAGE_SIZE]/exp_weights[PAGE_SIZE] register arrays removed to ease
    register pressure. Requires SM 80+ (Ampere+) for cp.async, same as the
    pipelined kernel.
    """
    global _split_kv_v4_ext
    if _split_kv_v4_ext is not None:
        return _split_kv_v4_ext

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires a GPU.")

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        raise RuntimeError(
            f"split_kv_kernelv4.cu requires SM 80+ (Ampere+) for "
            f"cp.async; detected sm_{props.major}{props.minor}."
        )

    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).resolve().parent

    _split_kv_v4_ext = load(
        name="split_kv_v4_attention_cuda",
        sources=[str(src_dir / "split_kv_kernelv4.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", _sm_flag(), "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lnvToolsExt"],
        verbose=verbose,
    )
    return _split_kv_v4_ext


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

    print("Building split-KV v1 kernel …")
    ext = get_split_kv_ext(verbose=True)
    print(f"split_kv_v1: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV v2 kernel …")
    ext = get_split_kv_v2_ext(verbose=True)
    print(f"split_kv_v2: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV v2.5 kernel …")
    ext = get_split_kv_v2_5_ext(verbose=True)
    print(f"split_kv_v2_5: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV pipelined kernel …")
    ext = get_split_kv_pipelined_ext(verbose=True)
    print(f"split_kv_pipelined: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV v3.5 kernel …")
    ext = get_split_kv_v3_5_ext(verbose=True)
    print(f"split_kv_v3_5: {[f for f in dir(ext) if not f.startswith('_')]}")

    print("Building split-KV v4 kernel …")
    ext = get_split_kv_v4_ext(verbose=True)
    print(f"split_kv_v4: {[f for f in dir(ext) if not f.startswith('_')]}")
