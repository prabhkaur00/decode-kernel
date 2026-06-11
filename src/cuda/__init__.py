# Makes src/cuda/ a Python package so `from cuda.build import get_cuda_ext` works.
from .build import get_cuda_ext

__all__ = ["get_cuda_ext"]
