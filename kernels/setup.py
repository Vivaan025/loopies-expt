"""
Build script for custom CUDA kernels.

Usage:
    cd kernels/
    python setup.py install

Or for development (rebuild on import):
    pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="loop_transformer_kernels",
    ext_modules=[
        CUDAExtension(
            name="loop_transformer_kernels",
            sources=[
                "binding.cpp",
                "layernorm_kernel.cu",
                "fused_ln_qkv.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
