from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='deGE_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'deGE_cuda', ['marlin/dequant_GEMM.cpp', 'marlin/dequant_GEMM_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)