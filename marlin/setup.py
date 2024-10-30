
from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='marlin',
    version='0.1.1',
    author='Zelei Shao',
    author_email='zelei2@illinois.edu',
    description='Optimized FP16xINT3 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['marlin'],
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_cuda', ['marlin/marlin_cuda.cpp','marlin/marlin_cuda_kernel.cu', 'marlin/marlin_3bit_cuda_kernel.cu','marlin/marlin_3bit_cuda_kernel_faster.cu','marlin/marlin_3bit_with_zero_cuda_kernel.cu','marlin/marlin_3bit_256_64_kernel.cu','marlin/marlin_3bit_256_64_with_zero_kernel.cu','marlin/marlin_3bit_64_256_with_zero_kernel.cu']
    )],
    extra_compile_args={
        'gcc': ['-g', '-O0'],
        'nvcc': ['-G', '-g', '-O0', '-lineinfo', '-arch=sm_80']
    },
    extra_link_args=['-lcudart'],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
