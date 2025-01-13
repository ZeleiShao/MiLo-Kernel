from glob import glob
from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='milo',
    version='0.1.1',
    author='Zelei Shao',
    author_email='zelei2@illinois.edu',
    description='Optimized FP16xINT3 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['milo_kernel'],
    ext_modules=[cpp_extension.CUDAExtension(
    name = 'milo_cuda', 
    sources=[
            'milo_kernel/milo_cuda.cpp',
            'milo_kernel/marlin_cuda_kernel.cu',
            'milo_kernel/milo_3bit_cuda_kernel.cu',
            'milo_kernel/milo_3bit_cuda_kernel_faster.cu',
            'milo_kernel/milo_3bit_with_zero_cuda_kernel.cu',
            'milo_kernel/milo_3bit_256_64_kernel.cu',
            'milo_kernel/milo_3bit_256_64_with_zero_kernel.cu',
            'milo_kernel/milo_3bit_64_256_with_zero_kernel.cu'
        ],
    extra_compile_args={
    'gcc': ['-g', '-O0'],
    'nvcc': ['-g', '-O0', '-lineinfo', '-arch=sm_80']  #-G 和 -lineinfo不能一起使用； -G优先级更高
    },
    extra_link_args=['-lcudart'])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)

