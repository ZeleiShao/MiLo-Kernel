
from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='marlin',
    version='0.1.1',
    author='Elias Frantar',
    author_email='elias.frantar@ist.ac.at',
    description='Highly optimized FP16xINT4 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['marlin'],

    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_cuda', ['marlin/marlin_cuda.cpp','marlin/marlin_cuda_kernel.cu', 'marlin/marlin_3bit_cuda_kernel.cu']
    )],
    extra_compile_args={
        'gcc': ['-g', '-O0'],
        'nvcc': ['-G', '-g', '-O0', '-lineinfo', '-arch=sm_80']
    },
    extra_link_args=['-lcudart'],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)

'''
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if self.compiler.compiler_type == 'unix':
                ext.extra_compile_args = {
            'gcc': ['-g', '-O0'],
            'nvcc': ['-G', '-g', '-O0', '-lineinfo', '-arch=sm_80']
            }
            elif self.compiler.compiler_type == 'msvc':
                ext.extra_compile_args = {
                'cl': ['/Zi', '/Od'],
                'nvcc': ['-G', '-g', '-O0', '-lineinfo', '-arch=sm_80']
            }
        build_ext.build_extensions(self)

ext_modules = [
Extension(
cpp_extension.CUDAExtension(
        'marlin_cuda', ['marlin/marlin_cuda.cpp','marlin/marlin_cuda_kernel.cu', 'marlin/marlin_3bit_cuda_kernel.cu']),
include_dirs=[
# Path to pybind11 headers
get_pybind_include(),
],
language='c++',
extra_compile_args={
'gcc': ['-g', '-O0'],
'nvcc': ['-G', '-g', '-O0', '-lineinfo', '-arch=sm_80']
},
extra_link_args=['-lcudart']
),
]

setup(
name='marlin',
version='0.1',
author='Amy',
author_email='your.email@example.com',
description='A Python package with C++ and CUDA extension',
ext_modules=ext_modules,
cmdclass={'build_ext': BuildExt},
zip_safe=False,
)
'''