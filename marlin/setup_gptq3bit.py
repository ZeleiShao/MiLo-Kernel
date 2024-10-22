from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['marlin/gptq_3bit.cpp', 'marlin/gptq_3bitkernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)