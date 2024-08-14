from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fusedconv',
    ext_modules=[
        CUDAExtension(
            name='fusedconv',
            sources=['fusedConv.cu'],
            include_dirs=['include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

