from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fusedconv',
    ext_modules=[
        CUDAExtension(
            name='fusedconv',
            sources=['src/fusedConv.cu', 'bindings/fusedConvWrapper.cpp'],
            include_dirs=['include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

