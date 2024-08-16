from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup( name='fusedconv',
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

setup(
    name='fftconv',
    ext_modules=[
        CUDAExtension(
            name='fftconv',
            sources=['src/fftConv.cu', 'bindings/fftConvWrapper.cpp'],
            include_dirs=['include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='gemmconv',
    ext_modules=[
        CUDAExtension(
            name='gemmconv',
            sources=['src/gemmConv.cu', 'bindings/gemmConvWrapper.cpp'],
            include_dirs=['include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

