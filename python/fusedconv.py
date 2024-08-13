from torch import nn, Tensor
from torch.autograd import Function
from dataclasses import dataclass
from .typing import singleton, duplet, triplet
from .padding import *

class fusedConv1d(Function):
    ## All of them in the complex plane
    def forward(self, X: Tensor, W: Tensor) -> Tensor:
        return ...
    def backward(self, X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return ...

class fusedConv2d(Function):
    ## All of them in the complex plane
    def forward(self, X: Tensor, W: Tensor) -> Tensor:
        return ...
    def backward(self, X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return ...

class fusedConv3d(Function):
    ## All of them in the complex plane
    def forward(self, X: Tensor, W: Tensor) -> Tensor:
        return ...
    def backward(self, X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return ...

class FusedConvolution(nn.Module):
    def __init__(self, args: FusedArgs, convMethod: Function) -> None:
        super().__init__()
        self.parameters = args
        self.convMethod = convMethod
        self.init_weights() ## Initialize weights for all of them

    def init_weights(self) -> None:

    def forward(self, input: Tensor) -> Tensor:
        return self.convMethod(
            input,
            self.weight,
            self.bias,
            self.kernel_params,
            self.stride_params,
            self.padding_params,
        )

class FusedConv1D(FusedConvolution):
    def __init__(self, args: FusedConv1dArgs) -> None:
        super().__init__(args, fusedConv1d)

class FusedConv2D(FusedConvolution):
    def __init__(self, args: FusedConv2dArgs) -> None:
        super().__init__(args, fusedConv2d)

class FusedConv3D(FusedConvolution):
    def __init__(self, args: FusedConv3dArgs) -> None:
        super().__init__(args, fusedConv3d)

