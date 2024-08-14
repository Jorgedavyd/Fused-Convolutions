from torch import nn, Tensor
from torch.autograd import Function
from .typing import *
import fusedconv

class fusedConv1d(Function):
    ## All of them in the complex plane
    @staticmethod
    def forward(X: Tensor, W: Tensor) -> Tensor:
        return fusedconv.fwd_1D(X, W)
    @staticmethod
    def backward(X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return fusedconv.bwd_1D(X, W)

class fusedConv2d(Function):
    ## All of them in the complex plane
    @staticmethod
    def forward(X: Tensor, W: Tensor) -> Tensor:
        return fusedconv.fwd_2D(X, W)
    @staticmethod
    def backward(X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return fusedconv.bwd_2D(X, W)

class fusedConv3d(Function):
    ## All of them in the complex plane
    @staticmethod
    def forward(X: Tensor, W: Tensor) -> Tensor:
        return fusedconv.fwd_3D(X, W)
    @staticmethod
    def backward(X: Tensor, W: Tensor, out: Tensor) -> Tensor:
        return fusedconv.bwd_3D(X, W)

class FusedConvolution(nn.Module):
    def __init__(self, args: FusedArgs, convMethod: Function) -> None:
        super().__init__()
        self.parameters = args
        self.convMethod = convMethod
        self.init_weights() ## Initialize weights for all of them

    def init_weights(self) -> None:

    def forward(self, input: Tensor) -> Tensor:
        return self.convMethod.apply(
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

