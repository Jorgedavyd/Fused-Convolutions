from torch import nn, Tensor
from torch.autograd import Function
from .typing import *
from fusedconv import *

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
        super().__init__(args, FusedConv1D)

class FusedConv2D(FusedConvolution):
    def __init__(self, args: FusedConv2dArgs) -> None:
        super().__init__(args, FusedConv2D)

class FusedConv3D(FusedConvolution):
    def __init__(self, args: FusedConv3dArgs) -> None:
        super().__init__(args, FusedConv3D)

