from torch import nn, Tensor
from fftconv import *
import torch

class FFTConvolution(nn.Module):
    def __init__(self, kernel_size, stride, padding, convMethod) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.convMethod = convMethod
        self.init_weights() ## Initialize weights for all of them
    def init_weights(self) -> None:
        # TODO
    def forward(self, x: Tensor) -> Tensor:
        return self.convMethod(x, self.weight, self.bias) # TODO

class Conv1d(DefaultConvolution):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__(kernel_size, stride, padding, FFTConv1d)

class Conv2d(DefaultConvolution):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__(kernel_size, stride, padding, FFTConv2d)

class Conv3d(DefaultConvolution):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__(kernel_size, stride, padding, FFTConv3d)
