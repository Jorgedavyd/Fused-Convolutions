from typing import Tuple, List, Union

## 1D, 2D, 3D CNN Kernel sizes
singleton = Union[int, Tuple[int]]
duplet = Tuple[int, int]
triplet = Tuple[int, int, int]

class FusedArgs:
    in_channels: int
    out_channels: int

class FusedConvArgs(FusedArgs):
    kernel_size: Union[singleton, duplet, triplet]
    padding: Union[singleton, duplet, triplet]
    stride: Union[singleton, duplet, triplet]
    parameters: List[Tuple[int, int, Union[singleton, duplet, triplet], Union[singleton, duplet, triplet], Union[singleton, duplet, triplet]]]

class FusedConv1DArgs(FusedConvArgs):
    kernel_size = singleton
    padding = singleton
    stride = singleton

class FusedConv2DArgs(FusedConvArgs):
    kernel_size = duplet
    padding = duplet
    stride = duplet

class FusedConv3DArgs(FusedConvArgs):
    kernel_size = triplet
    padding = triplet
    stride = triplet

