from fusedconv import FusedCovn1D, FusedCovn2D, FusedCovn3D
from typing import Callable, Tuple
from ..python.typing import FusedArgs
from fusedconv import *
import yaml
from torch import nn, Tensor
from torch.autograd.profiler import profiler
import torch
import time

def createArgs(data: Dict) -> FusedArgs:
    return data

class testConfig:
    path: Callable[[int], str] = lambda x: f'./tests/config{x}.yml'
    @staticmethod
    def __call__(idx: int) -> FusedArgs | None:
        config_file: str = testConfig.path(idx)
        with open(config_file) as stream:
            try:
                data = yaml.safe_load(stream)
                return createArgs(data)
            except yaml.YAMLError as exc:
                print(exc)

VALID_FUSION = [FusedConv1D, FusedConv2D, FusedConv3D]
VALID_FFT = [FFTConv1D, FFTConv2D, FFTConv3D]
VALID_GEMM = [DefaultConv1D, DefaultConv2D, DefaultConv3D]

def measure_forward_time(model: nn.Module, input_tensor: Tensor, iterations: int = 100) -> None:
    model.eval()

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f'The forward time for {model.__class__.__name__} is: {avg_time}')

def measure_backward_time(model: nn.Module, input_tensor: Tensor, iterations: int = 100) -> None:
    model.train()

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    start_time = time.time()
    for _ in range(iterations):
        model.zero_grad()
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average Backward Pass Time: {avg_time}")

def run_sample(config: FusedArgs, algo: nn.Module) -> Tuple[float,float]:
    forward_time = measure_forward_time(algo(**config.init_params))
    backward_time = measure_backward_time(algo(**config.init_params))
    return forward_time, backward_time

def make_attempt(config: FusedArgs) -> Tuple[Tuple[float, ...],...]:
    fused_algo = VALID_FUSION[config.d]
    fft_algo = VALID_FFT[config.d]
    gemm_algo = VALID_GEMM[config.d]

    gemm_time = run_sample(gemm_algo)
    fft_time = run_sample(fft_algo)
    fused_time = run_sample(fused_algo)

    return gemm_time, fft_time, fused_time

def n_sample(config_idx: int) -> None:
    config = testConfig.__call__(config_idx)
    default_time, fft_time, fusion_time = make_attempt(config)
    print('CuDNN GEMM convolution(forward, backward):', default_time)
    print('CuDNN FFT convolution(forward, backward):',fft_time )
    print('Fused convolution(forward, backward):', fusion_time)



