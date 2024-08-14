#include <torch/extensions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "fusedConv.cuh"

template <typename scalar_t>
__global__ void fwd1dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
};

template <typename scalar_t>
__global__ void fwd2dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
};

template <typename scalar_t>
__global__ void fwd3dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
};

template <typename scalar_t>
__global__ void bwd1dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
    };

template <typename scalar_t>
__global__ void bwd2dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
};

template <typename scalar_t>
__global__ void bwd3dKernel (
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
) {
};

template <typename scalar_t>
std::vector<torch::Tensor> fwd_1D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        fwd1dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

template <typename scalar_t>
std::vector<torch::Tensor> bwd_1D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        bwd1dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

template <typename scalar_t>
std::vector<torch::Tensor> fwd_2D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        fwd2dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

template <typename scalar_t>
std::vector<torch::Tensor> bwd_2D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        bwd2dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

template <typename scalar_t>
std::vector<torch::Tensor> fwd_3D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        fwd3dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

template <typename scalar_t>
std::vector<torch::Tensor> bwd_3D (
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& bias,
) {
        bwd3dKernel<scalar_t><<<BLOCKS, THREADS>>>
};

