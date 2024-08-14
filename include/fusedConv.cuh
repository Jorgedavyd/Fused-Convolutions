#pragma once
#ifndef FUSEDCONV_CUH
#define FUSEDCONV_CUH

#include <torch/extension.h>

torch::Tensor fwd_1D (torch::Tensor input, torch::Tensor weight);
torch::Tensor bwd_1D(torch::Tensor input, torch::Tensor weight);

torch::Tensor fwd_2D (torch::Tensor input, torch::Tensor weight);
torch::Tensor bwd_2D(torch::Tensor input, torch::Tensor weight);

torch::Tensor fwd_3D (torch::Tensor input, torch::Tensor weight);
torch::Tensor bwd_3D(torch::Tensor input, torch::Tensor weight);


#endif //FUSEDCONV_CUH
