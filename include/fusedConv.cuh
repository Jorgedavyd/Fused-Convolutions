#pragma once
#include <vector>
#ifndef FUSEDCONV_CUH
#define FUSEDCONV_CUH

#include <torch/extension.h>

class Conv1D : torch::autograd::Function<Conv1D> {
public:
    static torch::Tensor forward;
    static torch::autograd::variable_list backward;
};

class Conv2D : torch::autograd::Function<Conv2D> {
public:
    static torch::Tensor forward;
    static torch::autograd::variable_list backward;
};

class Conv3D : torch::autograd::Function<Conv3D> {
public:
    static torch::Tensor forward;
    static torch::autograd::variable_list backward;
}

#endif //FUSEDCONV_CUH
