#include <pybind11/pybind11.h>
#include "fusedConv.h"
#include <torch/extensions.h>

PYBIND11_MODULE(fusedconv, m) {
    m.def("fwd_2D", &fusedConv2D_fwd, "Forward method for 2D fused convolutions.");
    m.def("bwd_2D", &fusedConv2D_bwd, "Backward method for 2D fused convolutions.");
    m.def("fwd_1D", &fusedConv1D_fwd, "Forward method for 1D fused convolutions.");
    m.def("bwd_1D", &fusedConv1D_bwd, "Backward method for 1D fused convolutions.");
    m.def("fwd_3D", &fusedConv3D_fwd, "Forward method for 3D fused convolutions.");
    m.def("bwd_3D", &fusedConv3D_bwd, "Backward method for 3D fused convolutions.");
}



