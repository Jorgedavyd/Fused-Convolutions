#include <pybind11/pybind11.h>
#include "fusedConv.h"
#include <torch/extensions.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::class_<Conv1D>(m, "FusedConv1D")
        .def(pybind11::init<>())
        .def("forward", &Conv1D::forward)
        .def("backward", &Conv1D::backward);

    pybind11::class_<Conv2D>(m, "FusedConv2D")
        .def(pybind11::init<>())
        .def("forward", &Conv2D::forward)
        .def("backward", &Conv2D::backward);

    pybind11::class_<Conv3D>(m, "FusedConv3D")
        .def(pybind11::init<>())
        .def("forward", &Conv3D::forward)
        .def("backward", &Conv3D::backward);

}

