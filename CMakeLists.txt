cmake_minimum_required(VERSION 3.12)
project(fusedconv LANGUAGES CXX CUDA)

find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} include)

add_library(fusedconv SHARED src/fusedConv.cu bindings/fusedConvWrapper.cpp)
target_link_libraries(fusedConv PRIVATE pybind::module ${TORCH_LIBRARIES})

set_target_properties(fusedConv PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
