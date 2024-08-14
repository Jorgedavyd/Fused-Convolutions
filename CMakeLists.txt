cmake_minimum_required(VERSION 3.10)
project(FusedConvolutions)

find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(fusedConv SHARED src/fusedConv.cu bindings/fusedConvWrapper.cpp)
target_link_libraries(fusedConv PRIVATE pybind::module ${TORCH_LIBRARIES})

set_target_properties(fusedConv PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
