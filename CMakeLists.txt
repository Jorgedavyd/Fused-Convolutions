cmake_minimum_required(VERSION 3.12)
project(FusedConvolutions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(fusedConv STATIC src/fusedConv.cu)
add_library(fftConv STATIC src/fftConv.cu)
add_library(gemmConv STATIC src/gemmConv.cu)

target_include_directories(fusedConv PUBLIC include)
target_include_directories(fftConv PUBLIC include)
target_include_directories(gemmConv PUBLIC include)

pybind11_add_module(fusedconv MODULE bindings/fusedConvWrapper.cpp)
pybind11_add_module(fftconv MODULE bindings/fftConvWrapper.cpp)
pybind11_add_module(gemmconv MODULE bindings/gemmConvWrapper.cpp)

target_link_libraries(fusedconv PRIVATE "${TORCH_LIBRARIES}" fusedConv)
target_link_libraries(fftconv PRIVATE "${TORCH_LIBRARIES}" fftConv)
target_link_libraries(gemmconv PRIVATE "${TORCH_LIBRARIES}" gemmConv)

set_target_properties(fusedconv fftconv gemmconv PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

