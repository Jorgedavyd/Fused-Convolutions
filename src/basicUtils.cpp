#include <cstdint>
#include <torch/extensions.h>
// Parameters Utilities
// Conteiner types that are in charge of parameter initialization

template <typename index_t = uint32_t>
struct Stride {
    index_t x_, y_, z_;
    Stride (const index_t& x, const index_t& y, const index_t& z) : x_(x), y_(y), z_(z) {};
    Stride (const index_t& x, const index_t& y) : x_(x), y_(y){};
    Stride (const index_t& x) : x_(x){};
};

template <typename index_t = uint32_t>
struct Padding {
    index_t x_, y_, z_;
    Padding (const index_t& x, const index_t& y, const index_t& z) : x_(x), y_(y), z_(z) {};
    Padding (const index_t& x, const index_t& y) : x_(x), y_(y){};
    Padding (const index_t& x) : x_(x){};
};

template <typename index_t = uint32_t>
struct Kernel {
    index_t x_, y_, z_;
    Kernel (const index_t& x, const index_t& y, const index_t& z) : x_(x), y_(y), z_(z) {};
    Kernel (const index_t& x, const index_t& y) : x_(x), y_(y){};
    Kernel (const index_t& x) : x_(x){};
};
