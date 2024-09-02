#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <functional>
#include "ptrTraits.cpp"

// Accessor for fused convolution
template <typename scalar_t, size_t N, template <typename U> class PtrTrait = RestrictedPtrTrait, typename index_t = uint32_t>
class Accessor  {
public:
    typedef typename PtrTrait<scalar_t>::PtrType PtrType;

    explicit __device__ Accessor (PtrType data_ptr, const index_t* stride, const index_t* size) : data_(data_ptr), stride_(stride), size_(size) {};

    __device__ Accessor (Accessor<scalar_t, N, PtrTrait, index_t>&& input) = delete;

    __device__ Accessor (Accessor<scalar_t, N, PtrTrait, index_t>& input) = delete;

    __device__ virtual const Accessor<scalar_t, N - 1, PtrTrait, index_t> operator[] (const index_t& idx) const {
        const index_t new_stride[N-1] = this->stride_ + 1;
        const index_t new_size[N-1] = this->size_ + 1;
        return Accessor<scalar_t,N - 1, PtrTrait, index_t>(this->data_ + this->stride_[0]*idx, new_stride, new_size);
    }

    __device__ virtual Accessor<scalar_t, N - 1, PtrTrait, index_t> operator[] (const index_t& idx) {
        index_t new_stride[N-1] = this->stride_ + 1;
        index_t new_size[N-1] = this->size_ + 1;
        return Accessor<scalar_t,N - 1, PtrTrait, index_t>(this->data_ + this->stride_[0]*idx, new_stride, new_size);
    }

    __device__ ~Accessor (void) = default;

private:
    PtrType data_;
    const index_t* stride_[N], size_[N];
};

template <typename scalar_t, template <typename U> class PtrTrait = RestrictedPtrTrait, typename index_t = uint32_t>
class Accessor<scalar_t, 1, PtrTrait, index_t> : public Accessor<scalar_t, 1, PtrTrait, index_t> {
public:
    typedef typename PtrTrait<scalar_t>::PtrType PtrType;

    explicit __device__ Accessor (PtrType data, index_t* stride, index_t* size) : data_(data), stride_(stride) {};

    // const-correctness
    __device__ virtual const PtrType operator[] (const index_t& idx) const {
        return this->data_ + this->stride[0]*idx;
    }
    __device__ virtual PtrType operator[] (const index_t& idx) {
        return this->data_ + this->stride[0]*idx;
    }

private:
    PtrType data_;
    const index_t stride[1];
};

template <typename scalar_t, size_t N, template <typename U> class PtrTrait = RestrictedPtrTrait, typename index_t = uint32_t>
class FFTAccessor : Accessor<scalar_t, N, PtrTrait, index_t> {
public:
    typedef typename PtrTrait<scalar_t>::PtrType PtrType;

    explicit __device__ FFTAccessor(PtrType data, index_t stride[N], index_t size[N]) : Accessor<scalar_t, N, PtrTrait, index_t> {};

};
