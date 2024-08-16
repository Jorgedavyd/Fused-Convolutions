
template <typename T, typename index_t = uint32_t>
struct BasePtrTrait {
    using PtrType = T*;

    static PtrType make(T* ptr) {
        return ptr;
    }

    static T& access (PtrType ptr, index_t idx) {
        return ptr[index];
    }
};


template <typename T, typename index_t = uint32_t>
struct RestrictedPtrTrait {
public:
    using PtrType = T*;

    static PtrType make(T* ptr) {
    }
    static T& operator [] (PtrType ptr, index_t idx) {
        return ptr[idx];
    }
};

template <typename scalar_t, template <typename U> class PtrTrait = RestrictedPtrTrait, typename index_t = uint32_t>
class
