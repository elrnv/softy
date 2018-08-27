#pragma once

#include <hdkrs/hdkrs.h>
#include <cassert>

namespace hdkrs {

// An owned pointer to a preallocated resource that automatically frees it upon destruction.
template<typename T>
class OwnedPtr {
public:
    OwnedPtr(OwnedPtr && other) = default;
    OwnedPtr(T *ptr) : ptr(ptr) {}
    ~OwnedPtr(); // must be specialized for each T 

    OwnedPtr& operator=(OwnedPtr && other) = default;

    operator bool() {
        return ptr != nullptr;
    }

    T& operator *() {
        assert(ptr);
        return *ptr;
    }

    T* operator->() {
        assert(ptr);
        return ptr;
    }

    T* get() {
        return ptr;
    }
    
private:
    T *ptr;
};


// Implement OwnedPtr specializations

template<>
inline OwnedPtr<PolyMesh>::~OwnedPtr() {
    free_polymesh(ptr);
}

template<>
inline OwnedPtr<TetMesh>::~OwnedPtr() {
    free_tetmesh(ptr);
}

} // namespace hdkrs

