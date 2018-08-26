#pragma once

#include <cassert>

namespace hdkrs {

// An owned pointer to a preallocated resource that automatically frees it upon destruction.
template<typename T>
class OwnedPtr {
public:
    OwnedPtr(T *ptr) : ptr(ptr) {}
    ~OwnedPtr(); // must be specialized for each T 

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

} // namespace hdkrs

