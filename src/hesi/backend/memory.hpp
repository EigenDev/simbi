#ifndef HET_BACKEND_MEMORY_HPP
#define HET_BACKEND_MEMORY_HPP

#include "hesi/core/types.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::het::backend {

    // allocation dispatcher
    // returns raw pointer or throws on failure
    void* allocate(
        backend_type_t backend,
        std::size_t bytes,
        memory_type_t type,
        std::int32_t device_id = 0
    );

    // deallocation dispatcher
    void deallocate(backend_type_t backend, void* ptr, memory_type_t type);

    // query pointer attributes (useful for debugging)
    struct pointer_info_t {
        backend_type_t backend;
        memory_type_t type;
        std::int32_t device_id;
        bool is_valid;
    };

    pointer_info_t query_pointer(const void* ptr);

}   // namespace simbi::het::backend

#endif
