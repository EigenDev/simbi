#include "hesi/backend/memory.hpp"
#include "hesi/core/types.hpp"

#include <cstdlib>
#include <stdexcept>

namespace simbi::het::backend {

    void* allocate_cpu(std::size_t bytes, memory_type_t /*type*/)
    {
        if (bytes == 0) {
            return nullptr;
        }

        void* ptr = nullptr;

        // cpu only supports host_visible and pinned
        // pinned will be handled via cuda/hip if available.
        // if we are here, it means we are cpu only,
        // so it doesn't matter what the memory type is
        ptr = std::malloc(bytes);

        if (!ptr) {
            throw std::runtime_error("cpu allocation failed");
        }

        return ptr;
    }

    void deallocate_cpu(void* ptr, memory_type_t)
    {
        if (ptr) {
            std::free(ptr);
        }
    }

}   // namespace simbi::het::backend
