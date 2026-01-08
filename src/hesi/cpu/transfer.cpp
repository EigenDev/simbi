#include "hesi/backend/transfer.hpp"
#include <cstdint>
#include <cstring>

namespace simbi::het::backend {

    void copy_cpu(void* dst, const void* src, std::size_t bytes)
    {
        if (bytes > 0 && dst && src) {
            std::memcpy(dst, src, bytes);
        }
    }

    void fill_cpu(void* dst, std::uint8_t value, std::size_t bytes)
    {
        if (bytes > 0 && dst) {
            std::memset(dst, value, bytes);
        }
    }

}   // namespace simbi::het::backend
