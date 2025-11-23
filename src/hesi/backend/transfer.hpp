#ifndef HET_BACKEND_TRANSFER_HPP
#define HET_BACKEND_TRANSFER_HPP

#include "hesi/core/types.hpp"
#include "hesi/exec/token.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::het::backend {

    // synchronous copy (blocks until complete)
    void copy(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes
    );

    // asynchronous copy (returns token for synchronization)
    // stream must match dst/src backend
    exec::token_t copy_async(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        void* stream_handle,   // native stream (cudaStream_t, etc)
        backend_type_t stream_backend
    );

    // fill memory with byte pattern
    void
    fill(void* dst, locality_t dst_loc, std::uint8_t value, std::size_t bytes);

    // async fill
    exec::token_t fill_async(
        void* dst,
        locality_t dst_loc,
        std::uint8_t value,
        std::size_t bytes,
        void* stream_handle,
        backend_type_t stream_backend
    );

    // prefetch managed memory to device (hint for UVM)
    void prefetch(
        const void* ptr,
        locality_t target_loc,
        std::size_t bytes,
        void* stream_handle,
        backend_type_t stream_backend
    );

    // peer access management
    bool can_access_peer(locality_t from, locality_t to);
    void enable_peer_access(locality_t from, locality_t to);

}   // namespace simbi::het::backend

#endif
