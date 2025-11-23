#include "hesi/backend/transfer.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/token.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace simbi::het::backend {

    // forward declarations
    void copy_cpu(void* dst, const void* src, std::size_t bytes);
    void fill_cpu(void* dst, std::uint8_t value, std::size_t bytes);

#ifdef CUDA_ENABLED
    void copy_cuda(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes
    );
    exec::token_t copy_async_cuda(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        void* stream
    );
    void fill_cuda(void* dst, std::uint8_t value, std::size_t bytes);
    exec::token_t fill_async_cuda(
        void* dst,
        std::uint8_t value,
        std::size_t bytes,
        void* stream
    );
    void prefetch_cuda(
        const void* ptr,
        locality_t target,
        std::size_t bytes,
        void* stream
    );
    bool can_access_peer_cuda(locality_t from, locality_t to);
    void enable_peer_access_cuda(locality_t from, locality_t to);
#endif

#ifdef HIP_ENABLED
    void copy_hip(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes
    );
    exec::token_t copy_async_hip(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        void* stream
    );
    void fill_hip(void* dst, std::uint8_t value, std::size_t bytes);
    exec::token_t fill_async_hip(
        void* dst,
        std::uint8_t value,
        std::size_t bytes,
        void* stream
    );
    void prefetch_hip(
        const void* ptr,
        locality_t target,
        std::size_t bytes,
        void* stream
    );
    bool can_access_peer_hip(locality_t from, locality_t to);
    void enable_peer_access_hip(locality_t from, locality_t to);
#endif

    // public API: synchronous copy
    void copy(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes
    )
    {
        if (bytes == 0) {
            return;
        }

        // determine backend from localities
        // if both cpu, use cpu impl
        if (dst_loc.backend == backend_type_t::cpu &&
            src_loc.backend == backend_type_t::cpu) {
            copy_cpu(dst, src, bytes);
            return;
        }

        // otherwise dispatch to gpu backend
        auto backend = (dst_loc.backend != backend_type_t::cpu)
                           ? dst_loc.backend
                           : src_loc.backend;

        switch (backend) {
#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                copy_cuda(dst, dst_loc, src, src_loc, bytes);
                break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                copy_hip(dst, dst_loc, src, src_loc, bytes);
                break;
#endif

            default: throw std::runtime_error("unsupported backend for copy");
        }
    }

    // public API: async copy
    exec::token_t copy_async(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        void* stream_handle,
        backend_type_t stream_backend
    )
    {
        if (bytes == 0) {
            return exec::token_t::immediate(stream_backend);
        }

        switch (stream_backend) {
            case backend_type_t::cpu: {
                (void) dst_loc;
                (void) src_loc;
                (void) stream_handle;
                copy_cpu(dst, src, bytes);
                return exec::token_t::immediate(backend_type_t::cpu);
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                return copy_async_cuda(
                    dst,
                    dst_loc,
                    src,
                    src_loc,
                    bytes,
                    stream_handle
                );
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                return copy_async_hip(
                    dst,
                    dst_loc,
                    src,
                    src_loc,
                    bytes,
                    stream_handle
                );
#endif

            default:
                throw std::runtime_error("unsupported backend for async copy");
        }
    }

    // public API: fill
    void
    fill(void* dst, locality_t dst_loc, std::uint8_t value, std::size_t bytes)
    {
        if (bytes == 0) {
            return;
        }

        switch (dst_loc.backend) {
            case backend_type_t::cpu: fill_cpu(dst, value, bytes); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: fill_cuda(dst, value, bytes); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: fill_hip(dst, value, bytes); break;
#endif

            default: throw std::runtime_error("unsupported backend for fill");
        }
    }

    // public API: async fill
    exec::token_t fill_async(
        void* dst,
        locality_t dst_loc,
        std::uint8_t value,
        std::size_t bytes,
        void* stream_handle,
        backend_type_t stream_backend
    )
    {
        if (bytes == 0) {
            return exec::token_t::immediate(stream_backend);
        }

        switch (stream_backend) {
            case backend_type_t::cpu: {
                (void) dst_loc;
                (void) stream_handle;
                fill_cpu(dst, value, bytes);
                return exec::token_t::immediate(backend_type_t::cpu);
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                return fill_async_cuda(dst, value, bytes, stream_handle);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                return fill_async_hip(dst, value, bytes, stream_handle);
#endif

            default:
                throw std::runtime_error("unsupported backend for async fill");
        }
    }

    // public API: prefetch
    void prefetch(
        const void* ptr,
        locality_t target_loc,
        std::size_t bytes,
        void* stream_handle,
        backend_type_t stream_backend
    )
    {
        if (bytes == 0) {
            return;
        }

        switch (stream_backend) {
            case backend_type_t::cpu: {
                (void) target_loc;
                (void) stream_handle;
                (void) ptr;
                // cpu has no prefetch
                break;
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                prefetch_cuda(ptr, target_loc, bytes, stream_handle);
                break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                prefetch_hip(ptr, target_loc, bytes, stream_handle);
                break;
#endif

            default: break;   // ignore unsupported
        }
    }

    // public API: peer access query
    bool can_access_peer(locality_t from, locality_t to)
    {
        // same location = no peer needed
        if (from == to) {
            return false;
        }

        // different backends = no peer
        if (from.backend != to.backend) {
            return false;
        }

        // cpu has no peer concept
        if (from.backend == backend_type_t::cpu) {
            return false;
        }

        switch (from.backend) {
#ifdef CUDA_ENABLED
            case backend_type_t::cuda: return can_access_peer_cuda(from, to);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: return can_access_peer_hip(from, to);
#endif

            default: return false;
        }
    }

    // public API: enable peer access
    void enable_peer_access(locality_t from, locality_t to)
    {
        if (!can_access_peer(from, to)) {
            throw std::runtime_error("peer access not available");
        }

        switch (from.backend) {
#ifdef CUDA_ENABLED
            case backend_type_t::cuda: enable_peer_access_cuda(from, to); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: enable_peer_access_hip(from, to); break;
#endif

            default: throw std::runtime_error("peer access not supported");
        }
    }

}   // namespace simbi::het::backend
