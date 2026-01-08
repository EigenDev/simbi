#ifndef HET_MEM_TRANSFER_HPP
#define HET_MEM_TRANSFER_HPP

#include "compat.hpp"
#include "hesi/core/error_handling.hpp"
#include "hesi/core/traits.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"
#include "hesi/exec/token.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace simbi::het::mem {

    // main transfer function - handles all cases automatically
    inline exec::token_t copy_async(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        const exec::stream_t& stream
    )
    {
        // nada to copy
        if (bytes == 0) {
            return exec::token_t::create(stream.backend());
        }

        // case 1: same locality (same device)
        if (dst_loc == src_loc) {
            if (dst_loc.backend == backend_type_t::cpu) {
                // CPU → CPU
                std::memcpy(dst, src, bytes);
            }
#if defined(CUDA_ENABLED)
            else if (dst_loc.backend == backend_type_t::cuda) {
                // GPU → GPU (same device)
                check_error<cuda_backend_t>(
                    cudaMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            else if (dst_loc.backend == backend_type_t::hip) {
                // GPU → GPU (same device)
                check_error<hip_backend_t>(
                    hipMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        hipMemcpyDeviceToDevice,
                        static_cast<hipStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#endif
        }
        // case 2: different devices, same backend (peer transfer)
        else if (dst_loc.backend == src_loc.backend &&
                 dst_loc.device_id != src_loc.device_id) {
#if defined(CUDA_ENABLED)
            if (dst_loc.backend == backend_type_t::cuda) {
                // peer copy: GPU i → GPU j
                check_error<cuda_backend_t>(
                    cudaMemcpyPeerAsync(
                        dst,
                        static_cast<int>(dst_loc.device_id),
                        src,
                        static_cast<int>(src_loc.device_id),
                        bytes,
                        static_cast<cudaStream_t>(stream.native())
                    ),
                    "async peer memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (dst_loc.backend == backend_type_t::hip) {
                // peer copy: GPU i → GPU j
                check_error<hip_backend_t>(
                    hipMemcpyPeerAsync(
                        dst,
                        static_cast<int>(dst_loc.device_id),
                        src,
                        static_cast<int>(src_loc.device_id),
                        bytes,
                        static_cast<hipStream_t>(stream.native())
                    ),
                    "async peer memory copy"
                );
            }
#endif
        }
        // case 3: CPU → GPU
        else if (dst_loc.backend != backend_type_t::cpu &&
                 src_loc.backend == backend_type_t::cpu) {
#if defined(CUDA_ENABLED)
            if (dst_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (dst_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        hipMemcpyHostToDevice,
                        static_cast<hipStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#endif
        }
        // case 4: GPU → CPU
        else if (dst_loc.backend == backend_type_t::cpu &&
                 src_loc.backend != backend_type_t::cpu) {
#if defined(CUDA_ENABLED)
            if (src_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (src_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpyAsync(
                        dst,
                        src,
                        bytes,
                        hipMemcpyDeviceToHost,
                        static_cast<hipStream_t>(stream.native())
                    ),
                    "async memory copy"
                );
            }
#endif
        }
        // case 5: Different backends (not directly supported)
        else {
            throw std::runtime_error(
                "cross-backend transfers not supported "
                "(e.g., CUDA ↔ HIP)"
            );
        }

        // record completion event
        auto tok = exec::token_t::create(stream.backend());
        tok.record(stream);
        return tok;
    }

    // synchronous version (blocks until complete)
    inline void copy(
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

        // for synchronous, we can use the simpler non-async API
        if (dst_loc == src_loc) {
            if (dst_loc.backend == backend_type_t::cpu) {
                std::memcpy(dst, src, bytes);
            }
#if defined(CUDA_ENABLED)
            else if (dst_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),
                    "memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            else if (dst_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice),
                    "memory copy"
                );
            }
#endif
        }
        else if (dst_loc.backend == src_loc.backend &&
                 dst_loc.device_id != src_loc.device_id) {
#if defined(CUDA_ENABLED)
            if (dst_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpyPeer(
                        dst,
                        static_cast<int>(dst_loc.device_id),
                        src,
                        static_cast<int>(src_loc.device_id),
                        bytes
                    ),
                    "peer memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (dst_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpyPeer(
                        dst,
                        static_cast<int>(dst_loc.device_id),
                        src,
                        static_cast<int>(src_loc.device_id),
                        bytes
                    ),
                    "peer memory copy"
                );
            }
#endif
        }
        else if (dst_loc.backend != backend_type_t::cpu &&
                 src_loc.backend == backend_type_t::cpu) {
#if defined(CUDA_ENABLED)
            if (dst_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
                    "memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (dst_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice),
                    "memory copy"
                );
            }
#endif
        }
        else if (dst_loc.backend == backend_type_t::cpu &&
                 src_loc.backend != backend_type_t::cpu) {
#if defined(CUDA_ENABLED)
            if (src_loc.backend == backend_type_t::cuda) {
                check_error<cuda_backend_t>(
                    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
                    "memory copy"
                );
            }
#elif defined(HIP_ENABLED)
            if (src_loc.backend == backend_type_t::hip) {
                check_error<hip_backend_t>(
                    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost),
                    "memory copy"
                );
            }
#endif
        }
        else {
            throw std::runtime_error("cross-backend transfers not supported");
        }
    }

    // fill memory with a byte pattern
    inline exec::token_t fill_async(
        void* dst,
        locality_t dst_loc,
        std::uint8_t value,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        if (bytes == 0) {
            return exec::token_t::create(stream.backend());
        }

        if (dst_loc.backend == backend_type_t::cpu) {
            std::memset(dst, value, bytes);
        }
#if defined(CUDA_ENABLED)
        else if (dst_loc.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaMemsetAsync(
                    dst,
                    value,
                    bytes,
                    static_cast<cudaStream_t>(stream.native())
                ),
                "async memset"
            );
        }
#elif defined(HIP_ENABLED)
        else if (dst_loc.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(
                hipMemsetAsync(dst, value, bytes, stream.native()),
                "async memset"
            );
        }
#endif

        auto tok = exec::token_t::create(stream.backend());
        tok.record(stream);
        return tok;
    }

    // synchronous fill
    inline void
    fill(void* dst, locality_t dst_loc, std::uint8_t value, std::size_t bytes)
    {
        if (bytes == 0) {
            return;
        }

        if (dst_loc.backend == backend_type_t::cpu) {
            std::memset(dst, value, bytes);
        }
#if defined(CUDA_ENABLED)
        else if (dst_loc.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaMemset(dst, value, bytes),
                "memset"
            );
        }
#elif defined(HIP_ENABLED)
        else if (dst_loc.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(hipMemset(dst, value, bytes), "memset");
        }
#endif
    }

    // peer capability query
    inline bool can_access_peer(locality_t loc1, locality_t loc2)
    {
        // can't't peer across different backends
        if (loc1.backend != loc2.backend) {
            return false;
        }

        // cant''t peer to yourself
        if (loc1.device_id == loc2.device_id) {
            return false;
        }

        // cpu has no peer concept
        if (loc1.backend == backend_type_t::cpu) {
            return false;
        }

#if defined(CUDA_ENABLED)
        if (loc1.backend == backend_type_t::cuda) {
            int can_access  = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(
                &can_access,
                static_cast<int>(loc1.device_id),
                static_cast<int>(loc2.device_id)
            );
            check_error<cuda_backend_t>(err, "device can access peer");
            return can_access != 0;
        }
#elif defined(HIP_ENABLED)
        if (loc1.backend == backend_type_t::hip) {
            int can_access = 0;
            hipError_t err = hipDeviceCanAccessPeer(
                &can_access,
                static_cast<int>(loc1.device_id),
                static_cast<int>(loc2.device_id)
            );
            check_error<hip_backend_t>(err, "device can access peer");
            return can_access != 0;
        }
#endif

        return false;
    }

    // enable peer access (call during initialization)
    inline void enable_peer_access(locality_t from, locality_t to)
    {
        if (!can_access_peer(from, to)) {
            throw std::runtime_error(
                "peer access not supported between devices"
            );
        }

#if defined(CUDA_ENABLED)
        if (from.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaSetDevice(static_cast<int>(from.device_id)),
                "set device"
            );
            auto err =
                cudaDeviceEnablePeerAccess(static_cast<int>(to.device_id), 0);
            // Ignore "already enabled" errors
            if (err != cudaSuccess &&
                err != cudaErrorPeerAccessAlreadyEnabled) {
                throw std::runtime_error("failed to enable peer access");
            }
        }
#elif defined(HIP_ENABLED)
        if (from.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(
                hipSetDevice(static_cast<int>(from.device_id)),
                "set device"
            );
            auto err =
                hipDeviceEnablePeerAccess(static_cast<int>(to.device_id), 0);
            // Ignore "already enabled" errors
            if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
                throw std::runtime_error("failed to enable peer access");
            }
        }
#endif
    }

    // prefetch managed memory to a device (for UVM optimization)
    inline void prefetch_async(
        const void* ptr,
        locality_t target_loc,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        if (bytes == 0) {
            return;
        }

#if defined(CUDA_ENABLED)
        if (target_loc.backend == backend_type_t::cuda) {
            int device = (target_loc.device_id == -1)
                             ? cudaCpuDeviceId
                             : static_cast<int>(target_loc.device_id);
            cudaPointerAttributes attrs;
            cudaPointerGetAttributes(&attrs, ptr);
            if (attrs.type != cudaMemoryTypeManaged) {
                return;   // noop
            }
            check_error<cuda_backend_t>(
                cudaMemPrefetchAsync(
                    ptr,
                    bytes,
                    cudaMemLocation{.id = device},
                    0,
                    static_cast<cudaStream_t>(stream.native())
                ),
                "memprefetch async"
            );
        }
#elif defined(HIP_ENABLED)
        if (target_loc.backend == backend_type_t::hip) {
            int device = (target_loc.device_id == -1)
                             ? hipCpuDeviceId
                             : static_cast<int>(target_loc.device_id);
            check_error<hip_backend_t>(
                hipMemPrefetchAsync(ptr, bytes, device, stream.native()),
                "memprefetch async"
            );
        }
#else
        (void) ptr;
        (void) target_loc;
        (void) stream;
#endif
    }

}   // namespace simbi::het::mem

#endif   // HETERO_MEM_TRANSFER_HPP
