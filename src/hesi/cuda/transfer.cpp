#include "hesi/backend/transfer.hpp"
#include "hesi/core/error_handling.hpp"

#ifdef CUDA_ENABLED
#include "hesi/exec/token.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend {

    // determine copy kind based on localities
    cudaMemcpyKind infer_copy_kind(locality_t dst, locality_t src)
    {
        bool dst_is_device = dst.backend != backend_type_t::cpu;
        bool src_is_device = src.backend != backend_type_t::cpu;

        if (!dst_is_device && !src_is_device) {
            return cudaMemcpyHostToHost;
        }
        else if (dst_is_device && !src_is_device) {
            return cudaMemcpyHostToDevice;
        }
        else if (!dst_is_device && src_is_device) {
            return cudaMemcpyDeviceToHost;
        }
        else if (dst.device_id == src.device_id) {
            return cudaMemcpyDeviceToDevice;
        }
        else {
            return cudaMemcpyDefault;   // peer-to-peer
        }
    }

    void copy_cuda(
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

        auto kind = infer_copy_kind(dst_loc, src_loc);

        // peer copy requires special handling
        if (kind == cudaMemcpyDefault &&
            dst_loc.device_id != src_loc.device_id) {
            check_error<cuda_backend_t>(
                cudaMemcpyPeer(
                    dst,
                    dst_loc.device_id,
                    src,
                    src_loc.device_id,
                    bytes
                ),
                "peer memcpy"
            );
        }
        else {
            check_error<cuda_backend_t>(
                cudaMemcpy(dst, src, bytes, kind),
                "memcpy"
            );
        }
    }

    exec::token_t copy_async_cuda(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        cudaStream_t stream
    )
    {
        if (bytes == 0) {
            return exec::token_t::immediate(backend_type_t::cuda);
        }

        auto kind = infer_copy_kind(dst_loc, src_loc);

        // peer copy
        if (kind == cudaMemcpyDefault &&
            dst_loc.device_id != src_loc.device_id) {
            check_error<cuda_backend_t>(
                cudaMemcpyPeerAsync(
                    dst,
                    dst_loc.device_id,
                    src,
                    src_loc.device_id,
                    bytes,
                    stream
                ),
                "async peer memcpy"
            );
        }
        else {
            check_error<cuda_backend_t>(
                cudaMemcpyAsync(dst, src, bytes, kind, stream),
                "async memcpy"
            );
        }

        // create token and record completion
        auto token = exec::token_t::create(backend_type_t::cuda);
        check_error<cuda_backend_t>(
            cudaEventRecord(token.event_->native(), stream),
            "event record after copy"
        );

        return token;
    }

    void fill_cuda(void* dst, std::uint8_t value, std::size_t bytes)
    {
        if (bytes > 0 && dst) {
            check_error<cuda_backend_t>(
                cudaMemset(dst, value, bytes),
                "memset"
            );
        }
    }

    exec::token_t fill_async_cuda(
        void* dst,
        std::uint8_t value,
        std::size_t bytes,
        cudaStream_t stream
    )
    {
        if (bytes == 0) {
            return exec::token_t::immediate(backend_type_t::cuda);
        }

        check_error<cuda_backend_t>(
            cudaMemsetAsync(dst, value, bytes, stream),
            "async memset"
        );

        auto token = exec::token_t::create(backend_type_t::cuda);
        check_error<cuda_backend_t>(
            cudaEventRecord(token.event_->native(), stream),
            "event record after fill"
        );

        return token;
    }

    void prefetch_cuda(
        const void* ptr,
        locality_t target_loc,
        std::size_t bytes,
        cudaStream_t stream
    )
    {
        if (bytes == 0) {
            return;
        }

        int device = (target_loc.backend == backend_type_t::cpu)
                         ? cudaCpuDeviceId
                         : static_cast<int>(target_loc.device_id);

        check_error<cuda_backend_t>(
            cudaMemPrefetchAsync(ptr, bytes, device, stream),
            "prefetch"
        );
    }

    bool can_access_peer_cuda(locality_t from, locality_t to)
    {
        // can't peer across different backends
        if (from.backend != backend_type_t::cuda ||
            to.backend != backend_type_t::cuda) {
            return false;
        }

        // can't peer to yourself
        if (from.device_id == to.device_id) {
            return false;
        }

        int can_access  = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(
            &can_access,
            static_cast<int>(from.device_id),
            static_cast<int>(to.device_id)
        );

        // don't throw on query failure, just return false
        if (err != cudaSuccess) {
            cudaGetLastError();   // clear error
            return false;
        }

        return can_access != 0;
    }

    void enable_peer_access_cuda(locality_t from, locality_t to)
    {
        if (!can_access_peer_cuda(from, to)) {
            throw std::runtime_error(
                "peer access not supported between devices"
            );
        }

        // set device context
        check_error<cuda_backend_t>(
            cudaSetDevice(static_cast<int>(from.device_id)),
            "set device for peer access"
        );

        // enable access (may already be enabled)
        cudaError_t err =
            cudaDeviceEnablePeerAccess(static_cast<int>(to.device_id), 0);

        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            check_error<cuda_backend_t>(err, "enable peer access");
        }
        else if (err == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();   // clear error
        }
    }

}   // namespace simbi::het::backend

#endif   // CUDA_ENABLED
