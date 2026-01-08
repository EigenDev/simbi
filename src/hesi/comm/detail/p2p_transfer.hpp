#ifndef HETERO_COMM_DETAIL_P2P_HPP
#define HETERO_COMM_DETAIL_P2P_HPP

#include "compat.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/exec/token.hpp"
#include <cstring>
#include <stdexcept>

namespace simbi::comm::detail {

    using namespace het;
    using namespace het::exec;

    // performs a raw, asynchronous memory copy
    // ensures stream dependence and returns a token
    static inline token_t async_copy(
        const executor_t& exec,
        void* dst,
        const void* src,
        std::size_t bytes
    )
    {
        auto backend = exec.backend();

        if (backend == backend_type_t::cpu) {
            // synchronous cpu copy
            std::memcpy(dst, src, bytes);
            // return an empty token representing "completed now"
            return token_t::immediate(backend);
        }

// gpu copy (device, host, or p2p)
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

// determine kind based on backend type
#if defined(CUDA_ENABLED)
        cudaMemcpyKind kind        = cudaMemcpyDefault;
        cudaStream_t native_stream = exec.stream().native();
        cudaMemcpyAsync(dst, src, bytes, kind, native_stream);

        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("cuda async copy failed");
        }
#elif defined(HIP_ENABLED)
        hipMemcpyKind kind        = hipMemcpyDefault;
        hipStream_t native_stream = exec.stream().native();
        hipMemcpyAsync(dst, src, bytes, kind, native_stream);

        if (hipGetLastError() != hipSuccess) {
            throw std::runtime_error("hip async copy failed");
        }
#endif

        // record completion event on the stream and return the token
        auto t = token_t::create(backend);
        t.record(exec.stream());
        return t;

#else
        // if compiled without gpu support, we can't do async copy
        throw std::runtime_error("async copy attempted on disabled backend");
#endif
    }
}   // namespace simbi::comm::detail

#endif   // HETERO_COMM_DETAIL_P2P_HPP
