#ifndef HET_BACKEND_STREAM_HPP
#define HET_BACKEND_STREAM_HPP

#include "hesi/core/types.hpp"

#include <cstdint>

#if defined(CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include <hip/hip_runtime.h>
#endif

namespace simbi::het::backend {

    // opaque stream handle (backend-specific)
#if defined(CUDA_ENABLED)
    using stream_handle_t = cudaStream_t;
#elif defined(HIP_ENABLED)
    using stream_handle_t = hipStream_t;
#else
    using stream_handle_t = void*;
#endif

    // create stream for given backend
    stream_handle_t
    create_stream(backend_type_t backend, std::int32_t device_id = 0);

    // destroy stream
    void destroy_stream(backend_type_t backend, stream_handle_t handle);

    // block until all work in stream completes
    void synchronize_stream(backend_type_t backend, stream_handle_t handle);

    // query if stream has completed all work
    bool query_stream(backend_type_t backend, stream_handle_t handle);

}   // namespace simbi::het::backend

#endif
