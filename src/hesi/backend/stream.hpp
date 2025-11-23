#ifndef HET_BACKEND_STREAM_HPP
#define HET_BACKEND_STREAM_HPP

#include "hesi/core/types.hpp"

#include <cstdint>

namespace simbi::het::backend {

    // opaque stream handle (backend-specific)
    using stream_handle_t = void*;

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
