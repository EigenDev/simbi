#ifndef HET_BACKEND_EVENT_HPP
#define HET_BACKEND_EVENT_HPP

#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"

namespace simbi::het::backend {

    // opaque event handle (backend-specific)
    using event_handle_t = void*;

    // create event for given backend
    event_handle_t create_event(backend_type_t backend);

    // destroy event
    void destroy_event(backend_type_t backend, event_handle_t handle);

    // record event on stream (marks current point in stream)
    void record_event(
        backend_type_t backend,
        event_handle_t event,
        stream_handle_t stream
    );

    // make stream wait for event (cross-stream dependency)
    void wait_event(
        backend_type_t backend,
        stream_handle_t stream,
        event_handle_t event
    );

    // synchronize on event (block host until event completes)
    void synchronize_event(backend_type_t backend, event_handle_t event);

    // query if event has completed
    bool query_event(backend_type_t backend, event_handle_t event);

}   // namespace simbi::het::backend

#endif
