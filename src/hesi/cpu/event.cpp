#include "hesi/backend/event.hpp"
#include "hesi/backend/stream.hpp"

namespace simbi::het::backend {

    event_handle_t create_event_cpu()
    {
        // cpu has no event concept, return sentinel
        return nullptr;
    }

    void destroy_event_cpu(event_handle_t)
    {
        // noop
    }

    void record_event_cpu(event_handle_t, stream_handle_t)
    {
        // noop
    }

    void wait_event_cpu(stream_handle_t, event_handle_t)
    {
        // noop
    }

    void synchronize_event_cpu(event_handle_t)
    {
        // noop
    }

    bool query_event_cpu(event_handle_t)
    {
        // always complete
        return true;
    }

}   // namespace simbi::het::backend
