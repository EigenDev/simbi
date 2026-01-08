#include "hesi/backend/stream.hpp"

namespace simbi::het::backend {

    stream_handle_t create_stream_cpu()
    {
        // cpu has no stream concept, return sentinel
        return nullptr;
    }

    void destroy_stream_cpu(stream_handle_t)
    {
        // noop
    }

    void synchronize_stream_cpu(stream_handle_t)
    {
        // cpu is always synchronized
    }

    bool query_stream_cpu(stream_handle_t)
    {
        // always complete
        return true;
    }

}   // namespace simbi::het::backend
