#include "hesi/backend/stream.hpp"

namespace simbi::het::backend {

    void* create_stream_cpu()
    {
        // cpu has no stream concept, return sentinel
        return nullptr;
    }

    void destroy_stream_cpu(void*)
    {
        // noop
    }

    void synchronize_stream_cpu(void*)
    {
        // cpu is always synchronized
    }

    bool query_stream_cpu(void*)
    {
        // always complete
        return true;
    }

}   // namespace simbi::het::backend
