#include "hesi/backend/event.hpp"

namespace simbi::het::backend {

    void* create_event_cpu()
    {
        // cpu has no event concept, return sentinel
        return nullptr;
    }

    void destroy_event_cpu(void*)
    {
        // noop
    }

    void record_event_cpu(void*, void*)
    {
        // noop
    }

    void wait_event_cpu(void*, void*)
    {
        // noop
    }

    void synchronize_event_cpu(void*)
    {
        // noop
    }

    bool query_event_cpu(void*)
    {
        // always complete
        return true;
    }

}   // namespace simbi::het::backend
