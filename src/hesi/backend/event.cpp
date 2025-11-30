#include "hesi/backend/event.hpp"
#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"

#include <stdexcept>

namespace simbi::het::backend {

    // forward declarations (use canonical handle typedefs)
    event_handle_t create_event_cpu();
    void destroy_event_cpu(event_handle_t handle);
    void record_event_cpu(event_handle_t event, stream_handle_t stream);
    void wait_event_cpu(stream_handle_t stream, event_handle_t event);
    void synchronize_event_cpu(event_handle_t handle);
    bool query_event_cpu(event_handle_t handle);

#ifdef CUDA_ENABLED
    event_handle_t create_event_cuda();
    void destroy_event_cuda(event_handle_t handle);
    void record_event_cuda(event_handle_t event, stream_handle_t stream);
    void wait_event_cuda(stream_handle_t stream, event_handle_t event);
    void synchronize_event_cuda(event_handle_t handle);
    bool query_event_cuda(event_handle_t handle);
#endif

#ifdef HIP_ENABLED
    event_handle_t create_event_hip();
    void destroy_event_hip(event_handle_t handle);
    void record_event_hip(event_handle_t event, stream_handle_t stream);
    void wait_event_hip(stream_handle_t stream, event_handle_t event);
    void synchronize_event_hip(event_handle_t handle);
    bool query_event_hip(event_handle_t handle);
#endif

    event_handle_t create_event(backend_type_t backend)
    {
        switch (backend) {
            case backend_type_t::cpu: return create_event_cpu();

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: return create_event_cuda();
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: return create_event_hip();
#endif

            default: throw std::runtime_error("unsupported backend for event");
        }
    }

    void destroy_event(backend_type_t backend, event_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: destroy_event_cpu(handle); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: destroy_event_cuda(handle); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: destroy_event_hip(handle); break;
#endif

            default:
                // don't throw in destructor path
                break;
        }
    }

    void record_event(
        backend_type_t backend,
        event_handle_t event,
        stream_handle_t stream
    )
    {
        switch (backend) {
            case backend_type_t::cpu: record_event_cpu(event, stream); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: record_event_cuda(event, stream); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: record_event_hip(event, stream); break;
#endif

            default:
                throw std::runtime_error(
                    "unsupported backend for event record"
                );
        }
    }

    void wait_event(
        backend_type_t backend,
        stream_handle_t stream,
        event_handle_t event
    )
    {
        switch (backend) {
            case backend_type_t::cpu: wait_event_cpu(stream, event); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: wait_event_cuda(stream, event); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: wait_event_hip(stream, event); break;
#endif

            default:
                throw std::runtime_error("unsupported backend for event wait");
        }
    }

    void synchronize_event(backend_type_t backend, event_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: synchronize_event_cpu(handle); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: synchronize_event_cuda(handle); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: synchronize_event_hip(handle); break;
#endif

            default:
                throw std::runtime_error("unsupported backend for event sync");
        }
    }

    bool query_event(backend_type_t backend, event_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: return query_event_cpu(handle);

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: return query_event_cuda(handle);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: return query_event_hip(handle);
#endif

            default: return true;   // assume complete for unknown backend
        }
    }

}   // namespace simbi::het::backend
