#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"

#include <stdexcept>

namespace simbi::het::backend {

    // forward declarations (use platform-native handle type)
    stream_handle_t create_stream_cpu();
    void destroy_stream_cpu(stream_handle_t handle);
    void synchronize_stream_cpu(stream_handle_t handle);
    bool query_stream_cpu(stream_handle_t handle);

#ifdef CUDA_ENABLED
    stream_handle_t create_stream_cuda(std::int32_t device_id);
    void destroy_stream_cuda(stream_handle_t handle);
    void synchronize_stream_cuda(stream_handle_t handle);
    bool query_stream_cuda(stream_handle_t handle);
#endif

#ifdef HIP_ENABLED
    stream_handle_t create_stream_hip(std::int32_t device_id);
    void destroy_stream_hip(stream_handle_t handle);
    void synchronize_stream_hip(stream_handle_t handle);
    bool query_stream_hip(stream_handle_t handle);
#endif

    stream_handle_t
    create_stream(backend_type_t backend, std::int32_t device_id)
    {
        switch (backend) {
            case backend_type_t::cpu: {
                (void) device_id;
                return create_stream_cpu();
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: return create_stream_cuda(device_id);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: return create_stream_hip(device_id);
#endif

            default: throw std::runtime_error("unsupported backend for stream");
        }
    }

    void destroy_stream(backend_type_t backend, stream_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: destroy_stream_cpu(handle); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: destroy_stream_cuda(handle); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: destroy_stream_hip(handle); break;
#endif

            default:
                // don't throw in destructor path
                break;
        }
    }

    void synchronize_stream(backend_type_t backend, stream_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: synchronize_stream_cpu(handle); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: synchronize_stream_cuda(handle); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: synchronize_stream_hip(handle); break;
#endif

            default:
                throw std::runtime_error("unsupported backend for stream sync");
        }
    }

    bool query_stream(backend_type_t backend, stream_handle_t handle)
    {
        switch (backend) {
            case backend_type_t::cpu: return query_stream_cpu(handle);

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: return query_stream_cuda(handle);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: return query_stream_hip(handle);
#endif

            default: return true;   // assume complete for unknown backend
        }
    }

}   // namespace simbi::het::backend
