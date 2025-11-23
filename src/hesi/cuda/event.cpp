#include "hesi/backend/event.hpp"

#ifdef CUDA_ENABLED
#include "hesi/core/error_handling.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend {

    void* create_event_cuda()
    {
        cudaEvent_t event;
        check_error<cuda_backend_t>(cudaEventCreate(&event), "event create");
        return static_cast<void*>(event);
    }

    void destroy_event_cuda(void* handle)
    {
        if (!handle) {
            return;
        }

        auto event = static_cast<cudaEvent_t>(handle);
        check_error<cuda_backend_t>(cudaEventDestroy(event), "event destroy");
    }

    void record_event_cuda(void* event_handle, void* stream_handle)
    {
        auto event  = static_cast<cudaEvent_t>(event_handle);
        auto stream = static_cast<cudaStream_t>(stream_handle);

        check_error<cuda_backend_t>(
            cudaEventRecord(event, stream),
            "event record"
        );
    }

    void wait_event_cuda(void* stream_handle, void* event_handle)
    {
        auto stream = static_cast<cudaStream_t>(stream_handle);
        auto event  = static_cast<cudaEvent_t>(event_handle);

        check_error<cuda_backend_t>(
            cudaStreamWaitEvent(stream, event, 0),
            "stream wait event"
        );
    }

    void synchronize_event_cuda(void* handle)
    {
        if (!handle) {
            return;
        }

        auto event = static_cast<cudaEvent_t>(handle);
        check_error<cuda_backend_t>(
            cudaEventSynchronize(event),
            "event synchronize"
        );
    }

    bool query_event_cuda(void* handle)
    {
        if (!handle) {
            return true;
        }

        auto event      = static_cast<cudaEvent_t>(handle);
        cudaError_t err = cudaEventQuery(event);

        if (err == cudaSuccess) {
            return true;   // complete
        }
        else if (err == cudaErrorNotReady) {
            return false;   // still pending
        }
        else {
            // actual error
            check_error<cuda_backend_t>(err, "event query");
            return false;   // unreachable
        }
    }

}   // namespace simbi::het::backend

#endif   // CUDA_ENABLED
