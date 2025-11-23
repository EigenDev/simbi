#include "hesi/backend/stream.hpp"

#ifdef CUDA_ENABLED
#include "hesi/core/error_handling.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend {

    void* create_stream_cuda(std::int32_t device_id)
    {
        if (device_id >= 0) {
            check_error<cuda_backend_t>(
                cudaSetDevice(device_id),
                "set device for stream creation"
            );
        }

        cudaStream_t stream;
        check_error<cuda_backend_t>(cudaStreamCreate(&stream), "stream create");

        return static_cast<void*>(stream);
    }

    void destroy_stream_cuda(void* handle)
    {
        if (!handle) {
            return;
        }

        auto stream = static_cast<cudaStream_t>(handle);
        check_error<cuda_backend_t>(
            cudaStreamDestroy(stream),
            "stream destroy"
        );
    }

    void synchronize_stream_cuda(void* handle)
    {
        if (!handle) {
            return;
        }

        auto stream = static_cast<cudaStream_t>(handle);
        check_error<cuda_backend_t>(
            cudaStreamSynchronize(stream),
            "stream synchronize"
        );
    }

    bool query_stream_cuda(void* handle)
    {
        if (!handle) {
            return true;
        }

        auto stream     = static_cast<cudaStream_t>(handle);
        cudaError_t err = cudaStreamQuery(stream);

        if (err == cudaSuccess) {
            return true;   // complete
        }
        else if (err == cudaErrorNotReady) {
            return false;   // still working
        }
        else {
            // actual error
            check_error<cuda_backend_t>(err, "stream query");
            return false;   // unreachable
        }
    }

}   // namespace simbi::het::backend

#endif   // CUDA_ENABLED
