#include "kernel.hpp"   // for Kernel

namespace simbi {
    // Launch with no configuration
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args)
    {
        f(args...);
    }

    // Launch with explicit (or partial) configuration
    template <typename Function, typename... Arguments>
    void launch(
        const ExecutionPolicy<>& policy,
        const int device,
        Function f,
        Arguments... args
    )
    {

#if GPU_CODE
        // If streams are specified, use them
        if (!policy.streams.empty()) {
            Kernel<<<
                policy.get_device_gridSize(device),
                policy.blockSize,
                policy.sharedMemBytes,
                policy.streams[device % policy.streams.size()]>>>(f, args...);
        }
        else {
            Kernel<<<
                policy.get_device_gridSize(device),
                policy.blockSize,
                policy.sharedMemBytes>>>(f, args...);
        }
#else
        f(args...);
#endif
    }

}   // namespace simbi
