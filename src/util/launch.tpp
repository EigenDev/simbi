#include "kernel.hpp"   // for Kernel

namespace simbi {

    //
    // Launch with no configuration
    //
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args)
    {
        f(args...);
    }

    //
    // Launch with explicit (or partial) configuration
    //
    template <typename Function, typename... Arguments>
    void launch(const ExecutionPolicy<>& policy, Function f, Arguments... args)
    {

#if GPU_CODE
        Kernel<<<
            policy.gridSize,
            policy.blockSize,
            policy.sharedMemBytes,
            policy.stream>>>(f, args...);
#else
        f(args...);
#endif
    }

}   // namespace simbi
