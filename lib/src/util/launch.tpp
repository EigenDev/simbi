#include "kernel.hpp"
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
    void launch(const ExecutionPolicy<> &policy, Function f, Arguments... args)
    {
        #if GPU_CODE
        if constexpr(BuildPlatform == Platform::GPU)
        {
            ExecutionPolicy<> p = policy;
            Kernel<<<p.gridSize, 
                    p.blockSize, 
                    p.sharedMemBytes, 
                    p.stream>>>(f, args...);
        }
        #else 
        {
            f(args...);
        }
        #endif
    }

} // namespace simbi
