#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "build_options.hpp"

namespace simbi {
    template <typename Function, typename... Arguments>
    GPU_LAUNCHABLE void Kernel(Function f, Arguments... args)
    {
        f(args...);
    }
}   // namespace simbi

#endif