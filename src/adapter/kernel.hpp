#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "config.hpp"

namespace simbi::adapter {
    template <typename Func, typename... Args>
    KERNEL void generic_kernel(Func func, Args... args)
    {
        func(args...);
    }
}   // namespace simbi::adapter
#endif
