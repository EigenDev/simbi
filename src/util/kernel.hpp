#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <assert.h>
#include <stdlib.h>
#include "build_options.hpp"

namespace simbi {

template <typename Function, typename... Arguments>
GPU_LAUNCHABLE
void Kernel(Function f, Arguments... args)
{
    f(args...);
}

}

#endif