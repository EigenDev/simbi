/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 * @file       kernel.hpp
 * @brief      the generic gpu kernel that runs a generic functor
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */
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