/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       launch.hpp
 * @brief      houses calls for arch-specific calls on generic functors
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
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef LAUNCH_HPP
#define LAUNCH_HPP

#include "exec_policy.hpp"   // for ExecutionPolicy

namespace simbi {
    // Launch function object with no configuration
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args);

    // Launch function object with an explicit execution policy / configuration
    template <typename Function, typename... Arguments>
    void launch(const ExecutionPolicy<>& p, Function f, Arguments... args);
}   // namespace simbi

#include "launch.ipp"
#endif
