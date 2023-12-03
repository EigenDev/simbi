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

#include "launch.tpp"
#endif
