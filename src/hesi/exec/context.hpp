#ifndef HET_EXEC_CONTEXT_HPP
#define HET_EXEC_CONTEXT_HPP

#include "executor.hpp"

#include <stdexcept>

namespace simbi::het::exec {

    // thread-local storage for current executor
    namespace detail {
        inline thread_local executor_t* current_executor = nullptr;
    }

    // get current executor (throws if none set)
    inline executor_t& current_executor()
    {
        if (!detail::current_executor) {
            throw std::runtime_error(
                "No executor in current scope. "
                "Use executor_guard or call compute_from() explicitly."
            );
        }
        return *detail::current_executor;
    }

    // check if executor is set (non-throwing)
    inline bool has_executor() { return detail::current_executor != nullptr; }

    // RAII guard for scoped executor
    class executor_guard
    {
        executor_t* previous_;

      public:
        explicit executor_guard(executor_t& exec)
            : previous_(detail::current_executor)
        {
            detail::current_executor = &exec;
        }

        ~executor_guard() { detail::current_executor = previous_; }

        // Non-copyable, non-movable (strict RAII)
        executor_guard(const executor_guard&)            = delete;
        executor_guard& operator=(const executor_guard&) = delete;
        executor_guard(executor_guard&&)                 = delete;
        executor_guard& operator=(executor_guard&&)      = delete;
    };

}   // namespace simbi::het::exec

#endif   // HETERO_EXEC_CONTEXT_HPP
