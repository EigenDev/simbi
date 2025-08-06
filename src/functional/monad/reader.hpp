#ifndef READER_HPP
#define READER_HPP

#include <utility>
namespace simbi {
    /**
     * This is a reader monad. I just learned about it.
     * It's quite useful for passing around context
     * without explicitly threading it through every function call.
     *
     * srp: provide a way to read from an environment
     *      without explicitly passing it around.
     *
     * info:
     * https://stackoverflow.com/questions/14178889/what-is-the-purpose-of-the-reader-monad
     */
    template <typename Environment>
    class reader_t
    {
        thread_local static Environment* current_env;

      public:
        class scope_t
        {
            Environment* prev_;

          public:
            explicit scope_t(Environment& env) noexcept
                : prev_(std::exchange(current_env, &env))
            {
            }

            ~scope_t() noexcept { current_env = prev_; }

            // move-only semantics
            scope_t(const scope_t&)            = delete;
            scope_t& operator=(const scope_t&) = delete;
            scope_t(scope_t&& other) noexcept : prev_(other.prev_)
            {
                other.prev_ = nullptr;
            }
            scope_t& operator=(scope_t&& other) noexcept
            {
                if (this != &other) {
                    current_env = prev_;   // restore previous
                    prev_       = other.prev_;
                    other.prev_ = nullptr;
                }
                return *this;
            }
        };

        static Environment* ask() noexcept { return current_env; }

        template <typename F>
        static void with_env(F&& func) noexcept(
            noexcept(func(std::declval<Environment&>()))
        )
        {
            if (auto* env = ask()) {
                func(*env);
            }
        }
    };

    template <typename Environment>
    thread_local Environment* reader_t<Environment>::current_env = nullptr;

}   // namespace simbi
#endif
