/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       functional.hpp
 * @brief      a custom std::function-like case that can be used on the GPU
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Oct-27-2024     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include "smrt_ptr.hpp"
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace simbi {

    template <typename>
    class function;

    template <typename R, typename... Args>
    class function<R(Args...)>
    {
      public:
        function() noexcept : callable(nullptr) {}

        function(std::nullptr_t) noexcept : callable(nullptr) {}

        function(const function& other)
        {
            if (other.callable) {
                callable =
                    util::smart_ptr<callable_base>(other.callable->clone());
            }
            else {
                callable = nullptr;
            }
        }

        DUAL function(function&& other) noexcept
            : callable(std::move(other.callable))
        {
            other.callable = nullptr;
        }

        template <typename F>
        function(const F& f) : callable(new callable_impl<std::decay_t<F>>(f))
        {
        }

        template <typename F>
        function(F&& f)
            : callable(
                  f ? new callable_impl<std::decay_t<F>>(std::forward<F>(f))
                    : nullptr
              )
        {
        }

        ~function() = default;

        function& operator=(const function& other)
        {
            if (this != &other) {
                callable.reset(
                    other.callable ? other.callable->clone() : nullptr
                );
            }
            return *this;
        }

        function& operator=(function&& other) noexcept
        {
            if (this != &other) {
                callable       = std::move(other.callable);
                other.callable = nullptr;
            }
            return *this;
        }

        function& operator=(std::nullptr_t) noexcept
        {
            callable.reset();
            return *this;
        }

        DUAL R operator()(Args... args) const
        {

            if (!callable) {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
                    printf("Error: function is not callable on the GPU");
                }
                else {
                    throw std::bad_function_call();
                }
            }
            return callable->invoke(std::forward<Args>(args)...);
        }

        DUAL explicit operator bool() const noexcept
        {
            return callable != nullptr;
        }

        DUAL bool operator==(std::nullptr_t) const noexcept
        {
            return callable == nullptr;
        }

        DUAL bool operator!=(std::nullptr_t) const noexcept
        {
            return callable != nullptr;
        }

        void swap(function& other) noexcept
        {
            std::swap(callable, other.callable);
        }

      private:
        struct callable_base {
            virtual ~callable_base()                  = default;
            DUAL virtual R invoke(Args... args) const = 0;
            virtual callable_base* clone() const      = 0;
        };

        template <typename F, typename DecayType = std::decay_t<F>>
        struct callable_impl : callable_base {
            DecayType f;

            callable_impl(DecayType&& f) : f(std::forward<DecayType>(f)) {}

            callable_impl(const DecayType& f) : f(f) {}

            DUAL R invoke(Args... args) const override
            {
                return f(std::forward<Args>(args)...);
            }

            callable_base* clone() const override
            {
                return new callable_impl<DecayType>(f);
            }
        };

        util::smart_ptr<callable_base> callable;
    };

}   // namespace simbi

#endif   // FUNCTIONAL_HPP