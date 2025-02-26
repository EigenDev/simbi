/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            functional.hpp
 *  * @brief           a custom implementation of std::function for GPU/CPU
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include "smart_ptr.hpp"
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

        DEV function(function&& other) noexcept
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

        ~function() {};

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

        DEV R operator()(Args... args) const
        {

            if (!callable) {
                if constexpr (global::on_gpu) {
                    printf("Error: function is not callable on the GPU");
                }
                else {
                    throw std::bad_function_call();
                }
            }
            return callable->invoke(std::forward<Args>(args)...);
        }

        explicit operator bool() const noexcept { return callable != nullptr; }

        bool operator==(std::nullptr_t) const noexcept
        {
            return callable == nullptr;
        }

        bool operator!=(std::nullptr_t) const noexcept
        {
            return callable != nullptr;
        }

        void swap(function& other) noexcept
        {
            std::swap(callable, other.callable);
        }

      private:
        struct callable_base {
            virtual ~callable_base()                 = default;
            DEV virtual R invoke(Args... args) const = 0;
            virtual callable_base* clone() const     = 0;
        };

        template <typename F, typename DecayType = std::decay_t<F>>
        struct callable_impl : callable_base {
            DecayType f;

            callable_impl(DecayType&& f) : f(std::forward<DecayType>(f)) {}

            callable_impl(const DecayType& f) : f(f) {}

            DEV R invoke(Args... args) const override
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