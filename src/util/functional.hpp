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

#include "common/traits.hpp"   // for is_same_v
#include <exception>           // for bad_function_call
#include <optional>            // for optional
#include <utility>             // for forward

namespace simbi {
    // Exception class for bad function calls
    class bad_function_call : public std::exception
    {
      public:
        // Override the what() function to return a custom error message
        const char* what() const noexcept override
        {
            return "bad function call";
        }
    };

    // Forward declaration of the function template
    template <typename>
    class function;

    // Specialization of the function template for callable objects
    template <typename R, typename... Args>
    class function<R(Args...)>
    {
        // Base class for callable objects
        struct callable_base {
            virtual ~callable_base()                  = default;
            virtual DUAL R invoke(Args... args)       = 0;
            virtual callable_base* DUAL clone() const = 0;
        };

        // Derived class template for specific callable objects
        template <typename F>
        struct callable : callable_base {
            F f;
            // error code
            int error_code;

            // Constructor that takes an std::optional
            template <typename T>
            callable(std::optional<T> opt) : f(std::move(opt)), error_code(0)
            {
            }

            // Constructor that forwards the callable object
            callable(F&& f) : f(std::forward<F>(f)), error_code(0)
            {
                if constexpr (std::is_same_v<F, std::nullptr_t>) {
                    if constexpr (global::BuildPlatform ==
                                  global::Platform::GPU) {
                        error_code = 1;
                    }
                    else {
                        throw bad_function_call();
                    }
                }
            }

            // Override the invoke function to call the stored callable object
            DUAL R invoke(Args... args) override
            {
                if constexpr (std::is_same_v<F, std::nullptr_t>) {
                    if constexpr (global::BuildPlatform ==
                                  global::Platform::GPU) {
                        return R();
                    }
                    else {
                        throw bad_function_call();
                    }
                }
                else if constexpr (is_optional<std::optional<F>>::value) {
                    if (!f.has_value()) {
                        if constexpr (global::BuildPlatform ==
                                      global::Platform::GPU) {

                            return R();
                        }
                        else {
                            throw bad_function_call();
                        }
                    }
                    return (*f)(std::forward<Args>(args)...);
                }
                else {
                    return f(std::forward<Args>(args)...);
                }
            }

            // Override the clone function to create a copy of the callable
            // object
            DUAL callable_base* clone() const override
            {
                if constexpr (std::is_same_v<F, std::nullptr_t>) {
                    return nullptr;
                }
                else {
                    return new callable(f);
                }
            }
        };

        // Pointer to the base class of the callable object
        callable_base* callable_ptr;

      public:
        // Default constructor
        DUAL function() : callable_ptr(nullptr) {}

        // Constructor that takes a callable object
        template <typename F>
        DUAL function(F f) : callable_ptr(new callable<F>(std::forward<F>(f)))
        {
        }

        // Copy constructor
        DUAL function(const function& other)
            : callable_ptr(
                  other.callable_ptr ? other.callable_ptr->clone() : nullptr
              )
        {
        }

        // Move constructor
        DUAL function(function&& other) noexcept
            : callable_ptr(other.callable_ptr)
        {
            other.callable_ptr = nullptr;
        }

        // Copy assignment operator
        DUAL function& operator=(const function& other)
        {
            if (this != &other) {
                delete callable_ptr;
                callable_ptr =
                    other.callable_ptr ? other.callable_ptr->clone() : nullptr;
            }
            return *this;
        }

        // Move assignment operator
        DUAL function& operator=(function&& other) noexcept
        {
            if (this != &other) {
                delete callable_ptr;
                callable_ptr       = other.callable_ptr;
                other.callable_ptr = nullptr;
            }
            return *this;
        }

        // Destructor
        DUAL ~function() { delete callable_ptr; }

        // Function call operator
        DUAL R operator()(Args... args) const
        {
            if (!callable_ptr) {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
                    return R();
                }
                else {
                    throw bad_function_call();
                }
            }
            return callable_ptr->invoke(std::forward<Args>(args)...);
        }

        // Conversion operator to bool
        DUAL explicit operator bool() const noexcept
        {
            return callable_ptr != nullptr;
        }

        // Equality operator
        DUAL bool operator==(const function& other) const noexcept
        {
            return callable_ptr == other.callable_ptr;
        }

        // Inequality operator
        DUAL bool operator!=(const function& other) const noexcept
        {
            return !(*this == other);
        }

        // Equality operator for nullptr
        DUAL bool operator==(std::nullptr_t) const
        {
            return callable_ptr == nullptr;
        }

        // Inequality operator for nullptr
        DUAL bool operator!=(std::nullptr_t) const
        {
            return callable_ptr != nullptr;
        }

        // Support for std::optional
        DUAL bool has_value() const noexcept { return callable_ptr != nullptr; }

        DUAL void reset() noexcept
        {
            delete callable_ptr;
            callable_ptr = nullptr;
        }
    };
}   // namespace simbi

#endif