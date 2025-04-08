/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            maybe.hpp
 *  * @brief           Maybe monad for handling optional values
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
#ifndef MAYBE_HPP
#define MAYBE_HPP

#include "build_options.hpp"
#include <bitset>
#include <type_traits>

namespace simbi {
    struct nothing_t {
        constexpr explicit nothing_t(int) : error_message(nullptr) {}
        constexpr explicit nothing_t(const char* message)
            : error_message(message)
        {
        }

        const char* error_message;
    };

    inline constexpr nothing_t Nothing{0};
    using None = nothing_t;

    template <typename T>
    class Maybe
    {
      public:
        using value_type = T;

        DUAL constexpr Maybe() : valid{false}, error_message(nullptr) {}

        DUAL constexpr Maybe(nothing_t nothing)
            : valid{false}, error_message(nothing.error_message)
        {
        }

        DUAL constexpr Maybe(const T& value)
            : valid{true}, this_value{value}, error_message(nullptr)
        {
        }

        DUAL constexpr Maybe(T&& value)
            : valid{true}, this_value{std::move(value)}, error_message(nullptr)
        {
        }

        // Constructor deduced from the value
        template <typename U>
        DUAL constexpr Maybe(U&& value)
            : valid{true},
              this_value{std::forward<U>(value)},
              error_message(nullptr)
        {
            static_assert(std::is_convertible_v<U, T>);
        }

        DUAL constexpr bool has_value() const { return valid; }

        DUAL constexpr const T& value() const& { return this_value; }

        DUAL constexpr T& value() & { return this_value; }

        DUAL constexpr T value() && { return std::move(this_value); }

        DUAL constexpr const char* error() const { return error_message; }

        template <typename U>
        DUAL constexpr T unwrap_or(U&& default_value) const&
        {
            static_assert(
                std::is_convertible_v<U, T>,
                "U must be convertible to T"
            );
            return valid ? this_value
                         : static_cast<T>(std::forward<U>(default_value));
        }

        template <typename U>
        DUAL constexpr T unwrap_or(U&& default_value) &&
        {
            static_assert(
                std::is_convertible_v<U, T>,
                "U must be convertible to T"
            );
            return valid ? std::move(this_value)
                         : static_cast<T>(std::forward<U>(default_value));
        }

        template <typename F>
        DUAL constexpr auto map(F&& f) const&
        {
            using result_type = std::invoke_result_t<F, const T&>;
            if (valid) {
                return Maybe<result_type>{f(this_value)};
            }
            else {
                return Maybe<result_type>{Nothing};
            }
        }

        template <typename F>
        DUAL constexpr auto map(F&& f) &&
        {
            using result_type = std::invoke_result_t<F, T&&>;
            if (valid) {
                return Maybe<result_type>{f(std::move(this_value))};
            }
            else {
                return Maybe<result_type>{Nothing};
            }
        }

        template <typename F>
        DUAL constexpr auto and_then(F&& f) const&
        {
            using result_type = std::invoke_result_t<F, const T&>;
            if (valid) {
                return f(this_value);
            }
            else {
                return Maybe<result_type>{Nothing};
            }
        }

        template <typename F>
        DUAL constexpr auto and_then(F&& f) &&
        {
            using result_type = std::invoke_result_t<F, T&&>;
            if (valid) {
                return f(std::move(this_value));
            }
            else {
                return Maybe<result_type>{Nothing};
            }
        }

        template <typename F>
        DUAL constexpr auto or_else(F&& f) const&
        {
            using result_type = std::invoke_result_t<F>;
            if (valid) {
                return Maybe<T>{this_value};
            }
            else {
                return Maybe<result_type>{f()};
            }
        }

        template <typename F>
        DUAL constexpr auto or_else(F&& f) &&
        {
            using result_type = std::invoke_result_t<F>;
            if (valid) {
                return Maybe<T>{std::move(this_value)};
            }
            else {
                return Maybe<result_type>{f()};
            }
        }

        template <typename F>
        DUAL constexpr T unwrap_or_else(F&& f) const&
        {
            if (valid) {
                return this_value;
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr T unwrap_or_else(F&& f) &&
        {
            if (valid) {
                return std::move(this_value);
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr auto match(F&& f) const&
        {
            if (valid) {
                return f(this_value);
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr auto match(F&& f) &&
        {
            if (valid) {
                return f(std::move(this_value));
            }
            else {
                return f();
            }
        }

        template <typename U>
        DUAL constexpr bool operator==(const Maybe<U>& other) const
        {
            if (valid != other.valid) {
                return false;
            }
            if (valid) {
                return this_value == other.this_value;
            }
            return true;
        }

        template <typename U>
        DUAL constexpr bool operator!=(const Maybe<U>& other) const
        {
            return !(*this == other);
        }

        template <typename U>
        DUAL constexpr bool operator<(const Maybe<U>& other) const
        {
            if (valid != other.valid) {
                return valid < other.valid;
            }
            if (valid) {
                return this_value < other.this_value;
            }
            return false;
        }

        template <typename U>
        DUAL constexpr bool operator>(const Maybe<U>& other) const
        {
            return other < *this;
        }

        template <typename U>
        DUAL constexpr bool operator<=(const Maybe<U>& other) const
        {
            return !(other < *this);
        }

        template <typename U>
        DUAL constexpr bool operator>=(const Maybe<U>& other) const
        {
            return !(*this < other);
        }

        // reference access operators
        DUAL constexpr T* operator->() { return std::addressof(this_value); }

        DUAL constexpr const T* operator->() const
        {
            return std::addressof(this_value);
        }

        DUAL constexpr T& operator*() & { return this_value; }

        DUAL constexpr const T& operator*() const& { return this_value; }

        // in-place construction
        template <typename... Args>
        DUAL T& emplace(Args&&... args)
        {
            this_value = T(std::forward<Args>(args)...);
            valid      = true;
            return this_value;
        }

        // Math overloads with scalar types
        template <typename U>
        DUAL constexpr Maybe<T> operator+(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{this_value + rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator-(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{this_value - rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator*(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{this_value * rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator/(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{this_value / rhs};
            }
            return Maybe<T>{Nothing};
        }

        // implicit conversion to bool
        DUAL constexpr explicit operator bool() const { return valid; }

        // implicit conversion to T
        DUAL constexpr operator T() const { return this_value; }

        // implicit conversion to T
        // DUAL constexpr operator T() && { return std::move(this_value); }

      private:
        bool valid;
        T this_value;
        const char* error_message;
    };

    // Deduction guide
    template <typename T>
    Maybe(T) -> Maybe<std::decay_t<T>>;

    // Hash support
    // template <typename T>
    // struct std::hash<simbi::Maybe<T>> {
    //     std::size_t operator()(const simbi::Maybe<T>& m) const
    //     {
    //         if (!m.has_value()) {
    //             return 0;
    //         }
    //         return std::hash<T>{}(*m);
    //     }
    // };

}   // namespace simbi

#endif
