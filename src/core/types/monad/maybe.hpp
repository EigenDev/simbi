/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       maybe.hpp
 * @brief      implementation of a Maybe monad
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Jan-11-2025     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 */
#ifndef MAYBE_HPP
#define MAYBE_HPP

#include "build_options.hpp"
#include <bitset>
#include <type_traits>

namespace simbi {
    struct nothing_t {
        constexpr explicit nothing_t(int) {}

        nothing_t() = delete;
    };

    inline constexpr nothing_t Nothing{0};

    template <typename T>
    class Maybe
    {
      public:
        using value_type = T;

        DUAL constexpr Maybe() : valid{false} {}

        DUAL constexpr Maybe(nothing_t) : valid{false} {}

        DUAL constexpr Maybe(const T& value) : valid{true}, thisValue{value} {}

        DUAL constexpr Maybe(T&& value)
            : valid{true}, thisValue{std::move(value)}
        {
        }

        // Constructor deduced from the value
        template <typename U>
        DUAL constexpr Maybe(U&& value)
            : valid{true}, thisValue{std::forward<U>(value)}
        {
            static_assert(std::is_convertible_v<U, T>);
        }

        DUAL constexpr bool has_value() const { return valid; }

        DUAL constexpr const T& value() const& { return thisValue; }

        DUAL constexpr T& value() & { return thisValue; }

        DUAL constexpr T value() && { return std::move(thisValue); }

        template <typename U>
        DUAL constexpr T unwrap_or(U&& default_value) const&
        {
            static_assert(
                std::is_convertible_v<U, T>,
                "U must be convertible to T"
            );
            return valid ? thisValue
                         : static_cast<T>(std::forward<U>(default_value));
        }

        template <typename U>
        DUAL constexpr T unwrap_or(U&& default_value) &&
        {
            static_assert(
                std::is_convertible_v<U, T>,
                "U must be convertible to T"
            );
            return valid ? std::move(thisValue)
                         : static_cast<T>(std::forward<U>(default_value));
        }

        template <typename F>
        DUAL constexpr auto map(F&& f) const&
        {
            using result_type = std::invoke_result_t<F, const T&>;
            if (valid) {
                return Maybe<result_type>{f(thisValue)};
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
                return Maybe<result_type>{f(std::move(thisValue))};
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
                return f(thisValue);
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
                return f(std::move(thisValue));
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
                return Maybe<T>{thisValue};
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
                return Maybe<T>{std::move(thisValue)};
            }
            else {
                return Maybe<result_type>{f()};
            }
        }

        template <typename F>
        DUAL constexpr T unwrap_or_else(F&& f) const&
        {
            if (valid) {
                return thisValue;
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr T unwrap_or_else(F&& f) &&
        {
            if (valid) {
                return std::move(thisValue);
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr auto match(F&& f) const&
        {
            if (valid) {
                return f(thisValue);
            }
            else {
                return f();
            }
        }

        template <typename F>
        DUAL constexpr auto match(F&& f) &&
        {
            if (valid) {
                return f(std::move(thisValue));
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
                return thisValue == other.thisValue;
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
                return thisValue < other.thisValue;
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
        DUAL constexpr T* operator->() { return std::addressof(thisValue); }

        DUAL constexpr const T* operator->() const
        {
            return std::addressof(thisValue);
        }

        DUAL constexpr T& operator*() & { return thisValue; }

        DUAL constexpr const T& operator*() const& { return thisValue; }

        // in-place construction
        template <typename... Args>
        DUAL T& emplace(Args&&... args)
        {
            thisValue = T(std::forward<Args>(args)...);
            valid     = true;
            return thisValue;
        }

        // Math overloads with scalar types
        template <typename U>
        DUAL constexpr Maybe<T> operator+(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{thisValue + rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator-(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{thisValue - rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator*(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{thisValue * rhs};
            }
            return Maybe<T>{Nothing};
        }

        template <typename U>
        DUAL constexpr Maybe<T> operator/(const U& rhs) const
        {
            if (valid) {
                return Maybe<T>{thisValue / rhs};
            }
            return Maybe<T>{Nothing};
        }

        // implicit conversion to bool
        DUAL constexpr explicit operator bool() const { return valid; }

        // implicit conversion to T
        DUAL constexpr operator T() const { return thisValue; }

        // implicit conversion to T
        // DUAL constexpr operator T() && { return std::move(thisValue); }

      private:
        bool valid;
        T thisValue;
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