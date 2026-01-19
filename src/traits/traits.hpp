// =============================================================================
// traits.hpp
//
// custom type traits for compile-time introspection.
// this file provides custom type traits, such as `is_maybe` and `is_maybe_v`,
// which can be used to check if a given type is a specialization of the
// `maybe_t` template at compile time.
//
// usage:
//   if constexpr (is_maybe_v<my_type>) { ... }
// =============================================================================
#pragma once

#include <type_traits>

namespace simbi {
    template <typename T>
    class maybe_t;

    template <typename T>
    struct is_maybe
    {
        static const bool value = false;
    };

    template <typename T>
    struct is_maybe<maybe_t<T>>
    {
        static const bool value = true;
    };

    template <typename T>
    inline constexpr bool is_maybe_v = is_maybe<std::decay_t<T>>::value;
} // namespace simbi
