// =============================================================================
// traits.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
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
