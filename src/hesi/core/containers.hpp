#ifndef HET_CONTAINERS_HPP
#define HET_CONTAINERS_HPP

#include "compat.hpp"

#include <cstddef>

namespace simbi::het {

    // portable pair
    template <typename T1, typename T2>
    struct pair_t {
        T1 first;
        T2 second;

        constexpr pair_t() = default;
        DUAL constexpr pair_t(const T1& a, const T2& b) : first(a), second(b) {}
    };
}   // namespace simbi::het

#endif   // HETERO_CONTAINERS_HPP
