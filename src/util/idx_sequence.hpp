#ifndef IDX_SEQUENCE_HPP
#define IDX_SEQUENCE_HPP

#include <cstdio>

namespace simbi {
    namespace detail {
        template <typename T, T val>
        struct integral_constant {
            static constexpr T value = val;
            using value_type         = T;
            using type               = integral_constant;

            constexpr operator value_type() const noexcept { return value; }

            constexpr value_type operator()() const noexcept { return value; }
        };

        template <typename T, T... Ints>
        struct index_sequence {
            using type       = index_sequence;
            using value_type = T;

            static constexpr T size() noexcept { return sizeof...(Ints); }
        };

        template <std::size_t N, std::size_t... Ints>
        struct make_index_sequence
            : make_index_sequence<N - 1, N - 1, Ints...> {
        };

        template <std::size_t... Ints>
        struct make_index_sequence<0, Ints...>
            : index_sequence<std::size_t, Ints...> {
        };

        template <typename T, T... Vals, typename F>
        constexpr void for_sequence(index_sequence<T, Vals...>, F f)
        {
            (static_cast<void>(f(integral_constant<T, Vals>{})), ...);
        };
    }   // namespace detail
}   // namespace simbi

#endif