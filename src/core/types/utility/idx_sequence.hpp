/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            idx_sequence.hpp
 *  * @brief           a custom implementation of index_sequence for GPU/CPU
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

            DUAL constexpr operator value_type() const noexcept
            {
                return value;
            }

            DUAL constexpr value_type operator()() const noexcept
            {
                return value;
            }
        };

        template <typename T, T... Ints>
        struct index_sequence {
            using type       = index_sequence;
            using value_type = T;

            DUAL static constexpr T size() noexcept { return sizeof...(Ints); }
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
        DUAL constexpr void for_sequence(index_sequence<T, Vals...>, F f)
        {
            (static_cast<void>(f(integral_constant<T, Vals>{})), ...);
        };
    }   // namespace detail
}   // namespace simbi

#endif