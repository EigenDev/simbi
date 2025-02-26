/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            collapsable.hpp
 *  * @brief           provides a collapsable array class for fixed-size arrays
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-21
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
 *  * 2025-02-21      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef COLLAPSABLE_HPP
#define COLLAPSABLE_HPP

#include "core/types/alias/alias.hpp"
#include <initializer_list>
namespace simbi {
    template <size_type Dims>
    struct collapsable {

        uarray<Dims> vals;
        constexpr collapsable()  = default;
        constexpr ~collapsable() = default;

        constexpr collapsable(std::initializer_list<size_type> init)
        {

            // fill from the back of the initialize list, since we
            // are in general inputting shapes like (nk, nj, ni)
            auto init_size = std::distance(init.begin(), init.end());
            auto start     = init.begin();
            if (init_size > Dims) {
                std::advance(start, init_size - Dims);
            }
            for (size_type i = 0; i < Dims && start != init.end();
                 ++i, ++start) {
                vals[i] = *start;
            }
        }

        // accesor to get the value at index
        constexpr size_type operator[](size_type ii) const { return vals[ii]; }

        // implicit conversion to uarray
        constexpr operator uarray<Dims>() const { return vals; }
    };
}   // namespace simbi

#endif