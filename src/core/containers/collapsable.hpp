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

#include "core/containers/array.hpp"
#include "core/types/alias/alias.hpp"
#include <initializer_list>

namespace simbi {
    template <typename T, size_type Dims>
    struct collapsable_t {
        array_t<T, Dims> vals;
        constexpr collapsable_t()  = default;
        constexpr ~collapsable_t() = default;

        constexpr collapsable_t(std::initializer_list<T> init)
        {
            for (size_type ii = 0; ii < Dims && ii < init.size(); ++ii) {
                vals[ii] = *(init.begin() + ii);
            }

            // fill from the back of the initialize list, since we
            // are in general inputting shapes like (nk, nj, ni)
            // size_type init_size = std::distance(init.begin(), init.end());
            // auto start          = init.begin();
            // if (init_size > Dims) {
            //     std::advance(start, init_size - Dims);
            // }
            // for (size_type ii = 0; ii < Dims && start != init.end();
            //      ++ii, ++start) {
            //     vals[ii] = *start;
            // }
        }

        // accesor to get the value at index
        constexpr size_type operator[](size_type ii) const { return vals[ii]; }

        // implicit conversion to uarray
        constexpr operator array_t<T, Dims>() const { return vals; }
    };
}   // namespace simbi

#endif
