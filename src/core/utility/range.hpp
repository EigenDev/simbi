/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            range.hpp
 *  * @brief           a custom implementation of range for GPU/CPU
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
#ifndef RANGE_HPP
#define RANGE_HPP

#include "config.hpp"                              // for HD, STATIC
#include "system/adapter/device_adapter_api.hpp"   // for globalThreadIdx
#include <iterator>                                // for input_iterator_tag

template <typename T>
struct range_t {
    struct iter {
        // inheriting from std::iterator deprecated
        // must include explicit category and value_type
        // Note: there are three other types one could include
        // such as difference_type, pointer, reference
        using iterator_category = std::input_iterator_tag;
        using value_type        = T;

        DUAL iter(T current, T step) : current(current), step(step) {}

        DUAL T operator*() const { return current; }

        DUAL T const* operator->() const { return &current; }

        DUAL iter& operator++()
        {
            current += step;
            return *this;
        }

        DUAL iter operator++(int)
        {
            auto copy = *this;
            ++*this;
            return copy;
        }

        // Loses commutativity. Iterator-based ranges are simply broken. :-(
        DUAL bool operator==(iter const& other) const
        {
            return step > 0 ? current >= other.current
                            : current < other.current;
        }

        DUAL bool operator!=(iter const& other) const
        {
            return not(*this == other);
        }

      private:
        T step, current;
    };

    DUAL range_t(T end) : rbegin(0, 1), rend(end, 1), rstep(1) {}

    DUAL range_t(T begin, T end, T step = 1)
        : rbegin(begin, step), rend(end, step), rstep(step)
    {
    }

    DUAL iter begin() const { return rbegin; }

    DUAL iter end() const { return rend; }

  private:
    iter rbegin, rend;
    T rstep;
};

template <typename T, typename U = int>
STATIC range_t<T> range(T begin, T end, U step = 1)
{
    begin += simbi::global_thread_idx();
    return range_t<T>{begin, end, static_cast<T>(step)};
}
#endif
