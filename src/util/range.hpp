/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       range.hpp
 * @brief      implements a custom python-like range generator
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef RANGE_HPP
#define RANGE_HPP

#include "build_options.hpp"   // for HD, STATIC
#include "device_api.hpp"      // for globalThreadIdx
#include <iterator>            // for input_iterator_tag

template <typename T>
struct range_t {
    struct iter {
        // inheriting from std::iterator deprecated
        // must include explicit category and value_type
        // Note: there are three other types one could include
        // such as difference_type, pointer, reference
        using iterator_category = std::input_iterator_tag;
        using value_type        = T;

        HD iter(T current, T step) : current(current), step(step) {}

        HD T operator*() const { return current; }

        HD T const* operator->() const { return &current; }

        HD iter& operator++()
        {
            current += step;
            return *this;
        }

        HD iter operator++(int)
        {
            auto copy = *this;
            ++*this;
            return copy;
        }

        // Loses commutativity. Iterator-based ranges are simply broken. :-(
        HD bool operator==(iter const& other) const
        {
            return step > 0 ? current >= other.current
                            : current < other.current;
        }

        HD bool operator!=(iter const& other) const
        {
            return not(*this == other);
        }

      private:
        T step, current;
    };

    HD range_t(T end) : rbegin(0, 1), rend(end, 1), rstep(1) {}

    HD range_t(T begin, T end, T step = 1)
        : rbegin(begin, step), rend(end, step), rstep(step)
    {
    }

    HD iter begin() const { return rbegin; }

    HD iter end() const { return rend; }

  private:
    iter rbegin, rend;
    T rstep;
};

template <typename T, typename U = int>
STATIC range_t<T> range(T begin, T end, U step = 1)
{
    begin += simbi::globalThreadIdx();
    return range_t<T>{begin, end, static_cast<T>(step)};
}
#endif