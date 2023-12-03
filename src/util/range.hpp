#ifndef RANGE_HPP
#define RANGE_HPP

#include "build_options.hpp"   // for GPU_CALLABLE, GPU_CALLABLE_INLINE
#include "device_api.hpp"      // for globalThreadIdx
#include <iterator>            // for input_iterator_tag

template <typename T> struct range_t {
    struct iter {
        // inheriting from std::iterator deperecated
        // must include explicity category and value_type
        // Note: there are three other types one could include
        // such as difference_type, pointer, reference
        using iterator_category = std::input_iterator_tag;
        using value_type        = T;

        GPU_CALLABLE
        iter(T current, T step) : current(current), step(step) {}

        GPU_CALLABLE
        T operator*() const { return current; }

        GPU_CALLABLE
        T const* operator->() const { return &current; }

        GPU_CALLABLE
        iter& operator++()
        {
            current += step;
            return *this;
        }

        GPU_CALLABLE
        iter operator++(int)
        {
            auto copy = *this;
            ++*this;
            return copy;
        }

        // Loses commutativity. Terator-based ranges are simply broken. :-(
        GPU_CALLABLE
        bool operator==(iter const& other) const
        {
            return step > 0 ? current >= other.current
                            : current < other.current;
        }

        GPU_CALLABLE
        bool operator!=(iter const& other) const { return not(*this == other); }

      private:
        T step, current;
    };

    GPU_CALLABLE
    range_t(T end) : rbegin(0, 1), rend(end, 1), rstep(1) {}

    GPU_CALLABLE
    range_t(T begin, T end, T step = 1)
        : rbegin(begin, step), rend(end, step), rstep(step)
    {
    }

    GPU_CALLABLE
    iter begin() const { return rbegin; }

    GPU_CALLABLE
    iter end() const { return rend; }

  private:
    iter rbegin, rend;
    T rstep;
};

template <typename T, typename U = int>
GPU_CALLABLE_INLINE range_t<T> range(T begin, T end, U step = 1)
{
    begin += simbi::globalThreadIdx();
    return range_t<T>{begin, end, static_cast<T>(step)};
}
#endif