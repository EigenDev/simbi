/**
@file           algorithms.hpp
@brief          Custom Algorithms for simbi code base
*/

#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include "build_options.hpp"
#include <algorithm>

namespace simbi::algorithms {
    template <typename T>
    DUAL constexpr T my_min(const T& a, const T& b)
    {
        return a < b ? a : b;
    }

    template <typename T>
    DUAL constexpr T my_max(const T& a, const T& b)
    {
        return a > b ? a : b;
    }

    // device-compatible copy function
    template <typename InputIt, typename OutputIt>
    DUAL OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
    {
        if constexpr (!global::on_gpu) {
            return std::copy(first, last, d_first);
        }

        while (first != last) {
            *d_first++ = *first++;
        }
        return d_first;
    }

    // device-compatible copy_n function
    template <typename InputIt, typename Size, typename OutputIt>
    DUAL OutputIt copy_n(InputIt first, Size count, OutputIt result)
    {
        if constexpr (!global::on_gpu) {
            return std::copy_n(first, count, result);
        }
        for (Size i = 0; i < count; ++i) {
            *result++ = *first++;
        }
        return result;
    }

    // idem for move
    template <typename InputIt, typename OutputIt>
    DUAL OutputIt move(InputIt first, InputIt last, OutputIt d_first)
    {
        if constexpr (!global::on_gpu) {
            return std::move(first, last, d_first);
        }
        while (first != last) {
            *d_first++ = std::move(*first++);
        }
        return d_first;
    }

    // idem for fill
    template <typename ForwardIt, typename T>
    DUAL void fill(ForwardIt first, ForwardIt last, const T& value)
    {
        if constexpr (!global::on_gpu) {
            std::fill(first, last, value);
        }
        while (first != last) {
            *first++ = value;
        }
    }

    // idem for fill_n
    template <typename OutputIt, typename Size, typename T>
    DUAL OutputIt fill_n(OutputIt first, Size count, const T& value)
    {
        if constexpr (!global::on_gpu) {
            return std::fill_n(first, count, value);
        }
        for (Size i = 0; i < count; ++i) {
            *first++ = value;
        }
        return first;
    }

    // idem for swap_ranges
    template <typename ForwardIt1, typename ForwardIt2>
    DUAL void
    swap_ranges(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
    {
        if constexpr (!global::on_gpu) {
            std::swap_ranges(first1, last1, first2);
        }
        else {
            while (first1 != last1) {
                std::swap(*first1++, *first2++);
            }
        }
    }

    // idem for swap
    template <typename T>
    DUAL void swap(T& a, T& b)
    {
        if constexpr (!global::on_gpu) {
            std::swap(a, b);
        }
        else {
            T tmp = std::move(a);
            a     = std::move(b);
            b     = std::move(tmp);
        }
    }

    // reverse
    template <typename BidirectionalIt>
    DUAL void reverse(BidirectionalIt first, BidirectionalIt last)
    {
        if constexpr (!global::on_gpu) {
            std::reverse(first, last);
        }
        else {
            while (first < last) {
                std::swap(*first++, *--last);
            }
        }
    }

    // invoke that calls std::invoke on cpu
    // and is customized on gpu
    // template <typename F, typename... Args>
    // DUAL constexpr auto invoke(F&& f, Args&&... args)
    // {
    //     if constexpr (!global::on_gpu) {
    //         return std::invoke(std::forward<F>(f),
    //         std::forward<Args>(args)...);
    //     }
    //     else {
    //         // gpu-compatible invoke
    //         if constexpr (std::is_member_function_pointer_v<
    //                           std::remove_cvref_t<F>>) {
    //             return (std::forward<F>(f)->*(std::forward<Args>(args)...));
    //         }
    //         else {
    //             return std::forward<F>(f)(std::forward<Args>(args)...);
    //         }
    //     }
    // }
}   // namespace simbi::algorithms

#endif
