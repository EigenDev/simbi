/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            array_t.hpp
 *  * @brief           provides a simple array class for fixed-size arrays
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-19
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
 *  * 2025-02-19      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "build_options.hpp"
#include <iostream>
#include <stdexcept>

namespace simbi {
    template <typename T, size_type N>
    class array_t
    {
      public:
        static constexpr size_type len() { return N; }

        // Default constructor
        constexpr array_t() = default;

        // Initialize with value
        DUAL constexpr array_t(const T& value)
        {
            for (size_type i = 0; i < N; ++i) {
                data_[i] = value;
            }
        }

        // Host-only initializer list constructor
        constexpr array_t(const std::initializer_list<T>& values)
        {
            if (values.size() != N) {
                throw std::length_error("Invalid initializer list size");
            }
            for (size_type i = 0; i < N; ++i) {
                data_[i] = *(values.begin() + i);
            }
        }

        // Copy constructor
        DUAL constexpr array_t(const array_t& other)
        {
            for (size_type i = 0; i < N; ++i) {
                data_[i] = other.data_[i];
            }
        }

        // Move constructor
        DUAL constexpr array_t(array_t&& other)
        {
            for (size_type i = 0; i < N; ++i) {
                data_[i] = other.data_[i];
            }
        }

        // Copy assignment
        DUAL constexpr array_t& operator=(const array_t& other)
        {
            if (this != &other) {
                for (size_type i = 0; i < N; ++i) {
                    data_[i] = other.data_[i];
                }
            }
            return *this;
        }

        // Accessors
        DUAL constexpr size_type size() const { return N; }
        DUAL T* data() { return data_; }
        DUAL const T* data() const { return data_; }

        // array_t access
        DUAL constexpr T& operator[](size_type i)
        {
#ifdef GPU_CODE
            if (i >= N) {
                // GPU-safe error handling
                printf("array_t index out of bounds\n");
                return data_[0];
            }
#else
            if (i >= N) {
                throw std::out_of_range("array_t index out of bounds");
            }
#endif
            return data_[i];
        }

        DUAL constexpr const T& operator[](size_type i) const
        {
#ifdef GPU_CODE
            if (i >= N) {
                // GPU-safe error handling
                printf("array_t index out of bounds\n");
                return data_[0];
            }
#else
            if (i >= N) {
                throw std::out_of_range("array_t index out of bounds");
            }
#endif
            return data_[i];
        }

        // Iterator support - host only
        DUAL T* begin() { return data_; }
        DUAL const T* begin() const { return data_; }
        DUAL T* end() { return data_ + N; }
        DUAL const T* end() const { return data_ + N; }

      private:
        T data_[N];
    };

    // overload ostream to print simbi::array_t
    template <typename T, size_type N>
    std::ostream& operator<<(std::ostream& os, const simbi::array_t<T, N>& arr)
    {
        os << "[";
        for (size_type ii = 0; ii < N; ++ii) {
            os << arr[ii];
            if (ii < N - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
}   // namespace simbi

namespace std {
    // 1. Specify that array_t can be decomposed
    template <typename T, size_type N>
    struct tuple_size<simbi::array_t<T, N>>
        : std::integral_constant<size_t, N> {
    };

    // 2. Specify the type of each element
    template <size_t I, typename T, size_type N>
    struct tuple_element<I, simbi::array_t<T, N>> {
        static_assert(I < N, "Index out of bounds");
        using type = T;
    };
}   // namespace std

namespace simbi {
    template <size_t I, typename T, size_type N>
    DUAL T& get(array_t<T, N>& a)
    {
        static_assert(I < N, "Index out of bounds");
        return a[I];
    }

    template <size_t I, typename T, size_type N>
    DUAL const T& get(const array_t<T, N>& a)
    {
        static_assert(I < N, "Index out of bounds");
        return a[I];
    }

    template <size_t I, typename T, size_type N>
    DUAL T&& get(array_t<T, N>&& a)
    {
        static_assert(I < N, "Index out of bounds");
        return std::move(a[I]);
    }
}   // namespace simbi

#endif
