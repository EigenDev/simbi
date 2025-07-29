#ifndef SIMBI_ARRAY_VIEW_HPP
#define SIMBI_ARRAY_VIEW_HPP

#include "config.hpp"
#include "core/base/domain.hpp"
#include <array>
#include <cstddef>

namespace simbi::nd {
    using namespace simbi::base;
    // =============================================================================
    // Array View: Zero-Copy Sliced Access
    // =============================================================================

    template <typename T, std::uint64_t Dims>
    class array_view_t
    {
      private:
        T* data_;
        domain_t<Dims> parent_domain_;
        slice_t<Dims> slice_;

      public:
        array_view_t(
            T* data,
            const domain_t<Dims>& parent_domain,
            const slice_t<Dims>& slice
        )
            : data_(data), parent_domain_(parent_domain), slice_(slice)
        {
        }

        DUAL T& operator[](const std::array<size_t, Dims>& local_point)
        {
            std::array<size_t, Dims> global_point;
            for (size_t i = 0; i < Dims; ++i) {
                global_point[i] =
                    slice_.start[i] + local_point[i] * slice_.stride[i];
            }
            return data_[parent_domain_.linear_index(global_point)];
        }

        DUAL const T&
        operator[](const std::array<size_t, Dims>& local_point) const
        {
            std::array<size_t, Dims> global_point;
            for (size_t i = 0; i < Dims; ++i) {
                global_point[i] =
                    slice_.start[i] + local_point[i] * slice_.stride[i];
            }
            return data_[parent_domain_.linear_index(global_point)];
        }

        // convenience for different dimensions
        DUAL T& operator()(size_t i)
            requires(Dims == 1)
        {
            return (*this)[{i}];
        }

        DUAL T& operator()(size_t i, size_t j)
            requires(Dims == 2)
        {
            return (*this)[{i, j}];
        }

        DUAL T& operator()(size_t i, size_t j, size_t k)
            requires(Dims == 3)
        {
            return (*this)[{i, j, k}];
        }

        DUAL const T& operator()(size_t i) const
            requires(Dims == 1)
        {
            return (*this)[{i}];
        }

        DUAL const T& operator()(size_t i, size_t j) const
            requires(Dims == 2)
        {
            return (*this)[{i, j}];
        }

        DUAL const T& operator()(size_t i, size_t j, size_t k) const
            requires(Dims == 3)
        {
            return (*this)[{i, j, k}];
        }

        T* data() { return data_; }
        const T* data() const { return data_; }

        domain_t<Dims> domain() const
        {
            return domain_t<Dims>{slice_.extents()};
        }
        slice_t<Dims> slice() const { return slice_; }
        size_t size() const { return slice_.total_size(); }
    };
}   // namespace simbi::nd
#endif
