// =============================================================================
// Complete ndarray_t System
// =============================================================================
#ifndef SIMBI_NDARRAY_SYSTEM_HPP
#define SIMBI_NDARRAY_SYSTEM_HPP

#include "array_view.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/domain.hpp"
#include "core/base/memory.hpp"
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace simbi::nd {
    using namespace simbi::concepts;
    using namespace simbi::base;
    // =============================================================================
    // Memory-Backed ndarray_t
    // =============================================================================
    template <typename T, size_t Dims = 1>
    class ndarray_t
    {
      private:
        domain_t<Dims> domain_;
        std::unique_ptr<unified_memory_t<T>> memory_;

      public:
        using value_type             = T;
        static constexpr size_t rank = Dims;

        ndarray_t() = default;

        // construction
        ndarray_t(const domain_t<Dims>& domain)
            : domain_(domain),
              memory_(
                  std::make_unique<unified_memory_t<T>>(domain.total_size())
              )
        {
        }

        // from vector
        ndarray_t(const std::vector<T>& vec)
            : memory_(std::make_unique<unified_memory_t<T>>(vec.size()))
        {
            if constexpr (Dims == 1) {
                domain_ = domain_t<Dims>{vec.size()};
            }
            else if constexpr (Dims == 2) {
                domain_ =
                    domain_t<Dims>{vec.size(), 1};   // 1D vector treated as 2D
            }
            else if constexpr (Dims == 3) {
                domain_ = domain_t<Dims>{
                  vec.size(),
                  1,
                  1
                };   // 1D vector treated as 3D
            }
            else {
                static_assert(Dims <= 3, "Only supports up to 3 dimensions");
            }
            std::copy(vec.begin(), vec.end(), memory_->data());
        }

        // factory methods
        static ndarray_t zeros(const domain_t<Dims>& domain)
        {
            auto arr = ndarray_t{domain};
            arr.fill(T{0});
            return arr;
        }

        static ndarray_t ones(const domain_t<Dims>& domain)
        {
            auto arr = ndarray_t{domain};
            arr.fill(T{1});
            return arr;
        }

        // Enhanced memory management methods
        void sync_to_device() { memory_->to_gpu(); }

        void sync_to_host() { memory_->to_cpu(); }

        void ensure_device_synced() { memory_->ensure_gpu_synced(); }

        void ensure_host_synced() { memory_->ensure_cpu_synced(); }

        void mark_host_dirty() { memory_->mark_cpu_dirty(); }

        void mark_device_dirty() { memory_->mark_gpu_dirty(); }

        bool is_host_synced() const { return memory_->is_cpu_valid(); }

        bool is_device_synced() const { return memory_->is_gpu_valid(); }

        // element access (immutable)
        DUAL const T& operator[](const std::array<size_t, Dims>& point) const
        {
            return memory_->data()[domain_.linear_index(point)];
        }

        DUAL T& operator[](const size_t idx) const
        {
            return memory_->data()[idx];
        }

        // convenience accessors
        DUAL const T& operator()(size_t ii) const
            requires(Dims == 1)
        {
            return (*this)[{ii}];
        }

        DUAL const T& operator()(size_t ii, size_t jj) const
            requires(Dims == 2)
        {
            return (*this)[{ii, jj}];
        }

        DUAL const T& operator()(size_t ii, size_t jj, size_t kk) const
            requires(Dims == 3)
        {
            return (*this)[{ii, jj, kk}];
        }

        DUAL auto at(const domain_t<Dims>& space) const
        {
            return memory_->data()[domain_.linear_index(space)];
        }

        // slicing - returns zero-copy view
        array_view_t<const T, Dims> slice(const slice_t<Dims>& slice_spec) const
        {
            return array_view_t<const T, Dims>{
              memory_->data(),
              domain_,
              slice_spec
            };
        }

        // convenient slicing methods
        array_view_t<T, Dims> interior()
        {
            return slice(interior_domain(domain_));
        }

        auto boundary(size_t dim, size_t face)
        {
            return slice(boundary_slice(domain_, dim, face));
        }

        void reserve(size_t new_size)
        {
            if (new_size > memory_->size()) {
                memory_->resize(new_size);
            }
        }

        void resize(const domain_t<Dims>& new_domain)
        {
            if (new_domain.total_size() > memory_->size()) {
                memory_->resize(new_domain.total_size());
            }
            domain_ = new_domain;
        }

        void push_back(const T& value)
        {
            if (domain_.total_size() >= memory_->size()) {
                throw std::runtime_error("Cannot push_back: array is full");
            }
            memory_->data()[domain_.total_size()] = value;
        }

        void push_back_with_sync(const T& value)
        {
            push_back(value);
            ensure_host_synced();   // sync after push_back
        }

        // memory management
        void to_cpu() { memory_->to_cpu(); }
        void to_gpu() { memory_->to_gpu(); }

        T* cpu_data() const { return memory_->cpu_data(); }
        T* gpu_data() const { return memory_->gpu_data(); }
        T* data() const { return memory_->data(); }

        // properties
        const domain_t<Dims>& domain() const { return domain_; }
        size_t size() const { return domain_.total_size(); }

        // utility
        void fill(const T& value)
        {
            T* ptr = memory_->data();
            for (size_t i = 0; i < size(); ++i) {
                ptr[i] = value;
            }
        }

        // initialization from function
        template <ArrayFunction<Dims> Func>
        void initialize_from(const Func& func)
        {
            T* ptr = memory_->data();
            for (size_t i = 0; i < size(); ++i) {
                auto point = domain_.point_from_index(i);
                ptr[i]     = func(point);
            }
        }
    };

    // =============================================================================
    // Functional Operations: The Heart of the System (kinda)
    // =============================================================================
    // map operation - transforms elements in place
    template <typename T, size_t Dims, typename UnaryOp>
    void map(array_view_t<T, Dims> view, UnaryOp op)
    {
        auto domain = view.domain();

        // in real implementation, this would launch GPU kernel if data is on
        // GPU
        if constexpr (Dims == 1) {
            for (size_t i = 0; i < domain.extent(0); ++i) {
                view(i) = op(view(i));
            }
        }
        else if constexpr (Dims == 2) {
            for (size_t j = 0; j < domain.extent(1); ++j) {
                for (size_t i = 0; i < domain.extent(0); ++i) {
                    view(i, j) = op(view(i, j));
                }
            }
        }
        else if constexpr (Dims == 3) {
            for (size_t k = 0; k < domain.extent(2); ++k) {
                for (size_t j = 0; j < domain.extent(1); ++j) {
                    for (size_t i = 0; i < domain.extent(0); ++i) {
                        view(i, j, k) = op(view(i, j, k));
                    }
                }
            }
        }
    }

    // transform operation - creates new array
    template <typename T, size_t Dims, typename UnaryOp>
    auto transform(const array_view_t<T, Dims>& view, UnaryOp op)
    {
        using result_type = decltype(op(std::declval<T>()));
        auto result       = ndarray_t<result_type, Dims>{view.domain()};
        auto result_view  = result.slice(
            slice_t<Dims>{std::array<size_t, Dims>{}, view.domain().extents}
        );

        auto domain = view.domain();
        if constexpr (Dims == 1) {
            for (size_t i = 0; i < domain.extent(0); ++i) {
                result_view(i) = op(view(i));
            }
        }
        else if constexpr (Dims == 2) {
            for (size_t j = 0; j < domain.extent(1); ++j) {
                for (size_t i = 0; i < domain.extent(0); ++i) {
                    result_view(i, j) = op(view(i, j));
                }
            }
        }
        else if constexpr (Dims == 3) {
            for (size_t k = 0; k < domain.extent(2); ++k) {
                for (size_t j = 0; j < domain.extent(1); ++j) {
                    for (size_t i = 0; i < domain.extent(0); ++i) {
                        result_view(i, j, k) = op(view(i, j, k));
                    }
                }
            }
        }

        return result;
    }

    // reduce operation
    template <typename T, size_t Dims, typename BinaryOp>
    T reduce(const array_view_t<T, Dims>& view, BinaryOp op, T init = T{})
    {
        T result    = init;
        auto domain = view.domain();

        if constexpr (Dims == 1) {
            for (size_t i = 0; i < domain.extent(0); ++i) {
                result = op(result, view(i));
            }
        }
        else if constexpr (Dims == 2) {
            for (size_t j = 0; j < domain.extent(1); ++j) {
                for (size_t i = 0; i < domain.extent(0); ++i) {
                    result = op(result, view(i, j));
                }
            }
        }
        else if constexpr (Dims == 3) {
            for (size_t k = 0; k < domain.extent(2); ++k) {
                for (size_t j = 0; j < domain.extent(1); ++j) {
                    for (size_t i = 0; i < domain.extent(0); ++i) {
                        result = op(result, view(i, j, k));
                    }
                }
            }
        }

        return result;
    }

    // =============================================================================
    // Factory Functions and Utilities
    // =============================================================================

    // common reduction operations
    struct max_op {
        template <typename T>
        DUAL T operator()(const T& a, const T& b) const
        {
            return (a > b) ? a : b;
        }
    };

    struct min_op {
        template <typename T>
        DUAL T operator()(const T& a, const T& b) const
        {
            return (a < b) ? a : b;
        }
    };

    struct sum_op {
        template <typename T>
        DUAL T operator()(const T& a, const T& b) const
        {
            return a + b;
        }
    };

    template <typename T>
    using ndarray1d_t = ndarray_t<T, 1>;
    template <typename T>
    using ndarray2d_t = ndarray_t<T, 2>;
    template <typename T>
    using ndarray3d_t = ndarray_t<T, 3>;

}   // namespace simbi::nd

#endif   // NDARRAY_SYSTEM_HPP
