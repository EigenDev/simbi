/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            ndarray.hpp
 *  * @brief           n-dimensional arrays for cpu/gpu manopulation
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
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "build_options.hpp"                       // for global::on_gpu, rea;
#include "core/managers/array_props.hpp"           // for array_properties
#include "core/managers/memory_manager.hpp"        // for memory_manager
#include "core/traits.hpp"                         // for is_maybe
#include "core/types/alias/alias.hpp"              // for uarray
#include "core/types/containers/array_view.hpp"    // for array_view"
#include "core/types/containers/collapsable.hpp"   // for collapsable
#include "core/types/utility/operation_traits.hpp"   // for OperationTraits, PoinwiseOp
#include "util/parallel/exec_policy.hpp"             // for ExecutionPolicy
#include "util/parallel/parallel_for.hpp"            // for parallel_for
#include "util/tools/helpers.hpp"                    // for unravel_index
#include <cassert>                                   // for assert
namespace simbi {

    template <typename T, size_type Dims = 1>
    class ndarray : public array_properties<T, Dims>
    {
      public:
        using value_type =
            typename std::conditional_t<is_maybe_v<T>, get_value_type_t<T>, T>;
        ndarray() = default;
        explicit ndarray(
            std::initializer_list<size_type> dims,
            T fill_value = T()
        )
        {
            size_type size = 1;
            for (auto dim : dims) {
                size *= dim;
            }
            mem_.allocate(size);
            this->shape_   = dims;
            this->strides_ = this->compute_strides(this->shape_);
            fill(fill_value);
        }
        ndarray(const ndarray&)                = default;
        ndarray(ndarray&&) noexcept            = default;
        ndarray& operator=(const ndarray&)     = default;
        ndarray& operator=(ndarray&&) noexcept = default;

        explicit ndarray(std::vector<T>&& data)
        {
            mem_.allocate(data.size());
            std::copy(data.begin(), data.end(), mem_.host_data());
            this->size_     = data.size();
            this->shape_[0] = data.size();
            // fill the remaining dimensions with 1
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 1;
            }
            this->strides_ = this->compute_strides(this->shape_);
            // check values after copy
            for (size_type ii = 0; ii < this->size(); ++ii) {
                assert(mem_[ii] == data[ii]);
            }
            // clear and release the vector
            data.clear();
            data.shrink_to_fit();
            sync_to_device();
        }

        // move assignment from std::vector
        ndarray& operator=(std::vector<T>&& data)
        {
            mem_.allocate(data.size());
            std::copy(data.begin(), data.end(), mem_.host_data());
            this->size_     = data.size();
            this->shape_[0] = data.size();
            // fill the remaining dimensions with 1
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 1;
            }
            this->strides_ = this->compute_strides(this->shape_);
            // check values after copy
            for (size_type ii = 0; ii < this->size(); ++ii) {
                assert(mem_[ii] == data[ii]);
            }
            // clear and release the vector
            data.clear();
            data.shrink_to_fit();
            sync_to_device();
            return *this;
        }

        ndarray(const size_type sz)
        {
            mem_.allocate(sz);
            this->size_     = sz;
            this->shape_[0] = sz;
            // fill the remaining dimensions with 1
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 1;
            }
            this->strides_ = this->compute_strides(this->shape_);
        }

        ndarray(const size_type sz, T fill_value)
        {
            mem_.allocate(sz);
            this->size_     = sz;
            this->shape_[0] = sz;
            // fill the remaining dimensions with 1
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 1;
            }
            this->strides_ = this->compute_strides(this->shape_);
            fill(fill_value);
        }

        void copy_from(const ndarray& source, const ExecutionPolicy<>& policy)
        {
            assert(
                this->size_ == source.size() &&
                "Source and destination arrays must have the same size"
            );
            auto copy_op = [this,
                            source_ptr = source.data()] DEV(size_type idx) {
                mem_[idx] = source_ptr[idx];
            };
            parallel_for(policy, copy_op);
        }

        auto data() -> T* { return mem_.data(); }
        DUAL auto data() const -> const T* { return mem_.data(); }
        auto fill(T value) -> void
        {
            std::fill(mem_.host_data(), mem_.host_data() + this->size(), value);
        }

        template <typename... Indices>
        DUAL T& at(Indices... indices)
        {
            collapsable<Dims> idx{static_cast<size_type>(indices)...};
            size_type offset = 0;

            for (size_type d = 0; d < Dims; ++d) {
                offset += idx[d] * this->strides_[d];
            }
            return access(mem_[offset]);
        }
        template <typename... Indices>
        DUAL T& at(Indices... indices) const
        {
            collapsable<Dims> idx{static_cast<size_type>(indices)...};
            size_type offset = 0;

            for (size_type d = 0; d < Dims; ++d) {
                offset += idx[d] * this->strides_[d];
            }
            return access(mem_[offset]);
        }
        DUAL auto& access(T& val)
        {
            if constexpr (is_maybe_v<T>) {
                return val.value();
            }
            else {
                return val;
            }
        }
        DUAL auto& access(T& val) const
        {
            if constexpr (is_maybe_v<T>) {
                return val.value();
            }
            else {
                return val;
            }
        }

        // access operators
        DUAL value_type& operator[](size_type ii)
        {
            // bounds check
            assert(ii < this->size() && "Index out of bounds");
            return access(mem_[ii]);
        }
        DUAL value_type& operator[](size_type ii) const
        {
            // bounds check
            assert(ii < this->size() && "Index out of bounds");
            return access(mem_[ii]);
        }

        void sync_to_device() { mem_.sync_to_device(); }
        void sync_to_host() { mem_.sync_to_host(); }
        auto& reshape(const collapsable<Dims>& new_shape)
        {
            // Verify total size matches
            assert(
                this->compute_size(new_shape.vals) == this->size_ &&
                "New shape must match total size"
            );

            // Update shape and strides
            this->shape_   = new_shape.vals;
            this->strides_ = this->compute_strides(this->shape_);
            return *this;
        }
        auto& reshape(
            const collapsable<Dims>& new_shape,
            const collapsable<Dims>& new_strides
        )
        {
            // Verify total size matches
            assert(
                this->compute_size(new_shape.vals) == this->size_ &&
                "New shape must match total size"
            );

            // Update shape and strides
            this->shape_   = new_shape.vals;
            this->strides_ = new_strides.vals;
            return *this;
        }
        auto& resize(size_type size, T fill_value = T())
        {
            // save any old date we had if it existed
            std::vector<T> old_data;
            if (this->size_ > 0) {
                old_data.resize(std::min(this->size_, size));
                std::copy(
                    mem_.data(),
                    mem_.data() + old_data.size(),
                    old_data.begin()
                );
            }
            mem_.allocate(size);

            // now we can restore any old date if there was any
            if (!old_data.empty()) {
                std::copy(old_data.begin(), old_data.end(), mem_.data());
            }

            this->size_     = size;
            this->shape_[0] = size;
            // fill the remaining dimensions with 1
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 1;
            }
            this->strides_ = this->compute_strides(this->shape_);
            fill(fill_value);
            return *this;
        }

        void resize_capacity(size_type new_capacity)
        {
            if (new_capacity <= capacity_) {
                return;
            }

            memory_manager<T> new_mem;
            new_mem.allocate(new_capacity);

            // initialize new memory to avoid garbage values
            std::fill(
                new_mem.host_data(),
                new_mem.host_data() + new_capacity,
                T()
            );

            // copy existing data
            if (this->size_ > 0) {
                std::copy(
                    mem_.host_data(),
                    mem_.host_data() + this->size_,
                    new_mem.host_data()
                );
            }

            mem_      = std::move(new_mem);
            capacity_ = new_capacity;

            if constexpr (global::on_gpu) {
                mem_.sync_to_device();
            }
        }

        auto push_back(T value) -> void
        {
            if (this->size_ >= capacity_) {
                // grow capacity
                size_type new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
                resize_capacity(new_capacity);
            }

            // use placement new for proper initialization
            new (mem_.host_data() + this->size_) T(value);

            // update size and shape
            this->size_ += 1;
            this->shape_[0] += 1;
            this->strides_ = this->compute_strides(this->shape_);
        }

        auto push_back_with_sync(T value) -> void
        {
            push_back(value);
            // If using GPU, synchronize just the changed element
            if constexpr (global::on_gpu) {
                mem_.sync_to_device();
            }
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            // Check if we need to grow capacity
            if (this->size_ >= capacity_) {
                // Grow by doubling capacity (common strategy)
                size_type new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
                resize_capacity(new_capacity);
            }

            // Construct the new value in place
            new (mem_.host_data() + this->size_) T(std::forward<Args>(args)...);

            // Increment size
            this->size_ += 1;

            // Update shape - increase first dimension
            this->shape_[0] += 1;

            // Recompute strides (may not be necessary if only size changes)
            this->strides_ = this->compute_strides(this->shape_);

            // If using GPU, synchronize just the changed element
            if constexpr (global::on_gpu) {
                mem_.sync_to_device();
            }
        }

        // emplace method
        template <typename... Args>
        auto emplace_back_with_sync(Args&&... args) -> void
        {
            emplace_back(std::forward<Args>(args)...);
            // If using GPU, synchronize just the changed element
            if constexpr (global::on_gpu) {
                mem_.sync_to_device();
            }
        }

        void reserve(size_type capacity)
        {
            if (capacity > capacity_) {
                resize_capacity(capacity);
            }
        }
        T* host_data() { return mem_.host_data(); }

        T* host_data() const { return mem_.host_data(); }

        DUAL bool empty() const { return this->size_ == 0; }

        DUAL void clear()
        {
            this->size_     = 0;
            this->shape_[0] = 0;
            // fill the remaining dimensions with 0
            for (size_type ii = 1; ii < Dims; ++ii) {
                this->shape_[ii] = 0;
            }
            this->strides_ = this->compute_strides(this->shape_);
        }

        class iterator
        {
          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = typename ndarray::value_type;
            using difference_type   = std::ptrdiff_t;
            using pointer           = value_type*;
            using reference         = value_type&;

            DUAL iterator(ndarray& arr, size_type pos = 0)
                : array_(arr), pos_(pos)
            {
            }

            DUAL reference operator*() { return array_[pos_]; }

            DUAL iterator& operator++()
            {
                ++pos_;
                return *this;
            }

            DUAL iterator operator++(int)
            {
                iterator tmp = *this;
                ++pos_;
                return tmp;
            }

            DUAL bool operator!=(const iterator& other) const
            {
                return pos_ != other.pos_;
            }

            DUAL bool operator==(const iterator& other) const
            {
                return pos_ == other.pos_;
            }

          private:
            ndarray& array_;
            size_type pos_;
        };

        class const_iterator
        {
          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = typename ndarray::value_type;
            using difference_type   = std::ptrdiff_t;
            using pointer           = value_type*;
            using reference         = value_type&;

            DUAL const_iterator(const ndarray& arr, size_type pos = 0)
                : array_(arr), pos_(pos)
            {
            }

            DUAL reference operator*() { return array_[pos_]; }

            DUAL const_iterator& operator++()
            {
                ++pos_;
                return *this;
            }

            DUAL const_iterator operator++(int)
            {
                const_iterator tmp = *this;
                ++pos_;
                return tmp;
            }

            DUAL bool operator!=(const const_iterator& other) const
            {
                return pos_ != other.pos_;
            }

            DUAL bool operator==(const const_iterator& other) const
            {
                return pos_ == other.pos_;
            }

          private:
            const ndarray& array_;
            size_type pos_;
        };

        // Add iterator support
        DUAL iterator begin() { return iterator(*this); }
        DUAL iterator end() { return iterator(*this, this->size_); }
        DUAL auto begin() const { return const_iterator(*this); }
        DUAL auto end() const { return const_iterator(*this, this->size_); }
        DUAL auto cbegin() const { return const_iterator(*this); }
        DUAL auto cend() const { return const_iterator(*this, this->size_); }

        // contraction method for viewing subarrays
        auto contract(const size_type radius) -> array_view<T, Dims>
        {
            uarray<Dims> new_shape;
            uarray<Dims> offsets;

            for (size_type ii = 0; ii < Dims; ++ii) {
                new_shape[ii] = this->shape_[ii] - 2 * radius;
                offsets[ii]   = radius;
            }

            return array_view<T, Dims>(
                *this,
                mem_.data(),
                new_shape,
                this->strides_,
                offsets
            );
        }
        auto contract(const collapsable<Dims>& radii) -> array_view<T, Dims>
        {
            uarray<Dims> new_shape;
            uarray<Dims> offsets;

            for (size_type ii = 0; ii < Dims; ++ii) {
                // Only contract dimensions with non-zero radius
                new_shape[ii] = radii.vals[ii] > 0
                                    ? this->shape_[ii] - 2 * radii.vals[ii]
                                    : this->shape_[ii];

                offsets[ii] =
                    radii.vals[ii];   // Zero for uncontracted dimensions
            }

            return array_view<T, Dims>(
                *this,
                mem_.data(),
                new_shape,
                this->strides_,
                offsets
            );
        }
        auto view(const collapsable<Dims>& ranges) const -> array_view<T, Dims>
        {
            uarray<Dims> offsets;
            for (size_type ii = 0; ii < Dims; ++ii) {
                // Offset is 0 for each dimension being viewed
                offsets[ii] = 0;
            }

            return array_view<T, Dims>(
                *this,
                mem_.data(),      // Start at beginning
                ranges.vals,      // New shape is the ranges
                this->strides_,   // Strides stay the same
                offsets           // Start at 0 offset
            );
        }

        template <typename F>
        void transform(F op, const ExecutionPolicy<>& policy)
        {
            if constexpr (global::on_gpu) {
                mem_.ensure_device_synced();
                parallel_for(policy, [=, this] DEV(size_type ii) {
                    mem_[ii] = op(mem_[ii]);
                });
                policy.synchronize();
            }
            else {
                parallel_for(policy, [=, this](size_type ii) {
                    mem_[ii] = op(mem_[ii]);
                });
            }
        }

        // transform with variadic dependent ndarrays
        template <typename... DependentArrays, typename F>
        void transform(
            F op,
            const ExecutionPolicy<>& policy,
            const DependentArrays&... arrays
        )
        {
            OperationTraits<PointwiseOp>::execute(
                mem_.data(),
                op,
                policy,
                arrays.data()...
            );
        }

        // transform with variadic dependent ndarrays
        // that are mutable
        template <typename... DependentArrays, typename F>
        void transform(
            F op,
            const ExecutionPolicy<>& policy,
            DependentArrays&... arrays
        )
        {
            OperationTraits<PointwiseOp>::execute(
                mem_.data(),
                op,
                policy,
                arrays.data()...
            );
        }

        // transform with no dependent ndarrays
        // but we now track the indices
        template <typename F>
        void transform_with_indices(F op, const ExecutionPolicy<>& policy)
        {
            if constexpr (global::on_gpu) {
                mem_.ensure_device_synced();
                parallel_for(policy, [=, this] DEV(size_type ii) {
                    mem_[ii] = op(mem_[ii], ii);
                });
                policy.synchronize();
            }
            else {
                parallel_for(policy, [=, this](size_type ii) {
                    mem_[ii] = op(mem_[ii], ii);
                });
            }
        }

        // transform with indices and variadic dependent ndarrays
        template <typename... DependentArrays, typename F>
        void transform_with_indices(
            F op,
            const ExecutionPolicy<>& policy,
            const DependentArrays&... arrays
        )
        {
            OperationTraits<PointwiseOpIdx>::execute(
                mem_.data(),
                op,
                policy,
                arrays.data()...
            );
        }

        template <typename U, typename F>
        U reduce(U init, F reduce_op, const ExecutionPolicy<>& policy) const
        {
            if constexpr (global::on_gpu) {
                ndarray<U> result(1, init);
                result.sync_to_device();
                auto result_ptr      = result.data();
                auto arr             = mem_.data();
                const size_type size = this->size_;

                // First pass: each thread block computes a partial reduction
                parallel_for(policy, [=] DEV(size_type idx) {
                    extern __shared__ U shared_data[];

                    // Each thread initializes with its own value
                    U thread_val = reduce_op(init, arr[idx], idx);

                    // Block-level reduction
                    const size_type tid = get_thread_id();
                    // const size_type bid   = get_block_id();
                    const auto block_size = get_threads_per_block();
                    shared_data[tid]      = thread_val;
                    gpu::api::synchronize();

                    // const size_type block_start_idx = bid * block_size;
                    // const size_type block_end_idx =
                    //     std::min(size, (bid + 1) * block_size);
                    // const auto active_threads = block_end_idx -
                    // block_start_idx;

                    // Reduce within block
                    for (size_type s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s && idx + s < size) {
                            shared_data[tid] =
                                my_min(shared_data[tid], shared_data[tid + s]);
                        }
                        gpu::api::synchronize();
                    }

                    // Write block result to global memory
                    if (tid == 0) {
                        gpu::api::atomicMin(&result_ptr[0], shared_data[0]);
                    }
                });

                result.sync_to_host();
                return result[0];
            }
            else {
                std::atomic<U> result(init);
                const size_type batch_size = policy.batch_size;
                const size_type num_batches =
                    policy.get_num_batches(this->size());

                parallel_for(policy, num_batches, [&](size_type bid) {
                    const size_type start = bid * batch_size;
                    const size_type end =
                        std::min(start + batch_size, this->size());

                    // Local reduction: (accumulator, element, index)
                    U local_result = init;
                    for (size_type ii = start; ii < end; ii++) {
                        local_result = reduce_op(local_result, mem_[ii], ii);
                    }

                    // Global merge: (accumulator, element, index)
                    bool success;
                    do {
                        U expected = result.load(std::memory_order_relaxed);
                        success    = result.compare_exchange_weak(
                            expected,
                            reduce_op(expected, mem_[start], start),
                            std::memory_order_release,
                            std::memory_order_relaxed
                        );
                    } while (!success);
                });

                return result.load(std::memory_order_acquire);
            }
        }

        // filter method to apply a function to each element
        template <typename F>
        ndarray<size_type, 1>
        filter_indices(F op, const ExecutionPolicy<>& policy) const
        {
            ndarray<size_type, 1> indices(this->size());

            if constexpr (global::on_gpu) {
                int count = 0;
                // mem_.ensure_device_synced();
                indices.sync_to_device();
                auto indices_ptr = indices.data();
                auto arr         = mem_.data();
                auto count_ptr   = &count;

                parallel_for(policy, [=, this] DEV(size_type idx) {
                    if (op(arr[idx])) {
                        auto current_count =
                            gpu::api::atomicAdd<int>(count_ptr, 1);
                        indices_ptr[current_count] = idx;
                    }
                });

                indices.resize(count);
                indices.sync_to_host();
            }
            else {
                std::atomic<size_type> count{0};
                parallel_for(policy, [&, this](size_type idx) {
                    if (op(mem_[idx])) {
                        size_type current_count =
                            count.fetch_add(1, std::memory_order_relaxed);
                        indices[current_count] = idx;
                    }
                });

                indices.resize(count);
            }

            return indices;
        }

      private:
        memory_manager<T> mem_;
        size_type capacity_ = 0;
    };
}   // namespace simbi
#endif
