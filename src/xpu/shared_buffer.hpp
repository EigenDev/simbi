// =============================================================================
// shared_buffer.hpp
//
// phase 2 spec-compliant shared buffer implementation for xpu framework.
// provides reference-counted shared buffer with memory coherency preservation,
// preserving hesi semantics while using clean xpu memory space abstractions.
//
// design principles:
//   - reference counting for shared ownership (hesi semantics)
//   - memory coherency preservation across spaces
//   - view integration (unchanged api from hesi)
//   - async staging operations with tokens
//   - raii resource management
//
// usage:
//   shared_buffer_t<float, unified_memory> buffer(1000);
//   auto view = buffer.view();
//   auto staged = buffer.stage_to<device_memory>(executor);
// =============================================================================

#pragma once

#include "buffer_ops.hpp"
#include "detail/control_block.hpp"
#include "memory_space.hpp"
#include "view.hpp"

#include <memory>
#include <utility>

namespace xpu {

    // forward declarations
    template <execution_space ExecutionSpace>
    class executor_t;

    template <execution_space ExecutionSpace>
    class token_t;

    // =============================================================================
    // shared buffer implementation (preserving hesi reference counting semantics)
    // =============================================================================

    template <typename T, memory_space MemorySpace = unified_memory>
    class shared_buffer_t
    {
      public:
        using value_type         = T;
        using memory_space_type  = MemorySpace;
        using size_type          = std::size_t;
        using pointer            = T*;
        using const_pointer      = const T*;
        using reference          = T&;
        using const_reference    = const T&;
        using control_block_type = detail::control_block_t<T, MemorySpace>;

      private:
        control_block_type* cb_ = nullptr;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        // default constructor (empty buffer)
        shared_buffer_t() = default;

        // construct with size (default-initialized elements)
        explicit shared_buffer_t(size_type size) : cb_(new control_block_type(size)) {}

        // construct with size and initial value
        template <typename... Args>
        shared_buffer_t(size_type size, Args&&... init_args)
            : cb_(new control_block_type(size, std::forward<Args>(init_args)...))
        {
        }

        // construct from existing memory (non-owning)
        shared_buffer_t(pointer data, size_type size)
        {
            // create control block that doesn't own the memory
            // note: this is a simplified implementation
            // production version would need more sophisticated non-owning semantics
            cb_ = new control_block_type(size);
            std::copy_n(data, size, cb_->data());
        }

        // copy constructor (preserves hesi shallow copy semantics)
        shared_buffer_t(const shared_buffer_t& other) : cb_(other.cb_)
        {
            if (cb_) {
                cb_->add_ref();
            }
        }

        // copy assignment (preserves hesi shallow copy semantics)
        shared_buffer_t& operator=(const shared_buffer_t& other)
        {
            if (cb_ != other.cb_) {
                release();
                cb_ = other.cb_;
                if (cb_) {
                    cb_->add_ref();
                }
            }
            return *this;
        }

        // move constructor
        shared_buffer_t(shared_buffer_t&& other) noexcept : cb_(std::exchange(other.cb_, nullptr))
        {
        }

        // move assignment
        shared_buffer_t& operator=(shared_buffer_t&& other) noexcept
        {
            if (this != &other) {
                release();
                cb_ = std::exchange(other.cb_, nullptr);
            }
            return *this;
        }

        // destructor
        ~shared_buffer_t()
        {
            release();
        }

        // =============================================================================
        // deep copy operations (preserves hesi semantics)
        // =============================================================================

        // deep copy when needed (preserves hesi clone semantics)
        shared_buffer_t clone() const
        {
            if (!cb_) {
                return shared_buffer_t{};
            }

            shared_buffer_t result(cb_->size());
            buffer_ops::copy_buffer(*this, result);
            return result;
        }

        // copy to different memory space
        template <memory_space DstSpace>
        shared_buffer_t<T, DstSpace> copy_to() const
        {
            if (!cb_) {
                return shared_buffer_t<T, DstSpace>{};
            }

            auto result = shared_buffer_t<T, DstSpace>(cb_->size());
            buffer_ops::copy_buffer(*this, result);
            return result;
        }

        // =============================================================================
        // async staging operations (preserves hesi coherency)
        // =============================================================================

        template <memory_space DstSpace, execution_space ExecutionSpace>
        std::pair<shared_buffer_t<T, DstSpace>, token_t<ExecutionSpace>>
        stage_to(executor_t<ExecutionSpace>& exec) const
        {
            if (!cb_) {
                return std::make_pair(
                    shared_buffer_t<T, DstSpace>{},
                    make_ready_token<ExecutionSpace>()
                );
            }

            // coherency management - set pending transfer without nested locking
            auto dst_buffer = shared_buffer_t<T, DstSpace>(cb_->size());

            // note: set_pending_transfer acquires its own lock, so don't nest locks here
            cb_->set_pending_transfer(std::this_thread::get_id());

            auto token = buffer_ops::copy_async(*this, dst_buffer, exec);

            // clear pending transfer when token completes
            // note: in production, would use token callback
            return std::make_pair(std::move(dst_buffer), std::move(token));
        }

        template <memory_space DstSpace>
        shared_buffer_t<T, DstSpace> stage_to_sync() const
        {
            if (!cb_) {
                return shared_buffer_t<T, DstSpace>{};
            }

            return buffer_ops::stage_buffer_to_sync<DstSpace>(*this);
        }

        // =============================================================================
        // data access (hesi compatibility)
        // =============================================================================

        pointer data() const noexcept
        {
            return cb_ ? cb_->data() : nullptr;
        }

        size_type size() const noexcept
        {
            return cb_ ? cb_->size() : 0;
        }

        size_type capacity() const noexcept
        {
            return cb_ ? cb_->capacity() : 0;
        }

        bool empty() const noexcept
        {
            return size() == 0;
        }

        // hesi view integration (unchanged api)
        view_t<T, 1> view() const
        {
            if (!cb_) {
                return view_t<T, 1>{nullptr, {0}, {0}, {1}};
            }
            return view_t<T, 1>{data(), {size()}, {0}, {1}};
        }

        // multi-dimensional view support
        template <std::size_t Rank>
        view_t<T, Rank> view(const std::array<size_type, Rank>& extents) const
        {
            static_assert(Rank > 0, "view rank must be positive");

            if (!cb_) {
                std::array<size_type, Rank> zero_extents{};
                std::array<size_type, Rank> zero_offsets{};
                std::array<size_type, Rank> unit_strides{};
                unit_strides.fill(1);
                return view_t<T, Rank>{nullptr, zero_extents, zero_offsets, unit_strides};
            }

            // verify total size matches
            size_type total_elements = 1;
            for (size_type extent : extents) {
                total_elements *= extent;
            }

            if (total_elements > size()) {
                throw std::out_of_range("view extents exceed buffer size");
            }

            // compute strides
            std::array<size_type, Rank> strides{};
            strides[Rank - 1] = 1;
            for (int ii = static_cast<int>(Rank) - 2; ii >= 0; --ii) {
                strides[ii] = strides[ii + 1] * extents[ii + 1];
            }

            std::array<size_type, Rank> zero_offsets{};
            return view_t<T, Rank>{data(), extents, zero_offsets, strides};
        }

        // =============================================================================
        // element access
        // =============================================================================

        reference operator[](size_type index) noexcept
        {
            return cb_->operator[](index);
        }

        const_reference operator[](size_type index) const noexcept
        {
            return cb_->operator[](index);
        }

        reference at(size_type index)
        {
            if (!cb_) {
                throw std::out_of_range("shared_buffer_t::at: null buffer");
            }
            return cb_->at(index);
        }

        const_reference at(size_type index) const
        {
            if (!cb_) {
                throw std::out_of_range("shared_buffer_t::at: null buffer");
            }
            return cb_->at(index);
        }

        reference front()
        {
            return at(0);
        }

        const_reference front() const
        {
            return at(0);
        }

        reference back()
        {
            return at(size() - 1);
        }

        const_reference back() const
        {
            return at(size() - 1);
        }

        // =============================================================================
        // buffer operations integration
        // =============================================================================

        // synchronous operations
        void fill(const T& value)
        {
            buffer_ops::fill_buffer(*this, value);
        }

        void zero()
        {
            buffer_ops::zero_buffer(*this);
        }

        template <typename Func>
        void transform(Func&& func)
        {
            buffer_ops::transform_buffer(*this, std::forward<Func>(func));
        }

        // reduction operations
        template <typename BinaryOp>
        T reduce(T init_value, BinaryOp&& op) const
        {
            return buffer_ops::reduce_buffer(*this, init_value, std::forward<BinaryOp>(op));
        }

        T sum() const
        {
            return buffer_ops::sum_buffer(*this);
        }

        T max() const
        {
            return buffer_ops::max_buffer(*this);
        }

        T min() const
        {
            return buffer_ops::min_buffer(*this);
        }

        // async operations
        template <execution_space ExecutionSpace>
        token_t<ExecutionSpace> fill_async(const T& value, executor_t<ExecutionSpace>& exec)
        {
            return buffer_ops::fill_async(*this, value, exec);
        }

        template <execution_space ExecutionSpace>
        token_t<ExecutionSpace> zero_async(executor_t<ExecutionSpace>& exec)
        {
            return buffer_ops::zero_async(*this, exec);
        }

        template <execution_space ExecutionSpace, typename Func>
        token_t<ExecutionSpace> transform_async(Func&& func, executor_t<ExecutionSpace>& exec)
        {
            return buffer_ops::transform_async(*this, std::forward<Func>(func), exec);
        }

        // =============================================================================
        // memory management and properties
        // =============================================================================

        // resize (only if buffer is unique)
        void resize(size_type new_size)
        {
            if (!cb_) {
                *this = shared_buffer_t(new_size);
                return;
            }

            cb_->resize(new_size);
        }

        void reserve(size_type new_capacity)
        {
            if (!cb_) {
                return;
            }

            cb_->reserve(new_capacity);
        }

        // reference counting information
        int use_count() const noexcept
        {
            return cb_ ? cb_->ref_count() : 0;
        }

        bool unique() const noexcept
        {
            return cb_ ? cb_->unique() : false;
        }

        void clear()
        {
            resize(0);
        }

        // memory space properties
        static constexpr bool is_host_accessible()
        {
            return control_block_type::is_host_accessible();
        }

        static constexpr bool is_device_accessible()
        {
            return control_block_type::is_device_accessible();
        }

        static constexpr std::string_view memory_space_name()
        {
            return MemorySpace::name();
        }

        // =============================================================================
        // coherency management
        // =============================================================================

        void mark_dirty() const
        {
            if (cb_) {
                cb_->mark_dirty();
            }
        }

        void mark_clean() const
        {
            if (cb_) {
                cb_->mark_clean();
            }
        }

        bool is_dirty() const noexcept
        {
            return cb_ ? cb_->is_dirty() : false;
        }

        bool has_pending_transfer() const
        {
            return cb_ ? cb_->has_pending_transfer() : false;
        }

        // =============================================================================
        // comparison operations
        // =============================================================================

        template <memory_space OtherSpace>
        bool operator==(const shared_buffer_t<T, OtherSpace>& other) const
        {
            return buffer_ops::buffers_equal(*this, other);
        }

        template <memory_space OtherSpace>
        bool operator!=(const shared_buffer_t<T, OtherSpace>& other) const
        {
            return !(*this == other);
        }

        // =============================================================================
        // utility
        // =============================================================================

        explicit operator bool() const noexcept
        {
            return cb_ != nullptr && !empty();
        }

        void swap(shared_buffer_t& other) noexcept
        {
            std::swap(cb_, other.cb_);
        }

        void reset()
        {
            release();
        }

        // get buffer statistics
        auto stats() const
        {
            return buffer_ops::compute_buffer_stats(*this);
        }

      private:
        void release()
        {
            if (cb_ && cb_->release()) {
                delete cb_;
                cb_ = nullptr;
            }
        }
    };

    // =============================================================================
    // factory functions
    // =============================================================================

    template <memory_space MemorySpace, typename T, typename... Args>
    shared_buffer_t<T, MemorySpace> make_buffer(std::size_t size, Args&&... args)
    {
        return shared_buffer_t<T, MemorySpace>(size, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    shared_buffer_t<T, unified_memory> make_unified_buffer(std::size_t size, Args&&... args)
    {
        return shared_buffer_t<T, unified_memory>(size, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    shared_buffer_t<T, host_memory> make_host_buffer(std::size_t size, Args&&... args)
    {
        return shared_buffer_t<T, host_memory>(size, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    shared_buffer_t<T, device_memory> make_device_buffer(std::size_t size, Args&&... args)
    {
        return shared_buffer_t<T, device_memory>(size, std::forward<Args>(args)...);
    }

    // =============================================================================
    // convenience aliases
    // =============================================================================

    template <typename T>
    using host_shared_buffer_t = shared_buffer_t<T, host_memory>;

    template <typename T>
    using device_shared_buffer_t = shared_buffer_t<T, device_memory>;

    template <typename T>
    using unified_shared_buffer_t = shared_buffer_t<T, unified_memory>;

    // =============================================================================
    // factory functions
    // =============================================================================

    template <typename T>
    auto make_host_buffer(std::size_t size)
    {
        return shared_buffer_t<T, host_memory>(size);
    }

    template <typename T>
    auto make_device_buffer(std::size_t size)
    {
        return shared_buffer_t<T, device_memory>(size);
    }

    template <typename T>
    auto make_unified_buffer(std::size_t size)
    {
        return shared_buffer_t<T, unified_memory>(size);
    }

    template <memory_space MemorySpace, typename T>
    auto make_shared_buffer(std::size_t size)
    {
        return shared_buffer_t<T, MemorySpace>(size);
    }

    // =============================================================================
    // free functions for buffer operations
    // =============================================================================

    template <typename T, memory_space MemorySpace>
    void swap(shared_buffer_t<T, MemorySpace>& lhs, shared_buffer_t<T, MemorySpace>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

} // namespace xpu
