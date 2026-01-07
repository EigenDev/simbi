// =============================================================================
// control_block.hpp
//
// reference counting control block for shared_buffer_t implementation.
// provides atomic reference counting, memory management, and coherency
// control for phase 2 shared buffer semantics.
//
// design principles:
//   - atomic reference counting for thread safety
//   - raii memory management with custom allocators
//   - coherency tracking for cross-space transfers
//   - hesi-compatible semantics preservation
//
// usage:
//   control_block_t<float, unified_memory> cb(1000);
//   cb.add_ref();
//   cb.release();
// =============================================================================

#pragma once

#include "../execution_space.hpp"
#include "../memory_space.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>

namespace simbi::xpu::detail {

    // forward declaration for token dependency
    template <execution_space ExecutionSpace>
    class token_wrapper_t;

    // =============================================================================
    // control block implementation
    // =============================================================================

    template <typename T, memory_space MemorySpace>
    class control_block_t
    {
      public:
        using value_type        = T;
        using memory_space_type = MemorySpace;
        using size_type         = std::size_t;
        using pointer           = T*;
        using const_pointer     = const T*;

      private:
        std::atomic<int>          ref_count_;
        pointer                   data_;
        size_type                 size_;
        size_type                 capacity_;
        mutable std::mutex        coherency_mutex_;
        mutable std::atomic<bool> coherency_dirty_{false};

        // pending transfer tracking for coherency
        mutable std::optional<std::thread::id> pending_transfer_thread_;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        explicit control_block_t(size_type size)
            : ref_count_(1), data_(nullptr), size_(size), capacity_(size)
        {
            allocate_storage();
            default_initialize();
        }

        template <typename... Args>
        control_block_t(size_type size, Args&&... init_args)
            : ref_count_(1), data_(nullptr), size_(size), capacity_(size)
        {
            allocate_storage();
            initialize_with_args(std::forward<Args>(init_args)...);
        }

        ~control_block_t()
        {
            destroy_elements();
            deallocate_storage();
        }

        // no copy or move - managed by shared_buffer_t
        control_block_t(const control_block_t&)            = delete;
        control_block_t& operator=(const control_block_t&) = delete;
        control_block_t(control_block_t&&)                 = delete;
        control_block_t& operator=(control_block_t&&)      = delete;

        // =============================================================================
        // reference counting
        // =============================================================================

        void add_ref() noexcept
        {
            ref_count_.fetch_add(1, std::memory_order_relaxed);
        }

        bool release() noexcept
        {
            // returns true if this was the last reference
            const int old_count = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
            return old_count == 1;
        }

        int ref_count() const noexcept
        {
            return ref_count_.load(std::memory_order_relaxed);
        }

        bool unique() const noexcept
        {
            return ref_count() == 1;
        }

        // =============================================================================
        // data access
        // =============================================================================

        pointer data() noexcept
        {
            mark_dirty();
            return data_;
        }

        const_pointer data() const noexcept
        {
            return data_;
        }

        size_type size() const noexcept
        {
            return size_;
        }

        size_type capacity() const noexcept
        {
            return capacity_;
        }

        bool empty() const noexcept
        {
            return size_ == 0;
        }

        // =============================================================================
        // element access with bounds checking
        // =============================================================================

        T& operator[](size_type index) noexcept
        {
            mark_dirty();
            return data_[index];
        }

        const T& operator[](size_type index) const noexcept
        {
            return data_[index];
        }

        T& at(size_type index)
        {
            if (index >= size_) {
                throw std::out_of_range("control_block_t::at: index out of range");
            }
            mark_dirty();
            return data_[index];
        }

        const T& at(size_type index) const
        {
            if (index >= size_) {
                throw std::out_of_range("control_block_t::at: index out of range");
            }
            return data_[index];
        }

        // =============================================================================
        // coherency management
        // =============================================================================

        void mark_dirty() noexcept
        {
            coherency_dirty_.store(true, std::memory_order_relaxed);
        }

        void mark_clean() noexcept
        {
            coherency_dirty_.store(false, std::memory_order_relaxed);
        }

        bool is_dirty() const noexcept
        {
            return coherency_dirty_.load(std::memory_order_relaxed);
        }

        // lock for coherency operations (staging, transfers)
        std::unique_lock<std::mutex> lock_coherency() const
        {
            return std::unique_lock<std::mutex>{coherency_mutex_};
        }

        // track pending transfers for coherency
        void set_pending_transfer(std::thread::id thread_id) const
        {
            std::lock_guard lock{coherency_mutex_};
            pending_transfer_thread_ = thread_id;
        }

        void clear_pending_transfer() const
        {
            std::lock_guard lock{coherency_mutex_};
            pending_transfer_thread_.reset();
        }

        bool has_pending_transfer() const
        {
            std::lock_guard lock{coherency_mutex_};
            return pending_transfer_thread_.has_value();
        }

        std::optional<std::thread::id> pending_transfer_thread() const
        {
            std::lock_guard lock{coherency_mutex_};
            return pending_transfer_thread_;
        }

        // =============================================================================
        // memory space information
        // =============================================================================

        static constexpr bool is_host_accessible()
        {
            if constexpr (std::is_same_v<MemorySpace, host_memory>) {
                return true;
            }
            else if constexpr (std::is_same_v<MemorySpace, unified_memory>) {
                return true;
            }
            else {
                return false;
            }
        }

        static constexpr bool is_device_accessible()
        {
            if constexpr (std::is_same_v<MemorySpace, device_memory>) {
                return true;
            }
            else if constexpr (std::is_same_v<MemorySpace, unified_memory>) {
                return true;
            }
            else {
                return false;
            }
        }

        static constexpr std::string_view memory_space_name()
        {
            return MemorySpace::name();
        }

        // =============================================================================
        // resizing (if buffer is unique)
        // =============================================================================

        void resize(size_type new_size)
        {
            if (!unique()) {
                throw std::runtime_error("cannot resize shared control block");
            }

            if (new_size <= capacity_) {
                // shrink in place
                if (new_size < size_) {
                    std::destroy(data_ + new_size, data_ + size_);
                }
                else if (new_size > size_) {
                    std::uninitialized_default_construct(data_ + size_, data_ + new_size);
                }
                size_ = new_size;
            }
            else {
                // need to reallocate
                reallocate(new_size);
            }
            mark_dirty();
        }

        void reserve(size_type new_capacity)
        {
            if (!unique()) {
                throw std::runtime_error("cannot reserve shared control block");
            }

            if (new_capacity > capacity_) {
                reallocate(new_capacity, size_); // preserve size
            }
        }

      private:
        void allocate_storage()
        {
            if (size_ > 0) {
                data_ = static_cast<pointer>(MemorySpace::allocate(size_ * sizeof(T)));
                if (!data_) {
                    throw std::bad_alloc{};
                }
            }
        }

        void deallocate_storage()
        {
            if (data_) {
                MemorySpace::deallocate(data_, size_ * sizeof(T));
                data_ = nullptr;
            }
        }

        void default_initialize()
        {
            if (data_ && size_ > 0) {
                std::uninitialized_default_construct_n(data_, size_);
            }
        }

        template <typename... Args>
        void initialize_with_args(Args&&... args)
        {
            if (data_ && size_ > 0) {
                // construct all elements with the same arguments
                for (size_type ii = 0; ii < size_; ++ii) {
                    new (data_ + ii) T{args...}; // copy args for each element
                }
            }
        }

        void destroy_elements()
        {
            if (data_ && size_ > 0) {
                std::destroy_n(data_, size_);
            }
        }

        void reallocate(size_type new_capacity, size_type preserve_size = 0)
        {
            if (preserve_size == 0) {
                preserve_size = size_;
            }

            // allocate new storage
            auto new_data = static_cast<pointer>(MemorySpace::allocate(new_capacity * sizeof(T)));
            if (!new_data) {
                throw std::bad_alloc{};
            }

            // move existing elements
            if (data_ && preserve_size > 0) {
                const size_type copy_count = std::min(preserve_size, size_);
                std::uninitialized_move_n(data_, copy_count, new_data);
                std::destroy_n(data_, size_); // destroy old elements
            }

            // deallocate old storage
            deallocate_storage();

            // update state
            data_     = new_data;
            size_     = preserve_size;
            capacity_ = new_capacity;

            // initialize new elements if growing
            if (preserve_size < new_capacity && preserve_size == size_) {
                std::uninitialized_default_construct(data_ + size_, data_ + new_capacity);
                size_ = new_capacity;
            }
        }
    };

    // =============================================================================
    // factory functions
    // =============================================================================

    template <typename T, memory_space MemorySpace, typename... Args>
    std::unique_ptr<control_block_t<T, MemorySpace>>
    make_control_block(std::size_t size, Args&&... args)
    {
        return std::make_unique<control_block_t<T, MemorySpace>>(size, std::forward<Args>(args)...);
    }

    // =============================================================================
    // type traits
    // =============================================================================

    template <typename T>
    struct is_control_block : std::false_type
    {
    };

    template <typename T, memory_space MemorySpace>
    struct is_control_block<control_block_t<T, MemorySpace>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_control_block_v = is_control_block<T>::value;

} // namespace simbi::xpu::detail
