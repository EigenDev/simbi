// =============================================================================
// handle.hpp
//
// reference-counted shared handle with explicit coherency tracking.
// one job: manage shared ownership and track dirty state.
// follows hesi pattern of minimal, focused components.
//
// usage:
//   auto handle = shared_handle_t<float>::make(42.0f);
//   handle.mark_host_dirty();
//   if (handle.needs_device_sync()) sync_to_device(handle);
// =============================================================================

#pragma once

#include <atomic>
#include <utility>

namespace simbi::xpu::mem {

    // =============================================================================
    // control block - data + reference count + coherency metadata
    // =============================================================================

    template <typename T>
    struct control_block_t
    {
        T                data;
        std::atomic<int> ref_count;

        // explicit coherency tracking
        std::atomic<bool> host_dirty   = false;
        std::atomic<bool> device_dirty = false;

        template <typename... Args>
        control_block_t(Args&&... args) : data(std::forward<Args>(args)...), ref_count(1)
        {
        }

        void mark_host_dirty() noexcept
        {
            host_dirty.store(true, std::memory_order_relaxed);
        }

        void mark_device_dirty() noexcept
        {
            device_dirty.store(true, std::memory_order_relaxed);
        }

        void mark_synchronized() noexcept
        {
            host_dirty.store(false, std::memory_order_relaxed);
            device_dirty.store(false, std::memory_order_relaxed);
        }

        bool needs_host_sync() const noexcept
        {
            return device_dirty.load(std::memory_order_relaxed);
        }

        bool needs_device_sync() const noexcept
        {
            return host_dirty.load(std::memory_order_relaxed);
        }
    };

    // =============================================================================
    // shared handle - reference counted pointer with explicit coherency
    // =============================================================================

    template <typename T>
    class shared_handle_t
    {
      private:
        using block_type = control_block_t<T>;
        block_type* cb_  = nullptr;

      public:
        using value_type = T;

        // default constructor
        shared_handle_t() = default;

        // private constructor - use factory methods
        explicit shared_handle_t(block_type* cb) : cb_(cb) {}

        // copy constructor
        shared_handle_t(const shared_handle_t& other) : cb_(other.cb_)
        {
            if (cb_) {
                cb_->ref_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // move constructor
        shared_handle_t(shared_handle_t&& other) noexcept : cb_(other.cb_)
        {
            other.cb_ = nullptr;
        }

        // copy assignment (copy-and-swap idiom)
        shared_handle_t& operator=(const shared_handle_t& other)
        {
            shared_handle_t temp(other);
            swap(temp);
            return *this;
        }

        // move assignment (copy-and-swap idiom)
        shared_handle_t& operator=(shared_handle_t&& other) noexcept
        {
            shared_handle_t temp(std::move(other));
            swap(temp);
            return *this;
        }

        // destructor
        ~shared_handle_t()
        {
            release();
        }

        // data accessors
        T* get() const noexcept
        {
            return cb_ ? &cb_->data : nullptr;
        }

        T* operator->() const noexcept
        {
            return get();
        }

        T& operator*() const noexcept
        {
            return *get();
        }

        explicit operator bool() const noexcept
        {
            return cb_ != nullptr;
        }

        // reference counting
        int use_count() const noexcept
        {
            return cb_ ? cb_->ref_count.load(std::memory_order_relaxed) : 0;
        }

        // coherency operations - explicit and predictable
        void mark_host_dirty() noexcept
        {
            if (cb_) {
                cb_->mark_host_dirty();
            }
        }

        void mark_device_dirty() noexcept
        {
            if (cb_) {
                cb_->mark_device_dirty();
            }
        }

        void mark_synchronized() noexcept
        {
            if (cb_) {
                cb_->mark_synchronized();
            }
        }

        bool needs_host_sync() const noexcept
        {
            return cb_ ? cb_->needs_host_sync() : false;
        }

        bool needs_device_sync() const noexcept
        {
            return cb_ ? cb_->needs_device_sync() : false;
        }

        // factory methods - ensure single allocation
        template <typename... Args>
        static shared_handle_t make(Args&&... args)
        {
            auto* cb = new block_type(std::forward<Args>(args)...);
            return shared_handle_t(cb);
        }

        // construct from existing data
        static shared_handle_t from_data(T&& data)
        {
            auto* cb = new block_type(std::move(data));
            return shared_handle_t(cb);
        }

        // raw control block access (for advanced operations)
        block_type* control_block() const noexcept
        {
            return cb_;
        }

        // swap implementation for copy-and-swap idiom
        void swap(shared_handle_t& other) noexcept
        {
            std::swap(cb_, other.cb_);
        }

      private:
        void release()
        {
            if (cb_ && cb_->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete cb_;
            }
            cb_ = nullptr;
        }
    };

    // =============================================================================
    // factory function
    // =============================================================================

    template <typename T, typename... Args>
    shared_handle_t<T> make_shared(Args&&... args)
    {
        return shared_handle_t<T>::make(std::forward<Args>(args)...);
    }

} // namespace simbi::xpu::mem
