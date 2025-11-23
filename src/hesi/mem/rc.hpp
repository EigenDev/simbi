#ifndef HET_MEM_RC_HPP
#define HET_MEM_RC_HPP

#include <atomic>
#include <cassert>
#include <utility>

namespace simbi::het::mem {

    // a combined storage for data and reference count
    // ensures single allocation for the handle + metadata
    template <typename T>
    struct control_block_t {
        T data;
        std::atomic<int> count;

        template <typename... Args>
        control_block_t(Args&&... args)
            : data(std::forward<Args>(args)...), count(1)
        {
        }
    };

    // a minimalist shared handle
    // behaves like std::shared_ptr but strictly for this architecture
    template <typename T>
    struct handle_t {
        using block_type = control_block_t<T>;

        block_type* cb_ = nullptr;

        // default constructor
        handle_t() = default;

        // construction via factory (preferred)
        // takes ownership of the allocated control block
        explicit handle_t(block_type* cb) : cb_(cb) {}

        // copy constructor (shallow copy, increments ref)
        handle_t(const handle_t& other) : cb_(other.cb_)
        {
            if (cb_) {
                cb_->count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // move constructor (steals ref)
        handle_t(handle_t&& other) noexcept : cb_(other.cb_)
        {
            other.cb_ = nullptr;
        }

        // copy assignment
        handle_t& operator=(const handle_t& other)
        {
            if (this != &other) {
                release();   // drop current
                cb_ = other.cb_;
                if (cb_) {
                    cb_->count.fetch_add(1, std::memory_order_acq_rel);
                }
            }
            return *this;
        }

        // move assignment
        handle_t& operator=(handle_t&& other) noexcept
        {
            if (this != &other) {
                release();
                cb_       = other.cb_;
                other.cb_ = nullptr;
            }
            return *this;
        }

        // destructor
        ~handle_t() { release(); }

        // accessors
        T* get() const { return cb_ ? &cb_->data : nullptr; }
        T* operator->() const { return get(); }
        T& operator*() const { return *get(); }

        explicit operator bool() const { return cb_ != nullptr; }

        int use_count() const
        {
            return cb_ ? cb_->count.load(std::memory_order_relaxed) : 0;
        }

        // factory function to ensure single allocation
        template <typename... Args>
        static handle_t make(Args&&... args)
        {
            auto* cb = new block_type(std::forward<Args>(args)...);
            return handle_t(cb);
        }

      private:
        void release()
        {
            if (cb_) {
                // fetch_sub returns the value BEFORE decrement
                // if it was 1, it is now 0, so we delete
                if (cb_->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    delete cb_;
                }
                cb_ = nullptr;
            }
        }
    };

}   // namespace simbi::het::mem

#endif   // HETERO_MEM_RC_HPP
