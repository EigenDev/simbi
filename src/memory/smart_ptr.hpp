#ifndef SMART_PTR_HPP
#define SMART_PTR_HPP

#include "config.hpp"

#include <atomic>
#include <type_traits>
#include <utility>

namespace simbi::mem {

    // forward declarations
    template <typename T>
    class weak_ptr;

    /**
     * default_delete - lightweight deleter for array types
     *
     * srp: provide default deletion strategy without overhead
     * zero cost abstraction - compiles to direct delete[] call
     */
    template <typename T>
    struct default_delete {
        constexpr default_delete() noexcept = default;

        template <typename U>
        constexpr default_delete(const default_delete<U>&) noexcept
            requires std::is_convertible_v<U*, T*>
        {
        }

        void operator()(T* ptr) const noexcept
        {
            static_assert(sizeof(T), "cannot delete incomplete type");
            delete[] ptr;
        }
    };

    /**
     * unique_ptr - device-callable unique ownership
     *
     * srp: exclusive ownership of memory with device-safe access
     * zero overhead in device code - compiles to raw pointer
     */
    template <typename T, typename Deleter = default_delete<T>>
    class unique_ptr
    {
        static_assert(
            !std::is_rvalue_reference_v<Deleter>,
            "deleter cannot be rvalue reference"
        );

        T* ptr_;
        [[no_unique_address]] Deleter deleter_;

      public:
        using element_type = T;
        using deleter_type = Deleter;
        using pointer      = T*;

        // construction
        constexpr unique_ptr() noexcept : ptr_(nullptr), deleter_() {}

        constexpr unique_ptr(std::nullptr_t) noexcept
            : ptr_(nullptr), deleter_()
        {
        }

        explicit unique_ptr(pointer ptr) noexcept : ptr_(ptr), deleter_() {}

        unique_ptr(pointer ptr, const Deleter& deleter) noexcept
            : ptr_(ptr), deleter_(deleter)
        {
        }

        unique_ptr(pointer ptr, Deleter&& deleter) noexcept
            : ptr_(ptr), deleter_(std::move(deleter))
        {
        }

        // move-only semantics
        unique_ptr(const unique_ptr&)            = delete;
        unique_ptr& operator=(const unique_ptr&) = delete;

        unique_ptr(unique_ptr&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr)),
              deleter_(std::move(other.deleter_))
        {
        }

        template <typename U, typename E>
        unique_ptr(unique_ptr<U, E>&& other) noexcept
            requires std::is_convertible_v<
                         typename unique_ptr<U, E>::pointer,
                         pointer> &&
                         std::is_convertible_v<E, Deleter>
            : ptr_(other.release()), deleter_(std::move(other.get_deleter()))
        {
        }

        unique_ptr& operator=(unique_ptr&& other) noexcept
        {
            if (this != &other) {
                reset();
                ptr_     = std::exchange(other.ptr_, nullptr);
                deleter_ = std::move(other.deleter_);
            }
            return *this;
        }

        template <typename U, typename E>
        unique_ptr& operator=(unique_ptr<U, E>&& other) noexcept
            requires std::is_convertible_v<
                         typename unique_ptr<U, E>::pointer,
                         pointer> &&
                     std::is_assignable_v<Deleter&, E&&>
        {
            reset(other.release());
            deleter_ = std::move(other.get_deleter());
            return *this;
        }

        unique_ptr& operator=(std::nullptr_t) noexcept
        {
            reset();
            return *this;
        }

        ~unique_ptr()
        {
            if (ptr_) {
                deleter_(ptr_);
            }
        }

        // device-callable access (zero overhead)
        DUAL pointer get() const noexcept { return ptr_; }
        DUAL T& operator*() const noexcept { return *ptr_; }
        DUAL pointer operator->() const noexcept { return ptr_; }
        DUAL T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
        DUAL explicit operator bool() const noexcept { return ptr_ != nullptr; }

        // host-only management
        pointer release() noexcept { return std::exchange(ptr_, nullptr); }

        void reset(pointer ptr = nullptr) noexcept
        {
            pointer old = std::exchange(ptr_, ptr);
            if (old) {
                deleter_(old);
            }
        }

        void swap(unique_ptr& other) noexcept
        {
            std::swap(ptr_, other.ptr_);
            std::swap(deleter_, other.deleter_);
        }

        Deleter& get_deleter() noexcept { return deleter_; }
        const Deleter& get_deleter() const noexcept { return deleter_; }
    };

    /**
     * control_block - shared ownership metadata with strong exception safety
     *
     * srp: manage reference counting and cleanup for shared pointers
     * thread-safe ref counting, exception-safe construction
     */
    template <typename T, typename Deleter>
    class control_block
    {
        std::atomic<std::size_t> shared_count_;
        std::atomic<std::size_t> weak_count_;
        T* ptr_;
        [[no_unique_address]] Deleter deleter_;

      public:
        explicit control_block(T* ptr, Deleter deleter) noexcept
            : shared_count_(1),
              weak_count_(1),
              ptr_(ptr),
              deleter_(std::move(deleter))
        {
        }

        // non-copyable, non-movable
        control_block(const control_block&)            = delete;
        control_block& operator=(const control_block&) = delete;

        void add_shared() noexcept
        {
            shared_count_.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_shared() noexcept
        {
            if (shared_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // last shared reference - destroy object
                destroy_object();
                release_weak();
                return true;
            }
            return false;
        }

        void add_weak() noexcept
        {
            weak_count_.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_weak() noexcept
        {
            if (weak_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // last weak reference - destroy control block
                delete this;
                return true;
            }
            return false;
        }

        std::size_t shared_count() const noexcept
        {
            return shared_count_.load(std::memory_order_acquire);
        }

        std::size_t weak_count() const noexcept
        {
            return weak_count_.load(std::memory_order_acquire) -
                   1;   // subtract the shared count
        }

        bool expired() const noexcept { return shared_count() == 0; }

        T* get() const noexcept { return ptr_; }

        // attempt to lock weak reference to shared
        bool try_add_shared() noexcept
        {
            std::size_t count = shared_count_.load(std::memory_order_relaxed);
            do {
                if (count == 0) {
                    return false;   // expired
                }
            } while (!shared_count_.compare_exchange_weak(
                count,
                count + 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed
            ));
            return true;
        }

      private:
        void destroy_object() noexcept
        {
            if (ptr_) {
                deleter_(ptr_);
                ptr_ = nullptr;
            }
        }
    };

    /**
     * shared_ptr - device-callable shared ownership
     *
     * srp: shared ownership of memory with device-safe access
     * thread-safe reference counting, exception-safe construction
     */
    template <typename T>
    class shared_ptr
    {
        template <typename U>
        friend class shared_ptr;
        template <typename U>
        friend class weak_ptr;

        T* ptr_;
        control_block<T, default_delete<T>>* control_;

        // private constructor for internal use
        template <typename Deleter>
        shared_ptr(T* ptr, control_block<T, Deleter>* control) noexcept
            : ptr_(ptr),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      control
                  )
              )
        {
        }

      public:
        using element_type = T;
        using pointer      = T*;

        // construction
        constexpr shared_ptr() noexcept : ptr_(nullptr), control_(nullptr) {}

        constexpr shared_ptr(std::nullptr_t) noexcept
            : ptr_(nullptr), control_(nullptr)
        {
        }

        template <typename Deleter = default_delete<T>>
        explicit shared_ptr(T* ptr, Deleter deleter = Deleter{})
            : ptr_(ptr), control_(nullptr)
        {
            if (ptr) {
                try {
                    control_ =
                        reinterpret_cast<control_block<T, default_delete<T>>*>(
                            new control_block<T, Deleter>(
                                ptr,
                                std::move(deleter)
                            )
                        );
                }
                catch (...) {
                    deleter(ptr);   // cleanup on failure
                    throw;
                }
            }
        }

        // aliasing constructor
        template <typename U>
        shared_ptr(const shared_ptr<U>& other, T* ptr) noexcept
            : ptr_(ptr),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      other.control_
                  )
              )
        {
            if (control_) {
                control_->add_shared();
            }
        }

        // copy semantics (host-only)
        shared_ptr(const shared_ptr& other) noexcept
            : ptr_(other.ptr_), control_(other.control_)
        {
            if (control_) {
                control_->add_shared();
            }
        }

        template <typename U>
        shared_ptr(const shared_ptr<U>& other) noexcept
            requires std::is_convertible_v<U*, T*>
            : ptr_(other.ptr_),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      other.control_
                  )
              )
        {
            if (control_) {
                control_->add_shared();
            }
        }

        shared_ptr& operator=(const shared_ptr& other) noexcept
        {
            shared_ptr(other).swap(*this);
            return *this;
        }

        template <typename U>
        shared_ptr& operator=(const shared_ptr<U>& other) noexcept
            requires std::is_convertible_v<U*, T*>
        {
            shared_ptr(other).swap(*this);
            return *this;
        }

        // move semantics
        shared_ptr(shared_ptr&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr)),
              control_(std::exchange(other.control_, nullptr))
        {
        }

        template <typename U>
        shared_ptr(shared_ptr<U>&& other) noexcept
            requires std::is_convertible_v<U*, T*>
            : ptr_(std::exchange(other.ptr_, nullptr)),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      std::exchange(other.control_, nullptr)
                  )
              )
        {
        }

        shared_ptr& operator=(shared_ptr&& other) noexcept
        {
            shared_ptr(std::move(other)).swap(*this);
            return *this;
        }

        template <typename U>
        shared_ptr& operator=(shared_ptr<U>&& other) noexcept
            requires std::is_convertible_v<U*, T*>
        {
            shared_ptr(std::move(other)).swap(*this);
            return *this;
        }

        // unique_ptr conversion
        template <typename U, typename Deleter>
        shared_ptr(unique_ptr<U, Deleter>&& other)
            requires std::is_convertible_v<
                         typename unique_ptr<U, Deleter>::pointer,
                         T*>
            : ptr_(other.get()), control_(nullptr)
        {
            if (ptr_) {
                try {
                    control_ =
                        reinterpret_cast<control_block<T, default_delete<T>>*>(
                            new control_block<U, Deleter>(
                                other.release(),
                                std::move(other.get_deleter())
                            )
                        );
                }
                catch (...) {
                    // restore unique_ptr state on failure
                    other.reset(ptr_);
                    throw;
                }
            }
        }

        shared_ptr& operator=(std::nullptr_t) noexcept
        {
            reset();
            return *this;
        }

        ~shared_ptr()
        {
            if (control_) {
                control_->release_shared();
            }
        }

        // device-callable access (zero overhead)
        DUAL pointer get() const noexcept { return ptr_; }
        DUAL T& operator*() const noexcept { return *ptr_; }
        DUAL pointer operator->() const noexcept { return ptr_; }
        DUAL T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
        DUAL explicit operator bool() const noexcept { return ptr_ != nullptr; }

        // host-only queries
        std::size_t use_count() const noexcept
        {
            return control_ ? control_->shared_count() : 0;
        }

        bool unique() const noexcept { return use_count() == 1; }

        void reset() noexcept { shared_ptr().swap(*this); }

        template <typename U, typename Deleter = default_delete<U>>
        void reset(U* ptr, Deleter deleter = Deleter{})
        {
            shared_ptr(ptr, std::move(deleter)).swap(*this);
        }

        void swap(shared_ptr& other) noexcept
        {
            std::swap(ptr_, other.ptr_);
            std::swap(control_, other.control_);
        }
    };

    /**
     * weak_ptr - non-owning observer of shared objects
     *
     * srp: observe shared objects without affecting lifetime
     * break circular dependencies, safe observation
     */
    template <typename T>
    class weak_ptr
    {
        template <typename U>
        friend class weak_ptr;

        T* ptr_;
        control_block<T, default_delete<T>>* control_;

      public:
        using element_type = T;

        // construction
        constexpr weak_ptr() noexcept : ptr_(nullptr), control_(nullptr) {}

        weak_ptr(const shared_ptr<T>& shared) noexcept
            : ptr_(shared.ptr_), control_(shared.control_)
        {
            if (control_) {
                control_->add_weak();
            }
        }

        template <typename U>
        weak_ptr(const shared_ptr<U>& shared) noexcept
            requires std::is_convertible_v<U*, T*>
            : ptr_(shared.ptr_),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      shared.control_
                  )
              )
        {
            if (control_) {
                control_->add_weak();
            }
        }

        // copy semantics
        weak_ptr(const weak_ptr& other) noexcept
            : ptr_(other.ptr_), control_(other.control_)
        {
            if (control_) {
                control_->add_weak();
            }
        }

        template <typename U>
        weak_ptr(const weak_ptr<U>& other) noexcept
            requires std::is_convertible_v<U*, T*>
            : ptr_(other.ptr_),
              control_(
                  reinterpret_cast<control_block<T, default_delete<T>>*>(
                      other.control_
                  )
              )
        {
            if (control_) {
                control_->add_weak();
            }
        }

        weak_ptr& operator=(const weak_ptr& other) noexcept
        {
            weak_ptr(other).swap(*this);
            return *this;
        }

        weak_ptr& operator=(const shared_ptr<T>& shared) noexcept
        {
            weak_ptr(shared).swap(*this);
            return *this;
        }

        // move semantics
        weak_ptr(weak_ptr&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr)),
              control_(std::exchange(other.control_, nullptr))
        {
        }

        weak_ptr& operator=(weak_ptr&& other) noexcept
        {
            weak_ptr(std::move(other)).swap(*this);
            return *this;
        }

        ~weak_ptr()
        {
            if (control_) {
                control_->release_weak();
            }
        }

        // observation
        std::size_t use_count() const noexcept
        {
            return control_ ? control_->shared_count() : 0;
        }

        bool expired() const noexcept { return use_count() == 0; }

        shared_ptr<T> lock() const noexcept
        {
            if (control_ && control_->try_add_shared()) {
                return shared_ptr<T>(ptr_, control_);
            }
            return shared_ptr<T>{};
        }

        void reset() noexcept { weak_ptr().swap(*this); }

        void swap(weak_ptr& other) noexcept
        {
            std::swap(ptr_, other.ptr_);
            std::swap(control_, other.control_);
        }
    };

    // factory functions
    template <typename T, typename... Args>
    unique_ptr<T> make_unique(std::size_t count)
    {
        return unique_ptr<T>{new T[count]{}};
    }

    template <typename T, typename Deleter>
    unique_ptr<T, Deleter> make_unique(T* ptr, Deleter&& deleter)
    {
        return unique_ptr<T, Deleter>{ptr, std::forward<Deleter>(deleter)};
    }

    template <typename T>
    shared_ptr<T> make_shared(std::size_t count)
    {
        return shared_ptr<T>{new T[count]{}};
    }

    template <typename T, typename Deleter>
    shared_ptr<T> make_shared(T* ptr, Deleter&& deleter)
    {
        return shared_ptr<T>{ptr, std::forward<Deleter>(deleter)};
    }

    // comparison operators
    template <typename T, typename U, typename D1, typename D2>
    bool
    operator==(const unique_ptr<T, D1>& a, const unique_ptr<U, D2>& b) noexcept
    {
        return a.get() == b.get();
    }

    template <typename T, typename U>
    bool operator==(const shared_ptr<T>& a, const shared_ptr<U>& b) noexcept
    {
        return a.get() == b.get();
    }

    template <typename T, typename D>
    bool operator==(const unique_ptr<T, D>& ptr, std::nullptr_t) noexcept
    {
        return !ptr;
    }

    template <typename T>
    bool operator==(const shared_ptr<T>& ptr, std::nullptr_t) noexcept
    {
        return !ptr;
    }

    template <typename T, typename D>
    auto
    operator<=>(const unique_ptr<T, D>& a, const unique_ptr<T, D>& b) noexcept
    {
        return std::compare_three_way{}(a.get(), b.get());
    }

    template <typename T>
    auto operator<=>(const shared_ptr<T>& a, const shared_ptr<T>& b) noexcept
    {
        return std::compare_three_way{}(a.get(), b.get());
    }

    // utility functions
    template <typename T, typename D>
    void swap(unique_ptr<T, D>& a, unique_ptr<T, D>& b) noexcept
    {
        a.swap(b);
    }

    template <typename T>
    void swap(shared_ptr<T>& a, shared_ptr<T>& b) noexcept
    {
        a.swap(b);
    }

    template <typename T>
    void swap(weak_ptr<T>& a, weak_ptr<T>& b) noexcept
    {
        a.swap(b);
    }

}   // namespace simbi::mem

#endif   // PTR_HPP
