#ifndef SMART_PTR_HPP
#define SMART_PTR_HPP

#include "compat.hpp"

#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace simbi::mem {

    // forward declarations
    template <typename T>
    class weak_ptr;

    /**
     * default_delete - lightweight deleter for array types
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
     * control_block_base - polymorphic base for type erasure
     */
    template <typename T>
    class control_block_base
    {
      public:
        virtual ~control_block_base() = default;

        virtual void add_shared() noexcept                = 0;
        virtual bool release_shared() noexcept            = 0;
        virtual void add_weak() noexcept                  = 0;
        virtual bool release_weak() noexcept              = 0;
        virtual std::size_t shared_count() const noexcept = 0;
        virtual std::size_t weak_count() const noexcept   = 0;
        virtual bool expired() const noexcept             = 0;
        virtual T* get() const noexcept                   = 0;
        virtual bool try_add_shared() noexcept            = 0;
    };

    /**
     * control_block - concrete implementation with custom deleter
     */
    template <typename T, typename Deleter>
    class control_block : public control_block_base<T>
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

        control_block(const control_block&)            = delete;
        control_block& operator=(const control_block&) = delete;

        void add_shared() noexcept override
        {
            shared_count_.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_shared() noexcept override
        {
            if (shared_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                destroy_object();
                release_weak();
                return true;
            }
            return false;
        }

        void add_weak() noexcept override
        {
            weak_count_.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_weak() noexcept override
        {
            if (weak_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete this;
                return true;
            }
            return false;
        }

        std::size_t shared_count() const noexcept override
        {
            return shared_count_.load(std::memory_order_acquire);
        }

        std::size_t weak_count() const noexcept override
        {
            return weak_count_.load(std::memory_order_acquire) - 1;
        }

        bool expired() const noexcept override { return shared_count() == 0; }

        T* get() const noexcept override { return ptr_; }

        bool try_add_shared() noexcept override
        {
            std::size_t count = shared_count_.load(std::memory_order_relaxed);
            do {
                if (count == 0) {
                    return false;
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
     * unique_ptr - device-callable unique ownership
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

        constexpr unique_ptr() noexcept : ptr_(nullptr), deleter_() {}
        constexpr unique_ptr(std::nullptr_t) noexcept
            : ptr_(nullptr), deleter_()
        {
        }
        explicit unique_ptr(pointer ptr) noexcept : ptr_(ptr), deleter_() {}

        template <typename CustomDeleter>
        unique_ptr(pointer ptr, CustomDeleter&& deleter) noexcept
            : ptr_(ptr), deleter_(std::forward<CustomDeleter>(deleter))
        {
        }

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

        DUAL pointer get() const noexcept { return ptr_; }
        DUAL T& operator*() const noexcept { return *ptr_; }
        DUAL pointer operator->() const noexcept { return ptr_; }
        DUAL T& operator[](std::size_t i) const noexcept { return ptr_[i]; }
        DUAL explicit operator bool() const noexcept { return ptr_ != nullptr; }

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

    template <typename T, typename Deleter>
    unique_ptr(T*, Deleter) -> unique_ptr<T, std::decay_t<Deleter>>;

    template <typename T>
    unique_ptr(T*) -> unique_ptr<T, default_delete<T>>;

    /**
     * shared_ptr - device-callable shared ownership with type-erased deleters
     */
    template <typename T>
    class shared_ptr
    {
        template <typename U>
        friend class shared_ptr;
        template <typename U>
        friend class weak_ptr;

        T* ptr_;
        control_block_base<T>* control_;

        // private constructor for internal use (from weak_ptr::lock)
        shared_ptr(T* ptr, control_block_base<T>* control) noexcept
            : ptr_(ptr), control_(control)
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
                        new control_block<T, Deleter>(ptr, std::move(deleter));
                }
                catch (...) {
                    deleter(ptr);
                    throw;
                }
            }
        }

        // aliasing constructor
        template <typename U>
        shared_ptr(const shared_ptr<U>& other, T* ptr) noexcept
            : ptr_(ptr), control_(other.control_)
        {
            if (control_) {
                control_->add_shared();
            }
        }

        // copy semantics
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
            : ptr_(other.ptr_), control_(other.control_)
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
              control_(std::exchange(other.control_, nullptr))
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
                    control_ = new control_block<T, Deleter>(
                        static_cast<T*>(other.release()),
                        std::move(other.get_deleter())
                    );
                }
                catch (...) {
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
     */
    template <typename T>
    class weak_ptr
    {
        template <typename U>
        friend class weak_ptr;

        T* ptr_;
        control_block_base<T>* control_;

      public:
        using element_type = T;

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
            : ptr_(shared.ptr_), control_(shared.control_)
        {
            if (control_) {
                control_->add_weak();
            }
        }

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
            : ptr_(other.ptr_), control_(other.control_)
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

#endif
