/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       smart_ptr.hpp
 * @brief
 *
 * @note adapted from:
 * https://www.experts-exchange.com/articles/1959/C-Smart-pointers.html
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Jun-26-2025     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 */
#ifndef SMRT_PTR_HPP
#define SMRT_PTR_HPP

#include "build_options.hpp"
#include <atomic>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace simbi {
    template <typename>
    class function;

    namespace util {

        struct refcnt {
            std::atomic<int> count;

            refcnt() : count(1) {}

            refcnt(const refcnt&)            = delete;
            refcnt& operator=(const refcnt&) = delete;

            void inc() { count.fetch_add(1, std::memory_order_relaxed); }

            int release()
            {
                return count.fetch_sub(1, std::memory_order_acq_rel) - 1;
            }

            int get() const { return count.load(std::memory_order_relaxed); }

            bool is_zero() const { return get() == 0; }
        };

        // Default deleter for scalar types
        template <typename ptrT>
        struct default_delete {
            void operator()(ptrT* ptr) const { delete ptr; }
        };

        // Default deleter for array types
        template <typename ptrT>
        struct default_delete<ptrT[]> {
            void operator()(ptrT* ptr) const { delete[] ptr; }
        };

        template <typename ptrT, typename delete_policy = default_delete<ptrT>>
        class smart_ptr
        {
          private:
            using deleter_t = delete_policy;
            using refcnt_t  = refcnt;

            ptrT* pData;
            refcnt_t* pRef;

            void release() noexcept
            {
                if (pRef && pRef->release() == 0) {
                    deleter_t()(pData);
                    delete pRef;
                }
            }

          public:
            using ptr_t = ptrT;

            // Default constructor
            constexpr smart_ptr() noexcept : pData(nullptr), pRef(nullptr) {}

            // Constructor from raw pointer
            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(new refcnt())
            {
            }

            // Constructor from nullptr
            constexpr smart_ptr(std::nullptr_t) noexcept
                : pData(nullptr), pRef(nullptr)
            {
            }

            // Copy Constructor
            smart_ptr(const smart_ptr& other) noexcept
                : pData(other.pData), pRef(other.pRef)
            {
                if (pRef) {
                    pRef->inc();
                }
            }

            // Move Constructor
            smart_ptr(smart_ptr&& other) noexcept
                : pData(other.pData), pRef(other.pRef)
            {
                other.pData = nullptr;
                other.pRef  = nullptr;
            }

            ~smart_ptr() { release(); }

            // Copy Assignment
            smart_ptr& operator=(const smart_ptr& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData = other.pData;
                    pRef  = other.pRef;
                    if (pRef) {
                        pRef->inc();
                    }
                }
                return *this;
            }

            // Move assignment operator
            smart_ptr& operator=(smart_ptr&& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData       = other.pData;
                    pRef        = other.pRef;
                    other.pData = nullptr;
                    other.pRef  = nullptr;
                }
                return *this;
            }

            // Assignment operator from nullptr
            smart_ptr& operator=(std::nullptr_t) noexcept
            {
                release();
                pData = nullptr;
                pRef  = nullptr;
                return *this;
            }

            void reset(ptr_t* ptr = nullptr) noexcept
            {
                smart_ptr temp(ptr);
                swap(temp);
            }

            void swap(smart_ptr& other) noexcept
            {
                if constexpr (global::on_gpu) {
                    refcnt_t* pr = pRef;
                    ptr_t* pd    = pData;

                    pRef  = other.pRef;
                    pData = other.pData;

                    other.pRef  = pr;
                    other.pData = pd;
                }
                else {
                    std::swap(pData, other.pData);
                    std::swap(pRef, other.pRef);
                }
            }

            // Dereference operator
            DUAL ptr_t& operator*() const
            {
                if (!pData) {
                    error_out();
                }
                return *pData;
            }

            // Arrow operator
            DUAL ptr_t* operator->() const
            {
                if (!pData) {
                    error_out();
                }
                return pData;
            }

            // Get raw pointer
            DUAL ptr_t* get() const noexcept { return pData; }

            // Check if the smart pointer is valid
            constexpr explicit operator bool() const noexcept
            {
                return pData != nullptr;
            }

            // comparison with nullptr
            constexpr bool operator==(std::nullptr_t) const noexcept
            {
                return pData == nullptr;
            }

            // comparison with nullptr
            constexpr bool operator!=(std::nullptr_t) const noexcept
            {
                return pData != nullptr;
            }

            DUAL void error_out() const
            {
                if constexpr (global::on_gpu) {
                    printf("[GPU ERROR]: DEREFERENCING NULL POINTER\n");
                }
                else {
                    throw std::runtime_error("Dereferencing null pointer");
                }
            }
        };

        // Specialization for array types
        template <typename ptrT, typename deleter>
        class smart_ptr<ptrT[], deleter>
        {
          private:
            using deleter_t = deleter;
            using refcnt_t  = refcnt;

            ptrT* pData;
            refcnt_t* pRef;

            void release() noexcept
            {
                if (pRef && pRef->release() == 0) {
                    deleter_t()(pData);
                    delete pRef;
                }
            }

          public:
            using ptr_t = ptrT;

            constexpr smart_ptr() noexcept : pData(nullptr), pRef(nullptr) {}

            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(new refcnt())
            {
            }

            smart_ptr(const smart_ptr& other) noexcept
                : pData(other.pData), pRef(other.pRef)
            {
                if (pRef) {
                    pRef->inc();
                }
            }

            smart_ptr(smart_ptr&& other) noexcept
                : pData(other.pData), pRef(other.pRef)
            {
                other.pData = nullptr;
                other.pRef  = nullptr;
            }

            ~smart_ptr() { release(); }

            smart_ptr& operator=(const smart_ptr& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData = other.pData;
                    pRef  = other.pRef;
                    if (pRef) {
                        pRef->inc();
                    }
                }
                return *this;
            }

            smart_ptr& operator=(smart_ptr&& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData       = other.pData;
                    pRef        = other.pRef;
                    other.pData = nullptr;
                    other.pRef  = nullptr;
                }
                return *this;
            }

            void reset(ptr_t* ptr = nullptr) noexcept
            {
                smart_ptr temp(ptr);
                swap(temp);
            }

            void swap(smart_ptr& other) noexcept
            {
                if constexpr (global::on_gpu) {
                    refcnt_t* pr = pRef;
                    ptr_t* pd    = pData;

                    pRef  = other.pRef;
                    pData = other.pData;

                    other.pRef  = pr;
                    other.pData = pd;
                }
                else {
                    std::swap(pData, other.pData);
                    std::swap(pRef, other.pRef);
                }
            }

            DUAL ptr_t& operator*() const
            {
                if (!pData) {
                    error_out();
                }
                return *pData;
            }

            DUAL ptr_t* operator->() const
            {
                if (!pData) {
                    error_out();
                }
                return pData;
            }

            DUAL ptr_t* get() const noexcept { return pData; }

            template <typename IndexType>
            DUAL ptrT& operator[](IndexType index)
            {
                if (!pData) {
                    error_out();
                }
                return pData[index];
            }

            template <typename IndexType>
            DUAL const ptrT& operator[](IndexType index) const
            {
                if (!pData) {
                    error_out();
                }
                return pData[index];
            }

            DUAL void error_out() const
            {
                if constexpr (global::on_gpu) {
                    printf("[GPU ERROR]: DEFERENCING NULL POINTER\n");
                }
                else {
                    throw std::runtime_error("Dereferencing null pointer");
                }
            }

            constexpr explicit operator bool() const noexcept
            {
                return pData != nullptr;
            }
        };

        template <typename T, typename... Args>
        constexpr smart_ptr<T> make_unique(Args&&... args)
        {
            return smart_ptr<T>(new T(std::forward<Args>(args)...));
        }

        template <typename T>
        constexpr smart_ptr<T> make_unique(std::size_t size)
        {
            return smart_ptr<T>(new std::remove_extent<T>::type[size]());
        }

    }   // namespace util
}   // namespace simbi

#endif   // SMRT_PTR_HPP