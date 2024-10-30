
/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       smrt_ptr.hpp
 * @brief
 *
 * @note adapted from:
 * https://www.experts-exchange.com/articles/1959/C-Smart-pointers.html
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Jun-26-2024     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef SMRT_PTR_HPP
#define SMRT_PTR_HPP

#include "build_options.hpp"
#include <atomic>
#include <exception>
#include <memory>
#include <utility>

namespace simbi {
    namespace util {

        struct refcnt {
            std::atomic<int> count;

            refcnt() : count(1) {}

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
            DUAL void operator()(ptrT* ptr) const { delete ptr; }
        };

        // Default deleter for array types
        template <typename ptrT>
        struct default_delete<ptrT[]> {
            DUAL void operator()(ptrT* ptr) const { delete[] ptr; }
        };

        template <typename ptrT, typename delete_policy = default_delete<ptrT>>
        class smart_ptr
        {
          private:
            using deleter_t = delete_policy;
            using refcnt_t  = refcnt;

            ptrT* pData;
            refcnt_t* pRef;

            void release()
            {
                if (pRef && pRef->release() == 0) {
                    deleter_t()(pData);
                    delete pRef;
                }
            }

          public:
            using ptr_t = ptrT;

            // Default constructor
            DUAL smart_ptr() : pData(nullptr), pRef(nullptr) {}

            // Constructor from raw pointer
            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(new refcnt())
            {
            }

            // Constructor from nullptr
            smart_ptr(std::nullptr_t) noexcept : pData(nullptr), pRef(nullptr)
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
            smart_ptr& operator=(const smart_ptr& other)
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

            void reset(ptr_t* ptr = nullptr)
            {
                smart_ptr temp(ptr);
                swap(temp);
            }

            void swap(smart_ptr& other) noexcept
            {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
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
            DUAL ptr_t* get() const { return pData; }

            // Check if the smart pointer is valid
            explicit operator bool() const noexcept { return pData != nullptr; }

            // comparison with nullptr
            bool operator==(std::nullptr_t) const noexcept
            {
                return pData == nullptr;
            }

            // comparison with nullptr
            bool operator!=(std::nullptr_t) const noexcept
            {
                return pData != nullptr;
            }

            void error_out() const
            {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
                    printf("[GPU ERROR]: DEFERENCING NULL POINTER");
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

            void release()
            {
                if (pRef && pRef->release() == 0) {
                    deleter_t()(pData);
                    delete pRef;
                }
            }

          public:
            using ptr_t = ptrT;

            smart_ptr() : pData(nullptr), pRef(nullptr) {}

            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(new refcnt())
            {
            }

            smart_ptr(const smart_ptr& other)
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

            smart_ptr& operator=(const smart_ptr& other)
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

            void reset(ptr_t* ptr = nullptr)
            {
                smart_ptr temp(ptr);
                swap(temp);
            }

            void swap(smart_ptr& other) noexcept
            {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
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

            DUAL ptr_t* get() const { return pData; }

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

            void error_out() const
            {
                if constexpr (global::BuildPlatform == global::Platform::GPU) {
                    printf("[GPU ERROR]: DEFERENCING NULL POINTER");
                }
                else {
                    throw std::runtime_error("Dereferencing null pointer");
                }
            }

            explicit operator bool() const noexcept { return pData != nullptr; }
        };

    }   // namespace util
}   // namespace simbi

#endif   // SMRT_PTR_HPP