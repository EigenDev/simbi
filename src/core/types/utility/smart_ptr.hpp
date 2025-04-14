/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            smart_ptr.hpp
 *  * @brief           a custom implementation of smart pointers for GPU/CPU
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
 *  * @note adapted from:
 * https://www.experts-exchange.com/articles/1959/C-Smart-pointers.html
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

    // control blocks for type erasure
    class control_block_base
    {
      public:
        virtual ~control_block_base()  = default;
        virtual void destroy()         = 0;
        virtual void increment()       = 0;
        virtual long decrement()       = 0;
        virtual long use_count() const = 0;
    };

    template <typename T, typename Deleter>
    class typed_control_block : public control_block_base
    {
      private:
        T* ptr;
        Deleter deleter;
        std::atomic<long> count;

      public:
        typed_control_block(T* p, Deleter d)
            : ptr(p), deleter(std::move(d)), count(1)
        {
        }

        void destroy() override { deleter(ptr); }
        void increment() override
        {
            count.fetch_add(1, std::memory_order_relaxed);
        }
        long decrement() override
        {
            return count.fetch_sub(1, std::memory_order_acq_rel) - 1;
        }
        long use_count() const override
        {
            return count.load(std::memory_order_relaxed);
        }
    };

    // control blocks for array types
    template <typename T, typename Deleter>
    class typed_control_block<T[], Deleter> : public control_block_base
    {
      private:
        T* ptr;
        Deleter deleter;
        std::atomic<long> count;

      public:
        typed_control_block(T* p, Deleter d)
            : ptr(p), deleter(std::move(d)), count(1)
        {
        }

        void destroy() override { deleter(ptr); }
        void increment() override
        {
            count.fetch_add(1, std::memory_order_relaxed);
        }
        long decrement() override
        {
            return count.fetch_sub(1, std::memory_order_acq_rel) - 1;
        }
        long use_count() const override
        {
            return count.load(std::memory_order_relaxed);
        }
    };

    namespace util {

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
            template <typename U, typename D>
            friend class smart_ptr;

            ptrT* pData;
            control_block_base* ctrl;

            void release()
            {
                if (ctrl && ctrl->decrement() == 0) {
                    ctrl->destroy();   // Type-erased destruction
                    delete ctrl;
                }
            }

          public:
            using ptr_t = ptrT;

            // Default constructor
            constexpr smart_ptr() noexcept : pData(nullptr), ctrl(nullptr) {}

            // Constructor from raw pointer
            explicit smart_ptr(ptr_t* pData)
                : pData(pData),
                  ctrl(new typed_control_block<ptrT, delete_policy>(
                      pData,
                      delete_policy()
                  ))
            {
            }

            // Constructor from nullptr
            constexpr smart_ptr(std::nullptr_t) noexcept
                : pData(nullptr), ctrl(nullptr)
            {
            }

            // Copy Constructor
            smart_ptr(const smart_ptr& other) noexcept
                : pData(other.pData), ctrl(other.ctrl)
            {
                if (ctrl) {
                    ctrl->increment();
                }
            }

            // Move Constructor
            smart_ptr(smart_ptr&& other) noexcept
                : pData(other.pData), ctrl(other.ctrl)
            {
                other.pData = nullptr;
                other.ctrl  = nullptr;
            }

            // template <typename D>
            // smart_ptr(ptr_t* pData, D&& deleter)
            //     : pData(pData),
            //       ctrl(new refcnt()),
            //       custom_deleter(std::forward<D>(deleter))
            // {
            // }

            ~smart_ptr() { release(); }

            // add support for derived-to-base conversion
            template <
                typename U,
                typename = std::enable_if_t<std::is_convertible_v<U*, ptr_t*>>>
            smart_ptr(const smart_ptr<U>& other) noexcept
                : pData(static_cast<ptrT*>(other.get())),
                  ctrl(other.get_control_block())
            {
                if (ctrl) {
                    ctrl->increment();
                }
            }

            // safehguard ref management
            long use_count() const noexcept
            {
                return ctrl ? ctrl->use_count() : 0;
            }

            bool unique() const noexcept { return use_count() == 1; }

            control_block_base* get_control_block() const noexcept
            {
                return ctrl;
            }

            // template <typename U = ptrT>
            // control_block<U>* get_control_block() const noexcept
            // {
            //     // For same type, just return the control block
            //     if constexpr (std::is_same_v<U, ptrT>) {
            //         return ctrl;
            //     }
            //     // For base-of relationship, perform conversion with
            //     // static_assert
            //     else {
            //         static_assert(
            //             std::is_convertible_v<ptrT*, U*>,
            //             "Invalid control block conversion - types are not "
            //             "compatible"
            //         );
            //         // This reinterpret_cast is safe because the control
            //         blocks
            //         // follow the same inheritance relationship
            //         return reinterpret_cast<control_block<U>*>(ctrl);
            //     }
            // }

            // Copy Assignment
            smart_ptr& operator=(const smart_ptr& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData = other.pData;
                    ctrl  = other.ctrl;
                    if (ctrl) {
                        ctrl->increment();
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
                    ctrl        = other.ctrl;
                    other.pData = nullptr;
                    other.ctrl  = nullptr;
                }
                return *this;
            }

            // Assignment operator from nullptr
            smart_ptr& operator=(std::nullptr_t) noexcept
            {
                release();
                pData = nullptr;
                ctrl  = nullptr;
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
                    control_block_base* pr = ctrl;
                    ptr_t* pd              = pData;

                    ctrl  = other.ctrl;
                    pData = other.pData;

                    other.ctrl  = pr;
                    other.pData = pd;
                }
                else {
                    std::swap(pData, other.pData);
                    std::swap(ctrl, other.ctrl);
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
            DUAL constexpr explicit operator bool() const noexcept
            {
                return pData != nullptr;
            }

            // comparison with nullptr
            DUAL constexpr bool operator==(std::nullptr_t) const noexcept
            {
                return pData == nullptr;
            }

            // comparison with nullptr
            DUAL constexpr bool operator!=(std::nullptr_t) const noexcept
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

            ptrT* pData;
            control_block_base* ctrl;

            void release() noexcept
            {
                if (ctrl && ctrl->decrement() == 0) {
                    ctrl->destroy();
                    delete ctrl;
                }
            }

          public:
            using ptr_t = ptrT;

            constexpr smart_ptr() noexcept : pData(nullptr), ctrl(nullptr) {}

            explicit smart_ptr(ptr_t* pData)
                : pData(pData),
                  ctrl(
                      new typed_control_block<ptrT[], deleter>(pData, deleter())
                  )
            {
            }

            smart_ptr(const smart_ptr& other) noexcept
                : pData(other.pData), ctrl(other.ctrl)
            {
                if (ctrl) {
                    ctrl->increment();
                }
            }

            smart_ptr(smart_ptr&& other) noexcept
                : pData(other.pData), ctrl(other.ctrl)
            {
                other.pData = nullptr;
                other.ctrl  = nullptr;
            }

            ~smart_ptr() { release(); }

            smart_ptr& operator=(const smart_ptr& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData = other.pData;
                    ctrl  = other.ctrl;
                    if (ctrl) {
                        ctrl->increment();
                    }
                }
                return *this;
            }

            smart_ptr& operator=(smart_ptr&& other) noexcept
            {
                if (this != &other) {
                    release();
                    pData       = other.pData;
                    ctrl        = other.ctrl;
                    other.pData = nullptr;
                    other.ctrl  = nullptr;
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
                    control_block_base* pr = ctrl;
                    ptr_t* pd              = pData;

                    ctrl  = other.ctrl;
                    pData = other.pData;

                    other.ctrl  = pr;
                    other.pData = pd;
                }
                else {
                    std::swap(pData, other.pData);
                    std::swap(ctrl, other.ctrl);
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
        smart_ptr<T> make_unique(Args&&... args)
        {
            return smart_ptr<T>(new T(std::forward<Args>(args)...));
        }

        template <typename T>
        constexpr smart_ptr<T> make_unique_array(size_t size)
        {
            using element_t = std::remove_extent_t<T>;
            return smart_ptr<T>(new element_t[size]());
        }

    }   // namespace util
}   // namespace simbi

#endif   // SMRT_PTR_HPP
