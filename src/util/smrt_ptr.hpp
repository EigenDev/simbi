/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       smrt_ptr.hpp
 * @brief
 *
 * @note adapted from:
 * https://www.experts-exchange.com/articles/1959/C-Smart-pointers.html
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Jun-26-2024     Marcus DuPont                   md4469@nyu.edu
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

namespace simbi {
    namespace util {
        struct refcnt {
            std::atomic<int> count;

            refcnt() : count(0) {}

            refcnt(int c) : count(c) {}

            refcnt(const refcnt& o) : count(o.count.load()) {}

            refcnt& operator=(const refcnt& o)
            {
                count.store(o.count.load());
                return *this;
            }

            ~refcnt() { count.store(0); }

            int get() const { return count.load(); }

            void inc() { count.fetch_add(1); }

            void dec() { count.fetch_sub(1); }

            int release() { return count.fetch_sub(1) - 1; }

            bool is_zero() const { return count.load() == 0; }
        };

        // Default deleter for scalar types
        template <typename ptrT>
        struct default_delete {
            default_delete() = default;

            template <typename U>
            default_delete(const default_delete<U>&)
            {
            }

            void operator()(ptrT* ptr) const { delete ptr; }
        };

        // Default deleter for array types
        template <typename ptrT>
        struct default_delete<ptrT[]> {
            void operator()(ptrT* ptr) const { delete[] ptr; }
        };

        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        // This is the smart pointer template, which takes pointer type and
        // a destruct policy, which it uses to destruct object(s) pointed to
        // when the reference counter for the object becomes zero.

        template <typename ptrT, typename delete_policy = default_delete<ptrT>>
        class smart_ptr
        {
          private:
            using deleter_t   = delete_policy;
            using safe_bool_t = void (smart_ptr::*)();
            using refcnt_t    = refcnt;

          public:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Make a nice typedef for the pointer type
            using ptr_t = ptrT;

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // dc_tor, c_tor and cc_tor
            HD smart_ptr() : pData(0), pRef(0)
            {
                // create new reference
                pRef = new refcnt();
                // increment the reference count
                pRef->inc();
            }

            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(0)
            {
                // create new reference
                pRef = new refcnt();
                // increment the reference count
                pRef->inc();
            }

            // copy constructor
            smart_ptr(smart_ptr const& oPtr)
                : pData(oPtr.pData), pRef(oPtr.pRef)
            {
                pRef->inc();
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // d_tor, deletes the pointer using the destruct policy when the
            // reference counter for the object reaches zero
            ~smart_ptr()
            {
                try {
                    if (pRef && pRef->release() == 0) {
                        deleter_t()(pData);
                        delete pRef;
                    }
                }
                catch (...) {
                    // Ignored. Prevent percolation during stack unwinding.
                }
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Assignment operator copies an existing pointer smart_ptr, but
            // in doing so will 'reset' the current pointer
            smart_ptr& operator=(smart_ptr const& sp)
            {
                if (&sp != this && pData != sp.pData) {
                    reset(sp.pData);
                    pRef = sp.pRef;
                    pRef->inc();
                }

                return *this;
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Performs a safe swap of two smart pointer.
            void swap(smart_ptr& sp)
            {
                refcnt_t* pr = pRef;
                ptr_t* pd    = pData;

                pRef  = sp.pRef;
                pData = sp.pData;

                sp.pRef  = pr;
                sp.pData = pd;
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Resets the current smart pointer. If a new pointer is provided
            // The reference counter will be set to one and the pointer will
            // be stored, if no pointer is provided the reference counter and
            // pointer wil be set to 0, setting this as a null pointer.
            void reset(ptr_t* ptr = 0)
            {
                smart_ptr sp(ptr);
                swap(sp);
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Returns a reference to the object pointed to
            HD ptr_t& operator*() const { return *pData; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Invokes the -> operator on the pointer pointed too
            // NB. When you call the -> operator, the compiler  automatically
            //     calls the -> on the entity returned. This is a special,
            //     case, done to preserve normal indirection semantics.
            HD ptr_t* operator->() const { return pData; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Get the pointer being managed
            HD ptr_t* get() const { return pData; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Conversion to bool operator to facilitate logical pointer tests.
            // Returns a value that will logically be true if get != 0 else
            // and value that is logically false. We don't return a real
            // bool to prevent un-wanted automatic implicit conversion for
            // instances where it would make no semantic sense, rather we
            // return a pointer to a member function as this will always
            // implicitly convert to true or false when used in a boolean
            // context but will not convert, for example, to an int type.
            operator safe_bool_t() const
            {
                return pData ? &smart_ptr::true_eval : 0;
            }

          private:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // A dummy member function used to represent a logically true
            // boolean value, used by the conversion to bool operator.
            void true_eval() {};

          private:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Poiners to the object being managed and the reference counter
            ptr_t* pData;
            refcnt_t* pRef;

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        };

        // Specialize for runtime array lengths
        template <typename ptrT, typename deleter>
        class smart_ptr<ptrT[], deleter>
        {
          private:
            using deleter_t   = deleter;
            using safe_bool_t = void (smart_ptr::*)();
            using refcnt_t    = refcnt;

          public:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Make a nice typedef for the pointer type
            using ptr_t = ptrT;

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // dc_tor, c_tor and cc_tor
            smart_ptr() : pData(0), pRef(0)
            {
                // create new reference
                pRef = new refcnt();
                // increment the reference count
                pRef->inc();
            }

            explicit smart_ptr(ptr_t* pData) : pData(pData), pRef(0)
            {
                // create new reference
                pRef = new refcnt();
                // increment the reference count
                pRef->inc();
            }

            // copy constructor
            smart_ptr(smart_ptr const& oPtr)
                : pData(oPtr.pData), pRef(oPtr.pRef)
            {
                pRef->inc();
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // d_tor, deletes the pointer using the destruct policy when the
            // reference counter for the object reaches zero
            ~smart_ptr()
            {
                try {
                    if (pRef && pRef->release() == 0) {
                        deleter_t()(pData);
                        delete pRef;
                    }
                }
                catch (...) {
                    // Ignored. Prevent percolation during stack unwinding.
                }
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Assignment operator copies an existing pointer smart_ptr, but
            // in doing so will 'reset' the current pointer
            smart_ptr& operator=(smart_ptr const& sp)
            {
                if (&sp != this && pData != sp.pData) {
                    reset(sp.pData);
                    pRef = sp.pRef;
                    pRef->inc();
                }

                return *this;
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Performs a safe swap of two smart pointer.
            void swap(smart_ptr& sp)
            {
                refcnt_t* pr = pRef;
                ptr_t* pd    = pData;

                pRef  = sp.pRef;
                pData = sp.pData;

                sp.pRef  = pr;
                sp.pData = pd;
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Resets the current smart pointer. If a new pointer is provided
            // The reference counter will be set to one and the pointer will
            // be stored, if no pointer is provided the reference counter and
            // pointer wil be set to 0, setting this as a null pointer.
            void reset(ptr_t* ptr = 0)
            {
                smart_ptr sp(ptr);
                swap(sp);
            }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Returns a reference to the object pointed to
            HD ptr_t& operator*() const { return *pData; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Invokes the -> operator on the pointer pointed too
            // NB. When you call the -> operator, the compiler  automatically
            //     calls the -> on the entity returned. This is a special,
            //     case, done to preserve normal indirection semantics.
            HD ptr_t* operator->() const { return pData; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Get the pointer being managed
            HD ptr_t* get() const { return pData; }

            // Accessors.
            HD ptrT& operator[](const std::size_t ii) { return get()[ii]; }

            HD ptrT operator[](const std::size_t ii) const { return get()[ii]; }

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Conversion to bool operator to facilitate logical pointer tests.
            // Returns a value that will logically be true if get != 0 else
            // and value that is logically false. We don't return a real
            // bool to prevent un-wanted automatic implicit conversion for
            // instances where it would make no semantic sense, rather we
            // return a pointer to a member function as this will always
            // implicitly convert to true or false when used in a boolean
            // context but will not convert, for example, to an int type.
            operator safe_bool_t() const
            {
                return pData ? &smart_ptr::true_eval : 0;
            }

          private:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // A dummy member function used to represent a logically true
            // boolean value, used by the conversion to bool operator.
            void true_eval() {};

          private:
            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            // Poiners to the object being managed and the reference counter
            ptr_t* pData;
            refcnt_t* pRef;

            //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        };

    }   // namespace util

}   // namespace simbi

#endif