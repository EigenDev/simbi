/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       ndarray.hpp
 * @brief      implementation of custom cpu-gpu translatable array class
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "build_options.hpp"   // for Platform, global::BuildPlatform
#include "device_api.hpp"      // for gpuFree, gpuMalloc, gpuMallocManaged
#include "smrt_ptr.hpp"        // for smart_ptr
#include <cstddef>             // for size_t
#include <initializer_list>    // for initializer_list
#include <iterator>            // for forward_iterator_tag
#include <memory>              // for unique_ptr
#include <vector>              // for vector

using size_type = std::size_t;

namespace simbi {
    // Template class to create array of different data_type
    template <typename DT, global::Platform build_mode = global::BuildPlatform>
    class ndarray
    {
        using value_type = DT;
        template <typename Deleter>
        using unique_p = util::smart_ptr<DT[], Deleter>;

      private:
        size_type sz;            // Variable to store the size of the array
        size_type nd_capacity;   // Variable to store the current capacity of
                                 // the array
        size_type dimensions;    // Number of dimensions
        util::smart_ptr<DT[]> arr;   // Host-side array

        // Device-side array allocation
        void* myGpuMalloc(size_type size)
        {
            if constexpr (build_mode == global::Platform::GPU) {
                void* ptr;
                gpu::api::gpuMalloc(&ptr, size);
                return ptr;
            }
            return nullptr;
        }

        // Device-side managed array allocation
        void* myGpuMallocManaged(size_type size)
        {
            if constexpr (build_mode == global::Platform::GPU) {
                void* ptr;
                gpu::api::gpuMallocManaged(&ptr, size);
                return ptr;
            }
            return nullptr;
        }

        struct gpuDeleter {
            void operator()(DT* ptr)
            {
                if constexpr (build_mode == global::Platform::GPU) {
                    gpu::api::gpuFree(ptr);
                }
            }
        };

        unique_p<gpuDeleter> dev_arr;   // Device-side array

      public:
        ndarray() noexcept
            : sz(0),
              nd_capacity(0),
              dimensions(1),
              arr(nullptr),
              dev_arr(nullptr) {};
        ~ndarray() = default;
        // Assignment operator
        ndarray& operator=(ndarray rhs);
        // Zeri-initialize the array with defined size
        ndarray(size_type size);
        ndarray(
            size_type size,
            const DT val
        );   // Fill-initialize the array with defined size
        ndarray(const ndarray& rhs);           // Copy-constructor for array
        ndarray(const std::vector<DT>& rhs);   // Copy-constructor for vector
        ndarray(ndarray&& rhs) noexcept;       // Move-constructor for array
        ndarray(std::vector<DT>&& rhs);        // Move-constructor for vector
        void swap(ndarray& rhs);               // Swap function

        // Function that returns the number of elements in array after pushing
        // the data
        constexpr void push_back(const DT&);

        // Function that returns the popped element
        constexpr void pop_back();

        // Function to resize ndarray
        constexpr void resize(size_type new_size);
        constexpr void resize(size_type new_size, const DT new_value);

        // Function that returns the size of array
        constexpr size_type size() const;
        constexpr size_type capacity() const;
        constexpr size_type ndim() const;

        // Access operator (mutable)
        template <typename IndexType>
        DUAL constexpr DT& operator[](IndexType);

        // Const-access operator (read-only)
        template <typename IndexType>
        DUAL constexpr DT operator[](IndexType) const;

        // Some math operator overloads
        constexpr ndarray& operator*(real);
        constexpr ndarray& operator*=(real);
        constexpr ndarray& operator/(real);
        constexpr ndarray& operator/=(real);
        constexpr ndarray& operator+=(const ndarray& rhs);

        // Check if ndarray is empty
        bool empty() const;

        // Get pointers to underlying data ambiguously, on host, or on gpu
        DUAL DT* data();
        DT* host_data();
        DUAL DT* dev_data();

        DUAL DT* data() const;
        DT* host_data() const;
        DUAL DT* dev_data() const;

        // Iterator Class
        class iterator
        {
          private:
            DT* ptr;   // Dynamic array using pointers

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = DT;
            using difference_type   = void;
            using pointer           = void;
            using reference         = void;

            DUAL explicit iterator() : ptr(nullptr) {}

            DUAL explicit iterator(DT* p) : ptr(p) {}

            DUAL bool operator==(const iterator& rhs) const
            {
                return ptr == rhs.ptr;
            }

            DUAL bool operator!=(const iterator& rhs) const
            {
                return !(*this == rhs);
            }

            DT operator*() const { return *ptr; }

            DUAL iterator& operator++()
            {
                ++ptr;
                return *this;
            }

            DUAL iterator operator++(int)
            {
                iterator temp(*this);
                ++*this;
                return temp;
            }
        };

        // Begin iterator
        iterator begin() const;

        // End iterator
        iterator end() const;

        // Back of container
        DT back() const;
        DT& back();
        DT front() const;
        DT& front();

        // GPU memory copy helpers
        void copyToGpu();
        void copyFromGpu();
        void copyBetweenGpu(const ndarray& rhs);

        // Additional utility methods
        void clear();
        void shrink_to_fit();
        void reserve(size_type new_capacity);
    };

}   // namespace simbi

// Type trait
template <typename T>
struct is_ndarray {
    static constexpr bool value = false;
};

template <typename T>
struct is_2darray {
    static constexpr bool value = false;
};

template <typename T>
struct is_3darray {
    static constexpr bool value = false;
};

template <typename T>
struct is_1darray {
    static constexpr bool value = false;
};

template <typename U>
struct is_ndarray<simbi::ndarray<U>> {
    static constexpr bool value = true;
};

template <typename U>
struct is_1darray<simbi::ndarray<U>> {
    static constexpr bool value = true;
};

template <typename U>
struct is_2darray<simbi::ndarray<simbi::ndarray<U>>> {
    static constexpr bool value = true;
};

template <typename U>
struct is_3darray<simbi::ndarray<simbi::ndarray<simbi::ndarray<U>>>> {
    static constexpr bool value = true;
};

#include "ndarray.ipp"
#endif