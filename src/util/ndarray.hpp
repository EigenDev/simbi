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
#include "maybe.hpp"           // for maybe
#include "parallel_for.hpp"    // for parallel_for
#include "smrt_ptr.hpp"        // for smart_ptr
#include <cstddef>             // for size_t
#include <initializer_list>    // for initializer_list
#include <iterator>            // for forward_iterator_tag
#include <memory>              // for unique_ptr
#include <vector>              // for vector

using size_type = std::size_t;

namespace simbi {
    // Template class to create array of different data_type
    template <typename DT>
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

        bool is_gpu_synced           = false;
        bool needs_gpu_sync          = false;
        simbiStream_t current_stream = nullptr;

        // Device-side array allocation
        void* myGpuMalloc(size_type size)
        {
            if constexpr (global::on_gpu) {
                void* ptr;
                gpu::api::malloc(&ptr, size);
                return ptr;
            }
            return nullptr;
        }

        // Device-side managed array allocation
        void* myGpuMallocManaged(size_type size)
        {
            if constexpr (global::on_gpu) {
                void* ptr;
                gpu::api::mallocManaged(&ptr, size);
                return ptr;
            }
            return nullptr;
        }

        struct gpuDeleter {
            void operator()(DT* ptr)
            {
                if constexpr (global::on_gpu) {
                    gpu::api::free(ptr);
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
        constexpr ndarray& push_back(const DT&);

        // Function that returns the popped element
        constexpr ndarray& pop_back();

        // Function to resize ndarray
        constexpr ndarray& resize(size_type new_size);
        constexpr ndarray& resize(size_type new_size, const DT new_value);

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

        // GPU-compatible slice view
        class slice_view
        {
          private:
            DT* data;
            size_type view_size;
            size_type offset;
            bool on_device;

          public:
            DUAL slice_view(DT* data, size_type start, size_type end)
                : data(data),
                  view_size(end - start),
                  offset(start),
                  on_device(global::on_gpu)
            {
            }

            DUAL DT& operator[](size_type i)
            {
                if constexpr (global::on_gpu) {
                    return data[offset + i];
                }
                else {
                    return data[offset + i];
                }
            }

            DUAL const DT& operator[](size_type i) const
            {
                if constexpr (global::on_gpu) {
                    return data[offset + i];
                }
                else {
                    return data[offset + i];
                }
            }

            DUAL size_type size() const { return view_size; }

            DUAL size_type index() const { return offset; }
        };

        DUAL slice_view slice(size_type start, size_type end)
        {
            if (start > end || end > sz) {
                // GPU-safe error handling
                return slice_view(nullptr, 0, 0);
            }
            if constexpr (global::on_gpu) {
                return slice_view(dev_arr.get(), start, end);
            }
            else {
                return slice_view(arr.get(), start, end);
            }
        }

        DUAL slice_view slice(size_type start, size_type end) const
        {
            if (start > end || end > sz) {
                // GPU-safe error handling
                return slice_view(nullptr, 0, 0);
            }
            if constexpr (global::on_gpu) {
                return slice_view(dev_arr.get(), start, end);
            }
            else {
                return slice_view(arr.get(), start, end);
            }
        }

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

        // Memory management methods
        void pin_memory();
        void unpin_memory();

        // Stream support
        void set_stream(simbiStream_t stream);
        void async_copy_to_gpu();

        // Memory optimization helpers
        void ensure_gpu_synced();

        // Aligned memory allocation
        void* aligned_alloc(size_type size, size_type alignment = 32);

        //======================================================================
        // Functional methods
        //======================================================================
        // Map method
        template <typename UnaryFunction>
        DUAL ndarray& map(UnaryFunction f);

        template <typename UnaryFunction>
        DUAL ndarray map(UnaryFunction f) const;
        // Filter method
        template <typename UnaryPredicate>
        DUAL ndarray& filter(UnaryPredicate pred);

        template <typename UnaryPredicate>
        DUAL ndarray filter(UnaryPredicate pred) const;

        // Composed operations
        template <typename F, typename G>
        auto compose(F f, G g) const;

        // Chain operations
        template <typename... Fs>
        auto then(Fs... fs) const;

        // Chain transformations and return new array
        template <typename... Transforms>
        ndarray transform_chain(Transforms... transforms) const;

        // Apply function to each element safely with bounds checking
        template <typename F>
        Maybe<ndarray> safe_map(F f) const;

        // Combine two arrays element-wise with a binary operation
        template <typename F>
        ndarray combine(const ndarray& other, F binary_op) const;

        // Split array into chunks for parallel processing
        template <typename F>
        ndarray
        parallel_chunks(const ExecutionPolicy<>& policy, F chunk_op) const;

        // transform_parallel method
        template <typename NewType, typename F>
        ndarray<NewType> transform_parallel(
            const ExecutionPolicy<>& policy,
            F transform_op
        ) const;
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