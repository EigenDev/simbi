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
    template <typename DT, int dim = 1>
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

        DUAL bool is_boundary_point(
            size_type ii,
            size_type jj,
            size_type kk,
            size_type nx,
            size_type ny,
            size_type nz,
            size_type radius
        ) const
        {
            if constexpr (dim == 1) {
                return ii < radius || ii > nx - radius - 1;
            }
            else if constexpr (dim == 2) {
                return ii < radius || jj < radius || ii > nx - radius - 1 ||
                       jj > ny - radius - 1;
            }
            else {
                return ii < radius || jj < radius || kk < radius ||
                       ii > nx - radius - 1 || jj >= ny - radius - 1 ||
                       kk > nz - radius - 1;
            }
        }

        // view for boundary operations
        class boundary_view
        {
            DT* data;
            size_type nx, ny, nz, ii, jj, kk, radius;

          public:
            DUAL boundary_view(
                DT* data,
                size_type nx,
                size_type ny,
                size_type nz,
                size_type ii,
                size_type jj,
                size_type kk,
                size_type radius
            )
                : data(data),
                  nx(nx),
                  ny(ny),
                  nz(nz),
                  ii(ii),
                  jj(jj),
                  kk(kk),
                  radius(radius)
            {
            }

            DUAL DT& interior_value() const
            {
                return data[idx3(ii, jj, kk, nx, ny, nz)];
            }

            DUAL DT& reflecting_value() const
            {
                if constexpr (dim == 1) {
                    return data[idx3(
                        ii < radius         ? 2 * radius - ii - 1
                        : ii >= nx - radius ? 2 * (nx - radius) - ii - 1
                                            : ii,
                        jj,
                        kk,
                        nx,
                        ny,
                        nz
                    )];
                }
                else if constexpr (dim == 2) {
                    return data[idx3(
                        ii < radius         ? 2 * radius - ii - 1
                        : ii >= nx - radius ? 2 * (nx - radius) - ii - 1
                                            : ii,
                        jj < radius         ? 2 * radius - jj - 1
                        : jj >= ny - radius ? 2 * (ny - radius) - jj - 1
                                            : jj,
                        kk,
                        nx,
                        ny,
                        nz
                    )];
                }
                else {
                    return data[idx3(
                        ii < radius         ? 2 * radius - ii - 1
                        : ii >= nx - radius ? 2 * (nx - radius) - ii - 1
                                            : ii,
                        jj < radius         ? 2 * radius - jj - 1
                        : jj >= ny - radius ? 2 * (ny - radius) - jj - 1
                                            : jj,
                        kk < radius         ? 2 * radius - kk - 1
                        : kk >= nz - radius ? 2 * (nz - radius) - kk - 1
                                            : kk,
                        nx,
                        ny,
                        nz
                    )];
                }
            }

            DUAL DT& periodic_value() const
            {
                return data[idx3(
                    ii < radius         ? ii + nx - 2 * radius
                    : ii >= nx - radius ? ii - nx + 2 * radius
                                        : ii,
                    jj < radius         ? jj + ny - 2 * radius
                    : jj >= ny - radius ? jj - ny + 2 * radius
                                        : jj,
                    kk < radius         ? kk + nz - 2 * radius
                    : kk >= nz - radius ? kk - nz + 2 * radius
                                        : kk,
                    nx,
                    ny,
                    nz
                )];
            }

            DUAL auto position() const { return std::make_tuple(ii, jj, kk); }

            DUAL bool is_lower_boundary(int dir) const
            {
                return dir == 0   ? ii < radius
                       : dir == 1 ? jj < radius
                                  : kk < radius;
            }

            DUAL bool is_upper_boundary(int dir) const
            {
                return dir == 0   ? ii >= nx - radius
                       : dir == 1 ? jj >= ny - radius
                                  : kk >= nz - radius;
            }
        };

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

        // Stencil view class for accessing neighboring elements
        template <typename T = DT>
        class stencil_view
        {
          private:
            T* data;
            size_type nx, ny, nz;   // Grid dimensions
            size_type i, j, k;      // Center indices
            size_type radius;       // Stencil radius

          public:
            DUAL stencil_view(
                T* data,
                size_type nx,
                size_type ny,
                size_type nz,
                size_type i,
                size_type j,
                size_type k,
                size_type radius
            )
                : data(data),
                  nx(nx),
                  ny(ny),
                  nz(nz),
                  i(i),
                  j(j),
                  k(k),
                  radius(radius)
            {
            }

            // Get neighboring value at offset
            DUAL T& at(int di, int dj = 0, int dk = 0)
            {
                size_type idx = (k + dk) * nx * ny + (j + dj) * nx + (i + di);
                return data[idx];
            }

            // const reference to neighboring value at offset
            DUAL const T& at(int di, int dj = 0, int dk = 0) const
            {
                size_type idx = (k + dk) * nx * ny + (j + dj) * nx + (i + di);
                return data[idx];
            }

            // Get center value
            DUAL T& center() { return at(0, 0, 0); }

            // const reference to center value
            DUAL const T& center() const { return at(0, 0, 0); }

            DUAL auto indices() const { return std::make_tuple(i, j, k); }
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

        // reductions
        template <typename U, typename BinaryOp>
        U reduce(const ExecutionPolicy<>& policy, U init, BinaryOp binary_op)
            const;

        // Composed operations
        template <typename F, typename G>
        auto compose(F f, G g) const;

        // Chain operations
        template <typename... Funcs>
        auto then(Funcs... fs) const;

        // Chain transformations and return new array
        template <typename... Transforms>
        ndarray transform_chain(Transforms... transforms) const;

        // Apply function to each element safely with bounds checking
        template <typename Func>
        Maybe<ndarray> safe_map(Func f) const;

        // Combine two arrays element-wise with a binary operation
        template <typename Func>
        ndarray combine(const ndarray& other, Func binary_op) const;

        // Split array into chunks for parallel processing
        template <typename Func>
        ndarray
        parallel_chunks(const ExecutionPolicy<>& policy, Func chunk_op) const;

        // transform_parallel method
        // only for functions that do not require index
        template <typename Func>
        auto
        transform_parallel(const ExecutionPolicy<>& policy, Func transform_op)
            const -> std::enable_if_t<
                !has_index_param<Func, const DT&>::value,
                ndarray<std::invoke_result_t<Func, const DT&>, dim>>;

        // only for functions that require index
        template <typename Func>
        auto
        transform_parallel(const ExecutionPolicy<>& policy, Func transform_op)
            const -> std::enable_if_t<
                has_index_param<Func, const DT&>::value,
                ndarray<std::invoke_result_t<Func, const DT&, size_type>, dim>>;

        // transform parallel alongside another ndarray of arbitrary type
        // template <typename T, typename Func>
        // auto transform_parallel_with(
        //     const ExecutionPolicy<>& policy,
        //     const ndarray<T, dim>& other,
        //     Func transform_op
        // ) const;

        template <typename T, typename Func>
        auto transform_parallel_with(
            const ExecutionPolicy<>& policy,
            ndarray<T>& other,
            Func transform_op
        ) const;

        // inplace stencil transform method
        template <typename T, typename Func>
        auto transform_stencil_with(
            const ExecutionPolicy<>& policy,
            const ndarray<T, dim>& stencil_array,
            size_type radius,
            Func stencil_op
        );

        // Boundary region operations
        template <typename BoundaryOp>
        void apply_to_boundaries(
            const ExecutionPolicy<>& policy,
            size_type radius,
            BoundaryOp&& boundary_op
        );
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