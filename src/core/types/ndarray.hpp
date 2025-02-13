#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "build_options.hpp"                // for global::on_gpu, rea;
#include "core/traits.hpp"                  // for is_maybe
#include "core/types/enums.hpp"             // for BoundaryCondition
#include "core/types/idx_sequence.hpp"      // for make_idx_sequence
#include "core/types/maybe.hpp"             // for Maybe
#include "core/types/smart_ptr.hpp"         // for smart_ptr<T[]>
#include "util/parallel/exec_policy.hpp"    // for ExecutionPolicy
#include "util/parallel/parallel_for.hpp"   // for parallel_for
#include "util/tools/helpers.hpp"           // for
#include <array>                            // for array
#include <cassert>                          // for assert
#include <span>                             // for span

namespace simbi {
    template <size_type T>
    using uarray = std::array<size_type, T>;

    template <size_type T>
    using iarray = std::array<lint, T>;

    // overload ostream to print std::array
    template <typename T, size_type N>
    std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr)
    {
        os << "[";
        for (size_type ii = 0; ii < N; ++ii) {
            os << arr[ii];
            if (ii < N - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

    template <typename T, typename Deleter>
    using unique_ptr = util::smart_ptr<T[], Deleter>;

    template <typename T>
    struct gpuDeleter {
        void operator()(T* ptr) { gpu::api::free(ptr); }
    };

    template <typename T, size_type Dims = 1>
    class ndarray;

    template <typename T, size_type Dims>
    class array_view;

    template <typename T, size_type Dims>
    class value_array_view;

    struct range {
        size_type start;
        size_type end;
        size_type step;
    };

    template <size_type Dims>
    DUAL static auto
    memory_layout_coordinates(auto idx, const uarray<Dims>& shape)
        -> uarray<Dims>
    {
        uarray<Dims> coords;
        auto stride = 1;
        if constexpr (global::col_major) {
            // Column major: shape=(nk,nj,ni)
            // Want [k,j,i] where i is fastest
            for (size_type ii = 0; ii < Dims; ++ii) {
                coords[Dims - 1 - ii] = (idx / stride) % shape[Dims - 1 - ii];
                stride *= shape[Dims - 1 - ii];
            }
        }
        else {
            // Row major: shape=(nk,nj,ni)
            // Want [i,j,k] where i is fastest
            for (size_type ii = Dims - 1; ii < Dims; --ii) {
                coords[Dims - 1 - ii] = (idx / stride) % shape[ii];
                stride *= shape[ii];
            }
        }

        return coords;
    }

    template <typename T, size_type Dims>
    struct array_properties {
        // Ctor
        array_properties(
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& offsets,
            size_type size
        )
            : shape_(shape), strides_(strides), offsets_(offsets), size_(size)
        {
        }

        // default ctor
        array_properties() = default;

        // copy ctor
        array_properties(const array_properties&) = default;
        // move ctor
        array_properties(array_properties&&) = default;

        // // copy assignment
        array_properties& operator=(const array_properties&) = default;

        // helper to compute local coordinates from linear index
        DUAL uarray<Dims> get_local_coords(size_type idx) const
        {
            return memory_layout_coordinates<Dims>(idx, shape_);
        }

        DUAL uarray<Dims>
        get_local_coords(size_type idx, const auto shape) const
        {
            return memory_layout_coordinates<Dims>(idx, shape);
        }

        DUAL static size_type compute_size(const uarray<Dims>& dims)
        {
            size_type size = 1;
#pragma unroll
            for (size_type ii = 0; ii < Dims; ++ii) {
                size *= dims[ii];
            }
            return size;
        }

        DUAL static auto compute_strides(const uarray<Dims>& dims)
            -> std::array<size_t, Dims>
        {
            uarray<Dims> strides;

            if constexpr (global::col_major) {
                // Column major (i,j,k): k fastest
                strides[Dims - 1] = 1;   // k stride
                for (size_type ii = Dims - 2; ii >= 0; --ii) {
                    strides[ii] = strides[ii + 1] * dims[ii + 1];
                }
                // Result: strides = {nj*nk, nk, 1}
                // For (i,j,k) input order
            }
            else {
                // Row major (i,j,k): i fastest
                strides[0] = 1;   // i stride
                for (size_type ii = 1; ii < Dims; ++ii) {
                    strides[ii] = strides[ii - 1] * dims[Dims - ii];
                }
                // Result: strides = {1, ni, ni*nj}
                // For (i,j,k) input order
            }

            return strides;
        };

        DUAL size_type compute_offset(const uarray<Dims>& offsets) const
        {
            size_type offset = 0;
#pragma unroll
            for (size_type ii = 0; ii < Dims; ++ii) {
                offset += offsets[ii] * strides_[ii];
            }
            return offset;
        }

        DUAL auto strides() const -> uarray<Dims> { return strides_; };

        DUAL auto offsets() const -> uarray<Dims> { return offsets_; };

        DUAL auto shape() const -> uarray<Dims> { return shape_; };

        DUAL auto size() const -> size_type { return size_; };

      protected:
        uarray<Dims> shape_;
        uarray<Dims> strides_;
        uarray<Dims> offsets_;
        size_type size_{0};
    };

    template <size_type Dims>
    struct collapsable {

        uarray<Dims> vals;

        constexpr collapsable() = default;

        constexpr collapsable(std::initializer_list<size_type> init)
        {
            // fill from the back of the initialize list, since we
            // are in general inputting shapes like (nk, nj, ni)
            auto init_size = std::distance(init.begin(), init.end());
            auto start     = init.begin();
            if (init_size > Dims) {
                std::advance(start, init_size - Dims);
            }
            for (size_type i = 0; i < Dims && start != init.end();
                 ++i, ++start) {
                vals[i] = *start;
            }
        }

        // accesor to get the value at index
        constexpr size_type operator[](size_type ii) const { return vals[ii]; }

        // implicit conversion to uarray
        constexpr operator uarray<Dims>() const { return vals; }

        constexpr ~collapsable() = default;
    };

    template <typename T>
    class memory_manager

    {
      public:
        void allocate(size_type size);
        DUAL T* data();
        DUAL T* data() const;

        void sync_to_device();
        void sync_to_host();
        void ensure_device_synced();

        // access operators
        DUAL T& operator[](size_type ii) { return data()[ii]; }

        DUAL T& operator[](size_type ii) const { return data()[ii]; }

        // host data accessors
        DUAL T* host_data() { return host_data_.get(); }

        DUAL T* host_data() const { return host_data_.get(); }

      private:
        util::smart_ptr<T[]> host_data_;
        unique_ptr<T, gpuDeleter<T>> device_data_;
        bool is_synced_{true};
        size_type size_{0};
    };

    template <typename T, size_type Dims>
    class array_view : public array_properties<T, Dims>
    {
      public:
        using raw_type = T;
        using value_type =
            typename std::conditional_t<is_maybe_v<T>, get_value_type_t<T>, T>;
        DUAL array_view(
            const ndarray<T, Dims>& source,
            T* data,
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& offsets
        );

        // Allow copying but track source
        array_view(const array_view&)            = default;
        array_view& operator=(const array_view&) = default;

        DUAL value_type& operator[](size_type ii);
        DUAL value_type& operator[](size_type ii) const;
        template <typename... Indices>
        DUAL T& at(Indices... idx);
        DUAL auto data() const -> T*;
        DUAL auto data() -> T*;
        DUAL auto& access(T& val);
        DUAL auto& access(T& val) const;

        // transform using stencil views of dependent ndarrays views
        template <typename... DependentViews, typename F>
        void stencil_transform(
            F op,
            const ExecutionPolicy<>& policy,
            const DependentViews&... arrays
        );

        DUAL auto source_size() const { return data_.size(); }

        DUAL auto source_shape() const { return source_->shape(); }

        // Position-aware element that can do relative indexing
        template <typename DT = T>
        class stencil_view
        {
          public:
            DUAL stencil_view(
                std::span<DT> data,
                const uarray<Dims>& center_pos,
                const uarray<Dims>& shape,
                const uarray<Dims>& strides,
                const uarray<Dims>& offsets
            )
                : data_(data),
                  center_(center_pos),
                  shape_(shape),
                  strides_(strides),
                  offsets_(offsets)
            {
            }

            // Relative indexing from center
            DUAL get_value_type_t<DT>& at(int i, int j = 0, int k = 0) const
            {
                iarray<Dims> coords;
                size_type idx = 0;

                // center_ is in interior coordinates, convert to global by
                // adding offset
                if constexpr (global::col_major) {
                    // Column major: (i,j,k) -> k fastest
                    // Add offset to get global position
                    coords[0] = center_[0] + offsets_[0] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[1] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[2] + k;
                    }
                }
                else {
                    // Row major: (i,j,k) -> i fastest
                    coords[0] = center_[0] + offsets_[Dims - 1] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[Dims - 2] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[Dims - 3] + k;
                    }
                }

                // Calculate global linear index
                for (size_type d = 0; d < Dims; ++d) {
                    idx += coords[d] * strides_[d];
                }

                return access(data_[idx]);
            }

            // Get center position
            DUAL const auto position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2];
                }
                return pos3d;
            }

            // get global position
            DUAL const auto global_position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0] + offsets_[Dims - 1];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1] + offsets_[Dims - 2];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2] + offsets_[Dims - 3];
                }
                return pos3d;
            }

            // Direct value access at center
            DUAL get_value_type_t<DT>& value() const { return at(0); }

            // structured binding support
            template <size_type I>
            DUAL auto get() const
            {
                if constexpr (I < Dims) {
                    return position()[I];
                }
                else {
                    throw size_type{0};
                }
            }

            // we also need special access method in case we get maybe type
            DUAL auto& access(DT& val) const
            {
                if constexpr (has_value_type<DT>::value) {
                    return val.value();
                }
                else {
                    return val;
                }
            }

          private:
            // DT* data_;
            std::span<DT> data_;
            uarray<Dims> center_;
            uarray<Dims> shape_;
            uarray<Dims> strides_;
            uarray<Dims> offsets_;
        };

      protected:
        // non-owning pointer to source data
        const ndarray<T, Dims>* source_;
        // view into data
        std::span<T> data_;
    };

    template <typename T, size_type Dims>
    class boundary_manager
    {
      public:
        void sync_boundaries(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions,
            const bool need_corners = false
        ) const;

      private:
        size_type
        reflecting_idx(size_type ii, size_type ni, size_type radius) const;
        size_type
        periodic_idx(size_type ii, size_type ni, size_type radius) const;
        size_type
        outflow_idx(size_type ii, size_type ni, size_type radius) const;
        uarray<Dims>
        unravel_idx(size_type idx, const uarray<Dims>& shape) const;
        void sync_faces(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions
        ) const;
        void sync_corners(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions
        ) const;

        DUAL static bool is_boundary_point(
            const uarray<Dims>& coordinates,
            const uarray<Dims>& shape,
            const uarray<Dims>& radii
        )
        {
            int boundary_count = 0;
            for (size_type ii = 0; ii < Dims; ++ii) {
                if (coordinates[ii] < radii[ii] ||
                    coordinates[ii] >= shape[ii] + radii[ii]) {
                    boundary_count++;
                }
            }

            // True only if exactly one dimension is at a boundary
            return boundary_count == 1;
        }

        DUAL static bool is_corner_point(
            const uarray<Dims>& coordinates,
            const uarray<Dims>& shape,
            const uarray<Dims>& radii
        )
        {
            size_type boundary_count = 0;

            // Count how many dimensions are at boundaries
            for (size_type ii = 0; ii < Dims; ++ii) {
                if (coordinates[ii] < radii[ii] ||
                    coordinates[ii] >= shape[ii] + radii[ii]) {
                    boundary_count++;
                }
                if (boundary_count >= 2) {
                    return true;   // We're at a corner when 2+ dimensions
                                   // are at boundaries
                }
            }
            return false;
        }

        DUAL size_type get_interior_idx(
            const uarray<Dims>& coords,
            size_type dim,
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& radii,
            BoundaryCondition bc
        ) const
        {
            // Copy coordinates
            auto int_coords = coords;

            auto tshape = shape;
            // get inverted copy of shape if we are in row major
            if constexpr (!global::col_major) {
                std::reverse(tshape.begin(), tshape.end());
            }

            // Adjust coordinate based on boundary condition
            switch (bc) {
                case BoundaryCondition::REFLECTING:
                    int_coords[dim] =
                        (coords[dim] < radii[dim])
                            ? 2 * radii[dim] - coords[dim] - 1
                            : 2 * (tshape[dim] + radii[dim]) - coords[dim] - 1;
                    break;

                case BoundaryCondition::PERIODIC:
                    int_coords[dim] = (coords[dim] < radii[dim])
                                          ? tshape[dim] + coords[dim]
                                          : coords[dim] - tshape[dim];
                    break;

                default:   // OUTFLOW
                    int_coords[dim] = (coords[dim] < radii[dim])
                                          ? radii[dim]
                                          : tshape[dim] + radii[dim] - 1;
            }

            // Calculate linear index directly
            size_type idx = 0;
            for (size_type i = 0; i < Dims; i++) {
                idx += int_coords[i] * strides[i];
            }
            return idx;
        }

        template <typename U>
        DUAL static U apply_reflecting(const U& val, int momentum_idx)
        {
            auto result = val;
            if constexpr (is_conserved_v<T>) {
                result.mcomponent(momentum_idx) *= -1.0;
            }
            return result;
        }

        template <typename U>
        DUAL static U apply_periodic(const U& val)
        {
            return val;
        }

        template <typename U>
        DUAL static U apply_outflow(const U& val)
        {
            return val;
        }

        template <typename U>
        DUAL static U apply_dynamic(
            const U& val,
            const auto& source_fn,
            const real x1,
            const real x2,
            const real x3,
            const real t
        )
        {
            return source_fn(val, x1, x2, x3, t);
        }

        // Add plane sequence helper
        template <int num_dims>
        struct PlaneSequence {
            using type = typename std::conditional_t<
                num_dims == 1,
                detail::index_sequence<int>,
                typename std::conditional_t<
                    num_dims == 2,
                    detail::index_sequence<int, static_cast<int>(Plane::IJ)>,
                    detail::index_sequence<
                        int,
                        static_cast<int>(Plane::IJ),
                        static_cast<int>(Plane::IK),
                        static_cast<int>(Plane::JK)>>>;
        };

        template <Plane P>
        struct PlaneInfo {
            static constexpr std::array<std::pair<int, int>, 4> bc_pairs = [] {
                if constexpr (P == Plane::IJ) {
                    return std::array{
                      std::make_pair(0, 2),   // (min_i, min_j)
                      std::make_pair(0, 3),   // (min_i, max_j)
                      std::make_pair(1, 2),   // (max_i, min_j)
                      std::make_pair(1, 3)    // (max_i, max_j)
                    };
                }
                else if constexpr (P == Plane::IK) {
                    return std::array{
                      std::make_pair(0, 4),   // (min_i, min_k)
                      std::make_pair(0, 5),   // (min_i, max_k)
                      std::make_pair(1, 4),   // (max_i, min_k)
                      std::make_pair(1, 5)    // (max_i, max_k)
                    };
                }
                else {   // Plane::JK
                    return std::array{
                      std::make_pair(2, 4),   // (min_j, min_k)
                      std::make_pair(2, 5),   // (min_j, max_k)
                      std::make_pair(3, 4),   // (max_j, min_k)
                      std::make_pair(3, 5)    // (max_j, max_k)
                    };
                }
            }();
        };
    };

    template <typename T, size_type Dims>
    class ndarray : public array_properties<T, Dims>
    {
      public:
        using value_type =
            typename std::conditional_t<is_maybe_v<T>, get_value_type_t<T>, T>;
        ndarray() = default;
        explicit ndarray(
            std::initializer_list<size_type> dims,
            T fill_value = T()
        );
        ndarray(const ndarray&)     = default;
        ndarray(ndarray&&) noexcept = default;
        explicit ndarray(std::vector<T>&& data);

        auto data() -> T*;
        auto data() const -> const T*;
        auto fill(T value) -> void;

        template <typename... Indices>
        DUAL T& at(Indices... idx);
        template <typename... Indices>
        DUAL T& at(Indices... idx) const;
        DUAL auto& access(T& val);
        DUAL auto& access(T& val) const;

        // contraction method for viewing subarrays
        auto contract(const size_type radius) -> array_view<T, Dims>;
        auto contract(const collapsable<Dims>& radii) -> array_view<T, Dims>;
        auto view(const collapsable<Dims>& ranges) const -> array_view<T, Dims>;

        template <typename F>
        void transform(F op, const ExecutionPolicy<>& policy);

        // transform with variadic dependent ndarrays
        template <typename... DependentArrays, typename F>
        void transform(
            F op,
            const ExecutionPolicy<>& policy,
            const DependentArrays&... arrays
        );

        // transform with variadic dependent ndarrays
        // that are mutable
        template <typename... DependentArrays, typename F>
        void transform(
            F op,
            const ExecutionPolicy<>& policy,
            DependentArrays&... arrays
        );

        template <typename U, typename F>
        U reduce(U init, F reduce_op, const ExecutionPolicy<>& policy) const;

        // access operators
        DUAL value_type& operator[](size_type i);
        DUAL value_type& operator[](size_type i) const;

        void sync_to_device();
        void sync_to_host();
        auto& reshape(const collapsable<Dims>& new_shape);
        auto& reshape(
            const collapsable<Dims>& new_shape,
            const collapsable<Dims>& new_strides
        );
        auto& resize(size_type size, T fill_value = T());

        T* host_data() { return mem_.host_data(); }

        T* host_data() const { return mem_.host_data(); }

        class iterator
        {
          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = typename ndarray::value_type;
            using difference_type   = std::ptrdiff_t;
            using pointer           = value_type*;
            using reference         = value_type&;

            DUAL iterator(ndarray& arr, size_type pos = 0)
                : array_(arr), pos_(pos)
            {
            }

            DUAL reference operator*() { return array_[pos_]; }

            DUAL iterator& operator++()
            {
                ++pos_;
                return *this;
            }

            DUAL iterator operator++(int)
            {
                iterator tmp = *this;
                ++pos_;
                return tmp;
            }

            DUAL bool operator!=(const iterator& other) const
            {
                return pos_ != other.pos_;
            }

            DUAL bool operator==(const iterator& other) const
            {
                return pos_ == other.pos_;
            }

          private:
            ndarray& array_;
            size_type pos_;
        };

        // Add iterator support
        DUAL iterator begin() { return iterator(*this); }

        DUAL iterator end() { return iterator(*this, this->size_); }

        DUAL auto cbegin() const { return iterator(*this); }

        DUAL auto cend() const { return iterator(*this, this->size_); }

      private:
        memory_manager<T> mem_;
    };
}   // namespace simbi

#include "ndarray.ipp"
#endif