// Full implementation file for new_ndarray.hpp
namespace simbi {
    //==========================================================================
    // Memory Manager
    //==========================================================================
    template <typename T>
    void memory_manager<T>::allocate(size_type size)
    {
        this->size_ = size;
        host_data_  = util::make_unique<T[]>(size);
        if constexpr (global::on_gpu) {
            void* ptr;
            gpu::api::malloc(&ptr, this->size_ * sizeof(T));
            device_data_ = unique_ptr<T, gpuDeleter<T>>(ptr);
        }
    }

    template <typename T>
    DUAL T* memory_manager<T>::data()
    {
        if constexpr (global::on_gpu) {
            if (!is_synced_) {
                sync_to_device();
            }
            return device_data_.get();
        }
        return host_data_.get();
    }

    template <typename T>
    DUAL T* memory_manager<T>::data() const
    {
        if constexpr (global::on_gpu) {
            if (!is_synced_) {
                sync_to_device();
            }
            return device_data_.get();
        }
        return host_data_.get();
    }

    template <typename T>
    void memory_manager<T>::sync_to_device()
    {
        if constexpr (global::on_gpu) {
            gpu::api::copyHostToDevice(
                device_data_.get(),
                host_data_.get(),
                this->size_ * sizeof(T)
            );
            is_synced_ = true;
        }
    }

    template <typename T>
    void memory_manager<T>::sync_to_host()
    {
        if constexpr (global::on_gpu) {
            gpu::api::copyDeviceToHost(
                host_data_.get(),
                device_data_.get(),
                this->size_ * sizeof(T)
            );
            is_synced_ = true;
        }
    }

    template <typename T>
    void memory_manager<T>::ensure_device_synced()
    {
        if constexpr (global::on_gpu) {
            if (!is_synced_) {
                sync_to_device();
            }
        }
    }

    //============================================================================
    // Boundary Manager
    //============================================================================
    // private methods
    template <typename T, size_type Dims>
    size_type boundary_manager<T, Dims>::reflecting_idx(
        size_type ii,
        size_type ni,
        size_type radius
    ) const
    {
        if (ii < radius) {
            return 2 * radius - ii - 1;
        }
        else if (ii >= ni + radius) {
            return 2 * (ni + radius) - ii - 1;
        }
        return ii;
    }

    template <typename T, size_type Dims>
    size_type boundary_manager<T, Dims>::periodic_idx(
        size_type ii,
        size_type ni,
        size_type radius
    ) const
    {
        if (ii < radius) {
            return ni + ii;
        }
        else if (ii >= ni + radius) {
            return ii - ni;
        }
        return ii;
    }

    template <typename T, size_type Dims>
    size_type boundary_manager<T, Dims>::outflow_idx(
        size_type ii,
        size_type ni,
        size_type radius
    ) const
    {
        if (ii < radius) {
            return radius;
        }
        else if (ii >= ni + radius) {
            return ni + radius - 1;
        }
        return ii;
    }

    template <typename T, size_type Dims>
    std::array<size_type, Dims> boundary_manager<T, Dims>::unravel_idx(
        size_type idx,
        const uarray<Dims>& shape
    ) const
    {
        return memory_layout_coordinates<Dims>(idx, shape);
    }

    template <typename T, size_type Dims>
    void boundary_manager<T, Dims>::sync_faces(
        const ExecutionPolicy<>& policy,
        ndarray<T, Dims>& full_array,
        const array_view<T, Dims>& interior_view,
        const ndarray<BoundaryCondition>& conditions
    ) const
    {
        auto* data = full_array.data();
        auto radii = [&]() {
            uarray<Dims> rad;
            for (size_type ii = 0; ii < Dims; ++ii) {
                if constexpr (global::col_major) {
                    rad[ii] =
                        (full_array.shape()[ii] - interior_view.shape()[ii]) /
                        2;
                }
                else {
                    rad[ii] = (full_array.shape()[Dims - (ii + 1)] -
                               interior_view.shape()[Dims - (ii + 1)]) /
                              2;
                }
            }
            return rad;
        }();

        parallel_for(policy, [=, this] DEV(size_type idx) {
            auto coordinates = unravel_idx(idx, full_array.shape());
            auto rshape      = interior_view.shape();
            // reverse shape if row major
            if constexpr (!global::col_major) {
                std::reverse(rshape.begin(), rshape.end());
            }

            // Only process boundary points (automatically excludes corners)
            if (!is_boundary_point(coordinates, rshape, radii)) {
                return;
            }

            // Find which dimension's boundary we're on
            int boundary_dim = -1;
            bool is_lower    = false;
            for (size_type dim = 0; dim < Dims; ++dim) {
                if (coordinates[dim] < radii[dim]) {
                    boundary_dim = dim;
                    is_lower     = true;
                    break;
                }
                if (coordinates[dim] >= rshape[dim] + radii[dim]) {
                    boundary_dim = dim;
                    is_lower     = false;
                    break;
                }
            }

            // Process boundary point
            if (boundary_dim >= 0) {
                size_t bc_idx           = 2 * boundary_dim + (is_lower ? 0 : 1);
                const auto interior_idx = get_interior_idx(
                    coordinates,
                    boundary_dim,
                    interior_view.shape(),
                    full_array.strides(),
                    radii,
                    conditions[bc_idx]
                );

                // convert interior index into coordinates
                // auto interior_coords =
                // unravel_idx(interior_idx, full_array.shape());

                // Apply boundary condition
                switch (conditions[bc_idx]) {
                    case BoundaryCondition::REFLECTING:
                        data[idx] = apply_reflecting(
                            data[interior_idx],
                            boundary_dim + 1
                        );
                        break;
                    case BoundaryCondition::PERIODIC:
                        data[idx] = apply_periodic(data[interior_idx]);
                        break;
                    default:   // OUTFLOW
                        data[idx] = data[interior_idx];
                }
            }
        });
    }

    template <typename T, size_type Dims>
    void boundary_manager<T, Dims>::sync_corners(
        const ExecutionPolicy<>& policy,
        ndarray<T, Dims>& full_array,
        const array_view<T, Dims>& interior_view,
        const ndarray<BoundaryCondition>& conditions
    ) const
    {
        auto* data = full_array.data();
        auto radii = [&]() {
            uarray<Dims> rad;
            for (size_type ii = 0; ii < Dims; ++ii) {
                rad[ii] =
                    (full_array.shape()[ii] - interior_view.shape()[ii]) / 2;
            }
            return rad;
        }();

        parallel_for(policy, [=, this] DEV(size_type idx) {
            auto coords = unravel_idx(idx, full_array.shape());
            auto rshape = interior_view.shape();
            // reverse shape if row major
            if constexpr (!global::col_major) {
                std::reverse(rshape.begin(), rshape.end());
            }

            // Only process corner points
            if (!is_corner_point(coords, rshape, radii)) {
                return;
            }

            // Find which dimensions are at boundaries
            std::array<std::pair<int, bool>, Dims> boundary_dims;
            int num_boundaries = 0;

            for (size_type dim = 0; dim < Dims; ++dim) {
                if (coords[dim] < radii[dim]) {
                    boundary_dims[num_boundaries++] = {
                      dim,
                      true
                    };   // true = lower bound
                }
                else if (coords[dim] >= rshape[dim] + radii[dim]) {
                    boundary_dims[num_boundaries++] = {
                      dim,
                      false
                    };   // false = upper bound
                }
            }

            // Get interior indices for each boundary dimension
            size_type interior_idx = 0;
            auto int_coords        = coords;

            for (int i = 0; i < num_boundaries; ++i) {
                auto [dim, is_lower] = boundary_dims[i];
                const auto bc_idx    = 2 * dim + (is_lower ? 0 : 1);
                const auto bc        = conditions[bc_idx];

                // Update coordinate based on boundary condition
                switch (bc) {
                    case BoundaryCondition::REFLECTING:
                        int_coords[dim] = (is_lower)
                                              ? 2 * radii[dim] - coords[dim] - 1
                                              : 2 * (rshape[dim] + radii[dim]) -
                                                    coords[dim] - 1;
                        break;
                    case BoundaryCondition::PERIODIC:
                        int_coords[dim] = (is_lower)
                                              ? rshape[dim] + coords[dim]
                                              : coords[dim] - rshape[dim];
                        break;
                    default:   // OUTFLOW
                        int_coords[dim] = (is_lower)
                                              ? radii[dim]
                                              : rshape[dim] + radii[dim] - 1;
                }
            }

            // Calculate interior linear index
            for (size_type d = 0; d < Dims; d++) {
                interior_idx += int_coords[d] * full_array.strides()[d];
            }
            // std::cout << coords << " -> " << int_coords << std::endl;

            // Apply boundary conditions
            data[idx] = data[interior_idx];

            // Handle reflecting conditions for momentum/magnetic components
            if constexpr (is_conserved_v<T>) {
                for (int i = 0; i < num_boundaries; ++i) {
                    auto [dim, is_lower] = boundary_dims[i];
                    const auto bc_idx    = 2 * dim + (is_lower ? 0 : 1);
                    if (conditions[bc_idx] == BoundaryCondition::REFLECTING) {
                        data[idx].mcomponent(dim + 1) *= -1.0;
                        if constexpr (is_relativistic_mhd<T>::value) {
                            data[idx].bcomponent(dim + 1) *= -1.0;
                        }
                    }
                }
            }
        });
    }

    // public methods
    template <typename T, size_type Dims>
    void boundary_manager<T, Dims>::sync_boundaries(
        const ExecutionPolicy<>& policy,
        ndarray<T, Dims>& full_array,
        const array_view<T, Dims>& interior_view,
        const ndarray<BoundaryCondition>& conditions,
        const bool need_corners
    ) const
    {
        // Sync faces
        sync_faces(policy, full_array, interior_view, conditions);

        // Sync corners if needed
        if constexpr (comp_ct_type != CTTYPE::MdZ) {
            if constexpr (is_conserved_v<T>) {
                if (need_corners) {
                    sync_corners(policy, full_array, interior_view, conditions);
                }
            }
        }
    }

    //==========================================================================
    // Array View
    //==========================================================================
    template <typename T, size_type Dims>
    array_view<T, Dims>::array_view(
        const ndarray<T, Dims>& source,
        T* data,
        const uarray<Dims>& shape,
        const uarray<Dims>& strides,
        const uarray<Dims>& offsets
    )
        : array_properties<T, Dims>(
              shape,
              strides,
              offsets,
              this->compute_size(shape)
          ),
          source_(&source),
          data_(data, source.size())
    {
    }

    template <typename T, size_type Dims>
    DUAL array_view<T, Dims>::value_type&
    array_view<T, Dims>::operator[](size_type ii)
    {
        return access(data_[ii] + this->compute_offset(this->offsets_));
    }

    template <typename T, size_type Dims>
    DUAL array_view<T, Dims>::value_type&
    array_view<T, Dims>::operator[](size_type ii) const
    {
        return access(data_[ii] + this->compute_offset(this->offsets_));
    }

    template <typename T, size_type Dims>
    template <typename... Indices>
    T& array_view<T, Dims>::at(Indices... indices)
    {
        static_assert(sizeof...(Indices) == Dims);
        uarray<Dims> idx{static_cast<size_type>(indices)...};
        size_type offset = 0;

        if constexpr (global::col_major) {
            // Column major (k,j,i)
            for (size_type d = 0; d < Dims; ++d) {
                if (idx[d] >= this->shape_[d]) {
                    return data_[0];   // bounds check
                }
                offset += idx[d] * this->strides_[d];
            }
        }
        else {
            // Row major (i,j,k)
            for (size_type d = Dims - 1; d >= 0; --d) {
                if (idx[d] >= this->shape_[d]) {
                    return data_[0];   // bounds check
                }
                offset += idx[d] * this->strides_[d];
            }
        }
        return access(data_[offset]);
    }

    template <typename T, size_type Dims>
    DUAL auto array_view<T, Dims>::data() -> T*
    {
        return data_.data();
    }

    template <typename T, size_type Dims>
    DUAL auto& array_view<T, Dims>::access(T& val)
    {
        if constexpr (is_maybe_v<T>) {
            return val.value();
        }
        else {
            return val;
        }
    }

    template <typename T, size_type Dims>
    DUAL auto& array_view<T, Dims>::access(T& val) const
    {
        if constexpr (is_maybe_v<T>) {
            return val.value();
        }
        else {
            return val;
        }
    }

    template <typename T, size_type Dims>
    DUAL auto array_view<T, Dims>::data() const -> T*
    {
        return data_.data();
    }

    // function to print std::array elements
    template <typename T, size_type N>
    void print_array(const std::array<T, N>& arr)
    {
        for (const auto& elem : arr) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    template <typename T, size_type Dims>
    template <typename... DependentViews, typename F>
    void array_view<T, Dims>::stencil_transform(
        F op,
        const ExecutionPolicy<>& policy,
        const DependentViews&... arrays
    )
    {
        if constexpr (global::on_gpu) {
            // TODO: Implement GPU version
        }
        else {
            parallel_for(policy, [=, this](size_type idx) {
                // Get global coordinates
                auto pos = this->get_local_coords(idx);

                // Create span from data pointer
                std::span<T> data_span(this->data(), this->source_size());
                stencil_view center_view(
                    data_span,
                    pos,
                    this->shape(),
                    this->strides(),
                    this->offsets()
                );

                auto all_views = std::make_tuple(
                    center_view,
                    stencil_view<typename array_raw_type<DependentViews>::type>(
                        std::span<
                            typename array_raw_type<DependentViews>::type>(
                            arrays.data(),
                            arrays.source_size()
                        ),
                        arrays.get_local_coords(idx),
                        arrays.shape(),
                        arrays.strides(),
                        arrays.offsets()
                    )...
                );
                // center_view.value();
                // std::apply(op, all_views);

                center_view.value() = std::apply(op, all_views);
                // data_[idx] = std::apply(op, all_views);
            });
        }
    }

    //==========================================================================
    // NDArray
    //==========================================================================

    template <typename T, size_type Dims>
    ndarray<T, Dims>::ndarray(
        std::initializer_list<size_type> dims,
        T fill_value
    )
    {
        size_type size = 1;
        for (auto dim : dims) {
            size *= dim;
        }
        mem_.allocate(size);
        this->shape_   = dims;
        this->strides_ = this->compute_strides(this->shape_);
        fill(fill_value);
    }

    template <typename T, size_type Dims>
    ndarray<T, Dims>::ndarray(std::vector<T>&& data)
    {
        mem_.allocate(data.size());
        std::copy(data.begin(), data.end(), mem_.data());
        this->size_    = data.size();
        this->shape_   = {data.size()};
        this->strides_ = this->compute_strides(this->shape_);
        // check values after copy
        for (size_type ii = 0; ii < this->size(); ++ii) {
            assert(mem_[ii] == data[ii]);
        }
        // clear and release the vector
        data.clear();
        data.shrink_to_fit();
    }

    template <typename T, size_type Dims>
    void ndarray<T, Dims>::sync_to_device()
    {
        mem_.sync_to_device();
    }

    template <typename T, size_type Dims>
    void ndarray<T, Dims>::sync_to_host()
    {
        mem_.sync_to_host();
    }

    template <typename T, size_type Dims>
    auto& ndarray<T, Dims>::reshape(const collapsable<Dims>& new_shape)
    {
        // Verify total size matches
        size_type new_size = this->compute_size(new_shape.vals);
        assert(new_size == this->size_ && "New shape must match total size");

        // Update shape and strides
        this->shape_   = new_shape.vals;
        this->strides_ = this->compute_strides(this->shape_);
        return *this;
    }

    template <typename T, size_type Dims>
    auto& ndarray<T, Dims>::reshape(
        const collapsable<Dims>& new_shape,
        const collapsable<Dims>& new_strides
    )
    {
        // Verify total size matches
        size_type new_size = this->compute_size(new_shape.vals);
        assert(new_size == this->size_ && "New shape must match total size");

        // Update shape and strides
        this->shape_   = new_shape.vals;
        this->strides_ = new_strides.vals;
        return *this;
    }

    template <typename T, size_type Dims>
    auto& ndarray<T, Dims>::resize(size_type size, T fill_value)
    {
        mem_.allocate(size);
        this->size_ = size;
        return *this;
        // fill(fill_value);
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::data() -> T*
    {
        return mem_.data();
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::data() const -> const T*
    {
        return mem_.data();
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::fill(T value) -> void
    {
        if constexpr (global::on_gpu) {
            mem_.ensure_device_synced();
            parallel_for(
                [=, this] DEV(size_type ii) { mem_[ii] = value; },
                this->size()
            );
            mem_.sync_to_host();
        }
        else {
            std::fill(mem_.data(), mem_.data() + this->size(), value);
        }
    }

    template <typename T, size_type Dims>
    template <typename... Indices>
    DUAL T& ndarray<T, Dims>::at(Indices... indices)
    {
        collapsable<Dims> idx{static_cast<size_type>(indices)...};
        size_type offset = 0;

        for (size_type d = 0; d < Dims; ++d) {
            offset += idx[d] * this->strides_[d];
        }
        return access(mem_[offset]);
    }

    template <typename T, size_type Dims>
    template <typename... Indices>
    DUAL T& ndarray<T, Dims>::at(Indices... indices) const
    {
        collapsable<Dims> idx{static_cast<size_type>(indices)...};
        size_type offset = 0;

        for (size_type d = 0; d < Dims; ++d) {
            offset += idx[d] * this->strides_[d];
        }
        return access(mem_[offset]);
    }

    // contract dimensions uniformly
    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::contract(size_type radius) -> array_view<T, Dims>
    {
        uarray<Dims> new_shape;
        uarray<Dims> offsets;

#pragma unroll
        for (size_type ii = 0; ii < Dims; ++ii) {
            new_shape[ii] = this->shape_[ii] - 2 * radius;
            offsets[ii]   = radius;
        }

        return array_view<T, Dims>(
            *this,
            mem_.data(),
            new_shape,
            this->strides_,
            offsets
        );
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::contract(const collapsable<Dims>& radii)
        -> array_view<T, Dims>
    {
        uarray<Dims> new_shape;
        uarray<Dims> offsets;

#pragma unroll
        for (size_type ii = 0; ii < Dims; ++ii) {
            // Only contract dimensions with non-zero radius
            new_shape[ii] = radii.vals[ii] > 0
                                ? this->shape_[ii] - 2 * radii.vals[ii]
                                : this->shape_[ii];

            offsets[ii] = radii.vals[ii];   // Zero for uncontracted dimensions
        }

        return array_view<T, Dims>(
            *this,
            mem_.data(),
            new_shape,
            this->strides_,
            offsets
        );
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::view(const collapsable<Dims>& ranges) const
        -> array_view<T, Dims>
    {
        uarray<Dims> offsets;
        // Verify ranges are valid
        // for (size_type ii = 0; ii < Dims; ++ii) {
        //     if (ranges.vals[ii] > this->shape_[ii]) {
        //         throw std::runtime_error("View range exceeds array
        //         dimensions");
        //     }
        // }

        for (size_type ii = 0; ii < Dims; ++ii) {
            // Offset is 0 for each dimension being viewed
            offsets[ii] = 0;
        }

        return array_view<T, Dims>(
            *this,
            mem_.data(),      // Start at beginning
            ranges.vals,      // New shape is the ranges
            this->strides_,   // Strides stay the same
            offsets           // Start at 0 offset
        );
    }

    template <typename T, size_type Dims>
    template <typename F>
    void ndarray<T, Dims>::transform(F op, const ExecutionPolicy<>& policy)
    {
        if constexpr (global::on_gpu) {
            mem_.ensure_device_synced();
            parallel_for(policy, [=, this] DEV(size_type ii) {
                mem_[ii] = op(mem_[ii]);
            });
            policy.synchronize();
        }
        else {
            parallel_for(policy, [=, this](size_type ii) {
                mem_[ii] = op(mem_[ii]);
            });
        }
    }

    template <typename T, size_type Dims>
    template <typename... DependentArrays, typename F>
    void ndarray<T, Dims>::transform(
        F op,
        const ExecutionPolicy<>& policy,
        const DependentArrays&... arrays
    )
    {
        if constexpr (global::on_gpu) {
            mem_.ensure_device_synced();
            parallel_for(policy, [=, this] DEV(size_type ii) {
                mem_[ii] = op(mem_[ii], arrays[ii]...);
            });
            policy.synchronize();
        }
        else {
            parallel_for(policy, [=, this](size_type ii) {
                mem_[ii] = op(mem_[ii], arrays[ii]...);
            });
        }
    }

    template <typename T, size_type Dims>
    template <typename... DependentArrays, typename F>
    void ndarray<T, Dims>::transform(
        F op,
        const ExecutionPolicy<>& policy,
        DependentArrays&... arrays
    )
    {
        if constexpr (global::on_gpu) {
            mem_.ensure_device_synced();
            parallel_for(policy, [=, this] DEV(size_type ii) {
                mem_[ii] = op(mem_[ii], arrays[ii]...);
            });
            policy.synchronize();
        }
        else {
            parallel_for(policy, [&](size_type ii) {
                mem_[ii] = op(mem_[ii], arrays[ii]...);
            });
        }
    }

    // reduce with operation that takes a, b, and the index
    template <typename T, size_type Dims>
    template <typename U, typename F>
    U ndarray<T, Dims>::reduce(
        U init,
        F reduce_op,
        const ExecutionPolicy<>& policy
    ) const
    {
        if constexpr (global::on_gpu) {
            ndarray<U> result(1, init);
            result.sync_to_device();
            auto result_ptr = result.mem_.data();
            auto arr        = mem_.data();

            parallel_for(policy, [=, this] DEV(size_type idx) {
                U current_min = result_ptr[0];
                while (true) {
                    U next_min = reduce_op(current_min, arr[idx], idx);
                    if (result_ptr[0] == current_min) {
                        result_ptr[0] = next_min;
                        break;
                    }
                    current_min = result_ptr[0];
                }
            });

            result.sync_to_host();
            return result[0];
        }
        else {
            std::atomic<U> result(init);
            const size_type batch_size  = policy.batch_size;
            const size_type num_batches = policy.get_num_batches(this->size());

            parallel_for(policy, num_batches, [&](size_type bid) {
                const size_type start = bid * batch_size;
                const size_type end =
                    std::min(start + batch_size, this->size());

                // Local reduction: (accumulator, element, index)
                U local_result = init;
                for (size_type ii = start; ii < end; ii++) {
                    local_result = reduce_op(local_result, mem_[ii], ii);
                }

                // Global merge: (accumulator, element, index)
                bool success;
                do {
                    U expected = result.load(std::memory_order_relaxed);
                    success    = result.compare_exchange_weak(
                        expected,
                        reduce_op(expected, mem_[start], start),
                        std::memory_order_release,
                        std::memory_order_relaxed
                    );
                } while (!success);
            });

            return result.load(std::memory_order_acquire);
        }
    }

    template <typename T, size_type Dims>
    DUAL auto& ndarray<T, Dims>::access(T& val)
    {
        if constexpr (is_maybe_v<T>) {
            return val.value();
        }
        else {
            return val;
        }
    }

    template <typename T, size_type Dims>
    DUAL auto& ndarray<T, Dims>::access(T& val) const
    {
        if constexpr (is_maybe_v<T>) {
            return val.value();
        }
        else {
            return val;
        }
    }

    template <typename T, size_type Dims>
    DUAL ndarray<T, Dims>::value_type& ndarray<T, Dims>::operator[](size_type ii
    )
    {
        return access(mem_[ii]);
    }

    template <typename T, size_type Dims>
    DUAL ndarray<T, Dims>::value_type& ndarray<T, Dims>::operator[](size_type ii
    ) const
    {
        return access(mem_[ii]);
    }

}   // namespace simbi
