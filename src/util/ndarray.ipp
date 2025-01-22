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
    )
    {
        if (ii < radius) {
            return 2 * radius - ii - 1;
        }
        else if (ii >= ni + radius) {
            printf("ii: %zu, ni: %zu, radius: %zu\n", ii, ni, radius);
            return 2 * (ni + radius) - ii - 1;
        }
        return ii;
    }

    template <typename T, size_type Dims>
    size_type boundary_manager<T, Dims>::periodic_idx(
        size_type ii,
        size_type ni,
        size_type radius
    )
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
    )
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
    )
    {
        auto idx_shift = [&](const size_type ii) {
            if constexpr (global::col_major) {
                return ii;
            }
            else {
                return -(ii + 1) % Dims;
            }
        };
        uarray<Dims> coordinates;
        for (size_type ii = 0; ii < Dims; ++ii) {
            coordinates[ii] = idx % shape[idx_shift(ii)];
            idx /= shape[idx_shift(ii)];
        }
        return coordinates;
    }

    template <typename T, size_type Dims>
    void boundary_manager<T, Dims>::sync_faces(
        const ExecutionPolicy<>& policy,
        ndarray<T, Dims>& full_array,
        const array_view<T, Dims>& interior_view,
        const ndarray<BoundaryCondition>& conditions
    )
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
            auto coordinates = unravel_idx(idx, full_array.shape());
            auto rshape      = interior_view.shape();
            // reverse shape if row major
            if constexpr (!global::col_major) {
                std::reverse(rshape.begin(), rshape.end());
            }
            // For each dimension, check boundaries
            for (size_type dim = 0; dim < Dims; ++dim) {
                if (coordinates[dim] < radii[dim] ||
                    coordinates[dim] >= rshape[dim] + radii[dim]) {
                    // Get interior index and apply boundary condition
                    const auto interior_idx = get_interior_idx(
                        coordinates,
                        dim,
                        interior_view.shape(),
                        full_array.strides(),
                        radii,
                        coordinates[dim] < radii[dim] ? conditions[2 * dim]
                                                      :   // Lower boundary
                            conditions[2 * dim + 1]       // Upper boundary
                    );
                    // printf("idx: %zu, interior_idx: %zu\n", idx,
                    // interior_idx); std::cin.get();

                    // Apply boundary condition
                    switch (
                        conditions
                            [2 * dim + (coordinates[dim] < radii[dim] ? 0 : 1)]
                    ) {
                        case BoundaryCondition::REFLECTING:
                            data[idx] =
                                apply_reflecting(data[interior_idx], dim + 1);
                            break;
                        case BoundaryCondition::PERIODIC:
                            data[idx] = apply_periodic(data[interior_idx]);
                            break;
                        default:   // OUTFLOW
                            data[idx] = data[interior_idx];
                    }
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
    )
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

        // Use PlaneSequence to handle different dimensions
        using planes = typename PlaneSequence<Dims>::type;

        // Iterate through planes using index sequence
        detail::for_sequence(planes{}, [&](auto plane_idx) {
            constexpr Plane P = static_cast<Plane>(plane_idx());

            // Get boundary condition pairs for this plane
            constexpr auto& bc_pairs = PlaneInfo<P>::bc_pairs;

            parallel_for(policy, [=, this] DEV(size_type idx) {
                auto coords = unravel_idx(idx, full_array.shape());

                // Check if we're at a corner
                if (is_corner_point(coords, interior_view.shape(), radii)) {
                    // For each corner configuration in this plane
                    for (const auto& [bc1_idx, bc2_idx] : bc_pairs) {
                        const auto bc1 = conditions[bc1_idx];
                        const auto bc2 = conditions[bc2_idx];

                        // Apply corner boundary conditions
                        if constexpr (P == Plane::IJ) {
                            helpers::handle_corner<T, Plane::IJ>(
                                data,
                                idx,
                                coords[0],
                                coords[1],
                                coords[2],
                                full_array.shape()[0],
                                full_array.shape()[1],
                                full_array.shape()[2],
                                radii[0],
                                bc1,
                                bc2,
                                1,
                                2
                            );
                        }
                        else if constexpr (P == Plane::IK) {
                            helpers::handle_corner<T, Plane::IK>(
                                data,
                                idx,
                                coords[0],
                                coords[1],
                                coords[2],
                                full_array.shape()[0],
                                full_array.shape()[1],
                                full_array.shape()[2],
                                radii[0],
                                bc1,
                                bc2,
                                1,
                                3
                            );
                        }
                        else {   // Plane::JK
                            helpers::handle_corner<T, Plane::JK>(
                                data,
                                idx,
                                coords[0],
                                coords[1],
                                coords[2],
                                full_array.shape()[0],
                                full_array.shape()[1],
                                full_array.shape()[2],
                                radii[0],
                                bc1,
                                bc2,
                                2,
                                3
                            );
                        }
                    }
                }
            });
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
    )
    {
        // Sync faces
        sync_faces(policy, full_array, interior_view, conditions);

        // Sync corners if needed
        if (need_corners) {
            sync_corners(policy, full_array, interior_view, conditions);
        }
    }

    //==========================================================================
    // Array View
    //==========================================================================
    template <typename T, size_type Dims>
    array_view<T, Dims>::array_view(
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
          data_(data, this->size_)
    {
    }

    template <typename T, size_type Dims>
    DUAL array_view<T, Dims>::value_type&
    array_view<T, Dims>::operator[](size_type ii)
    {
        return access(data_[ii]);
    }

    template <typename T, size_type Dims>
    DUAL array_view<T, Dims>::value_type&
    array_view<T, Dims>::operator[](size_type ii) const
    {
        return access(data_[ii]);
    }

    template <typename T, size_type Dims>
    template <typename... Indices>
    T& array_view<T, Dims>::at(Indices... indices)
    {
        static_assert(sizeof...(Indices) == Dims, "invalid number of indices");

        uarray<Dims> idx{static_cast<size_type>(indices)...};
        size_type offset = 0;

#pragma unroll
        for (size_type ii = 0; ii < Dims; ++ii) {
            if (idx[ii] >= this->shape_[ii]) {
                // GPU-safe bounds check
                // Return first element on invalid access
                return data_[0];
            }
            offset += idx[ii] * this->strides_[ii];
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

                // Create centered view for main array
                stencil_view
                    center_view(data(), pos, this->shape(), this->strides());
                auto all_views = std::make_tuple(
                    center_view,
                    stencil_view<typename array_raw_type<DependentViews>::type>(
                        arrays.data(),
                        arrays.get_local_coords(idx),
                        arrays.shape(),
                        arrays.strides()
                    )...
                );

                // auto active_idx = global_linear_idx(idx);
                // printf("idx: %zu, aidx: %zu\n", idx, active_idx);
                // Create centered views for dependent arrays
                center_view.value() = std::apply(op, all_views);
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
        // printf("reshaping\n");
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
        static_assert(sizeof...(Indices) == Dims, "Invalid number of indices");

        uarray<Dims> idx{static_cast<size_type>(indices)...};
        size_type offset = 0;

#pragma unroll
        for (size_type ii = 0; ii < Dims; ++ii) {
            if (idx[ii] >= this->shape_[ii]) {
                // GPU-safe bounds check
                // Return first element on invalid access
                return mem_.data[0];
            }
            offset += idx[ii] * this->strides_[ii];
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
            mem_.data() + this->compute_offset(offsets),
            new_shape,
            this->strides_,
            offsets
        );
    }

    template <typename T, size_type Dims>
    auto ndarray<T, Dims>::view(const uarray<Dims>& ranges) const
        -> array_view<T, Dims>
    {
        uarray<Dims> offsets;
        for (size_type ii = 0; ii < Dims; ++ii) {
            offsets[ii] = this->shape_[ii] - ranges[ii];
        }
        return array_view<T, Dims>(
            mem_.data() + this->compute_offset(offsets),
            this->shape_,
            this->strides_,
            offsets
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

                // Thread-local reduction
                U local_result = init;
                for (size_type ii = start; ii < end; ii++) {
                    local_result = reduce_op(local_result, mem_[ii], ii);
                }

                // Atomic reduction using actual array element
                U current_min = result.load();
                while (true) {
                    U next_min = reduce_op(current_min, mem_[start], start);
                    if (result.compare_exchange_weak(current_min, next_min)) {
                        break;
                    }
                }
            });

            return result.load();
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
