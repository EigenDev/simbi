/**
@file dist_array.hpp
@brief Header file for the distribution manager class template.
*/
#ifndef DIST_ARRAY_HPP
#define DIST_ARRAY_HPP

#include "array_view.hpp"
#include "build_options.hpp"
#include <initializer_list>
#include <vector>

namespace simbi {
    template <size_type Dims, typename T>
    class distributed_ndarray
    {

      public:
        // ctors
        distributed_ndarray()                                      = default;
        distributed_ndarray(const distributed_ndarray&)            = default;
        distributed_ndarray(distributed_ndarray&&)                 = default;
        distributed_ndarray& operator=(const distributed_ndarray&) = default;
        distributed_ndarray& operator=(distributed_ndarray&&)      = default;

        explicit distributed_ndarray(
            std::initializer_list<size_type> dims,
            std::vector<size_type> devices = {0},
            T init_value                   = T()
        );

        // basic operations
        void fill(const T& value)
        {
            host_data_ = ndarray<T, Dims>(dims, fill_value);
            for (size_type ii = 0; ii < Dims; ii++) {
                // TODO: make haloes match spatial order
                halo_sizes_[ii] = 1;
            }

            if (devices.size() <= 1) {
                is_distributed_ = false;

                DeviceFragment fragment;
                fragment.device_id = devices.empty() ? 0 : devices[0];
                fragment.data      = host_data_;
                fragment.dirty     = false;

                // set up domain to cover the whole array
                for (size_type ii; ii < Dims; ii++) {
                    fragment.domain.start_indices[ii] = 0;
                    fragment.domain.end_indices[ii]   = host_data_.shape()[ii];
                    fragment.domain.local_shape[ii]   = host_data_.shape()[ii];
                    // no halos for single device
                    fragment.domain.halo_size[ii] = 0;
                }

                device_data_.push_back(fragment);
            }
            else {
                is_distributed_ = true;
                initialize_distribution(devices);
                compute_device_domains();
            }

            sync_to_devices();
        }
        void sync_to_devices()
        {
            for (size_type ii = 0; ii < device_fragments_.size(); ++ii) {
                device_fragments_[ii].data.copy_from(host_data_);
            }
            host_dirty_ = false;
        }
        void sync_to_host();
        void exchange_halos();

        // accessors
        T& operator[](size_type ii);
        const T& operator[](size_type ii) const;

        template <typename... Indices>
        DUAL T& at(Indices... indices);

        // device-specific view
        array_view<T, Dims> device_view(size_type device) const;

        // size
        size_type size() const;
        size_type dim(size_type i) const;
        const uarray<Dims>& shape() const;

        // operations
        template <typename F>
        void transform(F op, const ExecutionPolicy<>& policy);

        template <typename F>
        void transform(F op, const ExecutionPolicy<>& policy) const;

        template <typename F, typename... Arrays>
        void transform(
            F op,
            const ExecutionPolicy<>& policy,
            const distributed_ndarray<N, T>& other,
            Arrays... others
        );

      private:
        ndarray<T, Dims> host_data_;   // host data

        // per-device fragments
        struct DeviceDomain {
            int device_id;
            uarray<Dims> start_indices;   // start indices of the fragment
            uarray<Dims> end_indices;     // end indices of the fragment
            uarray<Dims> local_shape;     // local shape of the fragment
            uarray<Dims> halo_size;
        };

        // per-device data
        struct DeviceFragment {
            int device_id;
            ndarray<T, Dims> data;
            DeviceDomain domain;
            bool dirty;
        };

        std::vector<DeviceFragment> device_fragments_;
        uarray<Dims> halo_sizes_;
        bool is_distributed_ = false;
        bool host_dirty_     = false;

        // helper functions
        void initialize_distribution(const std::vector<int>& devices);
        int get_owning_device(const uarray<Dims>& indices) const;
        void compute_device_domains();
        uarray<Dims> global_to_local(
            int device_idx,
            const uarray<Dims>& gloval_indices
        ) const;
        uarray<Dims> local_to_global(
            int device_idx,
            const uarray<Dims>& local_indices
        ) const;
    };
}   // namespace simbi

#endif   // DIST_ARRAY_HPP
