#ifndef ATOMIC_ACCUMULATOR_HPP
#define ATOMIC_ACCUMULATOR_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "util/tools/device_api.hpp"

namespace simbi {
    template <typename T, size_type Dims>
    class AtomicAccumulator
    {
      private:
        ndarray<T> data_;
        size_type size_;

      public:
        AtomicAccumulator(size_type size) : size_(size)
        {
            data_.resize(size, T(0));
            data_.sync_to_device();
        }

        // atomic addition for both CPU and GPU
        DEV void add(size_type idx, T value)
        {
            if (idx >= size_) {
                return;
            }
            if constexpr (global::on_gpu) {
                gpu::api::atomicAdd(&data_[idx], value);
            }
            else {
#pragma omp atomic
                data_[idx] += value;
            }
        }

        // get accumulated value (non-atomic read)
        DEV T get(size_t idx) const
        {
            if (idx >= size_) {
                return T(0);
            }
            return data_[idx];
        }

        // reset all values to zero
        void reset()
        {
            for (size_t i = 0; i < size_; ++i) {
                data_[i] = T(0);
            }
            data_.sync_to_device();
        }

        // sync data from device to host
        void sync_to_host() { data_.sync_to_host(); }

        constexpr auto& operator[](size_t idx) const { return data_[idx]; }
        constexpr const auto& operator[](size_t idx) { return data_[idx]; }
    };
}   // namespace simbi

#endif
