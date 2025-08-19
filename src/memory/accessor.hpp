#ifndef UNIFIED_ACCESSOR_HPP
#define UNIFIED_ACCESSOR_HPP

#include "adapter/device_adapter_api.hpp"
#include "arena.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "io/exceptions.hpp"
#include "memory/device.hpp"
#include "smart_ptr.hpp"
#include "traits/traits.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace simbi::mem {
    /**
     * accessor_t - host-device agnostic field storage
     *
     * srp: provide coordinate-based access to arena-backed memory
     * works on both cpu and gpu via DUAL annotations
     */
    template <typename T, std::uint64_t Dims>
    class accessor_t
    {
        mem::shared_ptr<T> data_;
        std::shared_ptr<arena_t<T>> arena_;
        domain_t<Dims> domain_;
        iarray<Dims> strides_;
        device_id_t device_;

      public:
        using value_type                          = T;
        static constexpr std::uint64_t dimensions = Dims;

        // ctors
        accessor_t() = default;

        accessor_t(
            domain_t<Dims> domain,
            device_id_t device = device_id_t::cpu_device()
        )
            : arena_(get_arena_for_device<T>(device)),
              domain_(domain),
              strides_(compute_strides(domain.shape())),
              device_(device)
        {
            data_ = arena_->get(domain_.size());
        }

        ~accessor_t() = default;
        // copy / move semantics (cheap shallow copy)
        accessor_t(const accessor_t& other)                = default;
        accessor_t& operator=(const accessor_t& other)     = default;
        accessor_t(accessor_t&& other) noexcept            = default;
        accessor_t& operator=(accessor_t&& other) noexcept = default;

        // host-device element access
        DUAL const T& operator[](coordinate_t<Dims> coord) const
        {
            return data_.get()[compute_offset(coord)];
        }

        DUAL T& operator[](coordinate_t<Dims> coord)
        {
            return data_.get()[compute_offset(coord)];
        }

        DUAL const T& operator()(coordinate_t<Dims> coord) const
        {
            return (*this)[coord];
        }

        DUAL T& operator()(coordinate_t<Dims> coord) { return (*this)[coord]; }

        // direct data access
        DUAL const T* data() const { return data_.get(); }

        DUAL T* data() { return data_.get(); }

        // queries
        device_id_t device() const { return device_; }
        bool on_cpu() const { return device_.type == device_type_t::cpu; }
        bool on_gpu() const { return device_.type == device_type_t::gpu; }
        bool on_same_device(const accessor_t& other) const
        {
            return (device_ == other.device_) &&
                   (device_.type == other.device_.type);
        }
        const domain_t<Dims>& domain() const { return domain_; }
        std::size_t size() const { return domain_.size(); }
        bool is_allocated() const { return static_cast<bool>(data_); }

        // materialization interface for compute fields
        template <typename ComputeField, typename Executor>
        void commit(const ComputeField& computation, const Executor& executor)
        {
            // ensure we have a valid pool
            if (!arena_) {
                arena_ = global_arena<T>();
            }

            // ensure we have allocated memory
            if (!is_allocated()) {
                // allocate with same domain as computation
                data_    = arena_->get(computation.domain().size());
                domain_  = computation.domain();
                strides_ = compute_strides(domain_.shape());
            }

            using source_result_t = ComputeField::value_type;

            if constexpr (is_maybe_v<source_result_t>) {
                // maybe types - use parallel execution with error counting
                auto error_count =
                    executor
                        .reduce(
                            domain_,
                            std::size_t{0},
                            [this, computation] DUAL(coordinate_t<Dims> coord)
                                -> std::size_t {
                                auto maybe_result = computation(coord);
                                if (maybe_result.has_value()) {
                                    (*this)[coord] = maybe_result.value();
                                    return 0;   // no error
                                }
                                else {
                                    return 1;   // error occurred
                                }
                            },
                            std::plus{}
                        )
                        .wait();

                if (error_count > 0) {
                    throw exception::SimulationFailureException();
                }
            }
            else {
                // regular types - standard parallel execution
                executor
                    .for_each(
                        domain_,
                        [this, computation] DUAL(coordinate_t<Dims> coord) {
                            (*this)[coord] = computation(coord);
                        }
                    )
                    .wait();
            }
        }

        auto clone() const
        {
            auto new_accessor = accessor_t{domain_, device_};
            copy_data_to(new_accessor);
            return new_accessor;
        }

        // factory methods
        static auto zeros(
            const iarray<Dims>& shape,
            device_id_t device = device_id_t::cpu_device()
        )
        {
            auto domain   = make_domain(shape);
            auto accessor = accessor_t{domain, device};
            accessor.zero_fill();
            return accessor;
        }

        // static auto from_numpy(
        //     T* numpy_data,
        //     const iarray<Dims>& shape,
        //     std::shared_ptr<arena_t<T>> pool = global_arena<T>()
        // )
        // {
        //     auto domain   = make_domain(shape);
        //     auto accessor = accessor_t{domain, std::move(pool)};

        //     // copy from numpy
        //     std::copy_n(numpy_data, domain.size(), accessor.data());

        //     // deallocate the numpy array since we own the data now
        //     numpy_data = nullptr;

        //     return accessor;
        // }
        static auto from_numpy(
            T* numpy_data,
            const iarray<Dims>& shape,
            device_id_t device = device_id_t::cpu_device()
        )
        {
            auto domain   = make_domain(shape);
            auto accessor = accessor_t{domain, device};

            if (device.type == device_type_t::gpu) {
                gpu::api::set_device(device.device_id);
                gpu::api::copy_host_to_device(
                    accessor.data_.get(),
                    numpy_data,
                    domain.size() * sizeof(T)
                );
            }
            else {
                std::copy_n(numpy_data, domain.size(), accessor.data_.get());
            }

            return accessor;
        }

      private:
        DUAL std::size_t compute_offset(coordinate_t<Dims> coord) const
        {
            return vecops::dot(coord - domain_.start, strides_);
        }

        static iarray<Dims> compute_strides(const iarray<Dims>& shape)
        {
            iarray<Dims> strides;
            strides[Dims - 1] = 1;
            for (std::int64_t i = Dims - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }

        void zero_fill()
        {
            if (device_.type == device_type_t::gpu) {
                gpu::api::set_device(device_.device_id);
                gpu::api::memset(data_.get(), 0, domain_.size() * sizeof(T));
            }
            else {
                std::fill_n(data_.get(), domain_.size(), T{});
            }
        }

        void copy_data_to(const accessor_t& target) const
        {
            if (!on_same_device(target)) {
                throw std::runtime_error(
                    "Cross-device copy not implemented yet - use explicit "
                    "transfers"
                );
            }

            // same device copy
            if (device_.type == device_type_t::gpu) {
                gpu::api::set_device(device_.device_id);
                gpu::api::copy_device_to_device(
                    target.data_.get(),
                    data_.get(),
                    domain_.size() * sizeof(T)
                );
            }
            else {
                std::copy_n(data_.get(), domain_.size(), target.data_.get());
            }
        }
    };

}   // namespace simbi::mem

#endif
