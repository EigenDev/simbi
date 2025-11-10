#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "hetero/adapter.hpp"
#include "hetero/core/common_types.hpp"
#include "memory/arena.hpp"
#include "memory/device.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace simbi::mem {

    // memory allocation on a specific device
    template <typename T, std::uint64_t Dims>
    struct buffer_t {
        device_t device;
        std::shared_ptr<T[]> data_;
        domain_t<Dims> domain_;
        iarray<Dims> strides_;
        std::shared_ptr<arena_t<T>> arena_;

        // lightweight accessor for use in computations
        struct accessor_t {
            T* raw_data;
            iarray<Dims> strides;
            iarray<Dims> start;
            device_t device;

            DUAL T& operator()(const coordinate_t<Dims>& coord) const
            {
                auto offset = vecops::dot(coord - start, strides);
                return raw_data[offset];
            }
        };

        // compute row-major strides from shape
        static iarray<Dims> compute_strides(const iarray<Dims>& shape)
        {
            iarray<Dims> result;
            result[Dims - 1] = 1;
            for (std::int64_t ii = static_cast<std::int64_t>(Dims) - 2; ii >= 0;
                 --ii) {
                result[ii] = result[ii + 1] * shape[ii + 1];
            }
            return result;
        }

        // default ctor
        buffer_t() = default;

        // construct with arena allocation
        buffer_t(
            const domain_t<Dims>& domain,
            std::shared_ptr<arena_t<T>> arena
        )
            : device(arena->device()),
              data_(arena->get(domain.size())),
              domain_(domain),
              strides_(compute_strides(domain.shape())),
              arena_(arena)
        {
            if (!data_) {
                throw std::runtime_error("buffer allocation failed");
            }
        }

        // construct with explicit device (creates/reuses arena)
        buffer_t(const domain_t<Dims>& domain, device_t dev)
            : buffer_t(domain, arena<T>(dev))
        {
        }

        // construct wrapping existing memory (no arena)
        buffer_t(
            const domain_t<Dims>& domain,
            std::shared_ptr<T[]> data,
            device_t dev
        )
            : device(dev),
              data_(data),
              domain_(domain),
              strides_(compute_strides(domain.shape())),
              arena_(nullptr)
        {
            if (!data_) {
                throw std::runtime_error("buffer data is null");
            }
        }

        // queries
        const domain_t<Dims>& domain() const { return domain_; }
        const iarray<Dims>& strides() const { return strides_; }
        std::size_t size() const { return domain_.size(); }
        bool is_allocated() const { return data_ != nullptr; }

        // raw data accessf
        T* data() { return data_.get(); }
        const T* data() const { return data_.get(); }

        // get accessor for use in computations
        accessor_t accessor() const
        {
            return {data_.get(), strides_, domain_.start, device};
        }

        // explicit clone to same or different device
        buffer_t<T, Dims> clone(device_t target_device) const
        {
            auto target_arena = arena<T>(target_device);
            buffer_t<T, Dims> result(domain_, target_arena);

            if (device == target_device) {
                // same device - direct copy
                if (device.is_gpu) {
                    // gpu to same gpu
                    hetero::device::copy(
                        result.data(),
                        data_.get(),
                        domain_.size() * sizeof(T),
                        hetero::memory_direction_t::device_to_device
                    );
                }
                else {
                    // cpu to cpu
                    std::copy_n(data_.get(), domain_.size(), result.data());
                }
            }
            else if (device.is_gpu && target_device.is_gpu) {
                // gpu to gpu
                hetero::device::peer_copy(
                    result.data(),
                    target_device.device_id,
                    data_.get(),
                    device.device_id,
                    domain_.size() * sizeof(T)
                );
            }
            else if (device.is_gpu && !target_device.is_gpu) {
                // gpu to cpu
                hetero::device::copy(
                    result.data(),
                    data_.get(),
                    domain_.size() * sizeof(T),
                    hetero::memory_direction_t::device_to_host
                );
            }
            else if (!device.is_gpu && target_device.is_gpu) {
                // cpu to gpu
                hetero::device::copy(
                    result.data(),
                    data_.get(),
                    domain_.size() * sizeof(T),
                    hetero::memory_direction_t::host_to_device
                );
            }
            else {
                // cpu to cpu
                std::copy_n(data_.get(), domain_.size(), result.data());
            }

            return result;
        }

        // clone to same device
        buffer_t<T, Dims> clone() const { return clone(device); }

        buffer_t<T, Dims> slice(const domain_t<Dims>& subdomain) const
        {
            if (!domain_algebra::contains(domain_, subdomain)) {
                throw std::runtime_error(
                    "slice subdomain not contained in buffer domain"
                );
            }

            // create the view
            return buffer_t<T, Dims>(
                data_,
                domain_,
                strides_,
                subdomain,
                arena_,
                device
            );
        }

        DUAL T& operator()(const coordinate_t<Dims>& coord)
        {
            auto offset = vecops::dot(coord - domain_.start, strides_);
            return data_[offset];
        }

        DUAL const T& operator()(const coordinate_t<Dims>& coord) const
        {
            auto offset = vecops::dot(coord - domain_.start, strides_);
            return data_[offset];
        }

      private:
        buffer_t(
            std::shared_ptr<T[]> parent_data,      // parent's data ptr
            const domain_t<Dims>& parent_domain,   // parent's logical domain
            const iarray<Dims>& parent_strides,    // parent's strides
            const domain_t<Dims>& subdomain,   // the *absolute* domain to slice
            std::shared_ptr<arena_t<T>> arena,   // parent's arena
            device_t dev                         // parent's device
        )
            : device(dev),
              domain_(make_domain(subdomain.shape())),   // normalized domain
              strides_(parent_strides),   // inherit parent's strides
              arena_(arena)
        {
            // offset data pointer
            auto offset = vecops::dot(
                subdomain.start - parent_domain.start,
                parent_strides
            );
            data_ =
                std::shared_ptr<T[]>(parent_data, parent_data.get() + offset);
        }
    };

    // factory functions
    template <typename T, std::uint64_t Dims>
    buffer_t<T, Dims>
    allocate_buffer(const domain_t<Dims>& domain, device_t dev)
    {
        return buffer_t<T, Dims>(domain, dev);
    }

    template <typename T, std::uint64_t Dims>
    buffer_t<T, Dims> buffer_from_data(
        const T* host_data,
        const domain_t<Dims>& domain,
        device_t dev = device_t::cpu()
    )
    {
        buffer_t<T, Dims> result(domain, dev);

        if (dev.is_gpu) {
            hetero::device::copy(
                result.data(),
                host_data,
                domain.size() * sizeof(T),
                hetero::memory_direction_t::host_to_device
            );
        }
        else {
            std::copy_n(host_data, domain.size(), result.data());
        }

        return result;
    }

}   // namespace simbi::mem

#endif   // BUFFER_HPP
