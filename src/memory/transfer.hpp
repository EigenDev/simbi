#ifndef TRANSFER_HPP
#define TRANSFER_HPP

#include "accessor.hpp"
#include "arena.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "hetero/adapter.hpp"
#include "hetero/core/common_types.hpp"
#include "memory/device.hpp"
#include "memory_block.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace simbi::mem {
    //==========================================================================
    // Low-level memory block transfers
    //==========================================================================

    inline void
    copy(const memory_block_t& src, memory_block_t& dst, size_t bytes)
    {
        if (!src.data() || !dst.data()) {
            return;
        }

        bytes = std::min({bytes, src.size(), dst.size()});
        if (bytes == 0) {
            return;
        }

        // choose appropriate transfer method based on device types
        if (src.device().is_gpu && dst.device().is_gpu) {
            if (src.device().device_id == dst.device().device_id) {
                // same GPU to GPU
                hetero::device::set_device(src.device().device_id);
                hetero::device::copy(
                    dst.data(),
                    src.data(),
                    bytes,
                    hetero::memory_direction_t::device_to_device
                );
            }
            else {
                // peer GPU to GPU
                hetero::device::peer_copy_async(
                    dst.data(),
                    dst.device().device_id,
                    src.data(),
                    src.device().device_id,
                    bytes,
                    hetero::stream{}
                );   // default stream
            }
        }
        else if (src.device().is_gpu && !dst.device().is_gpu) {
            // GPU to CPU
            hetero::device::set_device(src.device().device_id);
            hetero::device::copy(
                dst.data(),
                src.data(),
                bytes,
                hetero::memory_direction_t::device_to_host
            );
        }
        else if (!src.device().is_gpu && dst.device().is_gpu) {
            // CPU to GPU
            hetero::device::set_device(dst.device().device_id);
            hetero::device::copy(
                dst.data(),
                src.data(),
                bytes,
                hetero::memory_direction_t::host_to_device
            );
        }
        else {
            // CPU to CPU
            std::memcpy(dst.data(), src.data(), bytes);
        }
    }

    // typed variants for convenience
    template <typename T>
    void
    copy_typed(const memory_block_t& src, memory_block_t& dst, size_t count)
    {
        copy(src, dst, count * sizeof(T));
    }

    template <typename T>
    void to_host(const memory_block_t& src, T* host_ptr, size_t count)
    {
        if (!src.data() || !host_ptr) {
            return;
        }

        const size_t bytes = count * sizeof(T);
        if (bytes > src.size()) {
            return;
        }

        if (src.device().is_gpu) {
            hetero::device::set_device(src.device().device_id);
            hetero::device::copy(
                host_ptr,
                src.data(),
                bytes,
                hetero::memory_direction_t::device_to_host
            );
        }
        else {
            std::memcpy(host_ptr, src.data(), bytes);
        }
    }

    template <typename T>
    void to_device(const T* host_ptr, memory_block_t& dst, size_t count)
    {
        if (!host_ptr || !dst.data()) {
            return;
        }

        const size_t bytes = count * sizeof(T);
        if (bytes > dst.size()) {
            return;
        }

        if (dst.device().is_gpu) {
            hetero::device::set_device(dst.device().device_id);
            hetero::device::copy(
                dst.data(),
                host_ptr,
                bytes,
                hetero::memory_direction_t::host_to_device
            );
        }
        else {
            std::memcpy(dst.data(), host_ptr, bytes);
        }
    }

    //==========================================================================
    // High-level accessor transfers
    //==========================================================================

    // copy entire accessor contents
    template <typename T, std::uint64_t Dims>
    void copy(const accessor_t<T, Dims>& src, accessor_t<T, Dims>& dst)
    {
        if (!src.is_allocated() || !dst.is_allocated()) {
            throw std::runtime_error(
                "Cannot transfer between unallocated accessors"
            );
        }

        if (src.size() != dst.size()) {
            throw std::runtime_error(
                "Cannot copy between accessors of different sizes"
            );
        }

        // create temporary memory blocks wrapping the data
        device_t src_device = src.arena()->device();
        device_t dst_device = dst.arena()->device();

        memory_block_t src_block(
            const_cast<T*>(src.data()),
            src.size() * sizeof(T),
            src_device
        );

        memory_block_t dst_block(
            dst.data(),
            dst.size() * sizeof(T),
            dst_device
        );

        // use low-level copy
        copy(src_block, dst_block, src.size() * sizeof(T));

        // prevent double-free when blocks go out of scope
        // src_block.data = nullptr;
        // dst_block.data = nullptr;
    }

    // copy a specific domain region
    template <typename T, std::uint64_t Dims>
    void copy_region(
        const accessor_t<T, Dims>& src,
        accessor_t<T, Dims>& dst,
        const domain_t<Dims>& region
    )
    {
        if (!src.is_allocated() || !dst.is_allocated()) {
            throw std::runtime_error(
                "Cannot transfer between unallocated accessors"
            );
        }

        // validate the region is contained within both source and destination
        // domains
        if (!domain_algebra::contains(src.domain(), region) ||
            !domain_algebra::contains(dst.domain(), region)) {
            throw std::runtime_error(
                "Region is not contained within both accessor domains"
            );
        }

        // detect if we can do a contiguous copy - only for 1D domains
        bool can_use_contiguous = (Dims == 1);

        // when devices are different or non-contiguous regions, we need
        // element-by-element copy
        if (!can_use_contiguous ||
            src.arena()->device().index != dst.arena()->device().index) {
            // create a temporary host buffer for the transfer
            std::vector<T> buffer(region.size());

            // copy region from source to host buffer
            std::size_t idx = 0;

            // iterate through all coordinates in the region
            iarray<Dims> coord = region.start;
            do {
                buffer[idx++] = src(coord);
            } while (increment_coord(coord, region.end));

            // copy from host buffer to destination
            idx   = 0;
            coord = region.start;
            do {
                dst(coord) = buffer[idx++];
            } while (increment_coord(coord, region.end));
        }
        else {
            // for contiguous 1D domains, we can do a direct memory copy
            std::size_t elem_count = region.size();

            device_t src_device = src.arena()->device();
            device_t dst_device = dst.arena()->device();

            memory_block_t src_block(
                const_cast<T*>(&src(region.start)),
                elem_count * sizeof(T),
                src_device
            );

            memory_block_t dst_block(
                &dst(region.start),
                elem_count * sizeof(T),
                dst_device
            );

            copy(src_block, dst_block, elem_count * sizeof(T));

            // prevent double-free
            // src_block.data = nullptr;
            // dst_block.data = nullptr;
        }
    }

    // generate ghost regions for a domain
    template <std::uint64_t Dims>
    domain_t<Dims>
    ghost_region(const domain_t<Dims>& domain, int ghost_width = 1)
    {
        iarray<Dims> ghost_amount;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            ghost_amount[ii] = ghost_width;
        }

        return domain_algebra::expand(domain, ghost_amount);
    }

    // exchange ghost regions between partitioned domains
    template <typename T, std::uint64_t Dims>
    void exchange_ghosts(
        std::vector<accessor_t<T, Dims>>& accessors,
        const std::vector<domain_t<Dims>>& partitions,
        int ghost_width = 1
    )
    {
        if (accessors.size() != partitions.size()) {
            throw std::runtime_error(
                "Number of accessors must match number of partitions"
            );
        }

        if (accessors.size() <= 1) {
            return;   // nothing to exchange
        }

        for (size_t ii = 0; ii < accessors.size(); ++ii) {
            // get ghost region for this partition
            domain_t<Dims> expanded = ghost_region(partitions[ii], ghost_width);

            // check each other partition for overlap with ghost region
            for (size_t jj = 0; jj < accessors.size(); ++jj) {
                if (ii == jj) {
                    continue;
                }

                // find intersection between expanded area and other partition
                domain_t<Dims> overlap =
                    domain_algebra::intersection(expanded, partitions[jj]);

                if (!overlap.empty()) {
                    // copy the overlapping region from other partition to this
                    // one
                    copy_region(accessors[jj], accessors[ii], overlap);
                }
            }
        }
    }

    // multi-device operations

    // copy an accessor to multiple devices, partitioning by domain
    template <typename T, std::uint64_t Dims>
    std::vector<accessor_t<T, Dims>> distribute(
        const accessor_t<T, Dims>& src,
        const std::vector<domain_t<Dims>>& partitions,
        const std::vector<std::shared_ptr<arena_t<T>>>& device_arenas
    )
    {
        if (partitions.size() != device_arenas.size()) {
            throw std::runtime_error(
                "Number of partitions must match number of device arenas"
            );
        }

        std::vector<accessor_t<T, Dims>> result;
        result.reserve(partitions.size());

        for (size_t ii = 0; ii < partitions.size(); ++ii) {
            // create accessor for this partition on the target device
            accessor_t<T, Dims> dst(partitions[ii], device_arenas[ii]);

            // copy region from source to this partition
            copy_region(src, dst, partitions[ii]);

            result.push_back(std::move(dst));
        }

        return result;
    }

    // gather results from multiple devices back to a single accessor
    template <typename T, std::uint64_t Dims>
    void gather(
        const std::vector<accessor_t<T, Dims>>& srcs,
        const std::vector<domain_t<Dims>>& partitions,
        accessor_t<T, Dims>& dst
    )
    {
        if (srcs.size() != partitions.size()) {
            throw std::runtime_error(
                "Number of source accessors must match number of partitions"
            );
        }

        for (size_t ii = 0; ii < srcs.size(); ++ii) {
            // copy each partition to the destination
            copy_region(srcs[ii], dst, partitions[ii]);
        }
    }

}   // namespace simbi::mem

#endif   // TRANSFER_HPP
