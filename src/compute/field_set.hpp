#ifndef FIELD_SET_HPP
#define FIELD_SET_HPP

#include "compat.hpp"
#include "computation.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "exec_context.hpp"
#include "execution/executor.hpp"
#include "execution/future.hpp"
#include "field.hpp"
#include "het/adapter.hpp"
#include "het/core/types.hpp"
#include "memory/device.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace simbi {

    // distributed field across multiple devices
    // owns multiple field_t shards, one per device
    template <typename T, std::uint64_t Rank>
    struct field_set_t {
        std::vector<field_t<T, Rank>> shards;
        std::vector<domain_t<Rank>> partitions;
        domain_t<Rank> global_domain_;

        // construction from devices

        // ctor using device pool
        explicit field_set_t(const domain_t<Rank>& domain, device_pool_t& pool)
            : global_domain_(domain)
        {
            std::size_t n_devices = pool.size();
            if (n_devices == 0) {
                throw executor_error_t(
                    "device pool for distributed_field is empty"
                );
            }

            partitions = partition_domain(domain, n_devices);

            shards.reserve(n_devices);
            for (std::size_t ii = 0; ii < n_devices; ++ii) {
                shards.emplace_back(partitions[ii], pool.get_next_device());
            }
        }
        field_set_t(
            const domain_t<Rank>& domain,
            const std::vector<mem::device_t>& devices
        )
            : global_domain_(domain)
        {
            if (devices.empty()) {
                throw std::runtime_error(
                    "field_set_t requires at least one device"
                );
            }

            // partition domain across devices
            partitions = partition_domain(domain, devices.size());

            // allocate shard on each device
            shards.reserve(devices.size());
            for (std::size_t ii = 0; ii < devices.size(); ++ii) {
                shards.emplace_back(partitions[ii], devices[ii]);
            }
        }

        // partition domain along largest axis
        static std::vector<domain_t<Rank>>
        partition_domain(const domain_t<Rank>& domain, std::size_t n_parts)
        {
            auto shape         = domain.shape();
            std::uint64_t axis = 0;
            for (std::uint64_t ii = 1; ii < Rank; ++ii) {
                if (shape[ii] > shape[axis]) {
                    axis = ii;
                }
            }

            std::vector<domain_t<Rank>> result;
            result.reserve(n_parts);

            for (std::size_t ii = 0; ii < n_parts; ++ii) {
                result.push_back(domain.partition(n_parts, ii, axis));
            }

            return result;
        }

        // queries
        const domain_t<Rank>& domain() const { return global_domain_; }
        std::size_t num_shards() const { return shards.size(); }
        const std::vector<domain_t<Rank>>& shard_domains() const
        {
            return partitions;
        }

        // find which shard contains a coordinate
        std::size_t find_shard(const coordinate_t<Rank>& coord) const
        {
            for (std::size_t ii = 0; ii < partitions.size(); ++ii) {
                if (partitions[ii].contains(coord)) {
                    return ii;
                }
            }
            throw std::runtime_error("coordinate not in any shard");
        }

        // access shard by index
        field_t<T, Rank>& operator[](std::size_t idx) { return shards[idx]; }

        const field_t<T, Rank>& operator[](std::size_t idx) const
        {
            return shards[idx];
        }

        // assignment materializes computation across all shards
        template <typename F>
        field_set_t& operator=(const computation_t<Rank, F>& comp)
        {
            // cost model assumptions:
            // - computation executed in parallel on all devices (N-way
            // parallel)
            // - no cross-device data movement (data stays local to each shard)
            // - total time ≈ max(shard_compute_times)
            if (comp.domain() != global_domain_) {
                throw std::runtime_error(
                    "distributed assignment: domain mismatch"
                );
            }

            auto& ctx = current_context();
            std::vector<exec::future_t<void>> futures;
            futures.reserve(shards.size());

            for (std::size_t ii = 0; ii < shards.size(); ++ii) {
                auto local_comp = comp[partitions[ii]];
                auto& shard     = shards[ii];

                // Use context executor with resource tracking
                if (shard.device().is_gpu) {
                    auto& executor =
                        ctx.get_executor<exec::gpu_executor_t>(shard.device());

                    if (!executor.check_and_update_limits(
                            shard.size() * sizeof(T)
                        )) {
                        throw resource_limit_error_t(
                            "GPU memory limit exceeded"
                        );
                    }

                    futures.push_back(executor.for_each(
                        shard.domain(),
                        [acc = shard.accessor(), local_comp] DUAL(auto coord) {
                            acc(coord) = local_comp(coord);
                        }
                    ));
                }
                else {
                    auto& executor =
                        ctx.get_executor<exec::cpu_executor_t>(shard.device());
                    futures.push_back(executor.for_each(
                        shard.domain(),
                        [acc = shard.accessor(), local_comp](auto coord) {
                            acc(coord) = local_comp(coord);
                        }
                    ));
                }
            }

            for (auto& future : futures) {
                future.wait();
            }

            return *this;
        }

        // gather: collect all shards onto a single device
        field_t<T, Rank> gather(mem::device_t target_device) const
        {
            field_t<T, Rank> result(global_domain_, target_device);

            // copy each shard to its location in result
            for (std::size_t ii = 0; ii < shards.size(); ++ii) {
                const auto& shard = shards[ii];
                auto target_slice = result.slice(partitions[ii]);

                // determine copy direction
                if (shard.device() == target_device) {
                    // same device - direct copy
                    std::copy_n(
                        shard.data(),
                        shard.size(),
                        target_slice.data()
                    );
                }
                else if (shard.device().is_gpu && target_device.is_gpu) {
                    // gpu to gpu
                    hetero::device::peer_copy(
                        target_slice.data(),
                        target_device.device_id,
                        shard.data(),
                        shard.device().device_id,
                        shard.size() * sizeof(T)
                    );
                }
                else if (shard.device().is_gpu && !target_device.is_gpu) {
                    // gpu to cpu
                    hetero::device::copy(
                        target_slice.data(),
                        shard.data(),
                        shard.size() * sizeof(T),
                        hetero::memory_direction_t::device_to_host
                    );
                }
                else if (!shard.device().is_gpu && target_device.is_gpu) {
                    // cpu to gpu
                    hetero::device::copy(
                        target_slice.data(),
                        shard.data(),
                        shard.size() * sizeof(T),
                        hetero::memory_direction_t::host_to_device
                    );
                }
                else {
                    // cpu to cpu
                    std::copy_n(
                        shard.data(),
                        shard.size(),
                        target_slice.data()
                    );
                }
            }

            return result;
        }

        // scatter: distribute single-device field across shards
        static field_set_t scatter(
            const field_t<T, Rank>& source,
            const std::vector<mem::device_t>& devices
        )
        {
            field_set_t result(source.domain(), devices);

            // copy each partition from source to corresponding shard
            for (std::size_t ii = 0; ii < result.shards.size(); ++ii) {
                auto source_slice  = source.slice(result.partitions[ii]);
                auto& target_shard = result.shards[ii];

                if (source.device() == target_shard.device()) {
                    // same device - direct copy
                    std::copy_n(
                        source_slice.data(),
                        source_slice.size(),
                        target_shard.data()
                    );
                }
                else if (source.device().is_gpu &&
                         target_shard.device().is_gpu) {
                    // gpu to gpu
                    hetero::device::copy(
                        target_shard.data(),
                        source_slice.data(),
                        source_slice.size() * sizeof(T),
                        target_shard.device().device_id,
                        source.device().device_id,
                        hetero::memory_direction_t::device_to_device
                    );
                }
                else if (source.device().is_gpu &&
                         !target_shard.device().is_gpu) {
                    // gpu to cpu
                    hetero::device::copy(
                        target_shard.data(),
                        source_slice.data(),
                        source_slice.size() * sizeof(T),
                        hetero::memory_direction_t::device_to_host
                    );
                }
                else if (!source.device().is_gpu &&
                         target_shard.device().is_gpu) {
                    // cpu to gpu
                    hetero::device::copy(
                        target_shard.data(),
                        source_slice.data(),
                        source_slice.size() * sizeof(T),
                        hetero::memory_direction_t::host_to_device
                    );
                }
                else {
                    // cpu to cpu
                    std::copy_n(
                        source_slice.data(),
                        source_slice.size(),
                        target_shard.data()
                    );
                }
            }

            return result;
        }

        // clone: create deep copy on same devices
        field_set_t clone() const
        {
            field_set_t result = *this;
            for (std::size_t ii = 0; ii < shards.size(); ++ii) {
                result.shards[ii] = shards[ii].clone();
            }
            return result;
        }
    };

    // factory functions
    template <typename T, std::uint64_t Rank>
    field_set_t<T, Rank>
    field_set(const domain_t<Rank>& domain, device_pool_t& pool)
    {
        return field_set_t<T, Rank>(domain, pool);
    }

}   // namespace simbi

#endif   // FIELD_SET_HPP
