#ifndef DECOMPOSE_HPP
#define DECOMPOSE_HPP

#include "containers/vector.hpp"
#include "core/base/buffer.hpp"
#include "domain/domain.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace simbi {
    // pure data structures
    template <std::uint64_t Dims>
    struct domain_shard_t {
        domain_t<Dims> compute_domain;   // where this shard does work
        domain_t<Dims> storage_domain;   // includes halo regions
        device_id_t target_device;       // which device owns this shard
        int shard_id;                    // for identification
    };

    template <std::uint64_t Dims>
    struct decomposition_t {
        std::vector<domain_shard_t<Dims>> shards;
        domain_t<Dims> global_domain;
        int halo_radius;
    };

    // the heart of the system :D
    template <std::uint64_t Dims>
    auto decompose_domain(
        domain_t<Dims> global_domain,
        const std::vector<device_id_t>& devices,
        int halo_radius
    ) -> decomposition_t<Dims>
    {
        auto num_devices = devices.size();
        auto grid_dims   = factor_devices(global_domain.shape(), num_devices);

        std::vector<domain_shard_t<Dims>> shards;
        int shard_id = 0;

        // iterate through grid positions
        for (auto grid_coord : domain_t<Dims>{{}, grid_dims}) {
            auto compute_domain =
                compute_subdomain(global_domain, grid_coord, grid_dims);
            auto storage_domain =
                add_halos(compute_domain, global_domain, halo_radius);

            shards.push_back(
                {.compute_domain = compute_domain,
                 .storage_domain = storage_domain,
                 .target_device  = devices[shard_id],
                 .shard_id       = shard_id,
                 .grid_coord     = grid_coord}
            );
            ++shard_id;
        }

        // compute neighbor relationships
        for (auto& shard : shards) {
            shard.neighbors = find_neighbors(shard.grid_coord, grid_dims);
        }

        return {shards, global_domain, halo_radius};
    }

    template <std::uint64_t Dims>
    auto factor_devices(const iarray<Dims>& shape, int num_devices)
    {
        // simple greedy factorization - split largest dimension first
        iarray<Dims> grid{1};
        int remaining = num_devices;

        while (remaining > 1) {
            auto max_dim =
                std::max_element(shape.begin(), shape.end()) - shape.begin();
            if (shape[max_dim] >= remaining) {
                grid[max_dim] *= remaining;
                break;
            }
            // factor out what we can from this dimension
            auto factor = std::min(remaining, static_cast<int>(shape[max_dim]));
            grid[max_dim] = factor;
            remaining /= factor;
        }
        return grid;
    }
}   // namespace simbi
#endif
