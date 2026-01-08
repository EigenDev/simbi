// =============================================================================
// domain_partition.hpp
//
// utilities for partitioning domains across multiple devices (multi-gpu).
// provides strategies for splitting iteration spaces to balance work across
// gpus while minimizing communication overhead.
//
// design principles:
//   - minimize cross-device communication
//   - balance work across devices
//   - preserve domain structure (start/end semantics)
//   - support 1d, 2d, 3d partitioning strategies
//
// usage:
//   auto parts = xpu::partition_uniform(domain, num_gpus);
//   auto parts = xpu::partition_along_axis<0>(domain, num_gpus);
// =============================================================================

#pragma once

#include "grid/domain.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace simbi::xpu {

    // =============================================================================
    // partitioning strategies
    // =============================================================================

    // uniform partition along first axis (typical for 1d or slab decomposition)
    // splits domain into n roughly-equal chunks, distributing remainder uniformly
    template <std::uint64_t Rank>
    std::vector<grid::domain_t<Rank>>
    partition_uniform(const grid::domain_t<Rank>& domain, std::size_t n)
    {
        if (n == 0 || domain.empty()) {
            return {};
        }

        if (n == 1) {
            return {domain};
        }

        std::vector<grid::domain_t<Rank>> result;
        result.reserve(n);

        auto               shape      = domain.shape();
        const std::int64_t total_size = shape[0];
        const std::int64_t chunk_size = total_size / n;
        const std::int64_t remainder  = total_size % n;

        std::int64_t current_start = domain.start[0];

        for (std::size_t ii = 0; ii < n; ++ii) {
            grid::domain_t<Rank> chunk = domain;
            chunk.start[0]             = current_start;
            const std::int64_t extra   = (ii < static_cast<std::size_t>(remainder)) ? 1 : 0;
            chunk.end[0]               = current_start + chunk_size + extra;

            result.push_back(chunk);
            current_start = chunk.end[0];
        }

        return result;
    }

    // partition along a specific axis (for more control over decomposition)
    template <std::uint64_t Axis, std::uint64_t Rank>
    std::vector<grid::domain_t<Rank>>
    partition_along_axis(const grid::domain_t<Rank>& domain, std::size_t n)
    {
        static_assert(Axis < Rank, "axis must be less than rank");

        if (n == 0 || domain.empty()) {
            return {};
        }

        if (n == 1) {
            return {domain};
        }

        std::vector<grid::domain_t<Rank>> result;
        result.reserve(n);

        auto               shape      = domain.shape();
        const std::int64_t total_size = shape[Axis];
        const std::int64_t chunk_size = total_size / n;
        const std::int64_t remainder  = total_size % n;

        std::int64_t current_start = domain.start[Axis];

        for (std::size_t ii = 0; ii < n; ++ii) {
            grid::domain_t<Rank> chunk = domain;
            chunk.start[Axis]          = current_start;
            const std::int64_t extra   = (ii < static_cast<std::size_t>(remainder)) ? 1 : 0;
            chunk.end[Axis]            = current_start + chunk_size + extra;

            result.push_back(chunk);
            current_start = chunk.end[Axis];
        }

        return result;
    }

    // 2d/3d decomposition: partition along multiple axes
    // useful for pencil or block decomposition in multi-dimensional hydro
    template <std::uint64_t Rank>
    std::vector<grid::domain_t<Rank>> partition_block(
        const grid::domain_t<Rank>&          domain,
        const std::array<std::size_t, Rank>& partitions_per_axis
    )
    {
        static_assert(Rank >= 2, "block partitioning requires rank >= 2");

        if (domain.empty()) {
            return {};
        }

        // calculate total number of blocks
        std::size_t total_blocks = 1;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            if (partitions_per_axis[ii] == 0) {
                return {};
            }
            total_blocks *= partitions_per_axis[ii];
        }

        std::vector<grid::domain_t<Rank>> result;
        result.reserve(total_blocks);

        // precompute chunk sizes per axis
        vector_t<std::int64_t, Rank> chunk_sizes;
        vector_t<std::int64_t, Rank> remainders;

        auto shape = domain.shape();
        for (std::uint64_t axis = 0; axis < Rank; ++axis) {
            chunk_sizes[axis] = shape[axis] / partitions_per_axis[axis];
            remainders[axis]  = shape[axis] % partitions_per_axis[axis];
        }

        // generate all blocks via multi-dimensional iteration
        std::array<std::size_t, Rank> indices{};

        for (std::size_t linear = 0; linear < total_blocks; ++linear) {
            grid::domain_t<Rank> block = domain;

            // compute start/end for each axis based on current indices
            for (std::uint64_t axis = 0; axis < Rank; ++axis) {
                std::int64_t base_start = domain.start[axis];

                // compute start position for this block
                std::int64_t offset = 0;
                for (std::size_t jj = 0; jj < indices[axis]; ++jj) {
                    const std::int64_t extra =
                        (jj < static_cast<std::size_t>(remainders[axis])) ? 1 : 0;
                    offset += chunk_sizes[axis] + extra;
                }

                block.start[axis] = base_start + offset;

                // compute end position
                const std::int64_t extra =
                    (indices[axis] < static_cast<std::size_t>(remainders[axis])) ? 1 : 0;
                block.end[axis] = block.start[axis] + chunk_sizes[axis] + extra;
            }

            result.push_back(block);

            // increment multi-dimensional index (row-major order)
            for (std::int64_t axis = Rank - 1; axis >= 0; --axis) {
                ++indices[axis];
                if (indices[axis] < partitions_per_axis[axis]) {
                    break;
                }
                indices[axis] = 0;
            }
        }

        return result;
    }

    // =============================================================================
    // halo region extraction (for ghost cell exchanges in hydro)
    // =============================================================================

    // extract boundary slice along specified axis and direction
    // useful for extracting data to send to neighboring ranks/devices
    template <std::uint64_t Axis, std::uint64_t Rank>
    grid::domain_t<Rank> extract_boundary(
        const grid::domain_t<Rank>& domain,
        std::int64_t                depth,
        bool                        lower_boundary // true = lower, false = upper
    )
    {
        static_assert(Axis < Rank, "axis must be less than rank");

        grid::domain_t<Rank> boundary = domain;

        if (lower_boundary) {
            // lower boundary: [start, start + depth)
            boundary.end[Axis] = std::min(boundary.start[Axis] + depth, domain.fin[Axis]);
        }
        else {
            // upper boundary: [end - depth, end)
            boundary.start[Axis] = std::max(boundary.end[Axis] - depth, domain.start[Axis]);
        }

        return boundary;
    }

    // extract interior domain (excluding ghost/halo cells)
    template <std::uint64_t Rank>
    grid::domain_t<Rank>
    extract_interior(const grid::domain_t<Rank>& domain, std::int64_t halo_depth)
    {
        grid::domain_t<Rank> interior = domain;

        for (std::uint64_t axis = 0; axis < Rank; ++axis) {
            interior.start[axis] += halo_depth;
            interior.end[axis] -= halo_depth;

            // ensure we don't create invalid domain
            if (interior.start[axis] >= interior.end[axis]) {
                interior.start[axis] = domain.start[axis];
                interior.end[axis]   = domain.start[axis]; // empty domain
            }
        }

        return interior;
    }

    // =============================================================================
    // load balancing utilities
    // =============================================================================

    // compute load-balanced partition based on work weights
    // useful when some regions require more computation than others
    template <std::uint64_t Rank>
    std::vector<grid::domain_t<Rank>> partition_weighted(
        const grid::domain_t<Rank>& domain,
        const std::vector<double>&  weights,
        std::size_t                 n_partitions
    )
    {
        if (n_partitions == 0 || domain.empty() || weights.empty()) {
            return {};
        }

        auto shape = domain.shape();
        if (static_cast<std::size_t>(shape[0]) != weights.size()) {
            // fallback to uniform if weights don't match
            return partition_uniform(domain, n_partitions);
        }

        // compute cumulative weights
        std::vector<double> cumulative(weights.size() + 1, 0.0);
        for (std::size_t ii = 0; ii < weights.size(); ++ii) {
            cumulative[ii + 1] = cumulative[ii] + weights[ii];
        }

        const double total_weight  = cumulative.back();
        const double target_weight = total_weight / n_partitions;

        std::vector<grid::domain_t<Rank>> result;
        result.reserve(n_partitions);

        std::int64_t current_start = domain.start[0];

        for (std::size_t partition = 0; partition < n_partitions; ++partition) {
            const double target = (partition + 1) * target_weight;

            // find index where cumulative weight exceeds target
            std::int64_t end_idx = current_start;
            for (std::int64_t ii = current_start; ii < shape[0]; ++ii) {
                if (cumulative[ii + 1] >= target) {
                    end_idx = ii + 1;
                    break;
                }
            }

            // ensure last partition gets remaining work
            if (partition == n_partitions - 1) {
                end_idx = shape[0];
            }

            grid::domain_t<Rank> chunk = domain;
            chunk.start[0]             = domain.start[0] + current_start;
            chunk.end[0]               = domain.start[0] + end_idx;

            result.push_back(chunk);
            current_start = end_idx;
        }

        return result;
    }

    // =============================================================================
    // query utilities
    // =============================================================================

    // check if two domains overlap (for detecting communication needs)
    template <std::uint64_t Rank>
    bool domains_overlap(const grid::domain_t<Rank>& a, const grid::domain_t<Rank>& b)
    {
        for (std::uint64_t axis = 0; axis < Rank; ++axis) {
            if (a.end[axis] <= b.start[axis] || b.end[axis] <= a.start[axis]) {
                return false;
            }
        }
        return true;
    }

    // compute intersection of two domains
    template <std::uint64_t Rank>
    grid::domain_t<Rank>
    domain_intersection(const grid::domain_t<Rank>& a, const grid::domain_t<Rank>& b)
    {
        grid::domain_t<Rank> result;

        for (std::uint64_t axis = 0; axis < Rank; ++axis) {
            result.start[axis] = std::max(a.start[axis], b.start[axis]);
            result.end[axis]   = std::min(a.end[axis], b.end[axis]);

            // check for empty intersection
            if (result.start[axis] >= result.end[axis]) {
                result.start[axis] = 0;
                result.end[axis]   = 0;
                return result; // empty domain
            }
        }

        return result;
    }

    // compute neighbors in a partitioned domain (for halo exchanges)
    // returns indices of partitions that share a boundary with partition_idx
    template <std::uint64_t Rank>
    std::vector<std::size_t> find_neighbors(
        const std::vector<grid::domain_t<Rank>>& partitions,
        std::size_t                              partition_idx,
        std::int64_t                             halo_depth = 1
    )
    {
        if (partition_idx >= partitions.size()) {
            return {};
        }

        std::vector<std::size_t> neighbors;
        const auto&              my_domain = partitions[partition_idx];

        // expand domain by halo depth to find overlapping neighbors
        grid::domain_t<Rank> expanded = my_domain;
        for (std::uint64_t axis = 0; axis < Rank; ++axis) {
            expanded.start[axis] -= halo_depth;
            expanded.end[axis] += halo_depth;
        }

        for (std::size_t ii = 0; ii < partitions.size(); ++ii) {
            if (ii == partition_idx) {
                continue;
            }

            if (domains_overlap(expanded, partitions[ii])) {
                neighbors.push_back(ii);
            }
        }

        return neighbors;
    }

} // namespace simbi::xpu
