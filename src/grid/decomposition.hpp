#ifndef GRID_DECOMPOSITION_HPP
#define GRID_DECOMPOSITION_HPP

#include "containers/vector.hpp"
#include "domain.hpp"
#include "functional/fp.hpp"

#include <cstdint>
#include <stdexcept>

namespace simbi::grid {

    // -------------------------------------------------------------------------
    // topology definition
    // describes how the ranks are arranged (e.g., 2x2x1)
    // -------------------------------------------------------------------------
    struct topology_t
    {
        vector_t<std::int64_t, 3> dims; // {px, py, pz}

        constexpr std::uint64_t size() const
        {
            std::uint64_t result = 1;
            for (std::uint64_t ii = 0; ii < 3; ++ii) {
                result *= dims[ii];
            }
            return result;
        }

        // map linear rank to 3d coordinate
        constexpr vector_t<std::int64_t, 3> coords(std::uint64_t rank) const
        {
            // row-major ordering
            vector_t<std::int64_t, 3> c = {0, 0, 0};
            c[0]                        = rank / (dims[0] * dims[1]);
            c[1]                        = (rank / dims[0]) % dims[1];
            c[2]                        = rank % dims[0];
            return c;
        }

        // map 3d coordinate to linear rank
        constexpr std::int64_t rank(std::int64_t x, std::int64_t y, std::int64_t z) const
        {
            return x + y * dims[2] + z * dims[2] * dims[1];
        }
    };

    // -------------------------------------------------------------------------
    // decomposer
    // factory that slices a global domain into a local rank's domain
    // -------------------------------------------------------------------------
    struct decomposer_t
    {

        struct interval_t
        {
            std::int64_t start;
            std::int64_t count;
        };

        // helper: calculates start/end for a specific dimension split
        // handles remainder distribution for load balancing
        static interval_t
        split_1d(std::int64_t total_cells, std::int64_t n_chunks, std::int64_t chunk_id)
        {
            std::int64_t base = total_cells / n_chunks;
            std::int64_t rem  = total_cells % n_chunks;

            // distribute remainder to the first 'rem' chunks
            std::int64_t start = 0;
            for (std::int64_t ii = 0; ii < chunk_id; ++ii) {
                start += base + (ii < rem ? 1 : 0);
            }

            std::int64_t size = base + (chunk_id < rem ? 1 : 0);
            return {start, size};
        }

        // the main factory
        template <std::uint64_t Rank>
        static domain_t<Rank>
        decompose(const domain_t<Rank>& global, const topology_t& topo, std::uint64_t my_rank)
        {
            if (my_rank >= topo.size()) {
                throw std::runtime_error("rank out of bounds of topology");
            }

            auto p_coords = topo.coords(my_rank);

            // start with global bounds, then contract
            domain_t<Rank> local        = global;
            auto           global_shape = global.shape();

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (topo.dims[dd] > 1 && dd >= Rank) {
                    throw std::runtime_error("topology dimension mismatch");
                }

                // calculate the local interval (0-based offset)
                interval_t split = split_1d(global_shape[dd], topo.dims[dd], p_coords[dd]);

                // shift to global coordinate space
                // local_start = global_start + offset
                local.start[dd] = global.start[dd] + split.start;
                local.fin[dd]   = local.start[dd] + split.count;
            }

            return local;
        }
    };

} // namespace simbi::grid

#endif // GRID_DECOMPOSITION_HPP
