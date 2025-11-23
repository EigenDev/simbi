#ifndef GRID_AMR_GEOMETRY_HPP
#define GRID_AMR_GEOMETRY_HPP

#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "grid/patch_id.hpp"

#include <cstdint>

namespace simbi::grid::amr {

    // -------------------------------------------------------------------------
    // geometry calculator
    // converts topological IDs into global integer coordinates
    // -------------------------------------------------------------------------
    struct geometry_calculator_t {
        // root configuration
        // we assume root blocks have a fixed size (e.g., 128^3)
        vector_t<std::int64_t, 3> block_size_;

        constexpr geometry_calculator_t(vector_t<std::int64_t, 3> block_size)
            : block_size_(block_size)
        {
        }

        // calculates the domain of a block at any level
        // returns domain in the global finest coordinate space?
        // or returns domain in the level's coordinate space?

        // crit decion (todo: revisit this later): coordinate systems.
        // the comment below is for my own education and may be removed later.
        // option a: all domains are in "level 0" coordinates (continuous-like).
        //           fine blocks have fractional coords (bad for int math).
        // option b: all domains are in "deepest level" coordinates.
        // option c: domains are in "level l" coordinates.

        // we choose option c (standard for amrex/flash).
        // a domain [0, 100] at level 0 covers the same physical space
        // as [0, 200] at level 1.

        template <std::uint64_t Rank>
        constexpr domain_t<Rank> get_domain(const patch_id_t& id) const
        {
            domain_t<Rank> d;

            for (std::uint64_t i = 0; i < Rank; ++i) {
                // start = topocoord * blocksize
                d.start[i] = id.coords[i] * block_size_[i];
                d.fin[i]   = d.start[i] + block_size_[i];
            }
            return d;
        }

        // map a domain from level src to level dst
        template <std::uint64_t Rank>
        constexpr domain_t<Rank> map_domain(
            const domain_t<Rank>& src_domain,
            std::int64_t src_level,
            std::int64_t dst_level
        ) const
        {
            if (src_level == dst_level) {
                return src_domain;
            }

            domain_t<Rank> result = src_domain;

            if (dst_level > src_level) {
                // refine (multiply by 2 per level delta)
                std::int64_t ratio = 1 << (dst_level - src_level);
                for (std::uint64_t i = 0; i < Rank; ++i) {
                    result.start[i] *= ratio;
                    result.fin[i] *= ratio;
                }
            }
            else {
                // coarsen (divide by 2 per level delta)
                std::int64_t ratio = 1 << (src_level - dst_level);
                for (std::uint64_t i = 0; i < Rank; ++i) {
                    result.start[i] /= ratio;
                    result.fin[i] /= ratio;
                }
            }
            return result;
        }
    };

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_GEOMETRY_HPP
