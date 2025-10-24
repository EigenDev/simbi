#ifndef FMR_GHOST_FILL_HPP
#define FMR_GHOST_FILL_HPP

#include "compat.hpp"                     // for real type
#include "compute/field.hpp"              // for field_t
#include "containers/vector.hpp"          // for iarray
#include "domain/algebra.hpp"             // for domain_algebra
#include "hierarchy.hpp"                  // for mesh_hierarchy_t
#include "level_descriptor.hpp"           // for level_descriptor_t
#include "mesh/refinement/transfer.hpp"   // for make_prolongation

#include <cstdint>   // for std::uint64_t

namespace simbi::mesh::refinement::fmr {

    // coordinate mapping: fine level → parent level
    template <std::uint64_t Dims>
    constexpr iarray<Dims> fine_to_parent_coord(
        const iarray<Dims>& fine_coord,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& parent_level
    )
    {
        iarray<Dims> parent_coord;

        // refinement ratio between these two levels
        auto ratio = fine_level.ref_ratio / parent_level.ref_ratio;

        // offset: where fine level starts in parent's index space
        auto offset = parent_level.parent_coverage.start;
        if (fine_level.level_id > 0) {
            offset = fine_level.parent_coverage.start;
        }

        for (std::uint64_t d = 0; d < Dims; ++d) {
            // map fine index to parent index
            parent_coord[d] =
                offset[d] + (fine_coord[d] - fine_level.domain.start[d]) /
                                static_cast<std::int64_t>(ratio);
        }

        return parent_coord;
    }

    // coordinate mapping: parent level → fine level
    template <std::uint64_t Dims>
    constexpr iarray<Dims> parent_to_fine_coord(
        const iarray<Dims>& parent_coord,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& parent_level
    )
    {
        iarray<Dims> fine_coord;

        auto ratio  = fine_level.ref_ratio / parent_level.ref_ratio;
        auto offset = fine_level.parent_coverage.start;

        for (std::uint64_t d = 0; d < Dims; ++d) {
            fine_coord[d] = fine_level.domain.start[d] +
                            (parent_coord[d] - offset[d]) *
                                static_cast<std::int64_t>(ratio);
        }

        return fine_coord;
    }

    // identify ghost regions that need filling from parent
    template <std::uint64_t Dims>
    auto identify_ghost_regions(const level_descriptor_t<Dims>& level)
    {
        // ghost regions = full_domain - domain
        return domain_algebra::difference(level.full_domain, level.domain);
    }

    // fill ghost cells from parent level using prolongation
    template <typename T, std::uint64_t Dims>
    void fill_from_parent(
        field_t<T, Dims>& fine_field,
        const field_t<T, Dims>& parent_field,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& parent_level,
        bool conservative = true
    )
    {
        // get ghost regions
        auto ghost_regions = identify_ghost_regions(fine_level);

        if (ghost_regions.empty()) {
            return;
        }

        // refinement ratio between levels
        auto ratio     = fine_level.ref_ratio / parent_level.ref_ratio;
        auto ratio_vec = ones<Dims, real>() * static_cast<real>(ratio);

        // fill each ghost region
        for (const auto& ghost_region : ghost_regions) {
            // create prolongation field for this ghost region
            auto prolonged = make_prolongation(
                parent_field,
                ghost_region,
                ratio_vec,
                conservative
            );

            // copy prolonged values to fine field
            for (const auto& coord : ghost_region) {
                fine_field(coord) = prolonged(coord);
            }
        }
    }

    // fill all ghost zones for a single level
    template <typename T, std::uint64_t Dims>
    void fill_level_ghosts(
        field_t<T, Dims>& level_field,
        const field_t<T, Dims>& parent_field,
        const mesh_hierarchy_t<Dims>& hierarchy,
        std::uint64_t level_id,
        bool conservative = true
    )
    {
        if (level_id == 0) {
            // base level - no parent to fill from
            // physical boundary conditions handled elsewhere
            return;
        }

        const auto& level  = hierarchy[level_id];
        const auto& parent = hierarchy[level.parent_level_id];

        fill_from_parent(
            level_field,
            parent_field,
            level,
            parent,
            conservative
        );
    }

}   // namespace simbi::mesh::refinement::fmr

#endif
