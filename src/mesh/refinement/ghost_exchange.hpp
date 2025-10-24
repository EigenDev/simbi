#ifndef GHOST_EXCHANGE_HPP
#define GHOST_EXCHANGE_HPP

#include "compute/field.hpp"              // for field_t
#include "domain/domain.hpp"              // for domain_t
#include "domain/ghost.hpp"               // for ghost analysis
#include "mesh/refinement/fmr_mesh.hpp"   // for fmr_mesh_t
#include "mesh/refinement/transfer.hpp"   // for prolongation/restriction
#include "utility/enums.hpp"              // for Geometry

#include <cstdint>   // for std::uint32_t, std::uint64_t
#include <vector>    // for std::vector

namespace simbi::mesh::refinement {

    template <typename T, std::uint64_t Dims, Geometry G>
    struct level_boundary_t {
        // region that needs filling
        domain_t<Dims> ghost_domain;
        // source level for filling
        std::uint32_t source_level_id;
        // whether this is a coarse->fine or fine->coarse fill
        bool is_coarse_to_fine;
        // boundary analysis info
        boundary::ghost_region_t<Dims> boundary_info;
    };

    // identifies level boundaries needing exchange
    template <typename T, std::uint64_t Dims, Geometry G>
    std::vector<level_boundary_t<T, Dims, G>> identify_level_boundaries(
        const fmr_mesh_t<Dims, G>& mesh,
        std::uint32_t level_id
    )
    {
        std::vector<level_boundary_t<T, Dims, G>> boundaries;

        // get current level
        const auto& level = mesh.level(level_id);

        // analyze ghost regions for this level
        auto ghost_regions = boundary::analyze_ghost_regions(
            level.mesh.full_domain,
            level.mesh.domain
        );

        // check each ghost region
        for (const auto& ghost : ghost_regions) {
            // skip if ghost region is contained in parent domain
            if (level.parent_domain.contains(ghost.domain)) {
                continue;
            }

            // this ghost needs coarse data if available
            if (level_id > 0) {
                boundaries.push_back(
                    {.ghost_domain      = ghost.domain,
                     .source_level_id   = level_id - 1,
                     .is_coarse_to_fine = true,
                     .boundary_info     = ghost}
                );
            }
        }

        // check if finer level needs data from this level
        if (level_id + 1 < mesh.num_levels()) {
            const auto& fine_level = mesh.level(level_id + 1);

            // analyze ghost regions of fine level
            auto fine_ghosts = boundary::analyze_ghost_regions(
                fine_level.mesh.full_domain,
                fine_level.mesh.domain
            );

            // check each fine ghost region
            for (const auto& ghost : fine_ghosts) {
                if (level.mesh.domain.contains(ghost.domain)) {
                    boundaries.push_back(
                        {.ghost_domain      = ghost.domain,
                         .source_level_id   = level_id + 1,
                         .is_coarse_to_fine = false,
                         .boundary_info     = ghost}
                    );
                }
            }
        }

        return boundaries;
    }

    // fills ghost cells at level boundaries
    template <typename T, std::uint64_t Dims, Geometry G>
    void fill_level_boundaries(
        std::vector<field_t<T, Dims>>& fields,
        const fmr_mesh_t<Dims, G>& mesh,
        std::uint32_t level_id,
        bool conservative = true
    )
    {
        // get boundaries for this level
        auto boundaries = identify_level_boundaries<T, Dims, G>(mesh, level_id);

        // fill each boundary region
        for (const auto& boundary : boundaries) {
            if (boundary.is_coarse_to_fine) {
                // coarse to fine (prolongation)
                const auto& coarse_level = mesh.level(boundary.source_level_id);
                const auto& fine_level   = mesh.level(level_id);

                fill_fine_region(
                    fields[level_id],
                    fields[boundary.source_level_id],
                    boundary.ghost_domain,
                    fine_level.ref_ratio,
                    conservative
                );
            }
            else {
                // fine to coarse (restriction)
                const auto& fine_level = mesh.level(boundary.source_level_id);

                fill_coarse_region(
                    fields[level_id],
                    fields[boundary.source_level_id],
                    boundary.ghost_domain,
                    fine_level.ref_ratio
                );
            }
        }
    }

    // synchronizes ghost cells across all levels
    template <typename T, std::uint64_t Dims, Geometry G>
    void synchronize_levels(
        std::vector<field_t<T, Dims>>& fields,
        const fmr_mesh_t<Dims, G>& mesh,
        bool conservative = true
    )
    {
        // fill boundaries level by level
        for (std::uint32_t level = 0; level < mesh.num_levels(); ++level) {
            fill_level_boundaries(fields, mesh, level, conservative);
        }
    }

}   // namespace simbi::mesh::refinement

#endif   // GHOST_EXCHANGE_HPP
