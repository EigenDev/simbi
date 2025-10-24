#ifndef MINIMAL_FMR_HPP
#define MINIMAL_FMR_HPP

#include "compat.hpp"                     // for real type
#include "compute/field.hpp"              // for_field_t
#include "containers/vector.hpp"          // for vector_t
#include "domain/algebra.hpp"             // for domain_algebra
#include "domain/domain.hpp"              // for domain_t, physical_region_t
#include "mesh/refinement/transfer.hpp"   // for make_prolongation, make_restriction

#include <cstdint>   // for std::uint64_t
#include <vector>    // for std::vector

namespace simbi::mesh::refinement {

    template <std::uint64_t Dims>
    struct level_descriptor_t {
        std::uint64_t level_id;
        domain_t<Dims> domain;        // this level's active cells
        domain_t<Dims> full_domain;   // including ghosts
        std::uint64_t ref_ratio;      // relative to level 0
        vector_t<real, Dims> dx;      // cell spacing

        // parent relationship for ghost fills
        std::uint64_t parent_level_id;
        domain_t<Dims> parent_coverage;   // what region of parent we refine
    };

    template <std::uint64_t Dims>
    struct mesh_hierarchy_t {
        // max 8 levels reasonable
        vector_t<level_descriptor_t<Dims>, 8> levels;
        std::uint64_t num_levels;

        const level_descriptor_t<Dims>& operator[](std::uint64_t id) const
        {
            return levels[id];
        }
    };

    template <std::uint64_t Dims>
    mesh_hierarchy_t<Dims> build_hierarchy(
        const vector_t<real, Dims>& base_dx,
        const domain_t<Dims>& base_domain,
        const std::vector<physical_region_t<Dims>>& refine_regions,
        const std::vector<std::uint64_t>& refine_ratios,
        std::uint64_t halo_radius
    )
    {
        mesh_hierarchy_t<Dims> hierarchy;
        hierarchy.num_levels = refine_regions.size() + 1;

        // level 0: base mesh
        hierarchy.levels[0] = {
          .level_id    = 0,
          .domain      = base_domain,
          .full_domain = domain_algebra::expand(
              base_domain,
              ones<Dims, std::int64_t>() * halo_radius
          ),
          .ref_ratio       = 1,
          .dx              = base_dx,
          .parent_level_id = 0,
          .parent_coverage = base_domain
        };

        // refined levels
        std::uint64_t cumulative_ratio = 1;
        for (std::uint64_t lvl = 1; lvl < hierarchy.num_levels; ++lvl) {
            cumulative_ratio *= refine_ratios[lvl - 1];

            // convert physical region to index space at this refinement
            auto refined_domain = to_index_space_at_level(
                refine_regions[lvl - 1],
                base_dx,
                cumulative_ratio
            );

            hierarchy.levels[lvl] = {
              .level_id    = lvl,
              .domain      = refined_domain,
              .full_domain = domain_algebra::expand(
                  refined_domain,
                  ones<Dims, std::int64_t>() * halo_radius
              ),
              .ref_ratio       = cumulative_ratio,
              .dx              = base_dx / static_cast<real>(cumulative_ratio),
              .parent_level_id = lvl - 1,
              .parent_coverage = refine_regions[lvl - 1]   // in parent coords
            };
        }

        return hierarchy;
    }

    template <typename T, std::uint64_t Dims>
    void fill_level_ghosts(
        field_t<T, Dims>& level_field,
        const field_t<T, Dims>& parent_field,
        const level_descriptor_t<Dims>& level,
        const level_descriptor_t<Dims>& parent
    )
    {
        // identify ghost regions
        auto ghost_regions =
            domain_algebra::difference(level.full_domain, level.domain);

        // fill each ghost region from parent via prolongation
        for (const auto& ghost_domain : ghost_regions) {
            auto prolonged = make_prolongation(
                parent_field,
                ghost_domain,
                ones<Dims, real>() * (level.ref_ratio / parent.ref_ratio),
                true   // conservative
            );

            for (const auto& coord : ghost_domain) {
                level_field(coord) = prolonged(coord);
            }
        }
    }

    template <typename T, std::uint64_t Dims>
    void correct_coarse_fluxes(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& coarse_level,
        std::uint64_t flux_dir
    )
    {
        // find interface: fine level boundary that touches coarse level
        auto interface = get_interface_region(
            fine_level.domain,
            coarse_level.domain,
            flux_dir
        );

        if (interface.empty()) {
            return;
        }

        // restrict fine fluxes to coarse resolution
        auto ratio = fine_level.ref_ratio / coarse_level.ref_ratio;
        auto restricted =
            make_restriction(fine_flux, interface, ones<Dims, real>() * ratio);

        // replace coarse fluxes at interface
        for (const auto& coord : interface) {
            coarse_flux(coord) = restricted(coord);
        }
    }

    template <typename Conserved, typename Primitive, std::uint64_t Dims>
    struct fmr_state_t {
        mesh_hierarchy_t<Dims> hierarchy;

        // per-level fields (level 0 is in parent hydro_state_t)
        vector_t<field_t<Conserved, Dims>, 7> cons;   // levels 1-7
        vector_t<field_t<Primitive, Dims>, 7> prim;
        vector_t<vector_t<field_t<Conserved, Dims>, Dims>, 7> flux;
    };

}   // namespace simbi::mesh::refinement

#endif   // MINIMAL_FMR_HPP
