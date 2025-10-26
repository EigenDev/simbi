#ifndef FMR_BUILDER_HPP
#define FMR_BUILDER_HPP

#include "compat.hpp"              // for real type
#include "containers/vector.hpp"   /// for vector_t, iarray
#include "domain/algebra.hpp"      // for domain_algebra
#include "domain/domain.hpp"       // for domain_t, physical_region_t
#include "hierarchy.hpp"           // for mesh_hierarchy_t

#include <cstdint>     // for std::uint64_t
#include <stdexcept>   // for std::runtime_error
#include <vector>      // for std::vector

namespace simbi::mesh::fmr {
    // helper: convert physical region to index space at a given refinement
    // level
    template <std::uint64_t Dims>
    domain_t<Dims> to_index_space_at_level(
        const physical_region_t<Dims>& phys_region,
        const vector_t<real, Dims>& bounds_min,
        const vector_t<real, Dims>& bounds_max,
        const iarray<Dims>& base_resolution,
        std::uint64_t refinement_ratio
    )
    {
        iarray<Dims> start, end;

        for (std::uint64_t d = 0; d < Dims; ++d) {
            // base cell size
            real base_dx = (bounds_max[d] - bounds_min[d]) / base_resolution[d];
            // refined cell size
            real refined_dx = base_dx / static_cast<real>(refinement_ratio);

            // convert to refined index space
            start[d] = static_cast<std::int64_t>(
                (phys_region.min[d] - bounds_min[d]) / refined_dx
            );
            end[d] = static_cast<std::int64_t>(
                (phys_region.max[d] - bounds_min[d]) / refined_dx
            );
        }

        return domain_t<Dims>{start, end};
    }

    // builder configuration
    template <std::uint64_t Dims>
    struct hierarchy_config_t {
        // base mesh properties
        vector_t<real, Dims> base_dx;
        domain_t<Dims> base_domain;
        vector_t<real, Dims> bounds_min;
        vector_t<real, Dims> bounds_max;
        iarray<Dims> base_resolution;

        // refinement specification
        std::vector<physical_region_t<Dims>> refine_regions;
        std::vector<std::uint64_t> refine_ratios;

        // ghost zone width
        std::uint64_t halo_radius;
    };

    // build hierarchy from configuration
    template <std::uint64_t Dims>
    mesh_hierarchy_t<Dims>
    build_hierarchy(const hierarchy_config_t<Dims>& config)
    {
        mesh_hierarchy_t<Dims> hierarchy;

        // validate input
        if (config.refine_regions.size() != config.refine_ratios.size()) {
            throw std::runtime_error(
                "refine_regions and refine_ratios must have same size"
            );
        }

        hierarchy.num_levels = config.refine_regions.size() + 1;

        if (hierarchy.num_levels > mesh_hierarchy_t<Dims>::max_levels) {
            throw std::runtime_error("too many refinement levels requested");
        }

        // level 0: base mesh
        hierarchy.levels[0] = {
          .level_id    = 0,
          .domain      = config.base_domain,
          .full_domain = domain_algebra::expand(
              config.base_domain,
              ones<Dims, std::int64_t>() *
                  static_cast<std::int64_t>(config.halo_radius)
          ),
          .ref_ratio       = 1,
          .dx              = config.base_dx,
          .parent_level_id = 0,
          .parent_coverage = config.base_domain
        };

        // build refined levels
        std::uint64_t cumulative_ratio = 1;
        for (std::uint64_t lvl = 1; lvl < hierarchy.num_levels; ++lvl) {
            cumulative_ratio *= config.refine_ratios[lvl - 1];

            // compute parent coverage (in parent's index space)
            auto parent_coverage = to_index_space_at_level(
                config.refine_regions[lvl - 1],
                config.bounds_min,
                config.bounds_max,
                config.base_resolution,
                cumulative_ratio / config.refine_ratios[lvl - 1]
            );

            // fine domain in its own local coordinate system
            // size = parent_size * refinement_ratio
            iarray<Dims> fine_size;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                fine_size[d] =
                    (parent_coverage.fin[d] - parent_coverage.start[d]) *
                    config.refine_ratios[lvl - 1];
            }
            auto refined_domain = make_domain(fine_size);

            hierarchy.levels[lvl] = {
              .level_id    = lvl,
              .domain      = refined_domain,
              .full_domain = domain_algebra::expand_end(
                  refined_domain,
                  ones<Dims, std::int64_t>() * 2 *
                      static_cast<std::int64_t>(config.halo_radius)
              ),
              .ref_ratio = cumulative_ratio,
              .dx        = config.base_dx / static_cast<real>(cumulative_ratio),
              .parent_level_id = lvl - 1,
              .parent_coverage = parent_coverage
            };
        }

        return hierarchy;
    }

}   // namespace simbi::mesh::fmr

#endif
