#ifndef FMR_FACTORY_HPP
#define FMR_FACTORY_HPP

#include "builder.hpp"                   // for build_hierarchy
#include "containers/vector.hpp"         // for vector_t
#include "hierarchy.hpp"                 // for mesh_hierarchy_t
#include "level_descriptor.hpp"          // for level_descriptor_t
#include "mesh/mesh_config.hpp"          // for mesh_config_t
#include "utility/enums.hpp"             // for Geometry
#include "utility/init_conditions.hpp"   // for initial_conditions_t

#include <cstdint>     // for std::uint64_t
#include <stdexcept>   // for std::runtime_error

namespace simbi::mesh::fmr {

    template <std::uint64_t Dims, Geometry G = Geometry::CARTESIAN>
    mesh_hierarchy_t<Dims> build_hierarchy_from_init(
        const initial_conditions_t& init,
        const mesh::mesh_config_t<Dims, G>& base_mesh
    )
    {
        if (!init.fmr_enabled) {
            throw std::runtime_error("FMR not enabled in config");
        }

        // extract physical regions
        auto regions = init.get_physical_regions<Dims>();

        // build config
        hierarchy_config_t<Dims> config{
          .base_dx         = base_mesh.dx,
          .base_domain     = base_mesh.domain,
          .bounds_min      = base_mesh.bounds_min,
          .bounds_max      = base_mesh.bounds_max,
          .base_resolution = base_mesh.shape,
          .refine_regions  = regions,
          .refine_ratios   = init.fmr_ratios,
          .halo_radius     = init.halo_radius,
          .face_domains    = base_mesh.face_domain
        };

        return build_hierarchy(config);
    }

    template <std::uint64_t Dims, Geometry G>
    mesh::mesh_config_t<Dims, G> create_level_mesh(
        const mesh::mesh_config_t<Dims, G>& base_mesh,
        const level_descriptor_t<Dims>& level_desc,
        const initial_conditions_t& init
    )
    {
        mesh::mesh_config_t<Dims, G> level_mesh = base_mesh;

        // update domains
        level_mesh.shape       = level_desc.domain.shape();
        level_mesh.full_shape  = level_desc.full_domain.shape();
        level_mesh.domain      = level_desc.domain;
        level_mesh.full_domain = level_desc.full_domain;
        level_mesh.halo_radius = init.halo_radius;

        // update spacing
        level_mesh.dx = level_desc.dx;

        // update face domains for this level
        for (std::uint64_t ii = 0; ii < Dims; ii++) {
            auto amount                = iarray<Dims>{0};
            amount[ii]                 = 1;
            level_mesh.face_domain[ii] = domain_algebra::expand_end(
                make_domain(level_mesh.domain.shape()),
                amount
            );

            if (init.is_mhd) {
                level_mesh.face_domain[ii].start[(ii + 1) % Dims] += 1;
                level_mesh.face_domain[ii].fin[(ii + 1) % Dims] += 1;
                level_mesh.face_domain[ii].start[(ii + 2) % Dims] += 1;
                level_mesh.face_domain[ii].fin[(ii + 2) % Dims] += 1;
            }
        }

        // use the physical bounds from the level descriptor
        level_mesh.bounds_min = level_desc.physical_min;
        level_mesh.bounds_max = level_desc.physical_max;

        // inherit time-dependent properties from base
        level_mesh.homologous       = base_mesh.homologous;
        level_mesh.mesh_motion      = base_mesh.mesh_motion;
        level_mesh.expansion_factor = base_mesh.expansion_factor;
        level_mesh.sf               = base_mesh.sf;
        level_mesh.sf_derivative    = base_mesh.sf_derivative;

        return level_mesh;
    }

}   // namespace simbi::mesh::fmr

#endif
