#ifndef FMR_FACTORY_HPP
#define FMR_FACTORY_HPP

#include "builder.hpp"                   // for build_hierarchy
#include "compute/field.hpp"             // for field_t
#include "containers/vector.hpp"         // for vector_t
#include "hierarchy.hpp"                 // for mesh_hierarchy_t
#include "level_descriptor.hpp"          // for level_descriptor_t
#include "mesh/mesh_config.hpp"          // for mesh_config_t
#include "utility/enums.hpp"             // for Geometry
#include "utility/init_conditions.hpp"   // for initial_conditions_t

#include <cstdint>     // for std::uint64_t
#include <stdexcept>   // for std::runtime_error

namespace simbi::mesh::refinement::fmr {

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
          .halo_radius     = init.halo_radius
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

        // compute physical bounds for this level
        // (map index domain back to physical space)
        auto phys_region = to_physical_space(
            level_desc.domain,
            base_mesh.bounds_min,
            base_mesh.bounds_max,
            base_mesh.shape
        );

        level_mesh.bounds_min = phys_region.min;
        level_mesh.bounds_max = phys_region.max;

        // inherit time-dependent properties from base
        level_mesh.homologous       = base_mesh.homologous;
        level_mesh.mesh_motion      = base_mesh.mesh_motion;
        level_mesh.expansion_factor = base_mesh.expansion_factor;
        level_mesh.sf               = base_mesh.sf;
        level_mesh.sf_derivative    = base_mesh.sf_derivative;

        return level_mesh;
    }

    template <
        typename Conserved,
        typename Primitive,
        std::uint64_t Dims,
        Geometry G = Geometry::CARTESIAN>
    struct fmr_data_t {
        mesh_hierarchy_t<Dims> hierarchy;

        struct level_data_t {
            field_t<Conserved, Dims> cons;
            field_t<Primitive, Dims> prim;
            vector_t<field_t<Conserved, Dims>, Dims> flux;
            mesh::mesh_config_t<Dims, G> mesh;
        };

        vector_t<level_data_t, 7> levels;   // levels 1-7

        std::uint64_t num_levels() const { return hierarchy.num_levels - 1; }

        const level_data_t& operator[](std::uint64_t level_id) const
        {
            return levels[level_id - 1];   // level 0 not stored here
        }

        level_data_t& operator[](std::uint64_t level_id)
        {
            return levels[level_id - 1];
        }
    };

    template <
        typename Conserved,
        typename Primitive,
        std::uint64_t Dims,
        Geometry G>
    fmr_data_t<Conserved, Primitive, Dims, G> create_fmr_data(
        const initial_conditions_t& init,
        const mesh::mesh_config_t<Dims, G>& base_mesh
    )
    {
        if (!init.fmr_enabled) {
            throw std::runtime_error("FMR not enabled in config");
        }

        fmr_data_t<Conserved, Primitive, Dims, G> fmr;

        // build hierarchy
        fmr.hierarchy = build_hierarchy_from_init<Dims>(init, base_mesh);

        // allocate data for each refined level
        for (std::uint64_t lvl = 1; lvl < fmr.hierarchy.num_levels; ++lvl) {
            const auto& level_desc = fmr.hierarchy[lvl];
            auto& level_data       = fmr.levels[lvl - 1];

            // create mesh config for this level
            level_data.mesh = create_level_mesh(base_mesh, level_desc, init);

            // allocate fields
            level_data.cons =
                stored_field<Conserved, Dims>(level_desc.full_domain);
            level_data.prim =
                stored_field<Primitive, Dims>(level_desc.full_domain);

            // allocate fluxes (face-centered in each direction)
            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                auto flux_domain = level_data.mesh.face_domain[dir];
                level_data.flux[dir] =
                    stored_field<Conserved, Dims>(flux_domain);
            }
        }

        return fmr;
    }

}   // namespace simbi::mesh::refinement::fmr

#endif
