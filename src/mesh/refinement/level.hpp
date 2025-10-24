#ifndef LEVEL_HPP
#define LEVEL_HPP

#include "compat.hpp"              // for real type
#include "containers/vector.hpp"   // for vector_t
#include "domain/domain.hpp"       // for domain_t, make_domain
#include "mesh/mesh_config.hpp"    // for mesh_config_t
#include "utility/enums.hpp"       // for Geometry

#include <cstdint>   // for std::uint32_t, std::uint64_t

namespace simbi::mesh::refinement {

    template <std::uint64_t Dims, Geometry G>
    struct level_t {
        std::uint32_t level_id;
        mesh_config_t<Dims, G> mesh;

        // track parent-child relationships
        domain_t<Dims> parent_domain;   // domain in parent's coordinate system
        std::uint64_t ref_ratio;        // refinement ratio to parent level

        // level-specific metadata
        vector_t<real, Dims> dx;   // grid spacing at this level

        // construct from parent mesh and refinement parameters
        static level_t create(
            std::uint32_t id,
            const mesh_config_t<Dims, G>& parent_mesh,
            const domain_t<Dims>& refined_region,
            const std::uint64_t refinement_ratio
        )
        {
            level_t level;
            level.level_id      = id;
            level.parent_domain = refined_region;
            level.ref_ratio     = refinement_ratio;

            // create refined mesh configuration
            level.mesh = refine_mesh_config(
                parent_mesh,
                refined_region,
                refinement_ratio
            );

            // calculate grid spacing
            for (std::uint64_t d = 0; d < Dims; d++) {
                level.dx[d] = parent_mesh.dx[d] / refinement_ratio;
            }

            return level;
        }

      private:
        static mesh_config_t<Dims, G> refine_mesh_config(
            const mesh_config_t<Dims, G>& parent,
            const domain_t<Dims>& region,
            const std::uint64_t ratio
        )
        {
            mesh_config_t<Dims, G> refined = parent;

            // adjust shape based on refinement region and ratio
            for (std::uint64_t d = 0; d < Dims; d++) {
                refined.shape[d] = region.shape()[d] * ratio;
                refined.full_shape[d] =
                    refined.shape[d] + 2 * refined.halo_radius;
            }

            // update domains
            refined.full_domain = make_domain(refined.full_shape);
            refined.domain      = domain_algebra::contract(
                refined.full_domain,
                ones<Dims, std::int64_t>() * refined.halo_radius
            );

            // update face domains
            // [TODO]: Handle face domains properly for MHD

            return refined;
        }
    };

}   // namespace simbi::mesh::refinement

#endif   // LEVEL_HPP
