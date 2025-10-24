#ifndef FLUX_CORRECTION_HPP
#define FLUX_CORRECTION_HPP

#include "compute/field.hpp"              // for field_t
#include "domain/domain.hpp"              // for domain_t
#include "domain/ghost.hpp"               // for ghost analysis
#include "mesh/refinement/fmr_mesh.hpp"   // for fmr_mesh_t
#include "mesh/refinement/transfer.hpp"   // for restriction
#include "utility/enums.hpp"              // for Geometry

#include <cstdint>   // for std::uint32_t, std::uint64_t
#include <vector>    // for std::vector

namespace simbi::mesh::refinement {

    template <typename T, std::uint64_t Dims, Geometry G>
    struct flux_interface_t {
        // interface region in coarse coordinates
        domain_t<Dims> interface_domain;
        // direction of interface normal
        std::uint64_t direction;
        // which levels are involved
        std::uint32_t coarse_level_id;
        std::uint32_t fine_level_id;
        // boundary info for classification
        boundary::ghost_region_t<Dims> boundary_info;
    };

    // identifies flux interfaces between refinement levels
    template <typename T, std::uint64_t Dims, Geometry G>
    std::vector<flux_interface_t<T, Dims, G>>
    identify_flux_interfaces(const fmr_mesh_t<Dims, G>& mesh)
    {
        std::vector<flux_interface_t<T, Dims, G>> interfaces;

        // check each level's boundaries
        for (std::uint32_t level_id = 0; level_id < mesh.num_levels();
             ++level_id) {
            const auto& level = mesh.level(level_id);

            // analyze ghost regions
            auto ghost_regions = boundary::analyze_ghost_regions(
                level.mesh.full_domain,
                level.mesh.domain
            );

            // check each ghost region
            for (const auto& ghost : ghost_regions) {
                // only interested in face ghosts for flux correction
                if (ghost.type != boundary::ghost_type_t::face) {
                    continue;
                }

                // get direction from ghost info
                std::uint64_t direction = 0;
                for (; direction < Dims; ++direction) {
                    if (ghost.directions[direction] !=
                        boundary::face_side_t::none) {
                        break;
                    }
                }

                // if this is a fine level, interface is with coarse level
                if (level_id > 0 &&
                    !level.parent_domain.contains(ghost.domain)) {
                    interfaces.push_back(
                        {.interface_domain = ghost.domain,
                         .direction        = direction,
                         .coarse_level_id  = level_id - 1,
                         .fine_level_id    = level_id,
                         .boundary_info    = ghost}
                    );
                }
            }
        }

        return interfaces;
    }

    // corrects coarse fluxes using fine level fluxes at interfaces
    template <typename T, std::uint64_t Dims, Geometry G>
    void correct_interface_fluxes(
        std::vector<field_t<T, Dims>>& fluxes,
        const fmr_mesh_t<Dims, G>& mesh,
        std::uint64_t flux_direction
    )
    {
        // get all interfaces
        auto interfaces = identify_flux_interfaces<T, Dims, G>(mesh);

        // process each interface
        for (const auto& interface : interfaces) {
            // only process interfaces normal to flux direction
            if (interface.direction != flux_direction) {
                continue;
            }

            const auto& fine_level = mesh.level(interface.fine_level_id);

            // get fine fluxes for this interface
            auto fine_fluxes = fluxes[interface.fine_level_id];

            // compute averaged fine fluxes (restricted to coarse grid)
            auto coarse_domain   = interface.interface_domain;
            auto restricted_flux = make_restriction(
                fine_fluxes,
                coarse_domain,
                fine_level.ref_ratio
            );

            // replace coarse fluxes with averaged fine fluxes
            auto& coarse_fluxes = fluxes[interface.coarse_level_id];
            for (const auto& coord : coarse_domain) {
                coarse_fluxes(coord) = restricted_flux(coord);
            }
        }
    }

    // updates fluxes at all refinement level interfaces
    template <typename T, std::uint64_t Dims, Geometry G>
    void synchronize_interface_fluxes(
        std::vector<field_t<T, Dims>>& fluxes,
        const fmr_mesh_t<Dims, G>& mesh
    )
    {
        // correct fluxes in each direction
        for (std::uint64_t dir = 0; dir < Dims; ++dir) {
            correct_interface_fluxes(fluxes, mesh, dir);
        }
    }

}   // namespace simbi::mesh::refinement

#endif   // FLUX_CORRECTION_HPP
