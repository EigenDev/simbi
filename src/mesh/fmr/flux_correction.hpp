#ifndef FMR_FLUX_CORRECTION_HPP
#define FMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "hierarchy.hpp"
#include "level_mapping.hpp"

#include <cstdint>

namespace simbi::mesh::fmr {

    // returns the coarse-fine interface faces for flux correction
    template <std::uint64_t Dims>
    struct flux_interface_t {
        domain_t<Dims> coarse_face;   // face in coarse coordinates
        domain_t<Dims> fine_face;     // corresponding face in fine coordinates
        bool is_valid;
    };

    // get lower and upper interface faces in a given direction
    template <std::uint64_t Dims>
    vector_t<flux_interface_t<Dims>, 2> get_flux_interfaces(
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction
    )
    {
        vector_t<flux_interface_t<Dims>, 2> interfaces;
        const auto& coverage = map.coarse_coverage;

        // lower interface (at coverage.start in flux_direction)
        {
            auto& lower = interfaces[0];
            lower.is_valid =
                (coverage.start[flux_direction] >
                 map.coarse_full.start[flux_direction]);

            if (lower.is_valid) {
                // coarse face at the boundary
                lower.coarse_face = coverage;
                lower.coarse_face.start[flux_direction] =
                    coverage.start[flux_direction];
                lower.coarse_face.fin[flux_direction] =
                    coverage.start[flux_direction] + 1;

                // corresponding fine face
                auto fine_base =
                    map.coarse_to_fine_base(lower.coarse_face.start);
                auto fine_end = map.coarse_to_fine_base(lower.coarse_face.fin);

                lower.fine_face = domain_t<Dims>{fine_base, fine_end};
            }
        }

        // upper interface (at coverage.fin in flux_direction)
        {
            auto& upper = interfaces[1];
            upper.is_valid =
                (coverage.fin[flux_direction] <
                 map.coarse_full.fin[flux_direction]);

            if (upper.is_valid) {
                // coarse face at the boundary
                upper.coarse_face = coverage;
                upper.coarse_face.start[flux_direction] =
                    coverage.fin[flux_direction];
                upper.coarse_face.fin[flux_direction] =
                    coverage.fin[flux_direction] + 1;

                // corresponding fine face
                auto fine_base =
                    map.coarse_to_fine_base(upper.coarse_face.start);
                auto fine_end = map.coarse_to_fine_base(upper.coarse_face.fin);

                upper.fine_face = domain_t<Dims>{fine_base, fine_end};
            }
        }

        return interfaces;
    }

    // correct fluxes at a single coarse-fine interface
    template <typename T, std::uint64_t Dims>
    void correct_interface_fluxes(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const flux_interface_t<Dims>& interface,
        const level_mapping_t<Dims>& map
    )
    {
        if (!interface.is_valid) {
            return;
        }

        // for each coarse flux face, average the corresponding fine fluxes
        // auto cf = coarse_flux[interface.coarse_face];
        // cf      = cf.coord_map([fine_flux, map](auto coarse_coord) {
        //     // get the fine flux faces that correspond to this coarse face
        //     auto fine_children = map.fine_children(coarse_coord);

        //     // average fine fluxes (conservative restriction)
        //     T sum{};
        //     real volume = 0;
        //     bool first  = true;

        //     for (const auto& fine_coord : fine_children) {
        //         if (fine_flux.domain().contains(fine_coord)) {
        //             if (first) {
        //                 sum   = fine_flux(fine_coord);
        //                 first = false;
        //             }
        //             else {
        //                 sum = sum | add_gas(fine_flux(fine_coord));
        //             }
        //             volume += 1.0;
        //         }
        //     }

        //     if (volume > 0) {
        //         return sum / volume;
        //     }
        // });

        for (const auto& coarse_coord : interface.coarse_face) {
            // get the fine flux faces that correspond to this coarse face
            auto fine_children = map.fine_children(coarse_coord);

            // average fine fluxes (conservative restriction)
            T sum{};
            real volume = 0;
            bool first  = true;

            for (const auto& fine_coord : fine_children) {
                if (fine_flux.domain().contains(fine_coord)) {
                    if (first) {
                        sum   = fine_flux(fine_coord);
                        first = false;
                    }
                    else {
                        sum = sum | add_gas(fine_flux(fine_coord));
                    }
                    volume += 1.0;
                }
            }

            if (volume > 0) {
                coarse_flux(coarse_coord) = sum / volume;
            }
        }
    }

    // correct fluxes in one direction for a level
    template <typename T, std::uint64_t Dims>
    void correct_fluxes_direction(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction
    )
    {
        auto interfaces = get_flux_interfaces(map, flux_direction);

        for (const auto& interface : interfaces) {
            correct_interface_fluxes(coarse_flux, fine_flux, interface, map);
        }
    }

    // correct all fluxes (all directions) for a level
    template <typename T, std::uint64_t Dims>
    void correct_level_fluxes(
        vector_t<field_t<T, Dims>, Dims>& coarse_fluxes,
        const vector_t<field_t<T, Dims>, Dims>& fine_fluxes,
        const level_mapping_t<Dims>& map
    )
    {
        // correct fluxes in each coordinate direction
        for (std::uint64_t dir = 0; dir < Dims; ++dir) {
            correct_fluxes_direction(
                coarse_fluxes[dir],
                fine_fluxes[dir],
                map,
                dir
            );
        }
    }

    // synchronize all flux interfaces in hierarchy
    template <typename T, std::uint64_t Dims, std::uint32_t MAX_LEVELS = 8>
    void synchronize_fluxes(
        vector_t<vector_t<field_t<T, Dims>, Dims>, MAX_LEVELS>& level_fluxes,
        const mesh_hierarchy_t<Dims>& hierarchy
    )
    {
        // work from finest to coarsest
        for (std::uint64_t lvl = hierarchy.num_levels - 1; lvl > 0; --lvl) {

            // create mapping for this level
            auto map = create_level_mapping(hierarchy, lvl);

            // correct fluxes
            correct_level_fluxes(level_fluxes[lvl - 1], level_fluxes[lvl], map);
        }
    }

}   // namespace simbi::mesh::fmr

#endif
