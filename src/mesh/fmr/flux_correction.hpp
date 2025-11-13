#ifndef FMR_FLUX_CORRECTION_HPP
#define FMR_FLUX_CORRECTION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
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
        const auto& coverage     = map.coarse_staggered_coverage;
        const auto& coarse_faces = map.coarse_face_domains[flux_direction];

        // lower interface
        {
            auto& lower = interfaces[0];
            lower.is_valid =
                (coverage.start[flux_direction] >
                 coarse_faces.start[flux_direction]);

            if (lower.is_valid) {
                // coarse face: full transverse extent of coverage, 1 cell in
                // flux direction
                lower.coarse_face = coverage;   // base for transverse dims

                //  j_start, the logical flux index
                lower.coarse_face.start[flux_direction] =
                    coverage.start[flux_direction];

                lower.coarse_face.fin[flux_direction] =
                    coverage.start[flux_direction] + 1;

                // map coarse face corners to fine coordinates
                auto fine_start =
                    map.coarse_to_fine_face_base(lower.coarse_face.start);
                auto fine_end =
                    map.coarse_to_fine_face_base(lower.coarse_face.fin);

                // build fine face domain
                lower.fine_face.start = fine_start;
                lower.fine_face.fin   = fine_end;

                // in flux direction: make it 1 cell thick
                lower.fine_face.fin[flux_direction] =
                    fine_start[flux_direction] + 1;
            }
        }

        // upper interface
        {
            auto& upper = interfaces[1];
            upper.is_valid =
                (coverage.fin[flux_direction] <
                 coarse_faces.fin[flux_direction]);

            if (upper.is_valid) {
                // set up coarse face exactly at the end of coverage
                upper.coarse_face = coverage;   // base for transverse

                // This is j_end, the logical flux index
                upper.coarse_face.start[flux_direction] =
                    coverage.fin[flux_direction];

                upper.coarse_face.fin[flux_direction] =
                    coverage.fin[flux_direction] + 1;

                auto fine_start =
                    map.coarse_to_fine_face_base(upper.coarse_face.start);
                auto fine_end =
                    map.coarse_to_fine_face_base(upper.coarse_face.fin);

                upper.fine_face.start = fine_start;
                upper.fine_face.fin   = fine_end;

                // make it 1 cell thick in flux direction
                upper.fine_face.fin[flux_direction] =
                    fine_start[flux_direction] + 1;
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
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction,
        real dt_coarse
    )
    {
        if (!interface.is_valid) {
            return;
        }

        for (const auto& coarse_coord : interface.coarse_face) {
            auto fine_children =
                map.fine_face_children(coarse_coord, flux_direction);

            T sum{};
            real count = 0;
            bool first = true;

            for (const auto& fine_coord : fine_children) {
                if (fine_flux.domain().contains(fine_coord)) {
                    if (first) {
                        sum   = fine_flux(fine_coord);
                        first = false;
                    }
                    else {
                        sum = sum | add_gas(fine_flux(fine_coord));
                    }
                    count += 1.0;
                }
            }

            if (count > 0) {
                coarse_flux(coarse_coord) = (sum / dt_coarse) / count;
            }
        }
    }

    // correct fluxes in one direction for a level
    template <typename T, std::uint64_t Dims>
    void correct_fluxes_direction(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const level_mapping_t<Dims>& map,
        std::uint64_t flux_direction,
        real dt_coarse
    )
    {
        auto interfaces = get_flux_interfaces(map, flux_direction);
        for (const auto& interface : interfaces) {
            correct_interface_fluxes(
                coarse_flux,
                fine_flux,
                interface,
                map,
                flux_direction,
                dt_coarse
            );
        }
    }

    // correct all fluxes (all directions) for a level
    template <typename T, std::uint64_t Dims>
    void correct_level_fluxes(
        vector_t<field_t<T, Dims>, Dims>& coarse_fluxes,
        const vector_t<field_t<T, Dims>, Dims>& fine_fluxes,
        const level_mapping_t<Dims>& map,
        real dt_coarse
    )
    {
        // correct fluxes in each coordinate direction
        for (std::uint64_t dir = 0; dir < Dims; ++dir) {
            correct_fluxes_direction(
                coarse_fluxes[dir],
                fine_fluxes[dir],
                map,
                dir,
                dt_coarse
            );
        }
    }
}   // namespace simbi::mesh::fmr

#endif
