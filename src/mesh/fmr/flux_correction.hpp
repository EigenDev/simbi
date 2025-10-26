#ifndef FMR_FLUX_CORRECTION_HPP
#define FMR_FLUX_CORRECTION_HPP

#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "hierarchy.hpp"
#include "level_descriptor.hpp"
#include "mesh/fmr/transfer.hpp"

#include <cstdint>

namespace simbi::mesh::fmr {

    // map fine domain to coarse index space
    template <std::uint64_t Dims>
    domain_t<Dims> map_fine_to_coarse_domain(
        const domain_t<Dims>& fine_domain,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& coarse_level
    )
    {
        auto ratio = fine_level.ref_ratio / coarse_level.ref_ratio;

        iarray<Dims> coarse_start, coarse_end;
        for (std::uint64_t d = 0; d < Dims; ++d) {
            // translate to fine level origin, scale, translate to coarse
            coarse_start[d] =
                fine_level.parent_coverage.start[d] +
                (fine_domain.start[d] - fine_level.domain.start[d]) /
                    static_cast<std::int64_t>(ratio);

            coarse_end[d] = fine_level.parent_coverage.start[d] +
                            (fine_domain.fin[d] - fine_level.domain.start[d]) /
                                static_cast<std::int64_t>(ratio);
        }

        return domain_t<Dims>{coarse_start, coarse_end};
    }

    // find coarse-fine interface for flux correction
    template <std::uint64_t Dims>
    domain_t<Dims> find_flux_interface(
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& coarse_level,
        std::uint64_t flux_direction
    )
    {
        // flux interfaces are at the boundaries of the fine level's parent
        // coverage we need the face-centered flux domain, which extends one
        // cell beyond

        // get parent coverage in coarse index space
        auto coverage = fine_level.parent_coverage;

        // for face-centered fluxes, the interface is the face at the boundary
        // check lower boundary
        auto lower_face =
            domain_algebra::get_lower_boundary(coverage, flux_direction, 1);

        // check upper boundary
        auto upper_face =
            domain_algebra::get_upper_boundary(coverage, flux_direction, 1);

        // determine which boundary is actually an interface
        // (touches coarse level but not another fine region)

        // for now, return lower face if it's at the start of coverage
        // this is simplified - you may need more logic for complex hierarchies
        if (coverage.start[flux_direction] >
            coarse_level.domain.start[flux_direction]) {
            return lower_face;
        }

        if (coverage.fin[flux_direction] <
            coarse_level.domain.fin[flux_direction]) {
            return upper_face;
        }

        return domain_t<Dims>{};   // no interface
    }

    // correct coarse fluxes using fine fluxes at interface
    template <typename T, std::uint64_t Dims>
    void correct_interface_flux(
        field_t<T, Dims>& coarse_flux,
        const field_t<T, Dims>& fine_flux,
        const level_descriptor_t<Dims>& fine_level,
        const level_descriptor_t<Dims>& coarse_level,
        std::uint64_t flux_direction
    )
    {
        // find interface in coarse coordinates
        auto interface =
            find_flux_interface(fine_level, coarse_level, flux_direction);

        if (interface.empty()) {
            return;   // no interface in this direction
        }

        // map interface to fine index space to get fine flux domain
        auto ratio = fine_level.ref_ratio / coarse_level.ref_ratio;

        iarray<Dims> fine_start, fine_end;
        for (std::uint64_t d = 0; d < Dims; ++d) {
            fine_start[d] =
                fine_level.domain.start[d] +
                (interface.start[d] - fine_level.parent_coverage.start[d]) *
                    static_cast<std::int64_t>(ratio);

            fine_end[d] =
                fine_level.domain.start[d] +
                (interface.fin[d] - fine_level.parent_coverage.start[d]) *
                    static_cast<std::int64_t>(ratio);
        }

        // auto fine_interface = domain_t<Dims>{fine_start, fine_end};

        auto restricted = make_restriction(fine_flux, interface, ratio);

        // replace coarse fluxes with restricted fine fluxes
        for (const auto& coord : interface) {
            coarse_flux(coord) = restricted(coord);
        }
    }

    // correct all flux interfaces for a single fine level
    template <typename T, std::uint64_t Dims>
    void correct_level_fluxes(
        vector_t<field_t<T, Dims>, Dims>& coarse_fluxes,
        const vector_t<field_t<T, Dims>, Dims>& fine_fluxes,
        const mesh_hierarchy_t<Dims>& hierarchy,
        std::uint64_t fine_level_id
    )
    {
        if (fine_level_id == 0) {
            return;   // no coarser level to correct
        }

        const auto& fine_level   = hierarchy[fine_level_id];
        const auto& coarse_level = hierarchy[fine_level.parent_level_id];

        // correct fluxes in each coordinate direction
        for (std::uint64_t dir = 0; dir < Dims; ++dir) {
            correct_interface_flux(
                coarse_fluxes[dir],
                fine_fluxes[dir],
                fine_level,
                coarse_level,
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
            correct_level_fluxes(
                level_fluxes[lvl - 1],
                level_fluxes[lvl],
                hierarchy,
                lvl
            );
        }
    }

}   // namespace simbi::mesh::fmr

#endif
