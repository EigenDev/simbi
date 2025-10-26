#ifndef TRANSFER_HPP
#define TRANSFER_HPP

#include "compat.hpp"              // for real type
#include "compute/field.hpp"       // for field_t
#include "containers/vector.hpp"   // for vector_t
#include "domain/domain.hpp"       // for domain_t
#include "functional/fp.hpp"       // for fp::partial
#include "interpolation.hpp"       // for interpolation utilities

#include <cstdint>   // for std::uint64_t

namespace simbi::mesh::fmr {

    // creates a prolongation transform for coarse->fine interpolation
    template <typename T, std::uint64_t Dims>
    auto make_prolongation(
        const field_t<T, Dims>& coarse_field,
        const domain_t<Dims>& fine_domain,
        const domain_t<Dims>& parent_coverage,
        std::uint64_t ref_ratio,
        bool conservative = true
    )
    {
        // create interpolation context
        interpolation_context_t<T, Dims> ctx{
          .coarse_field  = coarse_field,
          .coarse_domain = parent_coverage,
          .fine_domain   = fine_domain,
          .ref_ratio     = ref_ratio,
          .coarse_offset = parent_coverage.start,
          .fine_offset   = fine_domain.start
        };

        // create and return the interpolation field
        return make_interpolation_field(ctx, conservative);
    }

    // creates a restriction transform for fine->coarse averaging
    template <typename T, std::uint64_t Dims>
    struct restriction_transform_t {
        const field_t<T, Dims>& fine_field;
        std::uint64_t ref_ratio;

        DUAL T operator()(const coordinate_t<Dims>& coarse_coord) const
        {
            T sum{};
            real volume = 1.0;

            // compute fine cell range for this coarse cell
            coordinate_t<Dims> fine_start, fine_end;
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                fine_start[dd] = coarse_coord[dd] * ref_ratio;
                fine_end[dd]   = fine_start[dd] + ref_ratio;
                volume *= ref_ratio;
            }

            // sum over fine cells
            coordinate_t<Dims> fine_coord;
            for (fine_coord[0] = fine_start[0]; fine_coord[0] < fine_end[0];
                 ++fine_coord[0]) {
                if constexpr (Dims > 1) {
                    for (fine_coord[1] = fine_start[1];
                         fine_coord[1] < fine_end[1];
                         ++fine_coord[1]) {
                        if constexpr (Dims > 2) {
                            for (fine_coord[2] = fine_start[2];
                                 fine_coord[2] < fine_end[2];
                                 ++fine_coord[2]) {
                                sum += fine_field(fine_coord);
                            }
                        }
                        else {
                            sum += fine_field(fine_coord);
                        }
                    }
                }
                else {
                    sum += fine_field(fine_coord);
                }
            }

            // return volume-weighted average
            return sum / volume;
        }
    };

    // creates a restriction transform for fine->coarse averaging
    template <typename T, std::uint64_t Dims>
    auto make_restriction(
        const field_t<T, Dims>& fine_field,
        const domain_t<Dims>& coarse_domain,
        std::uint64_t ref_ratio
    )
    {
        auto transform =
            restriction_transform_t<T, Dims>{fine_field, ref_ratio};
        return field(coarse_domain, fp::partial(transform));
    }

    // applies prolongation to fill a fine region from coarse data
    template <typename T, std::uint64_t Dims>
    void fill_fine_region(
        field_t<T, Dims>& fine_field,
        const field_t<T, Dims>& coarse_field,
        const domain_t<Dims>& fine_region,
        std::uint64_t ref_ratio,
        bool conservative = true
    )
    {
        auto prolonged = make_prolongation(
            coarse_field,
            fine_region,
            ref_ratio,
            conservative
        );

        // copy prolonged values to fine field in the specified region
        for (const auto& coord : fine_region) {
            fine_field(coord) = prolonged(coord);
        }
    }

    // applies restriction to fill a coarse region from fine data
    template <typename T, std::uint64_t Dims>
    void fill_coarse_region(
        field_t<T, Dims>& coarse_field,
        const field_t<T, Dims>& fine_field,
        const domain_t<Dims>& coarse_region,
        std::uint64_t ref_ratio
    )
    {
        auto restricted =
            make_restriction(fine_field, coarse_region, ref_ratio);

        // copy restricted values to coarse field in the specified region
        for (const auto& coord : coarse_region) {
            coarse_field(coord) = restricted(coord);
        }
    }

    template <typename Sim>
    void prolongate_level_data(Sim& sim, std::uint64_t lvl)
    {
        auto& fine   = sim.hydro(lvl);
        auto& coarse = sim.hydro(lvl - 1);
        auto& meta   = sim.metadata();

        // get refinement info to know the ratio
        auto& refinement = sim.refinement(lvl);

        auto& level_info = sim.level_info(lvl);
        auto ratio       = level_info.refinement_ratio;

        // std::cout << "Prolongating from level " << (lvl - 1) << " to level "
        //           << lvl << " with ratio " << ratio << std::endl;
        // std::cout << "Coarse domain: " << coarse.prim.domain() << std::endl;
        // std::cout << "Fine domain: " << fine.prim.domain() << std::endl;
        // std::cout << "Refinement region: " << refinement.parent_coverage
        //           << std::endl;

        // prolongate primitives conservatively
        auto prolonged = make_prolongation(
            coarse.prim,
            fine.prim.domain(),
            refinement.parent_coverage,
            ratio,
            true   // conservative
        );

        for (const auto& coord : fine.prim.domain()) {
            fine.prim(coord) = prolonged(coord);
        }

        // convert to conserved
        for (const auto& coord : fine.cons.domain()) {
            fine.cons(coord) = to_conserved(fine.prim(coord), meta.gamma);
        }

        // if mhd mode, prolongate magnetic fields and flux ffields as well
        if constexpr (Sim::is_mhd) {
            for (std::uint64_t d = 0; d < Sim::dimensions; ++d) {
                auto prolonged_b = make_prolongation(
                    coarse.flux[d],
                    fine.flux[d].domain(),
                    refinement.parent_coverage,
                    ratio,
                    true   // conservative
                );

                for (const auto& coord : fine.flux[d].domain()) {
                    fine.flux[d](coord) = prolonged_b(coord);
                }
            }
        }
    }

}   // namespace simbi::mesh::fmr

#endif   // TRANSFER_HPP
