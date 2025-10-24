#ifndef TRANSFER_HPP
#define TRANSFER_HPP

#include "compat.hpp"                          // for real type
#include "compute/field.hpp"                   // for field_t
#include "containers/vector.hpp"               // for vector_t
#include "domain/domain.hpp"                   // for domain_t
#include "functional/fp.hpp"                   // for fp::partial
#include "mesh/refinement/interpolation.hpp"   // for interpolation utilities

#include <cstdint>   // for std::uint64_t

namespace simbi::mesh::refinement {

    // creates a prolongation transform for coarse->fine interpolation
    template <typename T, std::uint64_t Dims>
    auto make_prolongation(
        const field_t<T, Dims>& coarse_field,
        const domain_t<Dims>& fine_domain,
        const vector_t<real, Dims>& ref_ratio,
        bool conservative = true
    )
    {
        // create interpolation context
        interpolation_context_t<T, Dims> ctx{
          .coarse_field  = coarse_field,
          .coarse_domain = coarse_field.domain(),
          .fine_domain   = fine_domain,
          .ref_ratio     = ref_ratio,
          .coarse_offset = coarse_field.domain().start,
          .fine_offset   = fine_domain.start
        };

        // create and return the interpolation field
        return make_interpolation_field(ctx, conservative);
    }

    // creates a restriction transform for fine->coarse averaging
    template <typename T, std::uint64_t Dims>
    struct restriction_transform_t {
        const field_t<T, Dims>& fine_field;
        const vector_t<real, Dims> ref_ratio;

        DUAL T operator()(const coordinate_t<Dims>& coarse_coord) const
        {
            T sum{};
            real volume = 1.0;

            // compute fine cell range for this coarse cell
            coordinate_t<Dims> fine_start, fine_end;
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                fine_start[dd] = coarse_coord[dd] * ref_ratio[dd];
                fine_end[dd]   = fine_start[dd] + ref_ratio[dd];
                volume *= ref_ratio[dd];
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
        const vector_t<real, Dims>& ref_ratio
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
        const vector_t<real, Dims>& ref_ratio,
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
        const vector_t<real, Dims>& ref_ratio
    )
    {
        auto restricted =
            make_restriction(fine_field, coarse_region, ref_ratio);

        // copy restricted values to coarse field in the specified region
        for (const auto& coord : coarse_region) {
            coarse_field(coord) = restricted(coord);
        }
    }

}   // namespace simbi::mesh::refinement

#endif   // TRANSFER_HPP
