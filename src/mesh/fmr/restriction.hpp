#ifndef RESTRICTION_HPP
#define RESTRICTION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/state_ops.hpp"
#include "level_mapping.hpp"

#include <cstdint>

namespace simbi::mesh::fmr {
    // conservative restriction (volume-weighted averaging)
    template <typename T, std::uint64_t Dims>
    void restrict_conservative(
        const field_t<T, Dims>& fine,
        field_t<T, Dims>& coarse,
        const level_mapping_t<Dims>& map
    )
    {
        using namespace simbi::structs;

        // iterate over coarse cells that correspond to fine active region
        for (const auto& coarse_coord : map.coarse_coverage) {

            // get fine children of this coarse cell
            auto children = map.fine_children(coarse_coord);

            // average fine cells (conservative restriction)
            T sum{};
            real volume     = 0;
            bool first_cell = true;

            for (const auto& fine_coord : children) {
                // only use fine cells within the fine domain
                if (map.fine_full.contains(fine_coord)) {
                    if (first_cell) {
                        sum        = fine(fine_coord);
                        first_cell = false;
                    }
                    else {
                        sum = sum | add_gas(fine(fine_coord));
                    }
                    volume += 1.0;
                }
            }

            // update coarse cell
            if (volume > 0) {
                coarse(coarse_coord) = sum / volume;
            }
        }
    }

    // injection (just copy one fine cell to coarse)
    template <typename T, std::uint64_t Dims>
    void restrict_injection(
        const field_t<T, Dims>& fine,
        field_t<T, Dims>& coarse,
        const level_mapping_t<Dims>& map
    )
    {
        for (const auto& coarse_coord : map.coarse_coverage) {
            // just take the first fine child
            auto base_fine = map.coarse_to_fine_base(coarse_coord);
            if (map.fine_full.contains(base_fine)) {
                coarse(coarse_coord) = fine(base_fine);
            }
        }
    }

}   // namespace simbi::mesh::fmr

#endif
