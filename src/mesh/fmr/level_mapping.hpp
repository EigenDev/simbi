#ifndef LEVEL_MAPPING_HPP
#define LEVEL_MAPPING_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"

#include <cstdint>

namespace simbi::mesh::fmr {

    template <std::uint64_t Dims>
    struct level_mapping_t {
        domain_t<Dims> fine_full;
        domain_t<Dims> fine_active;
        domain_t<Dims> coarse_full;
        domain_t<Dims> coarse_coverage;
        std::uint64_t ratio;

        constexpr DUAL coordinate_t<Dims>
        fine_to_coarse(const coordinate_t<Dims>& fine_coord) const
        {
            coordinate_t<Dims> coarse_coord;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // map fine local coord to offset within fine active region
                auto fine_offset = fine_coord[d] - fine_active.start[d];

                // scale by ratio to get offset within coarse coverage
                auto coarse_offset =
                    fine_offset / static_cast<std::int64_t>(ratio);

                // add to coarse coverage start to get absolute coarse coord
                coarse_coord[d] = coarse_coverage.start[d] + coarse_offset;
            }
            return coarse_coord;
        }

        constexpr DUAL coordinate_t<Dims>
        coarse_to_fine_base(const coordinate_t<Dims>& coarse_coord) const
        {
            coordinate_t<Dims> fine_coord;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // map coarse coord to offset within coarse coverage
                auto coarse_offset = coarse_coord[d] - coarse_coverage.start[d];

                // scale by ratio to get offset within fine active region
                auto fine_offset =
                    coarse_offset * static_cast<std::int64_t>(ratio);

                // add to fine active start to get absolute fine coord
                fine_coord[d] = fine_active.start[d] + fine_offset;
            }
            return fine_coord;
        }

        constexpr domain_t<Dims>
        fine_children(const coordinate_t<Dims>& coarse_coord) const
        {
            auto base = coarse_to_fine_base(coarse_coord);
            auto end  = base;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                end[d] += static_cast<std::int64_t>(ratio);
            }
            return domain_t<Dims>{base, end};
        }

        constexpr DUAL coordinate_t<Dims>
        fine_offset_in_coarse(const coordinate_t<Dims>& fine_coord) const
        {
            coordinate_t<Dims> offset;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // offset relative to fine active start
                auto fine_offset = fine_coord[d] - fine_active.start[d];
                // modulo to get position within parent coarse cell
                offset[d] = fine_offset % static_cast<std::int64_t>(ratio);
            }
            return offset;
        }
    };

    // factory function to create mapping from hierarchy levels
    template <std::uint64_t dims>
    level_mapping_t<dims> make_level_mapping(
        const domain_t<dims>& fine_full,
        const domain_t<dims>& fine_active,
        const domain_t<dims>& coarse_full,
        const domain_t<dims>& coarse_coverage,
        std::uint64_t ratio
    )
    {
        return level_mapping_t<dims>{
          fine_full,
          fine_active,
          coarse_full,
          coarse_coverage,
          ratio
        };
    }

}   // namespace simbi::mesh::fmr

#endif
