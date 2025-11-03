#ifndef LEVEL_MAPPING_HPP
#define LEVEL_MAPPING_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <stdexcept>

namespace simbi::mesh::fmr {
    // compute mathematical floor(a / b) for integers
    constexpr DUAL std::int64_t floor_div(std::int64_t a, std::int64_t b)
    {
        // note: C++ division truncates toward zero, which is wrong for
        // negatives. we need true floor division.
        if (a >= 0) {
            return a / b;
        }
        // for negative a, C++'s (a / b) is wrong.
        // (a - b + 1) / b is also not floor.
        std::int64_t d = a / b;
        std::int64_t r = a % b;
        return (r != 0 && (a < 0 != b < 0)) ? d - 1 : d;

        // A simpler, correct alternative if you don't mind the cast:
        // return static_cast<std::int64_t>(std::floor(
        //     static_cast<double>(a) / static_cast<double>(b)
        // ));
    }

    // computes mathematical (a mod b) consistent with floor_div
    constexpr DUAL std::int64_t floor_mod(std::int64_t a, std::int64_t b)
    {
        // we want: a = b * floor_div(a, b) + floor_mod(a, b)
        // so: floor_mod(a, b) = a - b * floor_div(a, b)
        return a - b * floor_div(a, b);
    }

    template <std::uint64_t Dims>
    struct level_mapping_t {
        domain_t<Dims> fine_full;
        domain_t<Dims> fine_active;
        domain_t<Dims> coarse_full;
        domain_t<Dims> coarse_active;
        domain_t<Dims> coarse_coverage;
        domain_t<Dims> coarse_staggered_coverage;
        vector_t<domain_t<Dims>, Dims> coarse_face_domains;
        vector_t<domain_t<Dims>, Dims> fine_face_domains;
        std::uint64_t ratio;

        constexpr DUAL coordinate_t<Dims>
        fine_to_coarse(const coordinate_t<Dims>& fine_coord) const
        {
            coordinate_t<Dims> coarse_coord;
            const auto iratio = static_cast<std::int64_t>(ratio);

            for (std::uint64_t d = 0; d < Dims; ++d) {
                // map fine local coord to offset within fine active region
                auto fine_offset = fine_coord[d] - fine_active.start[d];

                // scale by ratio to get offset within coarse coverage
                // must use floor_div for correct negative offset mapping
                auto coarse_offset = floor_div(fine_offset, iratio);

                // add to coarse coverage start to get absolute coarse coord
                coarse_coord[d] = coarse_coverage.start[d] + coarse_offset;
            }
            return coarse_coord;
        }

        constexpr DUAL coordinate_t<Dims>
        coarse_to_fine_base(const coordinate_t<Dims>& coarse_coord) const
        {
            coordinate_t<Dims> fine_coord;
            const auto iratio = static_cast<std::int64_t>(ratio);
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // map coarse coord to offset within coarse coverage
                auto coarse_offset = coarse_coord[d] - coarse_coverage.start[d];

                // scale by ratio to get offset within fine active region
                auto fine_offset = coarse_offset * iratio;

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
            const auto iratio = static_cast<std::int64_t>(ratio);
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // offset relative to fine active start
                auto fine_offset = fine_coord[d] - fine_active.start[d];

                // modulo to get position within parent coarse cell
                // must use floor_mod for correct negative offset mapping
                offset[d] = floor_mod(fine_offset, iratio);
            }
            return offset;
        }

        constexpr DUAL coordinate_t<Dims> coarse_to_fine_face_base(
            const coordinate_t<Dims>& coarse_coord_logical
        ) const
        {
            coordinate_t<Dims> fine_coord_logical;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                // get the coarse logical index (j)
                auto j = coarse_coord_logical[d];

                // get the coarse logical index of the start of the fine patch
                auto j_start = coarse_staggered_coverage.start[d];

                // calculate the fine logical index (m)
                fine_coord_logical[d] =
                    (j - j_start) * static_cast<std::int64_t>(ratio);
            }
            return fine_coord_logical;
        }

        constexpr DUAL domain_t<Dims> fine_face_children(
            const coordinate_t<Dims>& coarse_coord,
            std::uint64_t direction
        ) const
        {
            auto base         = coarse_to_fine_face_base(coarse_coord);
            auto end          = base;
            const auto iratio = static_cast<std::int64_t>(ratio);
            for (std::uint64_t d = 0; d < Dims; ++d) {
                if (d == direction) {
                    end[d] += 1;   // face-centered, so only increment by 1
                }
                else {
                    end[d] += iratio;
                }
            }
            return domain_t<Dims>{base, end};
        }
    };

    template <std::uint64_t Dims>
    level_mapping_t<Dims> create_level_mapping(
        const mesh_hierarchy_t<Dims>& hierarchy,
        std::uint64_t fine_level_id
    )
    {
        if (fine_level_id == 0) {
            throw std::runtime_error("Cannot create mapping for base level");
        }

        const auto& fine_level   = hierarchy[fine_level_id];
        const auto& coarse_level = hierarchy[fine_level.parent_level_id];

        // convert parent_coverage from parent's active coordinate system
        // to parent's full domain coordinate system (which includes ghost
        // zones)
        auto coarse_coverage_adjusted = fine_level.parent_coverage;
        for (std::uint64_t d = 0; d < Dims; ++d) {
            // add offset from full domain origin to active domain start
            coarse_coverage_adjusted.start[d] += coarse_level.domain.start[d];
            coarse_coverage_adjusted.fin[d] += coarse_level.domain.start[d];
        }

        domain_t<Dims> logical_coverage;
        for (std::uint64_t d = 0; d < Dims; ++d) {
            // get the "ground truth" physical values
            real coarse_start = coarse_level.physical_min[d];
            real fine_start   = fine_level.physical_min[d];
            real fine_end     = fine_level.physical_max[d];
            real coarse_dx    = coarse_level.dx[d];

            // Calculate logical index of the start of the fine patch
            logical_coverage.start[d] = static_cast<std::int64_t>(
                std::round((fine_start - coarse_start) / coarse_dx)
            );

            // Calculate logical index of the end of the fine patch
            logical_coverage.fin[d] = static_cast<std::int64_t>(
                std::round((fine_end - coarse_start) / coarse_dx)
            );
        }

        return level_mapping_t<Dims>{
          .fine_full                 = fine_level.full_domain,
          .fine_active               = fine_level.domain,
          .coarse_full               = coarse_level.full_domain,
          .coarse_active             = coarse_level.domain,
          .coarse_coverage           = coarse_coverage_adjusted,
          .coarse_staggered_coverage = logical_coverage,
          .coarse_face_domains       = coarse_level.face_domains,
          .fine_face_domains         = fine_level.face_domains,
          .ratio = fine_level.ref_ratio / coarse_level.ref_ratio
        };
    }

}   // namespace simbi::mesh::fmr

#endif
