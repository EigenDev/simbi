#ifndef SIMBI_MHD_INTERP_HPP
#define SIMBI_MHD_INTERP_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "mesh/mesh_ops.hpp"
#include "utility/enums.hpp"
#include <cstdint>

namespace simbi::em {
    using namespace simbi::domain_algebra;
    template <typename HydroState>
    struct interpolate_face_to_cell_magnetic_t {
        const HydroState& state;
        static constexpr auto dimensions = HydroState::dimensions;

        template <typename Coord>
        auto operator()(Coord cell_coord, const auto& domain) const
        {
            return apply(cell_coord, domain);
        }

        template <typename Coord>
        auto apply(Coord cell_coord, const auto& /*domain*/) const
        {
            using vector_t = vector_t<real, dimensions>;
            vector_t b_cell;
            const auto& mesh = state.mesh;

            // for each component, average the face-centered values
            for (std::uint64_t dim = 0; dim < dimensions; ++dim) {
                auto face_domain  = active_staggered_domain(mesh.domain, dim);
                const auto b_face = state.bstaggs[dim][face_domain];
                // create offset vectors for this dimension
                iarray<dimensions> neg_offset{0};
                iarray<dimensions> pos_offset{0};
                neg_offset[dim] = 0;
                pos_offset[dim] = 1;

                // get the two face indices that bracket this cell
                const auto cm = cell_coord + neg_offset;
                const auto cp = cell_coord + pos_offset;

                // switch to physical index
                const auto phys_idx = 3 - (dim + 1);
                if constexpr (HydroState::geometry_t == Geometry::CARTESIAN) {
                    // simple arithmetic average for Cartesian
                    b_cell[phys_idx] = 0.5 * (b_face[cm] + b_face[cp]);
                }
                else {
                    // volume-weighted average for non-Cartesian
                    auto al = mesh::face_area(cm, dim, Dir::E, mesh);
                    auto ar = mesh::face_area(cp, dim, Dir::W, mesh);
                    b_cell[phys_idx] =
                        (b_face[cm] * al + b_face[cp] * ar) / (al + ar);
                }
            }

            return b_cell;
        }
    };

    // factory function
    template <typename HydroState>
    auto interpolate_face_to_cell_magnetic(const HydroState& state)
    {
        return interpolate_face_to_cell_magnetic_t<HydroState>{state};
    }
}   // namespace simbi::em

#endif
