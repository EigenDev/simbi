#ifndef SIMBI_CORE_BASE_STENCIL_SPACE_HPP
#define SIMBI_CORE_BASE_STENCIL_SPACE_HPP

#include "compute/math/field.hpp"
#include "compute/math/index_space.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/stencil.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "data/containers/vector.hpp"
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

namespace simbi::base {

    // convert stencil pattern to coordinate space relative to a base coordinate
    template <std::uint64_t Dims, Reconstruction Rec>
    class stencil_space_t
    {
      public:
        using stencil_type  = stencil_t<Dims, Rec>;
        using coord_array_t = typename stencil_type::coord_array_t;

        // get coordinate space for left reconstruction stencil
        static DEV index_space_t<Dims>
        left_space(const uarray<Dims>& base_coord, std::uint64_t direction)
        {
            auto pattern = stencil_type::left_pattern(direction);
            return pattern_to_space(pattern, base_coord);
        }

        // get coordinate space for right reconstruction stencil
        static DEV index_space_t<Dims>
        right_space(const uarray<Dims>& base_coord, std::uint64_t direction)
        {
            auto pattern = stencil_type::right_pattern(direction);
            return pattern_to_space(pattern, base_coord);
        }

        // get both left and right stencil spaces
        static DEV std::pair<index_space_t<Dims>, index_space_t<Dims>>
        neighbor_spaces(const uarray<Dims>& base_coord, std::uint64_t direction)
        {
            return {
              left_space(base_coord, direction),
              right_space(base_coord, direction)
            };
        }

        // get the coordinates directly as a vector
        static DEV vector_t<uarray<Dims>, stencil_type::size> left_coordinates(
            const uarray<Dims>& base_coord,
            std::uint64_t direction
        )
        {
            auto pattern = stencil_type::left_pattern(direction);
            return pattern_to_coordinates(pattern, base_coord);
        }

        static DEV vector_t<uarray<Dims>, stencil_type::size> right_coordinates(
            const uarray<Dims>& base_coord,
            std::uint64_t direction
        )
        {
            auto pattern = stencil_type::right_pattern(direction);
            return pattern_to_coordinates(pattern, base_coord);
        }

      private:
        // convert pattern offsets to actual coordinates
        static DEV vector_t<uarray<Dims>, stencil_type::size>
        pattern_to_coordinates(
            const coord_array_t& pattern,
            const uarray<Dims>& base
        )
        {
            vector_t<uarray<Dims>, stencil_type::size> coords;
            for (std::uint64_t i = 0; i < stencil_type::size; ++i) {
                coords[i] = base;
                for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                    coords[i][dim] += pattern[i][dim];
                }
            }
            return coords;
        }

        // convert pattern to bounding coordinate space
        static DEV index_space_t<Dims>
        pattern_to_space(const coord_array_t& pattern, const uarray<Dims>& base)
        {
            auto coords = pattern_to_coordinates(pattern, base);

            uarray<Dims> min_coord = coords[0];
            uarray<Dims> max_coord = coords[0];

            for (std::uint64_t i = 1; i < stencil_type::size; ++i) {
                for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                    min_coord[dim] = std::min(min_coord[dim], coords[i][dim]);
                    max_coord[dim] = std::max(max_coord[dim], coords[i][dim]);
                }
            }

            // coordinate space is [start, end) so add 1 to max
            for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                max_coord[dim] += 1;
            }

            return index_space_t<Dims>{min_coord, max_coord};
        }
    };

    // convenience factory functions
    namespace stencils {
        template DEV<std::uint64_t Dims, Reconstruction Rec> auto
        make_stencil_space(
            const uarray<Dims>& base_coord,
            std::uint64_t direction
        )
        {
            return stencil_space_t<Dims, Rec>::neighbor_spaces(
                base_coord,
                direction
            );
        }

        template <std::uint64_t Dims, Reconstruction Rec>
        DEV auto
        left_neighbors(const uarray<Dims>& base_coord, std::uint64_t direction)
        {
            return stencil_space_t<Dims, Rec>::left_coordinates(
                base_coord,
                direction
            );
        }

        template <std::uint64_t Dims, Reconstruction Rec>
        DEV auto
        right_neighbors(const uarray<Dims>& base_coord, std::uint64_t direction)
        {
            return stencil_space_t<Dims, Rec>::right_coordinates(
                base_coord,
                direction
            );
        }

        template <
            is_hydro_primitive_c prim_t,
            VectorLike Vector,
            std::uint64_t Dims = prim_t::dimensions>
        DEV auto gather_stencil(
            const field_t<prim_t, Dims>& prims,
            const Vector& pattern
        )
        {
            vector_t<prim_t, Vector::dimensions> gathered;
            for (std::uint64_t ii = 0; ii < Vector::dimensions; ++ii) {
                auto coord   = pattern[ii];
                gathered[ii] = prims[coord];
            }
            return gathered;
        }

        template <Reconstruction Rec>
        DEV auto primitive_recon(const auto& prims, real plm_theta)
        {
            if constexpr (Rec == Reconstruction::PCM) {
                return prims[0];   // PCM just returns the first element
            }
            else if constexpr (Rec == Reconstruction::PLM) {
                // PLM: average of left and right
                return helpers::plm_gradient(
                    prims[0],
                    prims[1],
                    prims[2],
                    plm_theta
                );
            }
            else {
                static_assert(
                    false,
                    "Unsupported reconstruction method for primitive recon"
                );
            }
        }

        template <
            Reconstruction Rec,
            is_hydro_primitive_c prim_t,
            std::uint64_t Dims = prim_t::dimensions>
        DEV std::tuple<prim_t, prim_t> reconstruct(
            const field_t<prim_t, Dims>& prims,
            const auto& face_coord,
            std::uint64_t dir,
            real plm_theta
        )
        {
            auto left_coords   = left_neighbors<Dims, Rec>(face_coord, dir);
            auto right_coords  = right_neighbors<Dims, Rec>(face_coord, dir);
            const auto pl      = gather_stencil(prims, left_coords);
            const auto pr      = gather_stencil(prims, right_coords);
            const auto plrecon = primitive_recon<Rec>(pl, plm_theta);
            const auto prrecon = primitive_recon<Rec>(pr, plm_theta);
            // get the left and right primitives adjecent to the interface
            // coordinate
            // const auto plc = prims[left_cell];
            // const auto prc = prims[right_cell];

            return {plrecon, prrecon};
        }
    }   // namespace stencils
}   // namespace simbi::base

#endif
