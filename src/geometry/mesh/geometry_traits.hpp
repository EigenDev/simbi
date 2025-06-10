/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            geometry_traits.hpp
 *  * @brief           provides geometry info (volume, area, etc) for mesh
 * geometry
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-18
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-18      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef GEOMETRY_TRAITS_HPP
#define GEOMETRY_TRAITS_HPP

#include "config.hpp"
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include "core/types/utility/enums.hpp"       // for Geometry

namespace simbi {
    enum class GridDirection {
        X1,
        X2,
        X3
    };

    enum class Side {
        X1L,
        X1R,
        X2L,
        X2R,
        X3L,
        X3R
    };

    // geometry traits
    template <Geometry G, int Dims>
    struct GeometryTraits {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell);
        static constexpr void calculate_areas(auto& faces, const auto& cell);
        static constexpr auto calculate_geometrical_source_terms(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        );
        static constexpr auto calculate_volume(auto& cell);
        static constexpr void calculate_widths(auto& widths, const auto& cell);
        static constexpr auto
        calculate_distance(const auto& pos, const auto& cell);
        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell);
    };

    // Spherical coordinate geometry specialization
    template <int Dims>
    struct GeometryTraits<Geometry::SPHERICAL, Dims> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            if constexpr (Dir == GridDirection::X1) {
                return cell.centroid_coordinate(0) *
                       cell.centroid_coordinate(0) *
                       (cell.normal(1) - cell.normal(0));
            }
            else if constexpr (Dir == GridDirection::X2) {
                if constexpr (Dims == 1) {
                    return 2.0;
                }
                else {
                    return std::cos(cell.normal(2)) - std::cos(cell.normal(3));
                }
            }
            else {
                if constexpr (Dims < 3) {
                    return 2.0 * M_PI;
                }
                else {
                    return cell.normal(5) - cell.normal(4);
                }
            }
        }

        static constexpr void calculate_areas(auto& faces, const auto& cell)
        {
            const auto rr   = cell.normal(Side::X1R);
            const auto rl   = cell.normal(Side::X1L);
            const auto tr   = cell.normal(Side::X2R);
            const auto tl   = cell.normal(Side::X2L);
            const auto dcos = get_differential<GridDirection::X2>(cell);
            const auto dphi = get_differential<GridDirection::X3>(cell);
            faces[0].area   = rl * rl * dcos * dphi;
            faces[1].area   = rr * rr * dcos * dphi;
            if constexpr (Dims > 1) {
                faces[2].area = 0.5 * (rr * rr - rl * rl) * std::sin(tl) * dphi;
                faces[3].area = 0.5 * (rr * rr - rl * rl) * std::sin(tr) * dphi;
                if constexpr (Dims > 2) {
                    faces[4].area = 0.5 * (rr * rr - rl * rl) * dcos;
                    faces[5].area = 0.5 * (rr * rr - rl * rl) * dcos;
                }
            }
        }

        static constexpr auto calculate_geometrical_source_terms(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        )
        {
            const auto& theta = cell.centroid_coordinate(1);
            const real sint   = std::sin(theta);
            const real cot    = std::cos(theta) / sint;

            // Grab central primitives
            const real v1    = prims.proper_velocity(1);
            const real v2    = prims.proper_velocity(2);
            const real v3    = prims.proper_velocity(3);
            const real pt    = prims.total_pressure();
            const auto bmu   = prims.calc_magnetic_four_vector();
            const real wt    = prims.enthalpy_density(gamma);
            const real gam2  = prims.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            // geometric source terms in momentum
            for (int qq = 0; qq < Dims; qq++) {
                const auto r = cell.centroid_coordinate(0);
                if (qq == 0) {
                    cons[qq + 1] =
                        pt * (cell.a1R() - cell.a1L()) / cell.volume() +
                        wgam2 * (v2 * v2 + v3 * v3) / r -
                        (bmu[1] * bmu[1] + bmu[3] * bmu[3]) / r;
                }
                else if (qq == 1) {
                    cons[qq + 1] =
                        pt * (cell.a2R() - cell.a2L()) / cell.volume() -
                        wgam2 * (v2 * v1 - v3 * v3 * cot) / r +
                        (bmu[1] * bmu[0] - bmu[3] * bmu[3] * cot) / r;
                }
                else {
                    cons[qq + 1] = -wgam2 * v3 * (v1 + cot * v2) / r +
                                   bmu[3] * (bmu[0] + cot * bmu[1]) / r;
                }
            }
        }

        static constexpr auto calculate_volume(auto& cell)
        {
            cell.set_volume(
                get_differential<GridDirection::X1>(cell) *
                get_differential<GridDirection::X2>(cell) *
                get_differential<GridDirection::X3>(cell)
            );
        }

        static constexpr void calculate_widths(auto& widths, const auto& cell)
        {
            widths[0] = cell.normal(1) - cell.normal(0);
            if constexpr (Dims > 1) {
                widths[1] = cell.centroid_coordinate(0) *
                            (cell.normal(3) - cell.normal(2));
            }
            if constexpr (Dims > 2) {
                widths[2] = cell.centroid_coordinate(0) *
                            std::sin(cell.centroid_coordinate(1)) *
                            (cell.normal(5) - cell.normal(4));
            }
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 1)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            return spatial_vector_t<real, 1>{x1 - x1prime};
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 2)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            const auto x2      = cell.centroid_coordinate(1);
            const auto x2prime = pos[1];
            const auto x       = x1 * std::sin(x2);
            const auto y       = x1 * std::cos(x2);
            const auto xp      = x1prime * std::sin(x2prime);
            const auto yp      = x1prime * std::cos(x2prime);
            return spatial_vector_t<real, 2>{x - xp, y - yp};
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 3)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            const auto x2      = cell.centroid_coordinate(1);
            const auto x2prime = pos[1];
            const auto x3      = cell.centroid_coordinate(2);
            const auto x3prime = pos[2];
            const auto x       = x1 * std::sin(x2) * std::cos(x3);
            const auto y       = x1 * std::sin(x2) * std::sin(x3);
            const auto z       = x1 * std::cos(x2);
            const auto xp = x1prime * std::sin(x2prime) * std::cos(x3prime);
            const auto yp = x1prime * std::sin(x2prime) * std::sin(x3prime);
            const auto zp = x1prime * std::cos(x2prime);
            return spatial_vector_t<real, 3>{x - xp, y - yp, z - zp};
        }

        static constexpr auto
        calculate_distance(const auto& pos, const auto& cell)
        {
            return calculate_distance_vector(pos, cell).norm();
        }
    };

    // Cylindrical coordinate geometry specialization
    template <int Dims>
    struct GeometryTraits<Geometry::CYLINDRICAL, Dims> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            if constexpr (Dir == GridDirection::X1) {
                return cell.centroid_coordinate(0) *
                       (cell.normal(1) - cell.normal(0));
            }
            else if constexpr (Dir == GridDirection::X2) {
                if (cell.geometry() == Geometry::AXIS_CYLINDRICAL) {
                    // this is dphi
                    return 2.0 * M_PI;
                }
                if constexpr (Dims > 1) {
                    return cell.normal(3) - cell.normal(2);
                }
                else {
                    return 2.0 * M_PI;
                }
            }
            else {
                if (cell.geometry() == Geometry::AXIS_CYLINDRICAL) {
                    // this is dz
                    return cell.normal(3) - cell.normal(2);
                }
                if constexpr (Dims < 3) {
                    return 1.0;
                }
                else {
                    return cell.normal(5) - cell.normal(4);
                }
            }
        }

        static constexpr void calculate_areas(auto& faces, const auto& cell)
        {
            const auto rr   = cell.normal(Side::X1R);
            const auto rl   = cell.normal(Side::X1L);
            const auto dphi = get_differential<GridDirection::X2>(cell);
            const auto dz   = get_differential<GridDirection::X3>(cell);
            faces[0].area   = rl * dz * dphi;
            faces[1].area   = rr * dz * dphi;
            if constexpr (Dims > 1) {
                if (cell.geometry() == Geometry::AXIS_CYLINDRICAL) {
                    faces[2].area = 0.5 * (rr * rr - rl * rl) * dphi;
                    faces[3].area = 0.5 * (rr * rr - rl * rl) * dphi;
                }
                else {
                    faces[2].area = (rr - rl) * dphi;
                    faces[3].area = (rr - rl) * dphi;
                }
                if constexpr (Dims > 2) {
                    const auto rmean = cell.centroid_coordinate(0);
                    faces[4].area    = rmean * (rr - rl) * dphi;
                    faces[5].area    = rmean * (rr - rl) * dphi;
                }
            }
        }

        static constexpr auto calculate_geometrical_source_terms(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        )
        {
            // special care must be taken for axisymmetry
            // or cylindrical-polar coordinates. In axisymmetry,
            // the phi velocity is zero, but out of convenience, we
            // store the z component in the second component of the
            // velocity vector. This is done to avoid having to
            // rearrange the velocity vector in the axisymmetric
            // case
            const bool axis  = cell.geometry() == Geometry::AXIS_CYLINDRICAL;
            const real v1    = prims.proper_velocity(1);
            const real v2    = axis ? 0.0 : prims.proper_velocity(2);
            const real pt    = prims.total_pressure();
            const auto bmu   = prims.calc_magnetic_four_vector();
            const real wt    = prims.enthalpy_density(gamma);
            const real gam2  = prims.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            for (int qq = 0; qq < Dims; qq++) {
                if (qq == 0) {
                    cons[qq + 1] = (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) /
                                   cell.centroid_coordinate(0);
                }
                else if (qq == 1) {
                    cons[qq + 1] = -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) /
                                   cell.centroid_coordinate(0);
                }
            }
        }

        static constexpr auto calculate_volume(auto& cell)
        {
            cell.set_volume(
                get_differential<GridDirection::X1>(cell) *
                get_differential<GridDirection::X2>(cell) *
                get_differential<GridDirection::X3>(cell)
            );
        }

        static constexpr void calculate_widths(auto& widths, const auto& cell)
        {
            widths[0] = cell.normal(1) - cell.normal(0);
            if constexpr (Dims > 1) {
                widths[1] = cell.centroid_coordinate(0) *
                            (cell.normal(3) - cell.normal(2));
            }
            if constexpr (Dims > 2) {
                widths[2] = cell.normal(5) - cell.normal(4);
            }
        };

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 1)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            return spatial_vector_t<real, 1>{x1 - x1prime};
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 2)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            const auto x2      = cell.centroid_coordinate(1);
            const auto x2prime = pos[1];
            const auto x       = x1 * std::cos(x2);
            const auto y       = x1 * std::sin(x2);
            const auto xp      = x1prime * std::cos(x2prime);
            const auto yp      = x1prime * std::sin(x2prime);
            return spatial_vector_t<real, 2>{x - xp, y - yp};
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 3)
        {
            const auto x1      = cell.centroid_coordinate(0);
            const auto x1prime = pos[0];
            const auto x2      = cell.centroid_coordinate(1);
            const auto x2prime = pos[1];
            const auto x3      = cell.centroid_coordinate(2);
            const auto x3prime = pos[2];
            const auto x       = x1 * std::cos(x2);
            const auto y       = x1 * std::sin(x2);
            const auto z       = x3;
            const auto xp      = x1prime * std::cos(x2prime);
            const auto yp      = x1prime * std::sin(x2prime);
            const auto zp      = x3prime;
            return spatial_vector_t<real, 3>{x - xp, y - yp, z - zp};
        }

        static constexpr auto
        calculate_distance(const auto& pos, const auto& cell)
        {
            return calculate_distance_vector(pos, cell).norm();
        }
    };

    // Cartesian coordinate geometry specialization
    template <int Dims>
    struct GeometryTraits<Geometry::CARTESIAN, Dims> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            if constexpr (Dir == GridDirection::X1) {
                return cell.width(0);
            }
            else if constexpr (Dir == GridDirection::X2) {
                return cell.width(1);
            }
            else {
                return cell.width(2);
            }
        }

        static constexpr void calculate_areas(auto& faces, const auto& cell)
        {
            // cartesian areas at x faces are y * z
            faces[0].area = cell.width(1) * cell.width(2);
            faces[1].area = cell.width(1) * cell.width(2);
            if constexpr (Dims > 1) {
                // cartesian areas at y faces are x * z
                faces[2].area = cell.width(0) * cell.width(2);
                faces[3].area = cell.width(0) * cell.width(2);
                if constexpr (Dims > 2) {
                    // cartesian areas at z faces are x * y
                    faces[4].area = cell.width(0) * cell.width(1);
                    faces[5].area = cell.width(0) * cell.width(1);
                }
            }
        }

        static constexpr auto calculate_geometrical_source_terms(
            const auto&,
            const auto&,
            auto& cons,
            auto
        )
        {
            // Do nothing
            for (size_type qq = 0; qq < Dims; qq++) {
                cons[qq + 1] = 0.0;
            }
        }

        static constexpr auto calculate_volume(auto& cell)
        {
            cell.set_volume(
                get_differential<GridDirection::X1>(cell) *
                get_differential<GridDirection::X2>(cell) *
                get_differential<GridDirection::X3>(cell)
            );
        }

        static constexpr void calculate_widths(auto& widths, const auto& cell)
        {
            widths[0] = cell.normal(1) - cell.normal(0);
            if constexpr (Dims > 1) {
                widths[1] = cell.normal(3) - cell.normal(2);
            }
            if constexpr (Dims > 2) {
                widths[2] = cell.normal(5) - cell.normal(4);
            }
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 1)
        {
            return spatial_vector_t<real, 1>{
              cell.centroid_coordinate(0) - pos[0]
            };
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 2)
        {
            return spatial_vector_t<real, 2>{
              cell.centroid_coordinate(0) - pos[0],
              cell.centroid_coordinate(1) - pos[1]
            };
        }

        static constexpr auto
        calculate_distance_vector(const auto& pos, const auto& cell)
            requires(Dims == 3)
        {
            return spatial_vector_t<real, 3>{
              cell.centroid_coordinate(0) - pos[0],
              cell.centroid_coordinate(1) - pos[1],
              cell.centroid_coordinate(2) - pos[2]
            };
        }

        static constexpr auto
        calculate_distance(const auto& pos, const auto& cell)
        {
            return calculate_distance_vector(pos, cell).norm();
        }
    };

}   // namespace simbi

#endif
