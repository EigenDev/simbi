/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            cell.hpp
 *  * @brief           a cell class with face, normal, and centroid information
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

#ifndef CELL_HPP
#define CELL_HPP

#include "config.hpp"
#include "core/types/utility/enums.hpp"
#include "geometry_manager.hpp"     // for GeometryManager
#include "geometry_traits.hpp"      // for GeomtryTraits
#include "grid_manager.hpp"         // for GridManager
#include "util/tools/helpers.hpp"   // for my_max

namespace simbi {
    constexpr real
    get_centroid(const auto xl, const auto xr, const Cellspacing& spacing)
    {
        if (spacing == Cellspacing::LOG) {
            return std::sqrt(xl * xr);
        }
        return 0.5 * (xl + xr);
    }
    // Cell
    template <size_type Dims>
    class Cell
    {
      private:
        struct FacePair {
            real normal;
            real area;
        };
        const GeometryManager& geo_info_;
        const GridManager& grid_info_;
        array_t<FacePair, Dims * 2> faces_;
        array_t<real, Dims> widths_;
        spatial_vector_t<real, Dims> centroid_;
        real dV_;
        static constexpr real POLAR_TOL = 1.e-10;

      public:
        DUAL Cell(
            const GeometryManager& geo_info,
            const GridManager& grid_info,
            const luint ii,
            const luint jj,
            const luint kk
        )
            : geo_info_(geo_info), grid_info_(grid_info)
        {
            calculate_metrics(ii, jj, kk);
        }

        template <Geometry G>
        DUAL void calculate_geometry_metrics_impl()
        {
            const auto traits = GeometryTraits<G, Dims>{};
            traits.calculate_widths(widths_, *this);
            traits.calculate_areas(faces_, *this);
            traits.calculate_volume(*this);
        }

        DUAL void calculate_geometry_metrics()
        {
            switch (geo_info_.geometry()) {
                case Geometry::SPHERICAL:
                    calculate_geometry_metrics_impl<Geometry::SPHERICAL>();
                    break;
                case Geometry::CARTESIAN:
                    calculate_geometry_metrics_impl<Geometry::CARTESIAN>();
                    break;
                default:
                    calculate_geometry_metrics_impl<Geometry::CYLINDRICAL>();
                    break;
            }
        }

        // Compile-time dispatch for geometry
        DUAL auto geometrical_sources(const auto& prims, const real gamma) const
        {
            using primitive_t = std::remove_cvref_t<decltype(prims)>;
            using conserved_t = primitive_t::counterpart_t;

            auto generic = conserved_t{};
            switch (geo_info_.geometry()) {
                case Geometry::SPHERICAL:
                    GeometryTraits<Geometry::SPHERICAL, Dims>::
                        calculate_geometrical_source_terms(
                            prims,
                            *this,
                            generic,
                            gamma
                        );
                    break;
                case Geometry::CARTESIAN:
                    GeometryTraits<Geometry::CARTESIAN, Dims>::
                        calculate_geometrical_source_terms(
                            prims,
                            *this,
                            generic,
                            gamma
                        );
                    break;
                default:
                    GeometryTraits<Geometry::CYLINDRICAL, Dims>::
                        calculate_geometrical_source_terms(
                            prims,
                            *this,
                            generic,
                            gamma
                        );
                    break;
            }
            return generic;
        }

        DUAL void calculate_centroid()
        {

            centroid_[0] = get_centroid(
                faces_[0].normal,
                faces_[1].normal,
                geo_info_.spacing_type(0)
            );
            if constexpr (Dims > 1) {
                centroid_[1] = get_centroid(
                    faces_[2].normal,
                    faces_[3].normal,
                    geo_info_.spacing_type(1)
                );
            }
            if constexpr (Dims > 2) {
                centroid_[2] = get_centroid(
                    faces_[4].normal,
                    faces_[5].normal,
                    geo_info_.spacing_type(2)
                );
            }
        }

        DUAL void
        calculate_normals(const luint ii, const luint jj, const luint kk)
        {
            faces_[0].normal = get_face<GridDirection::X1>(ii, jj, kk, 0);
            faces_[1].normal = get_face<GridDirection::X1>(ii, jj, kk, 1);
            if constexpr (Dims > 1) {
                faces_[2].normal = get_face<GridDirection::X2>(ii, jj, kk, 0);
                faces_[3].normal = get_face<GridDirection::X2>(ii, jj, kk, 1);
                if constexpr (Dims > 2) {
                    faces_[4].normal =
                        get_face<GridDirection::X3>(ii, jj, kk, 0);
                    faces_[5].normal =
                        get_face<GridDirection::X3>(ii, jj, kk, 1);
                }
            }
        }

        DUAL auto compute_distance(const auto& pos) const
        {
            switch (geo_info_.geometry()) {
                case Geometry::SPHERICAL:
                    return GeometryTraits<Geometry::SPHERICAL, Dims>::
                        calculate_distance(pos, *this);
                case Geometry::CARTESIAN:
                    return GeometryTraits<Geometry::CARTESIAN, Dims>::
                        calculate_distance(pos, *this);
                default:
                    return GeometryTraits<Geometry::CYLINDRICAL, Dims>::
                        calculate_distance(pos, *this);
            }
        }

        DUAL auto compute_distance_vector(const auto& pos) const
        {
            switch (geo_info_.geometry()) {
                case Geometry::SPHERICAL:
                    return GeometryTraits<Geometry::SPHERICAL, Dims>::
                        calculate_distance_vector(pos, *this);
                case Geometry::CARTESIAN:
                    return GeometryTraits<Geometry::CARTESIAN, Dims>::
                        calculate_distance_vector(pos, *this);
                default:
                    return GeometryTraits<Geometry::CYLINDRICAL, Dims>::
                        calculate_distance_vector(pos, *this);
            }
        }
        DUAL auto get_face_index(const spatial_vector_t<real, Dims>& nhat) const
        {
            // determine the face index based on the normal vector
            // This is used to determine the direction of the
            // surface normal vector. If the normal vector points
            // in the x1 direction, but in the negative direction,
            // then the face index will be 0. Same logic for the x2
            // and x3 directions.
            if (nhat[0] < 0) {
                return 0;
            }
            else if (nhat[0] > 0) {
                return 1;
            }
            else if (Dims > 1 && nhat[1] < 0) {
                return 2;
            }
            else if (Dims > 1 && nhat[1] > 0) {
                return 3;
            }
            else if (Dims > 2 && nhat[2] < 0) {
                return 4;
            }
            else if (Dims > 2 && nhat[2] > 0) {
                return 5;
            }
            return 0;
        }

        DUAL constexpr auto normal_vec(const size_type side) const
        {
            if (side < 2 * Dims) {
                const size_type dir = side / 2;
                const auto nhat     = unit_vectors::get<Dims>(dir + 1);
                // if side is even, then the normal vector
                // points in the negative direction
                return nhat;
            }
            return unit_vector_t<Dims>{};
        }

        DUAL auto area_normal(const spatial_vector_t<real, Dims>& nhat) const
        {
            // deterine the direction of the area normal
            // vector based on the normal vector. This is
            // used to determine the direction of the
            // surface normal vector. If the normal vector
            // points in the x1 direction, but in the negative
            // direction, then the area normal vector will
            // be the cell area to the left of the cell face.
            // Same logic for the x2 and x3 directions.
            const size_type face_idx = get_face_index(nhat);
            return nhat * faces_[face_idx].area;
        }

        // Unified accessor methods
        DUAL real area(const size_type norm) const
        {
            if (norm < 2 * Dims) {
                return faces_[norm].area;
            }
            return 0.0;
        }

        DUAL real dx(const size_type norm) const
        {
            if (norm < Dims) {
                return widths_[norm];
            }
            return 0.0;
        }

        DUAL real normal(Side s) const
        {
            if (static_cast<size_type>(s) < 2 * Dims) {
                return faces_[static_cast<size_type>(s)].normal;
            }
            return 0.0;
        }

        DUAL real normal(const size_type norm) const
        {
            if (norm < 2 * Dims) {
                return faces_[norm].normal;
            }
            return 0.0;
        }

        DUAL real velocity(Side s) const
        {
            if (static_cast<int>(s) <= 2 * Dims) {
                const real x = faces_[static_cast<int>(s)].normal;
                return geo_info_.homologous() ? x * geo_info_.expansion_term()
                                              : 1 * geo_info_.expansion_term();
            }
            return 0.0;
        }

        DUAL real velocity(const size_type norm) const
        {
            if (norm < 2 * Dims) {
                const real x = faces_[norm].normal;
                return geo_info_.homologous() ? x * geo_info_.expansion_term()
                                              : 1 * geo_info_.expansion_term();
            }
            return 0.0;
        }

        template <int dir>
        DUAL real get_face_linear(const luint idx, bool is_left) const
        {
            // Convert dir (1,2,3) to array index (0,1,2)
            const int axis     = dir - 1;
            const real min_val = geo_info_.min_bound(axis);
            const real dx      = (geo_info_.max_bound(axis) - min_val) /
                            grid_info_.active_gridsize(axis);

            if (is_left) {
                return min_val + idx * dx;
            }
            else {
                return min_val + (idx + 1) * dx;
            }
        }

        // Helper for log spacing
        template <int dir>
        DUAL real get_face_log(const luint idx, bool is_left) const
        {
            const int axis     = dir - 1;
            const real min_val = geo_info_.min_bound(axis);
            const real max_val = geo_info_.max_bound(axis);
            const real dlogx   = (std::log10(max_val / min_val)) /
                               grid_info_.active_gridsize(axis);

            if (is_left) {
                // Left face of cell idx
                return min_val * std::pow(10.0, idx * dlogx);
            }
            else {
                // Right face of cell idx
                return min_val * std::pow(10.0, (idx + 1) * dlogx);
            }
        }

        template <GridDirection Dir>
        DUAL real get_face(
            const luint ii,
            const luint jj,
            const luint kk,
            const int side
        ) const
        {
            constexpr int dir = Dir == GridDirection::X1   ? 1
                                : Dir == GridDirection::X2 ? 2
                                                           : 3;
            if constexpr (dir == 1) {
                if (geo_info_.spacing_type(0) == Cellspacing::LINEAR) {
                    return get_face_linear<dir>(ii, side == 0);
                }
                return get_face_log<dir>(ii, side == 0);
            }
            else if constexpr (dir == 2) {
                if (geo_info_.spacing_type(1) == Cellspacing::LINEAR) {
                    return get_face_linear<dir>(jj, side == 0);
                }
                return get_face_log<dir>(jj, side == 0);
            }
            else {
                if (geo_info_.spacing_type(2) == Cellspacing::LINEAR) {
                    return get_face_linear<dir>(kk, side == 0);
                }
                return get_face_log<dir>(kk, side == 0);
            }
        }

        DUAL void
        calculate_metrics(const luint ii, const luint jj, const luint kk)
        {
            calculate_normals(ii, jj, kk);
            calculate_centroid();
            calculate_geometry_metrics();
        }

        DUAL void set_volume(const real vol) { dV_ = vol; }

        // accessors
        DUAL real x1L() const { return faces_[0].normal; }

        DUAL real x1R() const { return faces_[1].normal; }

        DUAL real x2L() const
        {
            if constexpr (Dims > 1) {
                return faces_[2].normal;
            }
            return 0.0;
        }

        DUAL real x2R() const
        {
            if constexpr (Dims > 1) {
                return faces_[3].normal;
            }
            return 0.0;
        }

        DUAL real x3L() const
        {
            if constexpr (Dims > 2) {
                return faces_[4].normal;
            }
            return 0.0;
        }

        DUAL real x3R() const
        {
            if constexpr (Dims > 2) {
                return faces_[5].normal;
            }
            return 0.0;
        }

        // area accessors
        DUAL real a1L() const { return faces_[0].area; }

        DUAL real a1R() const { return faces_[1].area; }

        DUAL real a2L() const
        {
            if constexpr (Dims > 1) {
                return faces_[2].area;
            }
            return 0.0;
        }

        DUAL real a2R() const
        {
            if constexpr (Dims > 1) {
                return faces_[3].area;
            }
            return 0.0;
        }

        DUAL real a3L() const
        {
            if constexpr (Dims > 2) {
                return faces_[4].area;
            }
            return 0.0;
        }

        DUAL real a3R() const
        {
            if constexpr (Dims > 2) {
                return faces_[5].area;
            }
            return 0.0;
        }

        DUAL bool at_pole() const { return at_north_pole() || at_south_pole(); }

        DUAL bool at_north_pole() const
        {
            return std::abs(std::sin(x2L())) < POLAR_TOL;
        }

        DUAL bool at_south_pole() const
        {
            return std::abs(std::sin(x2R())) < POLAR_TOL;
        }

        DUAL auto max_cell_width() const
        {
            if constexpr (Dims == 1) {
                return widths_[0];
            }
            else if constexpr (Dims == 2) {
                return my_max<real>(widths_[0], widths_[1]);
            }
            else if constexpr (Dims == 3) {
                return my_max3<real>(widths_[0], widths_[1], widths_[2]);
            }
        }

        DUAL auto min_cell_width() const
        {
            if constexpr (Dims == 1) {
                return widths_[0];
            }
            else if constexpr (Dims == 2) {
                return my_min<real>(widths_[0], widths_[1]);
            }
            else if constexpr (Dims == 3) {
                return my_min3<real>(widths_[0], widths_[1], widths_[2]);
            }
        }

        // accessors
        DUAL constexpr auto geometry() const { return geo_info_.geometry(); }
        DUAL constexpr auto centroid() const { return centroid_; }
        DUAL constexpr auto cartesian_centroid() const
        {
            switch (geo_info_.geometry()) {
                case Geometry::SPHERICAL:
                    return vecops::spherical_to_cartesian(centroid_);
                case Geometry::CYLINDRICAL:
                    return vecops::cylindrical_to_cartesian(centroid_);
                default: return centroid_;
            }
        }
        DUAL constexpr auto volume() const { return dV_; }
        DUAL constexpr auto width(const size_type ii) const
        {
            if (ii >= Dims) {
                return 1.0;
            }
            return widths_[ii];
        }
        DUAL constexpr auto centroid_coordinate(const size_type ii) const
        {
            if (ii >= Dims) {
                return 0.0;
            }
            return centroid_[ii];
        }

        DUAL constexpr auto inverse_volume() const { return 1.0 / dV_; }

        DUAL spatial_vector_t<real, Dims> calculate_ghost_position(
            size_type boundary_dim,
            bool is_lower,
            size_type ghost_layer
        ) const
        {
            auto ghost_pos = this->centroid();

            // get the face position of the interior cell
            real face_pos = is_lower ? this->normal(Side(2 * boundary_dim + 0))
                                     : this->normal(Side(2 * boundary_dim + 1));

            // calc position based on spacing
            if (geo_info_.spacing_type(boundary_dim) == Cellspacing::LINEAR) {
                real dx = this->dx(boundary_dim);
                // for multiple ghost layers:
                // ghost_layer=1: one cell away from boundary
                // ghost_layer=2: two cells away from boundary, etc.
                ghost_pos[boundary_dim] =
                    is_lower ? face_pos - (ghost_layer - 0.5) * dx
                             : face_pos + (ghost_layer - 0.5) * dx;
            }
            else {
                // log spacing
                real log_ratio = this->normal(Side(2 * boundary_dim + 1)) /
                                 this->normal(Side(2 * boundary_dim + 0));

                // for multiple ghost layers with log spacing, apply ratio
                // multiple times
                if (is_lower) {
                    // for lower boundary, divide by the ratio ghost_layer
                    // times
                    for (size_type ii = 0; ii < ghost_layer; ii++) {
                        face_pos /= log_ratio;
                    }
                    // pos is at the center of the ghost cell
                    ghost_pos[boundary_dim] = face_pos * std::sqrt(log_ratio);
                }
                else {
                    // for upper boundary, multiply by the ratio ghost_layer
                    // times
                    for (size_type ii = 0; ii < ghost_layer; ii++) {
                        face_pos *= log_ratio;
                    }
                    // pos is at the center of the ghost cell
                    ghost_pos[boundary_dim] = face_pos / std::sqrt(log_ratio);
                }
            }

            // proper handling for different geometries
            if (geo_info_.geometry() != Geometry::CARTESIAN) {
                if (boundary_dim == 1) {   // theta (phi) coordinate in
                                           // spherical (cylindrical)
                    if (geo_info_.geometry() == Geometry::SPHERICAL) {
                        // handle spherical coordinates
                        if (is_lower && ghost_pos[boundary_dim] < 0) {
                            // Reflection across θ=0 pole
                            ghost_pos[boundary_dim] = -ghost_pos[boundary_dim];

                            // In spherical coordinates, crossing the pole
                            // also means phi changes by phi (if we have 3D)
                            if constexpr (Dims > 2) {
                                ghost_pos[2] += M_PI;
                                if (ghost_pos[2] > 2 * M_PI) {
                                    ghost_pos[2] -= 2 * M_PI;
                                }
                            }
                        }
                        else if (!is_lower && ghost_pos[boundary_dim] > M_PI) {
                            // reflection across \theta=\pi pole
                            ghost_pos[boundary_dim] =
                                2 * M_PI - ghost_pos[boundary_dim];

                            // Also adjust phi by \pi
                            if constexpr (Dims > 2) {
                                ghost_pos[2] += M_PI;
                                if (ghost_pos[2] > 2 * M_PI) {
                                    ghost_pos[2] -= 2 * M_PI;
                                }
                            }
                        }
                    }
                    else if (geo_info_.geometry() == Geometry::CYLINDRICAL) {
                        // For cylindrical, phi is periodic
                        if (ghost_pos[boundary_dim] < 0) {
                            ghost_pos[boundary_dim] += 2 * M_PI;
                        }
                        else if (ghost_pos[boundary_dim] > 2 * M_PI) {
                            ghost_pos[boundary_dim] -= 2 * M_PI;
                        }
                    }
                }
                else if (Dims > 2 &&
                         boundary_dim == 2) {   // phi coordinate in 3D
                    // Enforce periodicity for φ
                    while (ghost_pos[boundary_dim] < 0) {
                        ghost_pos[boundary_dim] += 2 * M_PI;
                    }
                    while (ghost_pos[boundary_dim] >= 2 * M_PI) {
                        ghost_pos[boundary_dim] -= 2 * M_PI;
                    }
                }
            }

            return ghost_pos;
        }
    };
}   // namespace simbi

#endif
