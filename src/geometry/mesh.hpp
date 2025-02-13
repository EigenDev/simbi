#ifndef MESH_HPP
#define MESH_HPP
#include "build_options.hpp"
#include "util/tools/helpers.hpp"
#include <type_traits>

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

    constexpr real
    get_centroid(const auto xl, const auto xr, const Cellspacing& spacing)
    {
        if (spacing == Cellspacing::LOGSPACE) {
            return std::sqrt(xl * xr);
        }
        return 0.5 * (xl + xr);
    }

    // geometry traits
    template <Geometry G, typename Derived, int dim>
    struct GeometryTraits {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell);
        static constexpr void calculate_areas(auto* areas, const auto& cell);
        static constexpr auto calculate_sources(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        );
        static constexpr auto calculate_volume(const auto& cell);
        static constexpr void calculate_widths(auto* widths, const auto& cell);
    };

    // Spherical coordinate geometry specialization
    template <typename Derived, int dim>
    struct GeometryTraits<Geometry::SPHERICAL, Derived, dim> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            if constexpr (Dir == GridDirection::X1) {
                return cell.x1mean * cell.x1mean *
                       (cell.normals[1] - cell.normals[0]);
            }
            else if constexpr (Dir == GridDirection::X2) {
                if constexpr (dim == 1) {
                    return 2.0;
                }
                else {
                    return std::cos(cell.normals[2]) -
                           std::cos(cell.normals[3]);
                }
            }
            else {
                if constexpr (dim < 3) {
                    return 2.0 * M_PI;
                }
                else {
                    return cell.normals[5] - cell.normals[4];
                }
            }
        }

        static constexpr void calculate_areas(auto* areas, const auto& cell)
        {
            const auto rr   = cell.normal(Side::X1R);
            const auto rl   = cell.normal(Side::X1L);
            const auto tr   = cell.normal(Side::X2R);
            const auto tl   = cell.normal(Side::X2L);
            const auto dcos = get_differential<GridDirection::X2>(cell);
            const auto dphi = get_differential<GridDirection::X3>(cell);
            areas[0]        = rl * rl * dcos * dphi;
            areas[1]        = rr * rr * dcos * dphi;
            if constexpr (dim > 1) {
                areas[2] = 0.5 * (rr * rr - rl * rl) * std::sin(tl) * dphi;
                areas[3] = 0.5 * (rr * rr - rl * rl) * std::sin(tr) * dphi;
                if constexpr (dim > 2) {
                    areas[4] = 0.5 * (rr * rr - rl * rl) * dcos;
                    areas[5] = 0.5 * (rr * rr - rl * rl) * dcos;
                }
            }
        }

        static constexpr auto calculate_sources(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        )
        {
            const real sint = std::sin(cell.x2mean);
            const real cot  = std::cos(cell.x2mean) / sint;

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
            for (int qq = 0; qq < dim; qq++) {
                if (qq == 0) {
                    cons[qq + 1] =
                        pt * (cell.a1R() - cell.a1L()) / cell.dV +
                        wgam2 * (v2 * v2 + v3 * v3) / cell.x1mean -
                        (bmu[1] * bmu[1] + bmu[3] * bmu[3]) / cell.x1mean;
                }
                else if (qq == 1) {
                    cons[qq + 1] =
                        pt * (cell.a2R() - cell.a2L()) / cell.dV -
                        wgam2 * (v2 * v1 - v3 * v3 * cot) / cell.x1mean +
                        (bmu[1] * bmu[0] - bmu[3] * bmu[3] * cot) / cell.x1mean;
                }
                else {
                    cons[qq + 1] =
                        -wgam2 * v3 * (v1 + cot * v2) / cell.x1mean +
                        bmu[3] * (bmu[0] + cot * bmu[1]) / cell.x1mean;
                }
            }
        }

        static constexpr auto calculate_volume(const auto& cell)
        {
            return get_differential<GridDirection::X1>(cell) *
                   get_differential<GridDirection::X2>(cell) *
                   get_differential<GridDirection::X3>(cell);
        }

        static constexpr void calculate_widths(auto* widths, const auto& cell)
        {
            widths[0] = cell.normals[1] - cell.normals[0];
            if constexpr (dim > 1) {
                widths[1] = cell.x1mean * (cell.normals[3] - cell.normals[2]);
            }
            if constexpr (dim > 2) {
                widths[2] = cell.x1mean * std::sin(cell.x2mean) *
                            (cell.normals[5] - cell.normals[4]);
            }
        }
    };

    // Cylindrical coordinate geometry specialization
    template <typename Derived, int dim>
    struct GeometryTraits<Geometry::CYLINDRICAL, Derived, dim> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            if constexpr (Dir == GridDirection::X1) {
                return cell.x1mean * (cell.normals[1] - cell.normals[0]);
            }
            else if constexpr (Dir == GridDirection::X2) {
                if constexpr (dim > 1) {
                    return cell.normals[3] - cell.normals[2];
                }
                else {
                    return 2.0 * M_PI;
                }
            }
            else {
                if constexpr (dim < 3) {
                    return 1.0;
                }
                else {
                    return cell.normals[5] - cell.normals[4];
                }
            }
        }

        static constexpr void calculate_areas(auto* areas, const auto& cell)
        {
            const auto rr   = cell.normal(Side::X1R);
            const auto rl   = cell.normal(Side::X1L);
            const auto dphi = get_differential<GridDirection::X2>(cell);
            const auto dz   = get_differential<GridDirection::X3>(cell);
            areas[0]        = rl * dz * dphi;
            areas[1]        = rr * dz * dphi;
            if constexpr (dim > 1) {
                areas[2] = (rr - rl) * dz;
                areas[3] = (rr - rl) * dz;
                if constexpr (dim > 2) {
                    const auto rmean = cell.x1mean;
                    areas[4]         = rmean * (rr - rl) * dphi;
                    areas[5]         = rmean * (rr - rl) * dphi;
                }
            }
        }

        static constexpr auto calculate_sources(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        )
        {
            const real v1    = prims.proper_velocity(1);
            const real v2    = prims.proper_velocity(2);
            const real pt    = prims.total_pressure();
            const auto bmu   = prims.calc_magnetic_four_vector();
            const real wt    = prims.enthalpy_density(gamma);
            const real gam2  = prims.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            for (int qq = 0; qq < dim; qq++) {
                if (qq == 0) {
                    cons[qq + 1] =
                        (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / cell.x1mean;
                }
                else if (qq == 1) {
                    cons[qq + 1] =
                        -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / cell.x1mean;
                }
            }
        }

        static constexpr auto calculate_volume(const auto& cell)
        {
            return get_differential<GridDirection::X1>(cell) *
                   get_differential<GridDirection::X2>(cell) *
                   get_differential<GridDirection::X3>(cell);
        }

        static constexpr void calculate_widths(auto* widths, const auto& cell)
        {
            widths[0] = cell.normals[1] - cell.normals[0];
            if constexpr (dim > 1) {
                widths[1] = cell.x1mean * (cell.normals[3] - cell.normals[2]);
            }
            if constexpr (dim > 2) {
                widths[2] = cell.normals[5] - cell.normals[4];
            }
        };
    };

    // Cartesian coordinate geometry specialization
    template <typename Derived, int dim>
    struct GeometryTraits<Geometry::CARTESIAN, Derived, dim> {
        template <GridDirection Dir>
        static constexpr auto get_differential(const auto& cell)
        {
            return (cell.normals[1] - cell.normals[0]);
        }

        static constexpr void calculate_areas(auto* areas, const auto& cell)
        {
            areas[0] = 1.0;
            areas[1] = 1.0;
            if constexpr (dim > 1) {
                areas[2] = 1.0;
                areas[3] = 1.0;
                if constexpr (dim > 2) {
                    areas[4] = 1.0;
                    areas[5] = 1.0;
                }
            }
        }

        static constexpr auto calculate_sources(
            const auto& prims,
            const auto& cell,
            auto& cons,
            auto gamma
        )
        {
            // Do nothing
            for (int qq = 0; qq < dim; qq++) {
                cons[qq + 1] = 0.0;
            }
        }

        static constexpr auto calculate_volume(const auto& cell)
        {
            return get_differential<GridDirection::X1>(cell) *
                   get_differential<GridDirection::X2>(cell) *
                   get_differential<GridDirection::X3>(cell);
        }

        static constexpr void calculate_widths(auto* widths, const auto& cell)
        {
            widths[0] = cell.normals[1] - cell.normals[0];
            if constexpr (dim > 1) {
                widths[1] = cell.normals[3] - cell.normals[2];
            }
            if constexpr (dim > 2) {
                widths[2] = cell.normals[5] - cell.normals[4];
            }
        }
    };

    // dimension traits
    template <int dim>
    struct DimensionTraits {

        static constexpr int area_count   = 2 * dim;
        static constexpr int normal_count = 2 * dim;
        static constexpr int width_count  = dim;
        static constexpr bool has_2d      = dim > 1;
        static constexpr bool has_3d      = dim > 2;
    };

    // CellParams
    template <
        typename Parent,
        typename Derived,
        int dim,
        typename C,
        typename P>
    struct CellParams {
        using child_t = Parent;
        const child_t& parent;
        alignas(32) real areas[DimensionTraits<dim>::area_count];
        alignas(32) real normals[DimensionTraits<dim>::normal_count];
        alignas(32) real widths[DimensionTraits<dim>::width_count];
        real dV, x1mean, x2mean, x3mean;
        Geometry geometry;
        static constexpr real POLAR_TOL = 1.e-10;

        DUAL CellParams(const child_t& parent)
            : parent(parent),
              dV(0.0),
              x1mean(0.0),
              x2mean(0.0),
              x3mean(0.0),
              geometry(parent.derived().geometry)
        {
        }

        // Compile-time dispatch for geometry
        DUAL void calculate_areas()
        {
            switch (parent.derived().geometry) {
                case Geometry::SPHERICAL:
                    GeometryTraits<Geometry::SPHERICAL, Derived, dim>::
                        calculate_areas(areas, *this);
                    break;
                case Geometry::CARTESIAN:
                    GeometryTraits<Geometry::CARTESIAN, Derived, dim>::
                        calculate_areas(areas, *this);
                    break;
                default:
                    GeometryTraits<Geometry::CYLINDRICAL, Derived, dim>::
                        calculate_areas(areas, *this);
                    break;
            }
        }

        DUAL auto geometrical_sources(const auto& prims) const
        {
            auto cons = typename Derived::conserved_t{};
            switch (parent.derived().geometry) {
                case Geometry::SPHERICAL:
                    GeometryTraits<Geometry::SPHERICAL, Derived, dim>::
                        calculate_sources(
                            prims,
                            *this,
                            cons,
                            parent.derived().gamma
                        );
                    break;
                case Geometry::CARTESIAN:
                    GeometryTraits<Geometry::CARTESIAN, Derived, dim>::
                        calculate_sources(
                            prims,
                            *this,
                            cons,
                            parent.derived().gamma
                        );
                    break;
                default:
                    GeometryTraits<Geometry::CYLINDRICAL, Derived, dim>::
                        calculate_sources(
                            prims,
                            *this,
                            cons,
                            parent.derived().gamma
                        );
                    break;
            }
            return cons;
        }

        DUAL void calculate_means()
        {
            x1mean = get_centroid(
                normals[0],
                normals[1],
                parent.derived().x1_cell_spacing
            );
            if constexpr (dim > 1) {
                x2mean = get_centroid(
                    normals[2],
                    normals[3],
                    parent.derived().x2_cell_spacing
                );
            }
            if constexpr (dim > 2) {
                x3mean = get_centroid(
                    normals[4],
                    normals[5],
                    parent.derived().x2_cell_spacing
                );
            }
        }

        DUAL void calculate_volume()
        {
            switch (parent.derived().geometry) {
                case Geometry::SPHERICAL:
                    dV = GeometryTraits<Geometry::SPHERICAL, Derived, dim>::
                        calculate_volume(*this);
                    break;
                case Geometry::CARTESIAN:
                    dV = GeometryTraits<Geometry::CARTESIAN, Derived, dim>::
                        calculate_volume(*this);
                    break;
                default:
                    dV = GeometryTraits<Geometry::CYLINDRICAL, Derived, dim>::
                        calculate_volume(*this);
                    break;
            }
        }

        DUAL void calculate_widths()
        {
            switch (parent.derived().geometry) {
                case Geometry::SPHERICAL:
                    GeometryTraits<Geometry::SPHERICAL, Derived, dim>::
                        calculate_widths(widths, *this);
                    break;
                case Geometry::CARTESIAN:
                    GeometryTraits<Geometry::CARTESIAN, Derived, dim>::
                        calculate_widths(widths, *this);
                    break;
                default:
                    GeometryTraits<Geometry::CYLINDRICAL, Derived, dim>::
                        calculate_widths(widths, *this);
                    break;
            }
        }

        DUAL void
        calculate_normals(const luint ii, const luint jj, const luint kk)
        {
            normals[0] = get_face<GridDirection::X1>(ii, jj, kk, 0);
            normals[1] = get_face<GridDirection::X1>(ii, jj, kk, 1);
            if constexpr (dim > 1) {
                normals[2] = get_face<GridDirection::X2>(ii, jj, kk, 0);
                normals[3] = get_face<GridDirection::X2>(ii, jj, kk, 1);
                if constexpr (dim > 2) {
                    normals[4] = get_face<GridDirection::X3>(ii, jj, kk, 0);
                    normals[5] = get_face<GridDirection::X3>(ii, jj, kk, 1);
                }
            }
        }

        // Unified accessor methods
        DUAL real area(const int norm) const
        {
            if (norm < DimensionTraits<dim>::area_count) {
                return areas[norm];
            }
            return 0.0;
        }

        DUAL real dx(const int norm) const
        {
            if (norm < DimensionTraits<dim>::normal_count) {
                return widths[norm];
            }
            return 0.0;
        }

        DUAL real normal(Side s) const
        {
            if (static_cast<int>(s) < DimensionTraits<dim>::normal_count) {
                return normals[static_cast<int>(s)];
            }
            return 0.0;
        }

        DUAL real velocity(Side s) const
        {
            if (static_cast<int>(s) <= 2 * dim - 1) {
                const real x = normals[static_cast<int>(s)];
                return parent.derived().homolog
                           ? x * parent.derived().hubble_param
                           : parent.derived().hubble_param;
            }
            return 0.0;
        }

        template <int dir>
        DUAL real get_face_linear(const luint idx, bool is_left) const
        {
            const real min_val = dir == 1   ? parent.derived().x1min
                                 : dir == 2 ? parent.derived().x2min
                                            : parent.derived().x3min;
            const real dx      = dir == 1   ? parent.derived().dx1
                                 : dir == 2 ? parent.derived().dx2
                                            : parent.derived().dx3;

            const real xl = my_max<real>(min_val + (idx - 0.5) * dx, min_val);
            if (is_left) {
                return xl;
            }

            return xl + dx * (idx == 0 ? 0.5 : 1.0);
        }

        // Helper for log spacing
        template <int dir>
        DUAL real get_face_log(const luint idx, bool is_left) const
        {
            const real min_val = dir == 1   ? parent.derived().x1min
                                 : dir == 2 ? parent.derived().x2min
                                            : parent.derived().x3min;
            const real dlog    = dir == 1   ? parent.derived().dlogx1
                                 : dir == 2 ? parent.derived().dlogx2
                                            : parent.derived().dlogx3;

            const real xl = my_max<real>(
                min_val * std::pow(10.0, (idx - 0.5) * dlog),
                min_val
            );

            if (is_left) {
                return xl;
            }

            return xl * std::pow(10.0, dlog * (idx == 0 ? 0.5 : 1.0));
        }

        template <GridDirection Dir>
        DUAL real
        get_face(const luint ii, const luint jj, const luint kk, const int side)
            const
        {
            constexpr int dir = Dir == GridDirection::X1   ? 1
                                : Dir == GridDirection::X2 ? 2
                                                           : 3;

            if constexpr (dir == 1) {
                const auto ireal = get_real_idx(
                    ii,
                    parent.derived().radius,
                    parent.derived().xag
                );
                if (parent.derived().x1_cell_spacing == Cellspacing::LINSPACE) {
                    return get_face_linear<dir>(ireal, side == 0);
                }
                return get_face_log<dir>(ireal, side == 0);
            }
            else if (dir == 2) {
                const auto jreal = get_real_idx(
                    jj,
                    parent.derived().radius,
                    parent.derived().yag
                );
                if (parent.derived().x2_cell_spacing == Cellspacing::LINSPACE) {
                    return get_face_linear<dir>(jreal, side == 0);
                }
                return get_face_log<dir>(jreal, side == 0);
            }
            else {
                const auto kreal = get_real_idx(
                    kk,
                    parent.derived().radius,
                    parent.derived().zag
                );
                if (parent.derived().x3_cell_spacing == Cellspacing::LINSPACE) {
                    return get_face_linear<dir>(kreal, side == 0);
                }
                return get_face_log<dir>(kreal, side == 0);
            }
        }

        DUAL void calculate_all(const luint ii, const luint jj, const luint kk)
        {
            calculate_normals(ii, jj, kk);
            calculate_areas();
            calculate_means();
            calculate_volume();
            calculate_widths();
        }

        // accessors
        // normal accessors
        DUAL real x1L() const { return normals[0]; }

        DUAL real x1R() const { return normals[1]; }

        DUAL real x2L() const
        {
            if constexpr (dim > 1) {
                return normals[2];
            }
            return 0.0;
        }

        DUAL real x2R() const
        {
            if constexpr (dim > 1) {
                return normals[3];
            }
            return 0.0;
        }

        DUAL real x3L() const
        {
            if constexpr (dim > 2) {
                return normals[4];
            }
            return 0.0;
        }

        DUAL real x3R() const
        {
            if constexpr (dim > 2) {
                return normals[5];
            }
            return 0.0;
        }

        // area accessors
        DUAL real a1L() const { return areas[0]; }

        DUAL real a1R() const { return areas[1]; }

        DUAL real a2L() const
        {
            if constexpr (dim > 1) {
                return areas[2];
            }
            return 0.0;
        }

        DUAL real a2R() const
        {
            if constexpr (dim > 1) {
                return areas[3];
            }
            return 0.0;
        }

        DUAL real a3L() const
        {
            if constexpr (dim > 2) {
                return areas[4];
            }
            return 0.0;
        }

        DUAL real a3R() const
        {
            if constexpr (dim > 2) {
                return areas[5];
            }
            return 0.0;
        }

        DUAL real idV1() const
        {
            if (parent.derived().geometry == Geometry::CARTESIAN) {
                return 1.0 / (x1R() - x1L());
            }
            return 1.0 / dV;
        }

        DUAL real idV2() const
        {
            if (parent.derived().geometry == Geometry::CARTESIAN) {
                return 1.0 / (x2R() - x2L());
            }
            return 1.0 / dV;
        }

        DUAL real idV3() const
        {
            if (parent.derived().geometry == Geometry::CARTESIAN) {
                return 1.0 / (x3R() - x3L());
            }
            return 1.0 / dV;
        }

        DUAL real idx1() const { return parent.derived().invdx1; }

        DUAL real idx2() const { return parent.derived().invdx2; }

        DUAL real idx3() const { return parent.derived().invdx3; }

        DUAL bool at_pole(real val) const
        {
            return std::abs(std::sin(val)) < POLAR_TOL;
        }
    };

    // Mesh class
    template <typename Derived, int dim, typename C, typename P>
    struct Mesh {
        // Static type checks
        static_assert(dim >= 1 && dim <= 3, "Invalid dimension");
        // static_assert(
        //     std::is_base_of_v<HydroBase, Derived>,
        //     "Must derive from HydroBase"
        // );

        using Params = CellParams<Mesh<Derived, dim, C, P>, Derived, dim, C, P>;

        // Simplified interface
        DUAL Params cell_geometry(
            const luint ii,
            const luint jj = 0,
            const luint kk = 0
        ) const
        {
            Params cell(*this);
            cell.calculate_all(ii, jj, kk);
            return cell;
        }

      private:
        DUAL Derived& derived() { return static_cast<Derived&>(*this); }

        DUAL const Derived& derived() const
        {
            return static_cast<const Derived&>(*this);
        }

        friend Params;

        // Replace function pointers with compile-time dispatch
        // template <Geometry G>
        // DUAL void initialize_geometry()
        // {
        //     geometry_calculator = &GeometryTraits<G>::calculate;
        // }
    };
}   // namespace simbi
#endif   // MESH_HPP