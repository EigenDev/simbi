#ifndef MESH_HPP
#define MESH_HPP
#include "build_options.hpp"
#include <type_traits>

// Forward declare helpers
namespace simbi {
    namespace helpers {
        template <typename T>
        KERNEL void hybrid_set_mesh_funcs(T geom_class);

        template <typename T>
        STATIC constexpr T my_max(const T a, const T b);

        template <typename T>
        STATIC constexpr T my_min(const T a, const T b);
    }   // namespace helpers

}   // namespace simbi

using namespace simbi::helpers;

template <typename Derived, int dim>
struct Mesh {
    struct CellParams {
        using rFunc2 = real (CellParams::*)(const real, const real) const;
        using AreaCalcFunc =
            void (CellParams::*)(const luint, const luint, const luint);
        using rFunc = real (CellParams::*)() const;

        static AreaCalcFunc area_func;
        static rFunc idV1func, idV2func, idV3func;
        static rFunc2 centroid_func;
        real areas[2 * dim];
        real normals[2 * dim];
        real dV, x1mean, x2mean, x3mean;

        const Mesh<Derived, dim>& parent;
        ~CellParams() = default;

        CellParams(const Mesh<Derived, dim>& parent) : parent(parent) {}

        // Singleton instance
        static CellParams& instance()
        {
            static CellParams instance;
            return instance;
        }

        //===================================================
        // AREA ACCESSORS
        //===================================================

        DUAL real a1L() const { return areas[0]; }

        DUAL real a1R() const { return areas[1]; }

        DUAL real a2L() const
        {
            if constexpr (dim > 1) {
                return areas[2];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real a2R() const
        {
            if constexpr (dim > 1) {
                return areas[3];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real a3L() const
        {
            if constexpr (dim > 2) {
                return areas[4];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real a3R() const
        {
            if constexpr (dim > 2) {
                return areas[5];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        //===================================================
        // NORMAL ACCESSORS
        //===================================================
        DUAL real x1L() const { return normals[0]; }

        DUAL real x1R() const { return normals[1]; }

        DUAL real x2L() const
        {
            if constexpr (dim > 1) {
                return normals[2];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real x2R() const
        {
            if constexpr (dim > 1) {
                return normals[3];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real x3L() const
        {
            if constexpr (dim > 2) {
                return normals[4];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL real x3R() const
        {
            if constexpr (dim > 2) {
                return normals[5];
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        // ==================================================
        // VOLUME CALCULATIONS
        // ==================================================
        DUAL real idx1() const { return parent.derived().invdx1; }

        DUAL real idx2() const { return parent.derived().invdx2; }

        DUAL real idx3() const { return parent.derived().invdx3; }

        DUAL real idV() const { return 1.0 / dV; }

        DUAL real idV1() const { return (this->*idV1func)(); }

        DUAL real idV2() const { return (this->*idV2func)(); }

        DUAL real idV3() const { return (this->*idV3func)(); }

        DUAL real get_x1_mean() const
        {
            return get_cell_centroid(x1L(), x1R());
        }

        DUAL real get_x2_mean() const
        {
            return calc_any_mean(
                x2L(),
                x2R(),
                parent.derived().x2_cell_spacing
            );
        }

        DUAL real get_x3_mean() const
        {
            return calc_any_mean(
                x3L(),
                x3R(),
                parent.derived().x3_cell_spacing
            );
        }

        //==================================================
        // MEAN CALCULATIONS
        //==================================================
        /**
         * @brief calculate the mean between any two values based on cell
         * spacing
         *
         * @param a
         * @param b
         * @param cellspacing
         * @return arithmetic or geometric mean between a and b
         */
        STATIC real calc_any_mean(
            const real a,
            const real b,
            const simbi::Cellspacing cellspacing
        ) const
        {
            switch (cellspacing) {
                case simbi::Cellspacing::LOGSPACE:
                    return std::sqrt(a * b);
                default:
                    return 0.5 * (a + b);
            }
        }

        STATIC real
        get_cell_centroid_cylindrical(const real xr, const real xl) const
        {
            return (2.0 / 3.0) * (xr * xr * xr - xl * xl * xl) /
                   (xr * xr - xl * xl);
        }

        STATIC real
        get_cell_centroid_spherical(const real xr, const real xl) const
        {
            return 0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) /
                   (xr * xr * xr - xl * xl * xl);
        }

        STATIC real
        get_cell_centroid_cartesian(const real xr, const real xl) const
        {
            return 0.5 * (xr + xl);
        }

        /**
         * @brief Get the geometric cell centroid for spherical or cylindrical
         * mesh
         *
         * @param xr left coordinates
         * @param xl right coordinate
         * @param geometry geometry of state
         * @return cell centroid
         */
        STATIC
        real get_cell_centroid(const real xr, const real xl) const
        {
            return (this->*centroid_func)(xr, xl);
        }

        //==================================================
        // Derived actions
        //==================================================

        DUAL constexpr real get_x1_differential(const lint ii) const
        {
            const real x1l   = get_x1face(ii, 0);
            const real x1r   = get_x1face(ii, 1);
            const real xmean = get_cell_centroid(x1r, x1l);
            switch (parent.derived().geometry) {
                case Geometry::SPHERICAL:
                    return xmean * xmean * (x1r - x1l);
                default:
                    return xmean * (x1r - x1l);
            }
        }

        DUAL constexpr real get_x2_differential(const lint ii) const
        {
            if constexpr (dim == 1) {
                switch (parent.derived().geometry) {
                    case Geometry::SPHERICAL:
                        return 2.0;
                    default:
                        return (2.0 * M_PI);
                }
            }
            else {
                switch (parent.derived().geometry) {
                    case Geometry::SPHERICAL: {
                        const real x2l  = get_x2face(ii, 0);
                        const real x2r  = get_x2face(ii, 1);
                        const real dcos = std::cos(x2l) - std::cos(x2r);
                        return dcos;
                    }
                    default: {
                        return parent.derived().dx2;
                    }
                }
            }
        }

        DUAL constexpr real get_x3_differential(const lint ii) const
        {
            if constexpr (dim == 1) {
                switch (parent.derived().geometry) {
                    case Geometry::SPHERICAL:
                        return (2.0 * M_PI);
                    default:
                        return 1.0;
                }
            }
            else if constexpr (dim == 2) {
                switch (parent.derived().geometry) {
                    case Geometry::PLANAR_CYLINDRICAL:
                        return 1.0;
                    default:
                        return (2.0 * M_PI);
                }
            }
            else {
                return parent.derived().dx3;
            }
        }

        DUAL real get_cell_volume(
            const lint ii,
            const lint jj = 0,
            const lint kk = 0
        ) const
        {
            // the volume in cartesian coordinates is only nominal
            if (parent.derived().geometry == Geometry::CARTESIAN) {
                return 1.0;
            }
            return get_x1_differential(ii) * get_x2_differential(jj) *
                   get_x3_differential(kk);
        }

        DUAL constexpr real get_x1face(const lint ii, const int side) const
        {
            switch (parent.derived().x1_cell_spacing) {
                case simbi::Cellspacing::LINSPACE: {
                    const real x1l = my_max<real>(
                        parent.derived().x1min +
                            (ii - 0.5) * parent.derived().dx1,
                        parent.derived().x1min
                    );
                    if (side == 0) {
                        return x1l;
                    }
                    return my_min<real>(
                        x1l + parent.derived().dx1 * (ii == 0 ? 0.5 : 1.0),
                        parent.derived().x1max
                    );
                }
                default: {
                    const real x1l = my_max<real>(
                        parent.derived().x1min *
                            std::pow(
                                10.0,
                                (ii - 0.5) * parent.derived().dlogx1
                            ),
                        parent.derived().x1min
                    );
                    if (side == 0) {
                        return x1l;
                    }
                    return my_min<real>(
                        x1l *
                            std::pow(
                                10.0,
                                parent.derived().dlogx1 * (ii == 0 ? 0.5 : 1.0)
                            ),
                        parent.derived().x1max
                    );
                }
            }
        }

        DUAL constexpr real get_x2face(const lint ii, const int side) const
        {
            switch (parent.derived().x2_cell_spacing) {
                case simbi::Cellspacing::LINSPACE: {
                    const real x2l = my_max<real>(
                        parent.derived().x2min +
                            (ii - 0.5) * parent.derived().dx2,
                        parent.derived().x2min
                    );
                    if (side == 0) {
                        return x2l;
                    }
                    return my_min<real>(
                        x2l + parent.derived().dx2 * (ii == 0 ? 0.5 : 1.0),
                        parent.derived().x2max
                    );
                }
                default: {
                    const real x2l = my_max<real>(
                        parent.derived().x2min *
                            std::pow(
                                10.0,
                                (ii - 0.5) * parent.derived().dlogx2
                            ),
                        parent.derived().x2min
                    );
                    if (side == 0) {
                        return x2l;
                    }
                    return my_min<real>(
                        x2l *
                            std::pow(
                                10.0,
                                parent.derived().dlogx2 * (ii == 0 ? 0.5 : 1.0)
                            ),
                        parent.derived().x2max
                    );
                }
            }
        }

        DUAL constexpr real get_x3face(const lint ii, const int side) const
        {
            switch (parent.derived().x3_cell_spacing) {
                case simbi::Cellspacing::LINSPACE: {
                    const real x3l = my_max<real>(
                        parent.derived().x3min +
                            (ii - 0.5) * parent.derived().dx3,
                        parent.derived().x3min
                    );
                    if (side == 0) {
                        return x3l;
                    }
                    return my_min<real>(
                        x3l + parent.derived().dx3 * (ii == 0 ? 0.5 : 1.0),
                        parent.derived().x3max
                    );
                }
                default: {
                    const real x3l = my_max<real>(
                        parent.derived().x3min *
                            std::pow(
                                10.0,
                                (ii - 0.5) * parent.derived().dlogx3
                            ),
                        parent.derived().x3min
                    );
                    if (side == 0) {
                        return x3l;
                    }
                    return my_min<real>(
                        x3l *
                            std::pow(
                                10.0,
                                parent.derived().dlogx3 * (ii == 0 ? 0.5 : 1.0)
                            ),
                        parent.derived().x3max
                    );
                }
            }
        }

        //===================================================
        // GEOMETRICAL SOURCE TERMS
        //===================================================
        // /**
        //  * @brief calculate the geometric source terms for the RMHD equations
        //  *
        //  * @tparam dim
        //  * @param prims
        //  * @param ii
        //  * @param jj
        //  * @param kk
        //  * @return DUAL
        //  */

        DUAL auto geom_sources_spherical_rmhd(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {

            const real sint = std::sin(x2mean);
            const real cot  = std::cos(x2mean) / sint;

            // Grab central primitives
            const real v1    = prb.get_v1();
            const real v2    = prb.get_v2();
            const real v3    = prb.get_v3();
            const real pt    = prb.total_pressure();
            const auto bmu   = typename Derived::mag_fourvec_t(prb);
            const real wt    = prb.enthalpy_density(parent.derived().gamma);
            const real gam2  = prb.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            // source terms in momentum
            // r-source terms are (T_{\theta\theta})
            const auto rsource = pt * (a1R() - a1L()) / dV +
                                 (wgam2 * (v2 * v2 + v3 * v3) -
                                  bmu.two * bmu.two - bmu.three * bmu.three) /
                                     x1mean;
            const auto tsource =
                pt * (a2R() - a2L()) / dV +
                cot * (wgam2 * v3 * v3 - bmu.three * bmu.three) / x1mean;
            const auto psource =
                -(wgam2 * v1 * v3 - bmu.one * bmu.three +
                  cot * (wgam2 * v2 * v3 - bmu.two * bmu.three)) /
                x1mean;

            const auto geom_source = typename Derived::conserved_t{
              0.0,
              rsource,
              tsource,
              psource,
              0.0,
              0.0,
              0.0,
              0.0
            };

            return geom_source;
        }

        /**
         * @brief geometriical source terms for non-mhd hydro (SRHD & Newtonian)
         *
         * @param prb
         * @param ii
         * @param jj
         * @param kk
         * @return geometrical source terms struct
         */
        DUAL auto geom_sources_spherical_nomhd(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            // Grab central primitives
            const real pt    = prb.p();
            const real wt    = prb.enthalpy_density(parent.derived().gamma);
            const real gam2  = prb.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            if constexpr (Derived::dimensions == 3) {
                const real sint    = std::sin(x2mean);
                const real cot     = std::cos(x2mean) / sint;
                const real v1      = prb.get_v1();
                const real v2      = prb.get_v2();
                const real v3      = prb.get_v3();
                const auto rsource = pt * (a1R() - a1L()) / dV +
                                     (wgam2 * (v2 * v2 + v3 * v3)) / x1mean;
                const auto tsource =
                    pt * (a2R() - a2L()) / dV +
                    (wgam2 * (cot * v3 * v3 - v1 * v2)) / x1mean;
                const auto psource = -(wgam2 * v3 * (v1 + v2 * cot)) / x1mean;
                return typename Derived::conserved_t{
                  0.0,
                  rsource,
                  tsource,
                  psource,
                  0.0
                };
            }
            else if constexpr (Derived::dimensions == 2) {
                const real v1 = prb.get_v1();
                const real v2 = prb.get_v2();
                const auto rsource =
                    pt * (a1R() - a1L()) / dV + (wgam2 * v2 * v2) / x1mean;
                const auto tsource =
                    pt * (a2R() - a2L()) / dV - (wgam2 * (v1 * v2)) / x1mean;
                return typename Derived::conserved_t{
                  0.0,
                  rsource,
                  tsource,
                  0.0,
                };
            }
            else {
                const auto rsource = pt * (a1R() - a1L()) / dV;
                return typename Derived::conserved_t{0.0, rsource, 0.0};
            }
        }

        DUAL auto geom_sources_spherical(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            if constexpr (Derived::regime == "srmhd") {
                return geom_sources_spherical_rmhd(prb, ii, jj, kk);
            }
            else if constexpr (Derived::regime == "srhd") {
                return geom_sources_spherical_srhd(prb, ii, jj, kk);
            }
            else {
                return typename Derived::conserved_t{};
            }
        }

        DUAL auto geom_sources_cylindrical_rmhd(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            // Grab central primitives
            const real v1    = prb.get_v1();
            const real v2    = prb.get_v2();
            const real pt    = prb.total_pressure();
            const auto bmuc  = typename Derived::mag_fourvec_t(prb);
            const real wt    = prb.enthalpy_density(parent.derived().gamma);
            const real gam2  = prb.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            const real rsource =
                (wgam2 * v2 * v2 - bmuc.two * bmuc.two + pt) / x1mean;
            const real psource =
                -(wgam2 * v1 * v2 - bmuc.one * bmuc.two) / x1mean;

            const auto geom_source = typename Derived::conserved_t{
              0.0,
              rsource,
              psource,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            };

            return geom_source;
        }

        DUAL auto geom_sources_cylindrical_nomhd(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            // Grab central primitives
            const real v1    = prb.get_v1();
            const real v2    = prb.get_v2();
            const real pt    = prb.total_pressure();
            const real wt    = prb.enthalpy_density(parent.derived().gamma);
            const real gam2  = prb.lorentz_factor_squared();
            const real wgam2 = wt * gam2;

            const real rsource = (wgam2 * v2 * v2 + pt) / x1mean;

            if constexpr (dim == 1) {
                const auto geom_source = typename Derived::conserved_t{
                  0.0,
                  rsource,
                  0.0,
                };
                return geom_source;
            }
            else if constexpr (dim == 2) {
                const real psource     = -(wgam2 * v1 * v2) / x1mean;
                const auto geom_source = typename Derived::conserved_t{
                  0.0,
                  rsource,
                  psource,
                  0.0,
                };
                return geom_source;
            }
            else {
                const real psource     = -(wgam2 * v1 * v2) / x1mean;
                const auto geom_source = typename Derived::conserved_t{
                  0.0,
                  rsource,
                  psource,
                  0.0,
                  0.0,
                };
                return geom_source;
            }
        }

        DUAL auto geom_sources_default(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            return typename Derived::conserved_t{};
        }

        DUAL auto geom_sources(
            const auto& prb,
            const luint ii,
            const luint jj,
            const luint kk
        ) const
        {
            if constexpr (Derived::regime == "srmhd") {
                switch (parent.derived().geometry) {
                    case Geometry::SPHERICAL:
                        return geom_sources_spherical_rmhd(prb, ii, jj, kk);
                    case Geometry::CYLINDRICAL:
                        return geom_sources_cylindrical_rmhd(prb, ii, jj, kk);
                    default:
                        return geom_sources_default(prb, ii, jj, kk);
                }
            }
            else {
                switch (parent.derived().geometry) {
                    case Geometry::SPHERICAL:
                        return geom_sources_spherical_nomhd(prb, ii, jj, kk);
                    case Geometry::CYLINDRICAL:
                        return geom_sources_cylindrical_nomhd(prb, ii, jj, kk);
                    default:
                        return geom_sources_default(prb, ii, jj, kk);
                }
            }
        }

        //===================================================
        // VOLUME ACCESSORS
        //===================================================

        //===================================================
        DUAL void calculate_means()
        {
            x1mean = get_x1_mean();
            x2mean = get_x2_mean();
            x3mean = get_x3_mean();
        }

        DUAL void
        calculate_normals(const luint ii, const luint jj, const luint kk)
        {
            normals[0] = get_x1face(ii, 0);
            normals[1] = get_x1face(ii, 1);
            if constexpr (dim > 1) {
                normals[2] = get_x2face(jj, 0);
                normals[3] = get_x2face(jj, 1);
                if constexpr (dim > 2) {
                    normals[4] = get_x3face(kk, 0);
                    normals[5] = get_x3face(kk, 1);
                }
            }
        }

        //===================================================
        // AREA CALCULATIONS
        //===================================================
        DUAL void calculate_areas_spherical(
            const luint ii,
            const luint jj,
            const luint kk
        )
        {
            const auto rr   = x1R();
            const auto rl   = x1L();
            const auto tr   = x2R();
            const auto tl   = x2L();
            const auto dcos = get_x2_differential(jj);
            const auto dphi = get_x3_differential(kk);
            areas[0]        = rl * rl * dcos * dphi;
            areas[1]        = rr * rr * dcos * dphi;
            if constexpr (dim > 1) {
                areas[2] = 0.5 * (rr * rr - rl * rl) * std::sin(tl) * dphi;
                areas[3] = 0.5 * (rr * rr - rl * rl) * std::sin(tr) * dphi;
                if constexpr (dim > 2) {
                    areas[4] = 0.5 * (rr * rr - rl * rl) * (tr - tl);
                    areas[5] = 0.5 * (rr * rr - rl * rl) * (tr - tl);
                }
            }
        }

        DUAL void calculate_areas_cylindrical(
            const luint ii,
            const luint jj,
            const luint kk
        )
        {
            const auto rr    = x1R();
            const auto rl    = x1L();
            const auto qr    = x2R();
            const auto ql    = x2L();
            const auto zr    = x3R();
            const auto zl    = x3L();
            const auto rmean = x1mean;
            areas[0]         = rl * (zr - zl) * (qr - ql);
            areas[1]         = rr * (zr - zl) * (qr - ql);
            if constexpr (dim > 1) {
                areas[2] = (rr - rl) * (zr - zl);
                areas[3] = (rr - rl) * (zr - zl);
                if constexpr (dim > 2) {
                    areas[4] = rmean * (rr - rl) * (zr - zl);
                    areas[5] = rmean * (rr - rl) * (zr - zl);
                }
            }
        }

        DUAL void calculate_areas_planar_cylindrical(
            const luint ii,
            const luint jj,
            const luint kk
        )
        {
            const real rl = x1L();
            const real rr = x1R();
            areas[0]      = rr * parent.derived().dx2;
            areas[1]      = rl * parent.derived().dx2;
            if constexpr (dim > 1) {
                areas[2] = (rr - rl);
                areas[3] = (rr - rl);
            }
        }

        DUAL void calculate_area_axis_cylindrical(
            const luint ii,
            const luint jj,
            const luint kk
        )
        {
            const real rl    = x1R();
            const real rr    = x1L();
            const real rmean = x1mean;
            areas[0]         = rr * parent.derived().dx2;
            areas[1]         = rl * parent.derived().dx2;
            if constexpr (dim > 1) {
                areas[2] = rmean * (rr - rl);
                areas[3] = rmean * (rr - rl);
            }
        }

        DUAL void
        calculate_areas_default(const luint ii, const luint jj, const luint kk)
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

        DUAL void
        calculate_areas(const luint ii, const luint jj, const luint kk)
        {
            (this->*area_func)(ii, jj, kk);
        }

        // Initialize function pointers
        static void
        initialize_function_pointers(const Mesh<Derived, dim>& parent)
        {
            if (parent.derived().geometry == Geometry::SPHERICAL) {
                area_func     = &CellParams::calculate_areas_spherical;
                centroid_func = &CellParams::get_cell_centroid_spherical;
            }
            else if (parent.derived().geometry == Geometry::CYLINDRICAL) {
                area_func     = &CellParams::calculate_areas_cylindrical;
                centroid_func = &CellParams::get_cell_centroid_cylindrical;
            }
            else if (parent.derived().geometry ==
                     Geometry::PLANAR_CYLINDRICAL) {
                area_func = &CellParams::calculate_areas_planar_cylindrical;
            }
            else if (parent.derived().geometry == Geometry::AXIS_CYLINDRICAL) {
                area_func = &CellParams::calculate_area_axis_cylindrical;
            }
            else {
                area_func     = &CellParams::calculate_areas_default;
                centroid_func = &CellParams::get_cell_centroid_cartesian;
            }

            if (parent.derived().geometry == Geometry::CARTESIAN) {
                idV1func = &CellParams::idx1;
                idV2func = &CellParams::idx2;
                idV3func = &CellParams::idx3;
            }
            else {
                idV1func = &CellParams::idV;
                idV2func = &CellParams::idV;
                idV3func = &CellParams::idV;
            }
        }

        static void set_mesh_funcs(const Mesh<Derived, dim>& parent)
        {
            SINGLE((helpers::hybrid_set_mesh_funcs<CellParams>), parent);
        }
    };

    Mesh()  = default;
    ~Mesh() = default;

    void initialize_cell_params() const { CellParams::set_mesh_funcs(*this); }

    DUAL CellParams compute_mesh_factors(
        const luint ii,
        const luint jj = 0,
        const luint kk = 0
    ) const
    {
        static bool initialized = false;
        if (!initialized) {
            initialize_cell_params();
            initialized = true;
        }
        CellParams cell(*this);
        cell.calculate_normals(ii, jj, kk);
        cell.calculate_areas(ii, jj, kk);
        cell.calculate_means();
        cell.dV = cell.get_cell_volume(ii, jj, kk);

        return cell;
    }

  private:
    Derived& derived() { return static_cast<Derived&>(*this); }

    const Derived& derived() const
    {
        return static_cast<const Derived&>(*this);
    }
};

// Define static members outside the class template
template <typename Derived, int dim>
typename Mesh<Derived, dim>::CellParams::AreaCalcFunc
    Mesh<Derived, dim>::CellParams::area_func = nullptr;

template <typename Derived, int dim>
typename Mesh<Derived, dim>::CellParams::rFunc2
    Mesh<Derived, dim>::CellParams::centroid_func = nullptr;

template <typename Derived, int dim>
typename Mesh<Derived, dim>::CellParams::rFunc
    Mesh<Derived, dim>::CellParams::idV1func = nullptr;

template <typename Derived, int dim>
typename Mesh<Derived, dim>::CellParams::rFunc
    Mesh<Derived, dim>::CellParams::idV2func = nullptr;

template <typename Derived, int dim>
typename Mesh<Derived, dim>::CellParams::rFunc
    Mesh<Derived, dim>::CellParams::idV3func = nullptr;
#endif