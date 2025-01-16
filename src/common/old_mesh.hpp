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

namespace simbi {
    /**
     * @brief Base mesh class for handling geometry-specific calculations
     *
     * @tparam Derived CRTP derived class type
     * @tparam dim  Number of spatial dimensions (1-3)
     * @tparam C Conserved variable type
     * @tparam P Primitive variable type
     */

    enum class Side {
        X1L,
        X1R,
        X2L,
        X2R,
        X3L,
        X3R
    };

    template <typename Derived, int dim, typename C, typename P>
    struct Mesh {
        Mesh()  = default;
        ~Mesh() = default;

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
        DUAL real calc_any_mean(
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

        DUAL real
        get_cell_centroid_cylindrical(const real xr, const real xl) const
        {
            return (2.0 / 3.0) * (xr * xr * xr - xl * xl * xl) /
                   (xr * xr - xl * xl);
        }

        DUAL real
        get_cell_centroid_spherical(const real xr, const real xl) const
        {
            return 0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) /
                   (xr * xr * xr - xl * xl * xl);
        }

        DUAL real
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

        struct CellParams {
            real areas[2 * dim];
            real normals[2 * dim];
            real dV, x1mean, x2mean, x3mean;

            const Mesh<Derived, dim, C, P>& parent;
            real hubble_param;
            bool homolog;
            ~CellParams() = default;

            DUAL CellParams(const Mesh<Derived, dim, C, P>& parent)
                : dV(0.0),
                  x1mean(0.0),
                  x2mean(0.0),
                  x3mean(0.0),
                  parent(parent),
                  hubble_param(parent.derived().hubble_param),
                  homolog(parent.derived().homolog)
            {
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

            DUAL bool at_pole(real val) const
            {
                return std::abs(std::sin(val)) < 1.0e-10;
            }

            // ==================================================
            // VOLUME CALCULATIONS
            // ==================================================
            DUAL real idx1() const { return parent.derived().invdx1; }

            DUAL real idx2() const { return parent.derived().invdx2; }

            DUAL real idx3() const { return parent.derived().invdx3; }

            DUAL real idV() const { return 1.0 / dV; }

            DUAL real idV1() const { return (this->*(parent.idV1func))(); }

            DUAL real idV2() const { return (this->*(parent.idV2func))(); }

            DUAL real idV3() const { return (this->*(parent.idV3func))(); }

            DUAL real get_x1_mean() const
            {
                return parent.get_cell_centroid(x1L(), x1R());
            }

            DUAL real get_x2_mean() const
            {
                return parent.calc_any_mean(
                    x2L(),
                    x2R(),
                    parent.derived().x2_cell_spacing
                );
            }

            DUAL real get_x3_mean() const
            {
                return parent.calc_any_mean(
                    x3L(),
                    x3R(),
                    parent.derived().x3_cell_spacing
                );
            }

            //==================================================
            // VELOCITY CALCULATIONS
            //==================================================
            DUAL real v1fL() const
            {
                return (homolog) ? x1L() * hubble_param : hubble_param;
            }

            DUAL real v1fR() const
            {
                return (homolog) ? x1R() * hubble_param : hubble_param;
            }

            DUAL real v2fL() const
            {
                if constexpr (dim > 1) {
                    return (homolog) ? x2L() * hubble_param : hubble_param;
                }
                else {
                    return static_cast<real>(0.0);
                }
            }

            DUAL real v2fR() const
            {
                if constexpr (dim > 1) {
                    return (homolog) ? x2R() * hubble_param : hubble_param;
                }
                else {
                    return static_cast<real>(0.0);
                }
            }

            DUAL real v3fL() const
            {
                if constexpr (dim > 2) {
                    return (homolog) ? x3L() * hubble_param : hubble_param;
                }
                else {
                    return static_cast<real>(0.0);
                }
            }

            DUAL real v3fR() const
            {
                if constexpr (dim > 2) {
                    return (homolog) ? x3R() * hubble_param : hubble_param;
                }
                else {
                    return static_cast<real>(0.0);
                }
            }

            template <Side Fc>
            DUAL real velocity() const
            {
                if constexpr (static_cast<int>(Fc) <= 2 * dim - 1) {
                    const real x = normals[static_cast<int>(Fc)];
                    return parent.derived().homolog
                               ? x * parent.derived().hubble_param
                               : parent.derived().hubble_param;
                }
                return 0.0;
            }

            //==================================================
            // Derived actions
            //==================================================

            DUAL constexpr real get_x1_differential(const lint ii) const
            {
                const real x1l   = get_x1face(ii, 0);
                const real x1r   = get_x1face(ii, 1);
                const real xmean = parent.get_cell_centroid(x1r, x1l);
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
                        return x1l +
                               parent.derived().dx1 * (ii == 0 ? 0.5 : 1.0);
                    }
                    default: {
                        const auto ia = [ii]() {
                            if (ii > parent.derived().xag - 1 +
                                         parent.derived().radius) {
                                return parent.derived().xag - 1;
                            }
                            return (ii - parent.derived().radius > 0) *
                                   (ii - parent.derived().radius);
                        };

                        const real x1l = my_max<real>(
                            parent.derived().x1min *
                                std::pow(
                                    10.0,
                                    (ia - 0.5) * parent.derived().dlogx1
                                ),
                            parent.derived().x1min
                        );
                        if (side == 0) {
                            return x1l;
                        }
                        return x1l * std::pow(
                                         10.0,
                                         parent.derived().dlogx1 *
                                             (ia == 0 ? 0.5 : 1.0)
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
                            x2l * std::pow(
                                      10.0,
                                      parent.derived().dlogx2 *
                                          (ii == 0 ? 0.5 : 1.0)
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
                            x3l + parent.derived().dx3,
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
                            x3l * std::pow(
                                      10.0,
                                      parent.derived().dlogx3 *
                                          (ii == 0 ? 0.5 : 1.0)
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
            //  * @brief calculate the geometric source terms for the RMHD
            //  equations
            //  *
            //  * @tparam dim
            //  * @param prims
            //  * @param ii
            //  * @param jj
            //  * @param kk
            //  * @return DUAL
            //  */

            DUAL auto geom_sources_spherical_rmhd(const auto& prb) const
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

                // geometric source terms in momentum
                const auto rsource =
                    pt * (a1R() - a1L()) / dV +
                    wgam2 * (v2 * v2 + v3 * v3) / x1mean -
                    (bmu.two * bmu.two + bmu.three * bmu.three) / x1mean;

                const auto tsource =
                    pt * (a2R() - a2L()) / dV -
                    wgam2 * (v2 * v1 - v3 * v3 * cot) / x1mean +
                    (bmu.two * bmu.one - bmu.three * bmu.three * cot) / x1mean;
                const auto psource =
                    -wgam2 * v3 * (v1 + cot * v2) / x1mean +
                    bmu.three * (bmu.one + cot * bmu.two) / x1mean;

                return typename Derived::conserved_t{
                  0.0,
                  rsource,
                  tsource,
                  psource,
                  0.0,
                  0.0,
                  0.0,
                  0.0
                };
            }

            /**
             * @brief geometriical source terms for non-mhd hydro (SRHD &
             * Newtonian)
             *
             * @param prb
             * @param ii
             * @param jj
             * @param kk
             * @return geometrical source terms struct
             */
            DUAL auto geom_sources_spherical_nomhd(const auto& prb) const
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
                    const auto psource =
                        -(wgam2 * v3 * (v1 + v2 * cot)) / x1mean;
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
                    const auto tsource = pt * (a2R() - a2L()) / dV -
                                         (wgam2 * (v1 * v2)) / x1mean;
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

            DUAL auto geom_sources_cylindrical_rmhd(const auto& prb) const
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

                return typename Derived::conserved_t{
                  0.0,
                  rsource,
                  psource,
                  0.0,
                  0.0,
                  0.0,
                  0.0,
                  0.0
                };
            }

            DUAL auto geom_sources_cylindrical_nomhd(const auto& prb) const
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
                    return typename Derived::conserved_t{
                      0.0,
                      rsource,
                      0.0,
                    };
                }
                else if constexpr (dim == 2) {
                    const real psource = -(wgam2 * v1 * v2) / x1mean;
                    return typename Derived::conserved_t{
                      0.0,
                      rsource,
                      psource,
                      0.0,
                    };
                }
                else {
                    const real psource = -(wgam2 * v1 * v2) / x1mean;
                    return typename Derived::conserved_t{
                      0.0,
                      rsource,
                      psource,
                      0.0,
                      0.0,
                    };
                }
            }

            DUAL auto geom_sources_axis_cylindrical_nomhd(const auto& prb) const
            {
                // Grab central primitives
                const real pt      = prb.total_pressure();
                const real rsource = pt / x1mean;
                if constexpr (dim == 2) {
                    return
                        typename Derived::conserved_t{0.0, rsource, 0.0, 0.0};
                }
                return typename Derived::conserved_t{};
            }

            DUAL auto geom_sources_default(const auto& prb) const
            {
                return typename Derived::conserved_t{};
            }

            DUAL auto geom_sources(const auto& prb) const
            {
                return (this->*(parent.geom_source_func))(prb);
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
                const auto rr = x1R() + v1fR() * parent.derived().step *
                                            parent.derived().dt;
                const auto rl = x1L() + v1fL() * parent.derived().step *
                                            parent.derived().dt;
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

            DUAL void calculate_areas_default(
                const luint ii,
                const luint jj,
                const luint kk
            )
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
                return (this->*(parent.area_func))(ii, jj, kk);
            }
        };

        DUAL void initialize_function_pointers()
        {
            if (derived().geometry == Geometry::SPHERICAL) {
                area_func     = &CellParams::calculate_areas_spherical;
                centroid_func = &Mesh::get_cell_centroid_spherical;
                if constexpr (Derived::regime == "srmhd") {
                    geom_source_func = &CellParams::geom_sources_spherical_rmhd;
                }
                else {
                    geom_source_func =
                        &CellParams::geom_sources_spherical_nomhd;
                }
            }
            else if (derived().geometry == Geometry::CYLINDRICAL) {
                area_func     = &CellParams::calculate_areas_cylindrical;
                centroid_func = &Mesh::get_cell_centroid_cylindrical;
                if constexpr (Derived::regime == "srmhd") {
                    geom_source_func =
                        &CellParams::geom_sources_cylindrical_rmhd;
                }
                else {
                    geom_source_func =
                        &CellParams::geom_sources_cylindrical_nomhd;
                }
            }
            else if (derived().geometry == Geometry::PLANAR_CYLINDRICAL) {
                area_func = &CellParams::calculate_areas_planar_cylindrical;
                if constexpr (Derived::regime != "srmhd") {
                    geom_source_func =
                        &CellParams::geom_sources_cylindrical_nomhd;
                }
            }
            else if (derived().geometry == Geometry::AXIS_CYLINDRICAL) {
                area_func = &CellParams::calculate_area_axis_cylindrical;
                if constexpr (Derived::regime != "srmhd") {
                    geom_source_func =
                        &CellParams::geom_sources_axis_cylindrical_nomhd;
                }
            }
            else {
                area_func        = &CellParams::calculate_areas_default;
                centroid_func    = &Mesh::get_cell_centroid_cartesian;
                geom_source_func = &CellParams::geom_sources_default;
            }

            if (derived().geometry == Geometry::CARTESIAN) {
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

        void set_mesh_funcs()
        {
            SINGLE((helpers::hybrid_set_mesh_funcs), this);
        }

        DUAL CellParams cell_geometry(
            const luint ii,
            const luint jj = 0,
            const luint kk = 0
        ) const
        {
            CellParams cell(*this);
            cell.calculate_normals(ii, jj, kk);
            cell.calculate_areas(ii, jj, kk);
            cell.calculate_means();
            cell.dV = cell.get_cell_volume(ii, jj, kk);

            return cell;
        }

      private:
        using CF = real (Mesh::*)(const real, const real) const;
        using AF = void (CellParams::*)(const luint, const luint, const luint);
        using VF = real (CellParams::*)() const;
        using GF = C (CellParams::*)(const P&) const;

        CF centroid_func;
        AF area_func;
        VF idV1func, idV2func, idV3func;
        GF geom_source_func;

        DUAL Derived& derived() { return static_cast<Derived&>(*this); }

        DUAL const Derived& derived() const
        {
            return static_cast<const Derived&>(*this);
        }
    };

}   // namespace simbi
#endif