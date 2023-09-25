/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHD_HPP
#define SRHD_HPP

#include <vector>
#include "common/hydro_structs.hpp"
#include "common/helpers.hip.hpp"
#include "base.hpp"

namespace simbi
{
    template<int dim>
    struct SRHD : public HydroBase
    {
        // set the primitive and conservative types at compile time
        using primitive_t = typename std::conditional_t<
        dim == 1,
        sr1d::Primitive,
        std::conditional_t<
        dim == 2,
        sr2d::Primitive,
        sr3d::Primitive>
        >;
        using conserved_t = typename std::conditional_t<
        dim == 1,
        sr1d::Conserved,
        std::conditional_t<
        dim == 2,
        sr2d::Conserved,
        sr3d::Conserved>
        >;//sr3d::Conserved;
        using primitive_soa_t = typename std::conditional_t<
        dim == 1,
        sr1d::PrimitiveSOA,
        std::conditional_t<
        dim == 2,
        sr2d::PrimitiveSOA,
        sr3d::PrimitiveSOA>
        >;
        using eigenvals_t = typename std::conditional_t<
        dim == 1,
        sr1d::Eigenvals,
        std::conditional_t<
        dim == 2,
        sr2d::Eigenvals,
        sr3d::Eigenvals>
        >;
        const static int dimensions = dim;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> sourceD, sourceS1, sourceS2, sourceS3, sourceTau, pressure_guess, dt_min;
        ndarray<bool> object_pos;
        bool scalar_all_zeros, quirk_smoothing;

        /* Methods */
        SRHD(
            std::vector<std::vector<real>> &state,
            InitialConditions &init_conditions);
        ~SRHD();

        void cons2prim(const ExecutionPolicy<> &p);

        void advance(
            const ExecutionPolicy<> &p,
            const luint xstride,
            const luint ystride);

        GPU_CALLABLE_MEMBER
        eigenvals_t calc_eigenvals(
            const primitive_t &primsL,
            const primitive_t &primsR,
            const luint nhat) const;

        GPU_CALLABLE_MEMBER
        conserved_t prims2cons(const primitive_t &prims) const;
        
        conserved_t calc_hll_state(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hllc_flux(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat,
            const real vface) const;

        GPU_CALLABLE_MEMBER
        conserved_t prims2flux(const primitive_t &prims, const luint nhat) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hll_flux(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat,
            const real vface) const;  

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<> &p);

        std::vector<std::vector<real>> simulate(
            std::function<double(double)> a,
            std::function<double(double)> adot,
            std::function<double(double, double)> d_outer  = nullptr,
            std::function<double(double, double)> s1_outer = nullptr,
            std::function<double(double, double)> s2_outer = nullptr,
            std::function<double(double, double)> s3_outer = nullptr,
            std::function<double(double, double)> e_outer  = nullptr
        );

        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const int side) const
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                    const real x1l = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                    if (side == 0) {
                        return x1l;
                    }
                    return helpers::my_min(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
            default:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
                break;
            }
        }


        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side) const
        {
            const real x2l = helpers::my_max(x2min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return x2l;
            } 
            return helpers::my_min(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x3face(const lint ii, const int side) const
        {

            const real x3l = helpers::my_max(x3min  + (ii - static_cast<real>(0.5)) * dx3,  x3min);
            if (side == 0) {
                return x3l;
            } 
            return helpers::my_min(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x1_differential(const lint ii) {
            const real x1l   = get_x1face(ii, 0);
            const real x1r   = get_x1face(ii, 1);
            const real xmean = helpers::get_cell_centroid(x1r, x1l, geometry);
            switch (geometry)
            {
            case Geometry::SPHERICAL:
                return xmean * xmean * (x1r - x1l);
            default:
                return xmean * (x1r - x1l);
            }
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x2_differential(const lint ii) {
            if constexpr(dim == 1) {
                switch (geometry)
                {
                case Geometry::SPHERICAL:
                    return 2;
                default:
                    return static_cast<real>(2 * M_PI);
                }
            } else {
                switch (geometry)
                {
                    case Geometry::SPHERICAL:
                    {
                        const real x2l = get_x2face(ii, 0);
                        const real x2r = get_x2face(ii, 1);
                        const real dcos = std::cos(x2l) - std::cos(x2r);
                        return dcos;  
                    }
                    default:
                    {
                        return dx2;
                    }
                }
            }
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x3_differential(const lint ii) {
            if constexpr(dim == 1) {
                switch (geometry)
                {
                case Geometry::SPHERICAL:
                    return static_cast<real>(2 * M_PI);
                default:
                    return 1;
                }
            } else if constexpr(dim == 2) {
                switch (geometry)
                {
                    case Geometry::PLANAR_CYLINDRICAL:
                         return 1;
                    default:
                        return static_cast<real>(2 * M_PI);
                        break;
                }
            } else {
                return dx3;
            }
        }


        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const lint kk) const
        {
            return get_x1_differential(ii) * get_x2_differential(jj) * get_x3_differential(kk);
        }

        // GPU_CALLABLE_INLINE
        // conserved_t get_geometric_source_terms(const primitive_t &prim, const real dV) const {
        //     // Grab central primitives
        //     const real rhoc = prim.rho;
        //     const real v1   = prim.v2omponent(1);
        //     const real v2   = prim.v2omponent(2);
        //     const real v3   = prim.v2omponent(3);
        //     const real pc   = prim.p;
        //     const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
        //     const real gam2 = 1/(1 - (v1 * v1 + v2 * v2 + v3 * v3));

        //     switch (geometry)
        //     {
        //     case Geometry::SPHERICAL:
        //         {
        //             if constexpr(dim == 1) {
        //                 return = conserved_t{
        //                     0, 
        //                     pc * (s1R - s1L) / dV,
        //                     0
        //                 };

        //             } else if constexpr(dim == 2) {
        //                 return = conserved_t{
        //                     0, 
        //                     (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV1,
        //                     rhoc * hc * gam2 * (-v1 * v2) / rmean + pc * (s2R - s2L)/dV2,
        //                     0
        //                 };

        //             } else {
        //                 return = conserved_t{
        //                     0, 
        //                     (rhoc * hc * gam2 * (v2 * v2 + v3 * v3)) / rmean + pc * (s1R - s1L) / dV1,
        //                     rhoc * hc * gam2 * (v3 * v3 * cot - v1 * v2) / rmean + pc * (s2R - s2L)/dV2 , 
        //                     - rhoc * hc * gam2 * v3 * (v1 + v2 * cot) / rmean, 
        //                     0
        //                 };
        //             }
        //         }
            
        //     case Geometry::AXIS_CYLINDRICAL:
        //     {

        //         return = conserved_t{
        //             0, 
        //             pc * (s1R - s1L) / dV,
        //             0,
        //             0
        //         };
        //     }
        //     case Geometry::PLANAR_CYLINDRICAL:
        //     {
        //         return = conserved_t{
        //             0, 
        //             (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV,
        //             rhoc * hc * gam2 * (-v1 * v2) / rmean + pc * (s2R - s2L) / dV,
        //             0
        //         };
        //     }
        //     case Geometry::CYLINDRICAL:
        //     {
        //         if constexpr(dim == 1) {
        //             return = conserved_t{
        //                 0, 
        //                 pc * (s1R - s1L) / dV,
        //                 0
        //             };

        //         } 
        //         } else {
        //             return = conserved_t{
        //                 0, 
        //                 (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV,
        //                 - rhoc * hc * gam2 * (v1 * v2) / rmean, 
        //                 0, 
        //                 0
        //             };
        //         }
        //     }
        //     default:
        //         return conserved_t;
        //     }
        // }

        GPU_CALLABLE_INLINE
        real get_x1r_face_area(const real x1l, const real x1r){

        }

        void emit_troubled_cells() {
            troubled_cells.copyFromGpu();
            cons.copyFromGpu();
            prims.copyFromGpu();
            for (luint gid = 0; gid < total_zones; gid++)
            {
                if (troubled_cells[gid] != 0) {
                    const luint xpg   = xphysical_grid;
                    const luint ypg   = yphysical_grid;
                    const luint kk    = detail::get_height(gid, xpg, ypg);
                    const luint jj    = detail::get_row(gid, xpg, ypg, kk);
                    const luint ii    = detail::get_column(gid, xpg, ypg, kk);
                    const lint ireal  = helpers::get_real_idx(ii, radius, xphysical_grid);
                    const lint jreal  = helpers::get_real_idx(jj, radius, yphysical_grid); 
                    const lint kreal  = helpers::get_real_idx(kk, radius, zphysical_grid); 
                    const real x1l    = get_x1face(ireal, 0);
                    const real x1r    = get_x1face(ireal, 1);
                    const real x2l    = get_x2face(jreal, 0);
                    const real x2r    = get_x2face(jreal, 1);
                    const real x3l    = get_x3face(kreal, 0);
                    const real x3r    = get_x3face(kreal, 1);
                    const real x1mean = helpers::calc_any_mean(x1l, x1r, x1cell_spacing);
                    const real x2mean = helpers::calc_any_mean(x2l, x2r, x2cell_spacing);
                    const real x3mean = helpers::calc_any_mean(x3l, x3r, x3cell_spacing);
                    const auto s1 = cons[gid].momentum(1);
                    const auto s2 = cons[gid].momentum(2);
                    const auto s3 = cons[gid].momentum(3);
                    const real et  = (cons[gid].d + cons[gid].tau + prims[gid].p);
                    const real s   = std::sqrt(s1 * s2 + s2 * s2 + s3 * s3);
                    const real v2  = (s * s) / (et * et);
                    const real w   = 1 / std::sqrt(1 - v2);
                    printf("\nCons2Prim cannot converge\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, x3coord: %.2e, iter: %d\n", 
                    cons[gid].d / w, prims[gid].p, v2, x1mean, x2mean, x3mean, troubled_cells[gid]);
                }
            }
        }
    };
}

template<>
struct is_relativistic<simbi::SRHD<1>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::SRHD<2>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::SRHD<3>>
{
    static constexpr bool value = true;
};

#include "srhd.tpp"
#endif