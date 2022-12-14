/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO2D_HIP_HPP
#define SRHYDRO2D_HIP_HPP

#include <vector>
#include "common/helpers.hpp"
#include "common/hydro_structs.hpp"
#include "util/exec_policy.hpp"
#include "util/ndarray.hpp"
#include "base.hpp"

namespace simbi
{
    struct SRHD2D : public HydroBase
    {
        using primitive_t = sr2d::Primitive;
        using conserved_t = sr2d::Conserved;
        using primitive_soa_t = sr2d::PrimitiveSOA;
        const static int dimensions = 2;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones;
        ndarray<real> sourceD, sourceS1, sourceS2, sourceTau, pressure_guess, dt_min;
        bool d_all_zeros, s1_all_zeros, s2_all_zeros, e_all_zeros, scalar_all_zeros;
        
        /* Methods */
        SRHD2D();
        SRHD2D(
            std::vector<std::vector<real>> state2D, 
            luint nx, 
            luint ny, 
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2,
            real cfl, 
            std::string coord_system);
        ~SRHD2D();

        //================================= Methods ==================================================
        GPU_CALLABLE_MEMBER
        sr2d::Eigenvals calc_eigenvals(
            const sr2d::Primitive &primsL,
            const sr2d::Primitive &primsR,
            const luint nhat) const;

        GPU_CALLABLE_MEMBER
        sr2d::Conserved prims2cons(const sr2d::Primitive &prims) const;

        GPU_CALLABLE_MEMBER
        sr2d::Conserved calc_hllc_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const luint nhat,
            const real vface) const;

        GPU_CALLABLE_MEMBER
        sr2d::Conserved prims2flux(const sr2d::Primitive &prims,  luint nhat) const;

        GPU_CALLABLE_MEMBER
        sr2d::Conserved calc_hll_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const luint nhat,
            const real vface) const;

        void adapt_dt();
        void adapt_dt(const ExecutionPolicy<> &p, luint bytes);
        
        void advance(
               const ExecutionPolicy<> &p, 
               const luint bx,
               const luint by);

        void cons2prim(const ExecutionPolicy<> &p);

        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::AXIS_CYLINDRICAL:
            case simbi::Geometry::CARTESIAN:
                {
                    const real xl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                    if (side == 0) {
                        return xl;
                    } 
                    return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                }
            case simbi::Geometry::PLANAR_CYLINDRICAL:
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    } 
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
            } // end switch
        }


        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side)
        {
            const real yl = helpers::my_max(x2min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return yl;
            } 
            return helpers::my_min(yl + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const simbi::Geometry geometry)
        {
            switch (geometry)
            {
            case simbi::Geometry::SPHERICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dcos   = std::cos(tl) - std::cos(tr);
                const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (xr * xr * xr - xl * xl * xl) * dcos);
                return dV;
            }
            case simbi::Geometry::PLANAR_CYLINDRICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dx2    = tr - tl;
                const real dV     = (1.0 / 2.0) * (xr * xr - xl * xl) * dx2;
                return dV;
            }

            case simbi::Geometry::AXIS_CYLINDRICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dx2    = zr - zl;
                const real dV     = (1.0 / 2.0) * (xr * xr - xl * xl) * dx2;
                return dV;
            }
            
            default:
                break;
            }
            
        }

        std::vector<std::vector<real>> simulate2D(
            std::vector<std::vector<real>> &sources,
            real tstart,
            real tend,
            real dlogt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            int  chkpt_idx,
            std::string data_directory,
            std::string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc,
            bool quirk_smoothing=true,
            std::function<double(double)> a = nullptr,
            std::function<double(double)> adot = nullptr,
            std::function<double(double, double)> d_outer = nullptr,
            std::function<double(double, double)> s1_outer = nullptr,
            std::function<double(double, double)> s2_outer = nullptr,
            std::function<double(double, double)> e_outer = nullptr);
    };
}

#endif