/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO2D_HIP_HPP
#define SRHYDRO2D_HIP_HPP

#include <vector>
#include "common/helpers.hip.hpp"
#include "common/hydro_structs.hpp"
#include "util/exec_policy.hpp"
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
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> sourceD, sourceS1, sourceS2, sourceTau, pressure_guess, dt_min;
        ndarray<bool> object_pos;
                
        std::function<double(double, double)> dens_outer;
        std::function<double(double, double)> mom1_outer;
        std::function<double(double, double)> mom2_outer;
        std::function<double(double, double)> nrg_outer;
        /* Methods */
        SRHD2D();
        SRHD2D(
            std::vector<std::vector<real>> &state2D, 
            luint nx, 
            luint ny, 
            real gamma, 
            std::vector<real> &x1,
            std::vector<real> &x2,
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
        sr2d::Conserved prims2flux(const sr2d::Primitive &prims,  const luint nhat) const;

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

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<> &p);
        
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

            default:
                return 0;
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
                const real rl     = get_x1face(ii, geometry, 0);
                const real rr     = get_x1face(ii, geometry, 1);
                const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dx2    = zr - zl;
                const real rmean  = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                const real dV     = rmean * (rr - rl) * dx2;
                return dV;
            }
            
            default:
                return 1;
            }
            
        }

        std::vector<std::vector<real>> simulate2D(
            std::vector<std::vector<real>> &sources,
            const std::vector<bool> &object_cells,
            std::vector<real> &gsource,
            real tstart,
            real tend,
            real dlogt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            int  chkpt_idx,
            std::string data_directory,
            std::vector<std::string> boundary_conditions,
            bool first_order,
            bool linspace,
            const std::string solver,
            bool quirk_smoothing,
            bool constant_sources,
            std::vector<std::vector<real>> boundary_sources,
            std::function<double(double)> a = nullptr,
            std::function<double(double)> adot = nullptr,
            std::function<double(double, double)> d_outer = nullptr,
            std::function<double(double, double)> s1_outer = nullptr,
            std::function<double(double, double)> s2_outer = nullptr,
            std::function<double(double, double)> e_outer = nullptr);


        
        void emit_troubled_cells() {
            troubled_cells.copyFromGpu();
            cons.copyFromGpu();
            prims.copyFromGpu();
            for (luint gid = 0; gid < total_zones; gid++)
            {
                if (troubled_cells[gid] != 0) {
                    const auto ii     = gid % nx;
                    const auto jj     = gid / nx;
                    const lint ireal  = helpers::get_real_idx(ii, radius, xphysical_grid);
                    const lint jreal  = helpers::get_real_idx(jj, radius, yphysical_grid); 
                    const real x1l    = get_x1face(ireal, geometry, 0);
                    const real x1r    = get_x1face(ireal, geometry, 1);
                    const real x2l    = get_x2face(jreal, 0);
                    const real x2r    = get_x2face(jreal, 1);
                    const real x1mean = helpers::calc_any_mean(x1l, x1r, x1cell_spacing);
                    const real x2mean = helpers::calc_any_mean(x2l, x2r, x2cell_spacing);
                    const real s      = std::sqrt(cons[gid].s1 * cons[gid].s1 + cons[gid].s2 * cons[gid].s2);
                    const real p      = prims[gid].p;
                    const real et     = (cons[gid].tau + cons[gid].d + prims[gid].p);
                    const real v2     = (s * s) / (et * et);
                    const real w      = 1 / std::sqrt(1 - v2);
                    printf("\nCons2Prim cannot converge:\nDensity: %.2e, Pressure: %.2e, Vsq: %.2f, et: %.2e, x1coord: %.2e, x2coord: %.2e, iter: %d\n", 
                    cons[gid].d / w, p, v2, et,  x1mean, x2mean, troubled_cells[gid]);
                }
            }
        }
    };
}

template<>
struct is_relativistic<simbi::SRHD2D>
{
    static constexpr bool value = true;
};
#endif