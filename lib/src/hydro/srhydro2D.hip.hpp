/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO2D_HIP_HPP
#define SRHYDRO2D_HIP_HPP

#include <vector>
#include <string>
#include "common/hydro_structs.hpp"
#include "common/clattice2D.hpp"
#include "util/exec_policy.hpp"

namespace simbi
{
    class SRHD2D
    {
    public:
        /* Shared Data Members */
        std::vector<sr2d::Primitive> prims;
        std::vector<sr2d::Conserved> cons;
        std::vector<std::vector<real>> state2D;
        real plm_theta, gamma, bipolar, hubble_param;
        bool first_order, periodic, hllc, linspace, inFailureState, mesh_motion;
        real cfl, dt, decay_const;
        luint nx, ny, nzones, n, block_size, xphysical_grid, yphysical_grid;
        luint active_zones, idx_active, x_bound, y_bound;
        luint i_start, i_bound, j_start, j_bound;
        std::string coord_system;
        std::vector<real> x1, x2, sourceD, sourceS1, sourceS2, sourceTau, pressure_guess;
        CLattice2D coord_lattice;
        simbi::Geometry geometry;
        simbi::BoundaryCondition bc;
        real x2max, x2min, x1min, x1max, dx2, dx1, dlogx1;
        bool d_all_zeros, s1_all_zeros, s2_all_zeros, e_all_zeros, scalar_all_zeros, quirk_smoothing;

        //==============GPU Mirrors================
        real *gpu_sourceD, *gpu_sourceS1, *gpu_sourceS2, *gpu_sourceTau, *gpu_pressure_guess;
        real *sys_state, *dt_min;
        sr2d::Primitive *gpu_prims;
        sr2d::Conserved *gpu_cons;
        

        /* Methods */
        SRHD2D();
        SRHD2D(std::vector<std::vector<real>> state2D, 
            luint nx, 
            luint ny, 
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2,
            real cfl, 
            std::string coord_system);
        ~SRHD2D();

        GPU_CALLABLE_MEMBER
        sr2d::Eigenvals calc_eigenvals(
            const sr2d::Primitive &prims_l,
            const sr2d::Primitive &prims_r,
            const luint nhat) const;

        GPU_CALLABLE_MEMBER
        sr2d::Conserved prims2cons(const sr2d::Primitive &prims) const;
        
        sr2d::Conserved calc_hll_state(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            luint nhat) const;

        sr2d::Conserved calc_intermed_statesSR2D(const sr2d::Primitive &prims,
                                                 const sr2d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 luint nhat);

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
        void adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, luint bytes);
        
        void advance(
               SRHD2D *s, 
               const ExecutionPolicy<> p, 
               const luint bx,
               const luint by,
               const luint radius, 
               const simbi::Geometry geometry, 
               const simbi::MemSide user = simbi::MemSide::Host);

        void cons2prim(
            ExecutionPolicy<> p, 
            SRHD2D *dev = nullptr, 
            simbi::MemSide user = simbi::MemSide::Host);

        void cons2prim(SRHD2D *s);

        GPU_CALLABLE_INLINE
        constexpr real get_xface(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                        return 1.0;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                        const real rl = (ii > 0 ) ? x1min * pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min;
                        if (side == 0) {
                            return rl;
                        } else {
                            return (ii < xphysical_grid - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                        }
                        break;
                }
            }
        }

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const simbi::Geometry geometry, const real step)
        {
            const real xl     = get_xface(ii, geometry, 0);
            const real xr     = get_xface(ii, geometry, 1);
            const real xlf    = xl * (1.0 + step * dt * hubble_param);
            const real xrf    = xr * (1.0 + step * dt * hubble_param);
            const real tl     = my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
            const real tr     = my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
            const real dcos   = std::cos(tl) - std::cos(tr);
            const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (xr * xr * xr - xl * xl * xl) * dcos);
            return dV;
            
        }

        std::vector<std::vector<real>> simulate2D(
            std::vector<std::vector<real>> &sources,
            real tstart,
            real tend,
            real init_dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
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