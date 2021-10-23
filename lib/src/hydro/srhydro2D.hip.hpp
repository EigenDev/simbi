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
        sr2d::Eigenvals lambda;
        std::vector<sr2d::Primitive> prims;
        std::vector<sr2d::Conserved> cons;
        std::vector<std::vector<real>> state2D;
        float tend, tstart;
        real plm_theta, gamma, bipolar;
        bool first_order, periodic, hllc, linspace;
        real CFL, dt, decay_const;
        int nx, ny, nzones, n, block_size, xphysical_grid, yphysical_grid;
        int active_zones, idx_active, x_bound, y_bound;
        int i_start, i_bound, j_start, j_bound;
        std::string coord_system;
        std::vector<real> x1, x2, sourceD, sourceS1, sourceS2, sourceTau, pressure_guess;
        CLattice2D coord_lattice;

        real x2max, x2min, x1min, x1max, dx2, dx1, dlogx1;
        bool d_all_zeros, s1_all_zeros, s2_all_zeros, e_all_zeros;

        //==============GPU Mirrors================
        real *gpu_sourceD, *gpu_sourceS1, *gpu_sourceS2, *gpu_sourceTau, *gpu_pressure_guess;
        real *sys_state, *dt_min;
        sr2d::Primitive *gpu_prims;
        sr2d::Conserved *gpu_cons;

        /* Methods */
        SRHD2D();
        SRHD2D(std::vector<std::vector<real>> state2D, int nx, int ny, real gamma, std::vector<real> x1,
               std::vector<real> x2,
               real CFL, std::string coord_system);
        ~SRHD2D();

        void cons2prim2D();

        GPU_CALLABLE_MEMBER
        sr2d::Eigenvals calc_Eigenvals(
            const sr2d::Primitive &prims_l,
            const sr2d::Primitive &prims_r,
            const unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr2d::Conserved prims2cons(const sr2d::Primitive &prims);
        
        sr2d::Conserved calc_hll_state(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            unsigned int nhat);

        sr2d::Conserved calc_intermed_statesSR2D(const sr2d::Primitive &prims,
                                                 const sr2d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 int nhat);

        GPU_CALLABLE_MEMBER
        sr2d::Conserved calc_hllc_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr2d::Conserved prims2flux(const sr2d::Primitive &prims, unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr2d::Conserved calc_hll_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const unsigned int nhat);

        sr2d::Conserved u_dot(unsigned int ii, unsigned int jj);

        std::vector<sr2d::Conserved> u_dot2D(const std::vector<sr2d::Conserved> &cons_state);

        void adapt_dt();
        void adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p);
        
        void advance(
               SRHD2D *s, 
               const ExecutionPolicy<> p, 
               const int bx,
               const int by,
               const int radius, 
               const simbi::Geometry geometry, 
               const simbi::MemSide user = simbi::MemSide::Host);

        void cons2prim(
            ExecutionPolicy<> p, 
            SRHD2D *dev = nullptr, 
            simbi::MemSide user = simbi::MemSide::Host);
        void cons2prim(SRHD2D *s);

        std::vector<std::vector<real>> simulate2D(
            std::vector<std::vector<real>> &sources,
            real tstart,
            real tend,
            real init_dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            std::string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc);
    };
}

#endif