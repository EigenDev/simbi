/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO3D_HIP_HPP
#define SRHYDRO3D_HIP_HPP

#include <vector>
#include <string>
#include "../common/hydro_structs.hpp"
#include "../common/clattice3D.hpp"

namespace simbi
{
    struct SRHD3D
    {
    public:
        /* Shared Data Members */
        sr3d::Eigenvals lambda;
        std::vector<sr3d::Primitive> prims;
        std::vector<sr3d::Conserved> cons;
        std::vector<std::vector<real>> state3D, sources;
        float tend, tstart;
        real plm_theta, gamma, bipolar;
        bool first_order, periodic, hllc, linspace, inFailureState;
        real cfl, dt, decay_const;
        luint nx, ny, nz, nzones, n, block_size, xphysical_grid, yphysical_grid, zphysical_grid;
        luint active_zones, idx_active, x_bound, y_bound;
        luint i_start, i_bound, j_start, j_bound, k_start, k_bound;
        std::string coord_system;
        std::vector<real> x1, x2, x3, sourceD, sourceS1, sourceS2, sourceS3, sourceTau, pressure_guess;
        CLattice3D coord_lattice;
        simbi::Geometry geometry;
        simbi::BoundaryCondition bc;

        real x3max, x3min, x2max, x2min, x1min, x1max, dx3, dx2, dx1, dlogx1;
        bool d_all_zeros, s1_all_zeros, s2_all_zeros, s3_all_zeros, e_all_zeros, scalar_all_zeros, quirk_smoothing;


        //==============GPU Mirrors================
        real *gpu_sourceD, *gpu_sourceS1, *gpu_sourceS2, *gpu_sourceS3, *gpu_sourceTau, *gpu_pressure_guess;
        real *sys_state, *dt_min;
        sr3d::Primitive *gpu_prims;
        sr3d::Conserved *gpu_cons;

        /* Methods */
        SRHD3D();
        SRHD3D(
            std::vector<std::vector<real>> state3D, 
            luint nx, 
            luint ny,
            luint nz, 
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            std::vector<real> x3,
            real cfl, 
            std::string coord_system);
        ~SRHD3D();

        void cons2prim();
        void cons2prim(
            const ExecutionPolicy<> p, 
            SRHD3D *dev = nullptr, 
            simbi::MemSide user = simbi::MemSide::Host);

        void advance(
            SRHD3D *dev, 
            const ExecutionPolicy<> p,
            const luint bx,
            const luint by,
            const luint bz,
            const luint radius, 
            const simbi::Geometry geometry, 
            const simbi::MemSide user);

        GPU_CALLABLE_MEMBER
        sr3d::Eigenvals calc_Eigenvals(
            const sr3d::Primitive &prims_l,
            const sr3d::Primitive &prims_r,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2cons(const sr3d::Primitive &prims);
        
        sr3d::Conserved calc_hll_state(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        sr3d::Conserved calc_luintermed_statesSR2D(const sr3d::Primitive &prims,
                                                 const sr3d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hllc_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_Flux(const sr3d::Primitive &prims, luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hll_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        sr3d::Conserved u_dot(luint ii, luint jj);

        std::vector<sr3d::Conserved> u_dot2D(const std::vector<sr3d::Conserved> &cons_state);

        void adapt_dt();
        void adapt_dt(SRHD3D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, const luint bytes);

        std::vector<std::vector<real>> simulate3D(
            const std::vector<std::vector<real>> sources,
            real tstart = 0, 
            real tend = 0.1, 
            real init_dt = 1.e-4, 
            real plm_theta = 1.5,
            real engine_duration = 10, 
            real chkpt_interval = 0.1,
            std::string data_directory = "data/", 
            std::string boundary_condition = "outflow",
            bool first_order = true,
            bool linspace = true, 
            bool hllc = false);
    };
}

#endif