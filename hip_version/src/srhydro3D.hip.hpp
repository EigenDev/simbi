/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/

#include <vector>
#include <string>
#include "hydro_structs.hpp"
#include "clattice3D.hpp"
#include "viscous_diff.hpp"

#ifndef SRHYDRO3D_HIP_HPP
#define SRHYDRO3D_HIP_HPP
namespace simbi
{
    struct SRHD3D
    {
    public:
        /* Shared Data Members */
        simbi::ArtificialViscosity aVisc;
        sr3d::Eigenvals lambda;
        std::vector<sr3d::Primitive> prims;
        std::vector<sr3d::Conserved> u0;
        std::vector<std::vector<real>> state3D, sources;
        float tend, tstart;
        real plm_theta, gamma, bipolar;
        bool first_order, periodic, hllc, linspace;
        real CFL, dt, decay_const;
        int NX, NY, NZ, nzones, n, block_size, xphysical_grid, yphysical_grid, zphysical_grid;
        int active_zones, idx_active, x_bound, y_bound;
        int i_start, i_bound, j_start, j_bound, k_start, k_bound;
        std::string coord_system;
        std::vector<real> x1, x2, x3, sourceD, source_S1, source_S2, source_S3, source_tau, pressure_guess;
        CLattice3D coord_lattice;

        //==============GPU Mirrors================
        real *gpu_sourceD, *gpu_sourceS1, *gpu_sourceS2, *gpu_sourceS3, *gpu_sourceTau, *gpu_pressure_guess;
        real *sys_state, *dt_min;
        sr3d::Primitive *gpu_prims;
        sr3d::Conserved *gpu_state3D;

        /* Methods */
        SRHD3D();
        SRHD3D(
            std::vector<std::vector<real>> state3D, 
            int NX, 
            int NY,
            int NZ, 
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            std::vector<real> x3,
            real CFL, 
            std::string coord_system);
        ~SRHD3D();

        void cons2prim2D();

        GPU_CALLABLE_MEMBER
        sr3d::Eigenvals calc_Eigenvals(
            const sr3d::Primitive &prims_l,
            const sr3d::Primitive &prims_r,
            const unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2cons(const sr3d::Primitive &prims);
        
        sr3d::Conserved calc_hll_state(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            unsigned int nhat);

        sr3d::Conserved calc_intermed_statesSR2D(const sr3d::Primitive &prims,
                                                 const sr3d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 int nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hllc_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_Flux(const sr3d::Primitive &prims, unsigned int nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hll_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const unsigned int nhat);

        sr3d::Conserved u_dot(unsigned int ii, unsigned int jj);

        std::vector<sr3d::Conserved> u_dot2D(const std::vector<sr3d::Conserved> &cons_state);

        real adapt_dt(const std::vector<sr3d::Primitive> &prims);

        std::vector<std::vector<real>> simulate3D(
            const std::vector<std::vector<real>> sources,
            float tstart,
            float tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            std::string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc);
    };

    struct SRHD3D_DualSpace
     {
           SRHD3D_DualSpace();
          ~SRHD3D_DualSpace();

          sr3d::Primitive *host_prims;
          sr3d::Conserved *host_u0;
          real            *host_pressure_guess;
          real            *host_source0;
          real            *host_sourceD;
          real            *host_sourceS1;
          real            *host_sourceS2;
          real            *host_sourceS3;
          real            *host_dtmin;
          real            *host_dx1, *host_x1m, *host_fas1, *host_dV1, *host_dx3, *host_sin;
          real            *host_dx2, *host_cot, *host_fas2, *host_dV2, *host_dV3, *host_fas3;
          CLattice2D      *host_clattice;

          real host_dt;
          real host_xmin;
          real host_xmax;
          real host_ymin;
          real host_ymax;
          real host_zmin;
          real host_zmax;

          void copyStateToGPU(const SRHD3D &host, SRHD3D *device);
          void copyGPUStateToHost(const SRHD3D *device, SRHD3D &host);
          void cleanUp();
     };

    // GPU_CALLABLE_MEMBER bool strong_shock(real pl, real pr);
    __global__ void gpu_advance(SRHD3D *s, const int n, const simbi::Geometry geometry);
    __global__ void shared_gpu_advance(SRHD3D *s, const int sh_block_size, const int sh_block_space, const int radius, const simbi::Geometry geometry);
    __global__ void shared_gpu_cons2prim(SRHD3D *s);
}

#endif