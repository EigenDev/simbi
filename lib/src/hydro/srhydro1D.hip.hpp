/*
 * Interface between python construction of the 1D SR state
 * and cpp. This is where the heavy lifting will occur when
 * computing the HLL derivative of the state vector
 * given the state itself.
 */

#ifndef SRHYDRO1D_HIP_HPP
#define SRHYDRO1D_HIP_HPP

#include <string>
#include <vector>
#include "common/clattice1D.hpp"
#include "common/config.hpp"
#include "common/hydro_structs.hpp"
#include "build_options.hpp"
#include "util/exec_policy.hpp"

namespace simbi
{
     struct SRHD
     {
          int xmin, xmax;
          real dt;
          real gamma, CFL;
          std::string coord_system;
          std::vector<real> r, dt_arr;
          std::vector<std::vector<real>> state;
          CLattice1D coord_lattice;


          SRHD();
          SRHD(std::vector<std::vector<real>> state, real gamma, real CFL,
               std::vector<real> r, std::string coord_system);
          ~SRHD();

          // SRHD* device_self;

          // Create vector instances that will live on host
          std::vector<sr1d::Conserved> cons;
          std::vector<sr1d::Primitive> prims;

          int nx, n, active_zones, idx_shift, i_start, i_bound;
          real tend;
          real plm_theta, engine_duration, t, decay_constant;
          bool first_order, periodic, linspace, hllc;

          std::vector<real> sourceD, sourceS, source0, pressure_guess;

          //===============================
          // Host Ptrs to underlying data
          //===============================
          sr1d::Conserved * cons_ptr;
          real      * pguess_ptr;
          sr1d::Primitive * prims_ptr;
          void cons2prim1D();
          void advance(
               SRHD *s, 
               const int sh_block_size, 
               const int radius, 
               const simbi::Geometry geometry, 
               const simbi::MemSide user = simbi::MemSide::Host);
          void set_mirror_ptrs();
          void initalizeSystem(
              std::vector<std::vector<real>> &sources,
              real tstart = 0.0,
              real tend = 0.1,
              real dt = 1.e-4,
              real plm_theta = 1.5,
              real engine_duration = 10,
              real chkpt_interval = 0.1,
              std::string data_directory = "data/",
              bool first_order = true,
              bool periodic = false,
              bool linspace = true,
              bool hllc = false);
          
          void cons2prim(ExecutionPolicy<> p, SRHD *dev = nullptr, simbi::MemSide user = simbi::MemSide::Host);

          GPU_CALLABLE_MEMBER
          sr1d::Eigenvals calc_eigenvals(const sr1d::Primitive &prims_l,
                                         const sr1d::Primitive &prims_r);

          void adapt_dt();
          void adapt_dt(SRHD *dev, int blockSize);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved prims2cons(const sr1d::Primitive &prim);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hll_state(
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux,
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims);

          
          sr1d::Conserved calc_intermed_state(
               const sr1d::Primitive &prims,
               const sr1d::Conserved &state,
               const real a, const real aStar,
               const real pStar);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved prims2flux(const sr1d::Primitive &prim);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hll_flux(
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims,
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hllc_flux(
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims,
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux);

          std::vector<sr1d::Conserved> u_dot1D(std::vector<sr1d::Conserved> &u_state);
          
          std::vector<std::vector<real>>
          simulate1D(
               std::vector<std::vector<real>> &sources, 
               real tstart,
               real tend, real dt, real plm_theta, real engine_duraction,
               real chkpt_interval, std::string data_directory,
               bool first_order, bool periodic, bool linspace, bool hllc);

          
          //==============================================================
          // Create dynamic array instances that will live on device
          //==============================================================
          //             GPU RESOURCES
          //==============================================================
          int blockSize;
          sr1d::Conserved *gpu_cons, *gpu_du_dt, *gpu_u1;
          sr1d::Primitive *gpu_prims;
          real            *gpu_pressure_guess, *gpu_sourceD, *gpu_sourceS, *gpu_source0, *dt_min;
          CLattice1D      *gpu_coord_lattice;
          
     };
     
} // namespace simbi
#endif