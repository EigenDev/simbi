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
          real dt, hubble_param;
          real gamma, cfl;
          std::string coord_system;
          std::vector<real> x1, dt_arr;
          std::vector<std::vector<real>> state;
          CLattice1D coord_lattice;
          simbi::BoundaryCondition bc;
          simbi::Geometry geometry;
          real dlogx1, dx1, x1min, x1max;  

          SRHD();
          SRHD(std::vector<std::vector<real>> state, real gamma, real cfl,
               std::vector<real> x1, std::string coord_system);
          ~SRHD();

          // SRHD* device_self;

          // Create vector instances that will live on host
          std::vector<sr1d::Conserved> cons;
          std::vector<sr1d::Primitive> prims;

          luint nx, n, active_zones, idx_active, i_start, i_bound;
          real tend;
          real plm_theta, engine_duration, t, decay_constant;
          bool first_order, periodic, linspace, hllc, inFailureState;

          std::vector<real> sourceD, sourceS, source0, pressure_guess;

          //===============================
          // Host Ptrs to underlying data
          //===============================
          sr1d::Conserved * cons_ptr;
          real      * pguess_ptr;
          sr1d::Primitive * prims_ptr;

          void advance(
               SRHD *s, 
               const luint sh_block_size, 
               const luint radius, 
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
          void adapt_dt(SRHD *dev, luint blockSize);

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
               const sr1d::Conserved &right_flux,
               const real             vface);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hllc_flux(
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims,
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux,
               const real             vface);

          std::vector<sr1d::Conserved> u_dot1D(std::vector<sr1d::Conserved> &u_state);
          
          std::vector<std::vector<real>>
          simulate1D(
               std::vector<std::vector<real>> &sources, 
               real tstart,
               real tend, 
               real dt,\
               real plm_theta, 
               real engine_duraction,
               real chkpt_interval, 
               std::string data_directory,
               std::string boundary_condition,
               bool first_order, 
               bool linspace, 
               bool hllc,
               std::function<double(double)> a = nullptr,
               std::function<double(double)> adot = nullptr,
               std::function<double(double)> d_outer = nullptr,
               std::function<double(double)> s_souter = nullptr,
               std::function<double(double)> e_outer = nullptr);

          GPU_CALLABLE_MEMBER
          real calc_vface(const lint ii, const real hubble_const, const simbi::Geometry geometry, const int side);

          GPU_CALLABLE_INLINE
          real get_xface(const lint ii, const simbi::Geometry geometry, const int side)
          {
               switch (geometry)
               {
               case simbi::Geometry::CARTESIAN:
                    {
                         return 1.0;
                    }
                    break;
               
               case simbi::Geometry::SPHERICAL:
                    {
                         const real rl = (ii > 0 ) ? x1min * pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min;
                         if (side == 0) {
                              return rl;
                         } else {
                              return (ii < active_zones - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                         }
                         break;
                    }
               }
          }

          GPU_CALLABLE_MEMBER
          real get_cell_volume(lint ii, const simbi::Geometry geometry, const bool mesh_motion = false)
          {
               if (!mesh_motion)
               {
                    return 1.0;
               } else {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                    {
                         if (ii >= active_zones - 1)
                              ii = active_zones - 1;
                              
                         const real rl     = (ii > 0 ) ? x1min * pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min;
                         const real rr     = (ii < active_zones - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                         const real rmean  = static_cast<real>(0.75) * (rr * rr * rr *rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                         return rmean * rmean * (rr - rl);
                    }
                    default:
                         return 1.0;
                    }
               }
               
          }
          //==============================================================
          // Create dynamic array instances that will live on device
          //==============================================================
          //             GPU RESOURCES
          //==============================================================
          luint blockSize;
          sr1d::Conserved *gpu_cons, *gpu_du_dt, *gpu_u1;
          sr1d::Primitive *gpu_prims;
          real            *gpu_pressure_guess, *gpu_sourceD, *gpu_sourceS, *gpu_source0, *dt_min;
          CLattice1D      *gpu_coord_lattice;
          
     };
     
} // namespace simbi

#endif