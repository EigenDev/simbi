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
#include "common/helpers.hpp"
#include "common/clattice1D.hpp"
#include "common/enums.hpp"
#include "common/hydro_structs.hpp"
#include "build_options.hpp"
#include "util/exec_policy.hpp"
#include "build_options.hpp"
namespace simbi
{
     struct SRHD
     {
          // Init Params (order matters)
          std::vector<std::vector<real>> state;
          real gamma;
          real cfl;
          std::vector<real> x1; 
          std::string coord_system;

          real dt, hubble_param;
          std::vector<real> dt_arr;
          CLattice1D coord_lattice;
          simbi::BoundaryCondition bc;
          simbi::Geometry geometry;
          simbi::Cellspacing xcell_spacing;
          real dlogx1, dx1, x1min, x1max;  
          luint radius, total_zones;

          // SRHD* device_self;

          // Create vector instances that will live on host
          std::vector<sr1d::Conserved> cons;
          std::vector<sr1d::Primitive> prims;

          luint nx, active_zones, idx_active, i_start, i_bound;
          real plm_theta, engine_duration, t, decay_constant, dlogt, tend;
          bool first_order, periodic, linspace, hllc, inFailureState, mesh_motion;

          std::vector<real> sourceD, sourceS, source0, pressure_guess;
          
          SRHD();
          SRHD(
               std::vector<std::vector<real>> state, 
               real gamma, 
               real cfl,
               std::vector<real> x1, 
               std::string coord_system);
          ~SRHD();

          //===============================
          // Host Ptrs to underlying data
          //===============================
          sr1d::Conserved * cons_ptr;
          real            * pguess_ptr;
          sr1d::Primitive * prims_ptr;

          void advance(
               SRHD *s, 
               const luint sh_block_size, 
               const luint radius, 
               const simbi::Geometry geometry, 
               const simbi::MemSide user = simbi::MemSide::Host);

          void set_mirror_ptrs();
          void cons2prim(ExecutionPolicy<> p, SRHD *dev = nullptr, simbi::MemSide user = simbi::MemSide::Host);

          GPU_CALLABLE_MEMBER
          sr1d::Eigenvals calc_eigenvals(const sr1d::Primitive &primsL,
                                         const sr1d::Primitive &primsR) const;

          void adapt_dt();
          void adapt_dt(SRHD *dev, luint blockSize);

          GPU_CALLABLE_MEMBER
          sr1d::Conserved prims2cons(const sr1d::Primitive &prim) const;

          GPU_CALLABLE_MEMBER
          sr1d::Conserved prims2flux(const sr1d::Primitive &prim) const;

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hll_flux(
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims,
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux,
               const real             vface) const;

          GPU_CALLABLE_MEMBER
          sr1d::Conserved calc_hllc_flux(
               const sr1d::Primitive &left_prims,
               const sr1d::Primitive &right_prims,
               const sr1d::Conserved &left_state,
               const sr1d::Conserved &right_state,
               const sr1d::Conserved &left_flux,
               const sr1d::Conserved &right_flux,
               const real             vface) const;
          
          std::vector<std::vector<real>>
          simulate1D(
               std::vector<std::vector<real>> &sources, 
               real tstart,
               real tend, 
               real dlogt,
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
          real calc_vface(const lint ii, const real hubble_const, const simbi::Geometry geometry, const int side) const;
          
          GPU_CALLABLE_INLINE
          constexpr real get_xface(const lint ii, const simbi::Geometry geometry, const int side) const
          {
               switch (geometry)
               {
               case simbi::Geometry::CARTESIAN:
                    {
                         const real xl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                         if (side == 0) {
                              return xl;
                         } else {
                              return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                         }
                    }
               case simbi::Geometry::SPHERICAL:
                    {
                         const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                         if (side == 0) {
                              return rl;
                         } else {
                              return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                         }
                    }
               case simbi::Geometry::CYLINDRICAL:
                    // TODO: Implement
                    break;
               }
          }

          GPU_CALLABLE_MEMBER
          constexpr real get_cell_volume(lint ii, const simbi::Geometry geometry) const
          {
               if (!mesh_motion)
               {
                    return 1.0;
               } else {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                    {         
                         const real rl     = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1), x1min);
                         const real rr     = helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
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
          sr1d::Conserved *gpu_cons;
          sr1d::Primitive *gpu_prims;
          real            *gpu_pressure_guess, *gpu_sourceD, *gpu_sourceS, *gpu_source0, *dt_min;
          
     };
     
} // namespace simbi

#endif