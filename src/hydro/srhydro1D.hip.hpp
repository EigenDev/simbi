/*
 * Interface between python construction of the 1D SR state
 * and cpp. This is where the heavy lifting will occur when
 * computing the HLL derivative of the state vector
 * given the state itself.
 */

#ifndef SRHYDRO1D_HIP_HPP
#define SRHYDRO1D_HIP_HPP

#include <vector>
#include "common/helpers.hpp"
#include "common/enums.hpp"
#include "common/hydro_structs.hpp"
#include "build_options.hpp"
#include "util/exec_policy.hpp"
#include "build_options.hpp"
#include "util/ndarray.hpp"
#include "base.hpp"

namespace simbi
{
     struct SRHD : public HydroBase
     {
          using conserved_t = sr1d::Conserved;
          using primitive_t = sr1d::Primitive;
          using primitive_soa_t = sr1d::PrimitiveSOA;
          const static int dimensions = 1;

          // Create vector instances that will live on host
          ndarray<conserved_t> cons, outer_zones; 
          ndarray<primitive_t> prims;
          ndarray<real> sourceD, sourceS, source0, pressure_guess, dt_min;
          
          SRHD(){};
          SRHD(
               std::vector<std::vector<real>> state, 
               real gamma, 
               real cfl,
               std::vector<real> x1, 
               std::string coord_system);
          ~SRHD(){};

          void advance(
               const ExecutionPolicy<> &p,
               const luint xstride);

          void cons2prim(const ExecutionPolicy<> &p);

          GPU_CALLABLE_MEMBER
          sr1d::Eigenvals calc_eigenvals(const sr1d::Primitive &primsL,
                                         const sr1d::Primitive &primsR) const;

          void adapt_dt();
          void adapt_dt(luint blockSize);

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
               int  chkpt_idx,
               std::string data_directory,
               std::string boundary_condition,
               bool first_order, 
               bool linspace, 
               bool hllc,
               bool constant_sources,
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
                         }
                         return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                    }
               case simbi::Geometry::SPHERICAL:
                    {
                         const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                         if (side == 0) {
                              return rl;
                         } 
                         return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                         
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
          
     };
     
} // namespace simbi

#endif