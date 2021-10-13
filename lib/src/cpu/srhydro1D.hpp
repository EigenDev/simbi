/*
 * Interface between python construction of the 1D SR state
 * and cpp. This is where the heavy lifting will occur when
 * computing the HLL derivative of the state vector
 * given the state itself.
 */

#ifndef SRHYDRO1D_H
#define SRHYDRO1D_H

#include <string>
#include <vector>
#include "../common/clattice1D.hpp"
#include "../common/hydro_structs.hpp"
#include "../common/helpers.hpp"

namespace simbi
{
     class SRHD
     {
     public:
          double gamma, CFL;
          std::string coord_system;
          std::vector<double> r;
          std::vector<std::vector<double>> state;
          CLattice1D coord_lattice;

          SRHD();
          SRHD(std::vector<std::vector<double>> state, double gamma, double CFL,
               std::vector<double> r, std::string coord_system);
          ~SRHD();

          std::vector<sr1d::Conserved> cons, cons_n;
          std::vector<sr1d::Primitive> prims;

          int NX, n, pgrid_size, idx_shift, i_start, i_bound;
          double tend, dt;
          double plm_theta, engine_duration, t, decay_constant;
          bool first_order, periodic, linspace, hllc;

          std::vector<double> sourceD, sourceS, source0, pressure_guess;

          void cons2prim1D();

          sr1d::Eigenvals calc_eigenvals(const sr1d::Primitive &prims_l,
                                         const sr1d::Primitive &prims_r);

          void adapt_dt();

          sr1d::Conserved calc_state(const sr1d::Primitive &prim);

          sr1d::Conserved calc_hll_state(const sr1d::Conserved &left_state,
                                         const sr1d::Conserved &right_state,
                                         const sr1d::Conserved &left_flux,
                                         const sr1d::Conserved &right_flux,
                                         const sr1d::Primitive &left_prims,
                                         const sr1d::Primitive &right_prims);

          sr1d::Conserved calc_intermed_state(const sr1d::Primitive &prims,
                                              const sr1d::Conserved &state,
                                              const double a, const double aStar,
                                              const double pStar);

          sr1d::Conserved calc_flux(double rho, double v, double pressure);

          sr1d::Conserved calc_hll_flux(const sr1d::Primitive &left_prims,
                                        const sr1d::Primitive &right_prims,
                                        const sr1d::Conserved &left_state,
                                        const sr1d::Conserved &right_state,
                                        const sr1d::Conserved &left_flux,
                                        const sr1d::Conserved &right_flux);

          sr1d::Conserved calc_hllc_flux(const sr1d::Primitive &left_prims,
                                         const sr1d::Primitive &right_prims,
                                         const sr1d::Conserved &left_state,
                                         const sr1d::Conserved &right_state,
                                         const sr1d::Conserved &left_flux,
                                         const sr1d::Conserved &right_flux);

          void advance();

          std::vector<std::vector<double>>
          simulate1D(
               std::vector<std::vector<double>> &sources, 
               double tstart               = 0.0,
               double tend                 = 0.1, 
               double init_dt              = 1e-4, 
               double plm_theta            = 1.5, 
               double engine_duraction     = 10.0,
               double chkpt_interval       = 1.0, 
               std::string data_directory  = "../data/",
               bool first_order = true, 
               bool periodic = false, 
               bool linspace = true, 
               bool hllc     = false);
     };
} // namespace simbi

#endif