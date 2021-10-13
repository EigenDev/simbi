/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHD_2D_H
#define SRHD_2D_H

#include <vector>
#include <string>
#include "../common/hydro_structs.hpp"
#include "../common/clattice2D.hpp"
#include "../common/viscous_diff.hpp"
#include "../common/helpers.hpp"

namespace simbi
{
    class SRHD2D
    {
    public:
        /* Shared Data Members */
        simbi::ArtificialViscosity aVisc;
        sr2d::Eigenvals lambda;
        std::vector<sr2d::Primitive> prims;
        std::vector<sr2d::Conserved> cons, cons_n;
        std::vector<std::vector<double>> state2D, sources;
        double tend, tstart;
        double plm_theta, gamma, bipolar;
        bool first_order, periodic, hllc, linspace;
        double CFL, dt, decay_const;
        int NX, NY, nzones, n, block_size, xphysical_grid, yphysical_grid;
        int active_zones, idx_active, x_bound, y_bound;
        std::string coord_system;
        std::vector<double> x1, x2, sourceD, source_S1, source_S2, source_tau, pressure_guess;
        std::vector<double> lorentz_gamma, xvertices, yvertices;
        CLattice2D coord_lattice;

        /* Methods */
        SRHD2D();
        SRHD2D(std::vector<std::vector<double>> state2D, int NX, int NY, double gamma, std::vector<double> x1,
               std::vector<double> x2,
               double CFL, std::string coord_system);
        ~SRHD2D();

        void cons2prim2D();

        sr2d::Eigenvals calc_Eigenvals(
            const sr2d::Primitive &prims_l,
            const sr2d::Primitive &prims_r,
            const unsigned int nhat);

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
                                                 double a,
                                                 double aStar,
                                                 double pStar,
                                                 int nhat);

        sr2d::Conserved calc_hllc_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const unsigned int nhat);

        sr2d::Conserved calc_Flux(const sr2d::Primitive &prims, unsigned int nhat);

        sr2d::Conserved calc_hll_flux(
            const sr2d::Conserved &left_state,
            const sr2d::Conserved &right_state,
            const sr2d::Conserved &left_flux,
            const sr2d::Conserved &right_flux,
            const sr2d::Primitive &left_prims,
            const sr2d::Primitive &right_prims,
            const unsigned int nhat);

        sr2d::Conserved u_dot(unsigned int ii, unsigned int jj);

        void evolve();

        void adapt_dt();

        std::vector<std::vector<double>> simulate2D(
            const std::vector<std::vector<double>> sources,
            double tstart = 0., 
            double tend = 0.1, 
            double init_dt = 1.e-4, 
            double plm_theta = 1.5,
            double engine_duration = 10, 
            double chkpt_interval = 0.1,
            std::string data_directory = "data/", 
            bool first_order = true,
            bool periodic = false, 
            bool linspace = true, 
            bool hllc = false);
    };
}

#endif