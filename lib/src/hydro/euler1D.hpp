/* 
* Interface between python construction of the 1D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER1D_HPP
#define EULER1D_HPP

#include <vector>
#include <string>
#include "common/hydro_structs.hpp"
#include "common/clattice1D.hpp"
#include "common/helpers.hpp"
#include "common/config.hpp"
namespace simbi {
    enum class SOLVER{HLLE, HLLC};

    class Newtonian1D {
        public: 
            std::vector<std::vector<real>> init_state;
            std::vector<hydro1d::Conserved> cons, cons_n; 
            std::vector<hydro1d::Primitive> prims;
            std::vector<real> r, xvertices, sourceRho, sourceMom, sourceE;
            real plm_theta, gamma, tend, dt, CFL, engine_duration, t, decay_constant;
            bool first_order, periodic, linspace, hllc;
            int nzones, active_zones, idx_active, i_start, i_bound, n, nx;
            std::string coord_system;
            simbi::SOLVER sim_solver;
            CLattice1D coord_lattice;

            
            Newtonian1D();
            Newtonian1D(
                std::vector<std::vector<real>> init_state, 
                real gamma, 
                real CFL,
                std::vector<real> r, 
                std::string coord_system);
            ~Newtonian1D();

            // Calculate the wave speeds from the Jacobian Matrix formed by the Euler Eqns
            hydro1d::Eigenvals calc_eigenvals(const hydro1d::Primitive &left_state, const hydro1d::Primitive &right_state);

            void cons2prim();
            void adapt_dt();

            hydro1d::Conserved prims2cons(const hydro1d::Primitive &prims);

            hydro1d::Conserved calc_flux(const hydro1d::Primitive &prims);

            hydro1d::Conserved calc_hll_flux(
                const hydro1d::Primitive &left_prims,
                const hydro1d::Primitive &right_prims,
                const hydro1d::Conserved &left_state,
                const hydro1d::Conserved &right_state,
                const hydro1d::Conserved &left_flux,
                const hydro1d::Conserved &right_flux);

            hydro1d::Conserved calc_hllc_flux(
                const hydro1d::Primitive &left_prims,
                const hydro1d::Primitive &right_prims,
                const hydro1d::Conserved &left_state,
                const hydro1d::Conserved &right_state,
                const hydro1d::Conserved &left_flux,
                const hydro1d::Conserved &right_flux);

            void evolve();

            std::vector<std::vector<real> > simulate1D(
                std::vector<std::vector<real>> &sources,
                real tstart = 0.0,
                real tend = 0.1,
                real init_dt = 1.e-4,
                real plm_theta = 1.5,
                real engine_duration = 10,
                real chkpt_interval = 0.1,
                std::string data_directory = "data/",
                bool first_order = true,
                bool periodic = false,
                bool linspace = true,
                bool hllc = false);

            
            

    };
}

#endif