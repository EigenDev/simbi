/* 
* Interface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER2D_HPP
#define EULER2D_HPP

#include <vector>
#include <string>
#include "common/hydro_structs.hpp"
#include "common/clattice2D.hpp"
#include "common/config.hpp"
#include "common/helpers.hpp"


namespace simbi {
    class Newtonian2D {
        public:
        std::vector<std::vector<real> > init_state, sources;
        std::vector<hydro2d::Conserved> cons, cons_n;
        std::vector<hydro2d::Primitive> prims;
        std::vector<real> sourceRho, sourceM1, sourceM2, sourceE;
        real plm_theta, gamma, tend, CFL, dt, decay_const;
        bool first_order, periodic, hllc, linspace;
        std::string coord_system;
        std::vector<real> x1, x2;
        int nzones, ny, nx, active_zones, idx_active, n;
        int xphysical_grid, yphysical_grid, x_bound, y_bound;
        CLattice2D coord_lattice;
        simbi::Solver solver;


        Newtonian2D();
        Newtonian2D(std::vector< std::vector<real> > init_state, 
            int nx, 
            int ny,
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            real CFL, 
            std::string coord_system);

        ~Newtonian2D();

        void cons2prim();

        hydro2d::Eigenvals calc_eigenvals(
            const hydro2d::Primitive &left_state, 
            const hydro2d::Primitive &right_state,
            const int ehat = 1);

        hydro2d::Conserved prims2cons(const hydro2d::Primitive &prims);

        hydro2d::Conserved calc_flux(const hydro2d::Primitive &prims, const int ehat = 1);

        hydro2d::Conserved calc_hll_flux(
            const hydro2d::Conserved &left_state,
            const hydro2d::Conserved &right_state,
            const hydro2d::Conserved &left_flux,
            const hydro2d::Conserved &right_flux,
            const hydro2d::Primitive &left_prims,
            const hydro2d::Primitive &right_prims,
            const int ehat);

        hydro2d::Conserved calc_hllc_flux(
            const hydro2d::Conserved &left_state,
            const hydro2d::Conserved &right_state,
            const hydro2d::Conserved &left_flux,
            const hydro2d::Conserved &right_flux,
            const hydro2d::Primitive &left_prims,
            const hydro2d::Primitive &right_prims,
            const int ehat = 1);

        void evolve();
        void adapt_dt();

        std::vector<std::vector<real> > simulate2D(
            const std::vector<std::vector<real>> sources,
            real tstart = 0., 
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