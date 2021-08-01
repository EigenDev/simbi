/* 
* Interface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef CLASSICAL_2D_HPP
#define CLASSICAL_2D_HPP

#include <vector>
#include <string>
#include "hydro_structs.hpp"
#include "clattice.hpp"
#include "config.hpp"



namespace simbi {
    class Newtonian2D {
        public:
        std::vector<std::vector<real> > init_state, sources;
        std::vector<hydro2d::Conserved> cons_state2D;
        std::vector<hydro2d::Primitive> prims;
        std::vector<real> sourceRho, sourceM1, sourceM2, sourceE;
        real theta, gamma, tend, CFL, dt;
        bool first_order, periodic, hllc, linspace;
        std::string coord_system;
        std::vector<real> x1, x2;
        int nzones, NY, NX, idx_shift, active_zones;
        int xphysical_grid, yphysical_grid, x_bound, y_bound;
        CLattice2D coord_lattice;
        simbi::Solver solver;

        Newtonian2D();
        Newtonian2D(std::vector< std::vector<real> > init_state, 
            int NX, 
            int NY,
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            real CFL, 
            std::string coord_system);

        ~Newtonian2D();

        std::vector<hydro2d::Primitive> cons2prim(const std::vector<hydro2d::Conserved > &cons_state2D);

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

        std::vector<hydro2d::Conserved> u_dot(const std::vector<hydro2d::Conserved> &cons_state);

        real adapt_dt(const std::vector<hydro2d::Primitive>  &prims);

        std::vector<std::vector<real> > simulate2D(
            const std::vector<std::vector<real> >  &sources,
            real tend, 
            bool periodic, 
            real dt, 
            bool linspace, 
            bool hllc,
            real theta = 1.5);
        
    };
}

#endif