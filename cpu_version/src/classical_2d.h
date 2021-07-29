/* 
* Interface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef CLASSICAL_2D_H
#define CLASSICAL_2D_H

#include <vector>
#include <string>
#include "hydro_structs.h"
#include "clattice.h"
#include "config.h"



namespace simbi {
    class Newtonian2D {
        public:
        std::vector<std::vector<double> > init_state, sources;
        std::vector<hydro2d::Conserved> cons_state2D;
        std::vector<hydro2d::Primitive> prims;
        std::vector<double> sourceRho, sourceM1, sourceM2, sourceE;
        double theta, gamma, tend, CFL, dt;
        bool first_order, periodic, hllc, linspace;
        std::string coord_system;
        std::vector<double> x1, x2;
        int nzones, NY, NX, idx_shift, active_zones;
        int xphysical_grid, yphysical_grid, x_bound, y_bound;
        CLattice coord_lattice;
        simbi::Solver solver;

        Newtonian2D();
        Newtonian2D(std::vector< std::vector<double> > init_state, 
            int NX, 
            int NY,
            double gamma, 
            std::vector<double> x1,
            std::vector<double> x2, 
            double CFL, 
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

        double adapt_dt(const std::vector<hydro2d::Primitive>  &prims);

        std::vector<std::vector<double> > simulate2D(
            const std::vector<std::vector<double> >  &sources,
            double tend, 
            bool periodic, 
            double dt, 
            bool linspace, 
            bool hllc,
            double theta = 1.5);
        
    };
}

#endif