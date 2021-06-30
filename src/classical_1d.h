/* 
* Interface between python construction of the 1D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef CLASSICAL_1D_H
#define CLASSICAL_1D_H

#include <vector>
#include <string>
#include <hydro_structs.h>
#include "clattice_1d.hpp"

namespace simbi {
    enum class SOLVER{HLLE, HLLC};

    class Newtonian1D {
        public: 
            std::vector<std::vector<double>> init_state;
            std::vector<hydro1d::Conserved> cons_state; 
            std::vector<double> r, xvertices;
            double theta, gamma, tend, dt, CFL;
            bool first_order, periodic, linspace, hllc;
            int nzones, active_zones, idx_shift;
            std::string coord_system;
            simbi::SOLVER sim_solver;
            CLattice1D coord_lattice;

            std::vector<hydro1d::Primitive> prims;
            Newtonian1D();
            Newtonian1D(std:: vector <std::vector <double> > init_state, 
                        double gamma, double CFL,std::vector<double> r, 
                        std::string coord_system);
            ~Newtonian1D();

            // Calculate the wave speeds from the Jacobian Matrix formed by the Euler Eqns
            hydro1d::Eigenvals calc_eigenvals(const hydro1d::Primitive &left_state, const hydro1d::Primitive &right_state);

            std::vector<hydro1d::Primitive> cons2prim(const std::vector < hydro1d::Conserved > &u_state);

            hydro1d::Conserved prims2cons(const hydro1d::Primitive &prims);

            hydro1d::Conserved calc_flux(const hydro1d::Primitive &prims);

            hydro1d::Conserved calc_hll_flux(
                                        const hydro1d::Conserved &left_state,
                                        const hydro1d::Conserved &right_state,
                                        const hydro1d::Conserved &left_flux,
                                        const hydro1d::Conserved &right_flux,
                                        const hydro1d::Primitive &left_prims,
                                        const hydro1d::Primitive &right_prims);

            hydro1d::Conserved calc_hllc_flux(
                                const hydro1d::Conserved &left_state,
                                const hydro1d::Conserved &right_state,
                                const hydro1d::Conserved &left_flux,
                                const hydro1d::Conserved &right_flux,
                                const hydro1d::Primitive &left_prims,
                                const hydro1d::Primitive &right_prims
                                );

            std::vector<hydro1d::Conserved> u_dot(std::vector<hydro1d::Conserved> &cons_state);

            std::vector<std::vector<double> > simulate1D(
                                                        float tend, 
                                                        float dt, 
                                                        float theta, 
                                                        bool first_order, 
                                                        bool periodic, 
                                                        bool linspace, 
                                                        bool hllc);

            double adapt_dt(std::vector<hydro1d::Primitive> &prims);
            

    };
}

#endif