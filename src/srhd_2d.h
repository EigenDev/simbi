/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/

#include <vector>
#include <string>
#include <hydro_structs.h>
#include <clattice.h>
#ifndef SRHD_2D_H
#define SRHD_2D_H

namespace simbi {
    class SRHD2D {
        private:
            int a;
        public:
            /* Shared Data Members */
            hydro2d::Eigenvals lambda;
            std::vector<hydro2d::Primitives> prims; 
            hydro2d::ConserveData u_state; 
            std::vector<std::vector<double> > state2D, sources;
            float tend, tstart;
            double theta, gamma;
            bool first_order, periodic, hllc, linspace;
            double CFL, dt, decay_const;
            int NX, NY, nzones, n, block_size, xphysical_grid, yphysical_grid;
            int active_zones, idx_active, x_bound, y_bound; 
            std::string coord_system;
            std::vector<double> x1, x2, sourceD, source_S1, source_S2, source_tau, pressure_guess;
            std::vector<double> lorentz_gamma, xvertices, yvertices;
            CLattice coord_lattice;

            /* Methods */
            SRHD2D();
            SRHD2D(std::vector<std::vector<double> > state2D, int NX, int NY, double gamma, std::vector<double> x1,
                                std::vector<double> x2,
                                double CFL, std::string coord_system);
            ~SRHD2D();

            hydro2d::Primitives cons2primSR(hydro2d::Conserved  &u_state,
                                    double lorentz_gamma,
                                    std::tuple<int, int>(coordinates));

            std::vector<hydro2d::Primitives> cons2prim2D(const std::vector<hydro2d::Conserved> &cons_state2D,
                                            std::vector<double> &lorentz_gamma);

            hydro2d::Eigenvals  calc_Eigenvals(const hydro2d::Primitives &prims_l,
                                            const hydro2d::Primitives &prims_r,
                                            const unsigned int nhat);

            hydro2d::Conserved  calc_stateSR2D(const hydro2d::Primitives &prims);

            hydro2d::Conserved    calc_hll_state(
                                    const hydro2d::Conserved   &left_state,
                                    const hydro2d::Conserved   &right_state,
                                    const hydro2d::Conserved   &left_flux,
                                    const hydro2d::Conserved   &right_flux,
                                    const hydro2d::Primitives  &left_prims,
                                    const hydro2d::Primitives  &right_prims,
                                    unsigned int nhat);

            hydro2d::Conserved calc_intermed_statesSR2D( const hydro2d::Primitives &prims,
                                                        const hydro2d::Conserved &state,
                                                        double a,
                                                        double aStar,
                                                        double pStar,
                                                        int nhat);

            hydro2d::Conserved      calc_hllc_flux(
                                    const hydro2d::Conserved    &left_state,
                                    const hydro2d::Conserved    &right_state,
                                    const hydro2d::Conserved         &left_flux,
                                    const hydro2d::Conserved         &right_flux,
                                    const hydro2d::Primitives   &left_prims,
                                    const hydro2d::Primitives   &right_prims,
                                    const unsigned int nhat);

            hydro2d::Conserved calc_Flux(const hydro2d::Primitives &prims, unsigned int nhat);

            hydro2d::Conserved   calc_hll_flux(
                            const hydro2d::Conserved    &left_state,
                            const hydro2d::Conserved    &right_state,
                            const hydro2d::Conserved    &left_flux,
                            const hydro2d::Conserved    &right_flux,
                            const hydro2d::Primitives   &left_prims,
                            const hydro2d::Primitives   &right_prims,
                            const unsigned int nhat);

            hydro2d::Conserved  u_dot(unsigned int ii, unsigned int jj);

            std::vector<hydro2d::Conserved>  u_dot2D(const std::vector<hydro2d::Conserved>  &cons_state);

            double adapt_dt(const std::vector<hydro2d::Primitives> &prims);

            std::vector<std::vector<double> >   simulate2D( const std::vector<double> lorentz_gamma,
                                                            const std::vector<std::vector<double> > sources,
                                                            float tstart,
                                                            float tend, 
                                                            double dt,
                                                            double theta,
                                                            double engine_duration,
                                                            double chkpt_interval,
                                                            std::string data_directory,
                                                            bool first_order,  
                                                            bool periodic,  
                                                            bool linspace,
                                                            bool hllc
                                                            );
                                                                    
    };
}

#endif