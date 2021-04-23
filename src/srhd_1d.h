/* 
* Interface between python construction of the 1D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/

#include <vector>
#include <string>
#include <hydro_structs.h>

#ifndef SRHD_1D_H
#define SRHD_1D_H


namespace simbi {
    class SRHD {
        public: 
            double gamma, CFL;
            std::string coord_system;
            std::vector<double> r; 
            std::vector< std::vector<double> > state;

            SRHD();
            SRHD(std:: vector <std::vector <double> > state, double gamma, double CFL,
                    std::vector<double> r, std::string coord_system);
            ~SRHD();


            hydro1d::ConservedArray cons_state; 
            hydro1d::PrimitiveArray prims;

            int Nx, n, pgrid_size, idx_shift;
            float tend, dt;
            double theta, engine_duration, t, decay_constant;
            bool first_order, periodic, linspace, hllc;
            
            std::vector<double> lorentz_gamma, sourceD, sourceS, source0, pressure_guess;

            hydro1d::PrimitiveArray cons2prim1D(const hydro1d::ConservedArray &cons_state, std::vector<double> &lorentz_gamma);

            hydro1d::Eigenvals calc_eigenvals(const hydro1d::Primitive &prims_l, const hydro1d::Primitive &prims_r);

            double adapt_dt(hydro1d::PrimitiveArray &prims);

            hydro1d::Conserved calc_state(double rho, double v, double pressure);

            hydro1d::Conserved calc_hll_state(
                                    const hydro1d::Conserved &left_state,
                                    const hydro1d::Conserved &right_state,
                                    const hydro1d::Flux &left_flux,
                                    const hydro1d::Flux &right_flux,
                                    const hydro1d::Primitive &left_prims,
                                    const hydro1d::Primitive &right_prims);

            hydro1d::Conserved calc_intermed_state(const hydro1d::Primitive &prims,
                                    const hydro1d::Conserved &state,
                                    const double a,
                                    const double aStar,
                                    const double pStar);

            hydro1d::Flux calc_flux(double rho, double v, double pressure);

            hydro1d::Flux calc_hll_flux(const hydro1d::Primitive &left_prims, 
                                        const hydro1d::Primitive &right_prims,
                                        const hydro1d::Conserved &left_state,
                                        const hydro1d::Conserved &right_state,
                                        const hydro1d::Flux &left_flux,
                                        const hydro1d::Flux &right_flux);

            hydro1d::Flux  calc_hllc_flux(  const hydro1d::Conserved &left_state,
                                            const hydro1d::Conserved &right_state,
                                            const hydro1d::Flux &left_flux,
                                            const hydro1d::Flux &right_flux,
                                            const hydro1d::Primitive &left_prims,
                                            const hydro1d::Primitive &right_prims);

            hydro1d::ConservedArray u_dot1D(hydro1d::ConservedArray &u_state);

            std::vector<std::vector<double> > simulate1D(std::vector<double> &lorentz_gamma, 
                                                            std::vector<std::vector<double> > &sources,
                                                            float tstart,
                                                            float tend, 
                                                            float dt, 
                                                            double theta, 
                                                            double engine_duraction,
                                                            double chkpt_interval,
                                                            std::string data_directory,
                                                            bool first_order, bool periodic, bool linspace, bool hllc);

            
            

    };
}

#endif 