/* 
* Passing the state tensor between cython and cpp 
* This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself. Will be later updated to 
* include the HLLC flux computation.
*/

#ifndef USTATE_H
#define USTATE_H 

#include <vector>
#include <array>
#include <iostream>
#include <tuple>
#include "hydro_structs.h"

namespace hydro {
    class Newtonian1D {
        public: 
            std::vector< std::vector<double> > state, cons_state; 
            std::vector<double> r;
            float theta, gamma, tend, dt;
            bool first_order, periodic, linspace;
            double CFL;
            std::string coord_system;
            Newtonian1D();
            Newtonian1D(std:: vector <std::vector <double> > state, float gamma, double CFL,
                    std::vector<double> r, std::string coord_system);
            ~Newtonian1D();
            std::vector < std::vector<double > > cons2prim1D(
                std::vector < std::vector<double > > cons_state);

            std::vector<std::vector<double> > u_dot1D(std::vector<std::vector<double> > &cons_state, bool first_order,
                bool periodic, float theta, bool linspace, bool hllc);

            std::vector<std::vector<double> > simulate1D(float tend, float dt, float theta, 
                                                            bool first_order, bool periodic, bool linspace, bool hllc);
            double adapt_dt(std::vector<std::vector<double> > &cons_state, 
                                    std::vector<double> &r, bool linspace, bool first_order);
            

    };

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

    class Newtonian2D {
        public:
        std::vector<std::vector<std::vector<double> > > state2D, cons_state2D;
        float theta, gamma, tend;
        bool first_order, periodic;
        double CFL, dt;
        std::string coord_system;
        std::vector<double> x1, x2;
        Newtonian2D();
        Newtonian2D(std::vector< std::vector< std::vector<double> > > state2D, float gamma, std::vector<double> x1,
                            std::vector<double> x2, double CFL, std::string coord_system);
        ~Newtonian2D();
        std::vector <std::vector < std::vector<double > > > cons2prim2D(
            std::vector<std::vector<std::vector<double> > > &cons_state2D);

        std::vector <std::vector < std::vector<double > > > cons2prim2D(
            std::vector<std::vector<std::vector<double> > > &cons_state2D, int xcoord, int ycoord);

        std::vector<double> u_dot2D1(float gamma, 
                                        std::vector<std::vector<std::vector<double> > >  &cons_state,
                                        std::vector<std::vector<std::vector<double> > >  &sources,
                                        int ii, int jj, bool periodic, float theta, bool linspace, bool hllc);

        std::vector<std::vector<std::vector<double> > > u_dot2D(float gamma, 
            std::vector<std::vector<std::vector<double> > >  &cons_state,
            std::vector<std::vector<std::vector<double> > >  &sources,
            bool periodic, float theta, bool linspace, bool hllc);

        double adapt_dt(std::vector<std::vector<std::vector<double> > >  &cons_state,
                                    bool linspace);

        std::vector<std::vector<std::vector<double> > > simulate2D(std::vector<std::vector<std::vector<double> > >  &sources,
                                    float tend, bool periodic, double dt, bool linspace, bool hllc);
    };

    class SRHD2D {
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