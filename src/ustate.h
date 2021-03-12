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
            double theta;
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
                                                            float tend, float dt, double theta, 
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

        /* Define Data Structures for the Fluid Properties. */
        struct Conserved
        {
            double D;
            double S1;
            double S2;
            double tau;
            Conserved() {}
            ~Conserved() {}
            double momentum(int nhat)
            {
                return (nhat == 1 ? S1 : S2);
            }
        };

        struct Flux
        {
            double D;
            double S1;
            double S2;
            double tau;

            Flux() {}
            ~Flux() {}
            double momentum(int nhat)
            {
                return (nhat == 1 ? S1 : S2);
            }
        };

        struct Primitives {
            double rho;
            double v1;
            double v2;
            double p;
        };

        struct Eigenvals{
            double aL;
            double aR;
        };
        struct ConserveData
        {
            std::vector<double> D, S1, S2, tau;
        };

        struct PrimitiveData
        {
            std::vector<double> rho, v1, v2, p;
        };

        /* Shared Data Members */
        Eigenvals lambda;
        PrimitiveData prims; 
        ConserveData u_state; 
        std::vector<std::vector<double> > state2D, sources;
        float tend, tstart;
        double theta, gamma;
        bool first_order, periodic, hllc, linspace;
        double CFL, dt;
        int NX, NY, nzones, n, block_size, xphysical_grid, yphysical_grid;
        int active_zones, idx_active, x_bound, y_bound; 
        std::string coord_system;
        std::vector<double> x1, x2, sourceD, source_S1, source_S2, source_tau, pressure_guess;
        std::vector<double> lorentz_gamma;

        /* Methods */
        SRHD2D();
        SRHD2D(std::vector<std::vector<double> > state2D, int NX, int NY, double gamma, std::vector<double> x1,
                            std::vector<double> x2,
                            double CFL, std::string coord_system);
        ~SRHD2D();

        Primitives cons2primSR(Conserved  &u_state,
                                 double lorentz_gamma,
                                 std::tuple<int, int>(coordinates));

        PrimitiveData cons2prim2D( const ConserveData &cons_state2D,
                                   std::vector<double> &lorentz_gamma);

        Eigenvals  calc_Eigenvals(const Primitives &prims_l,
                                  const Primitives &prims_r,
                                  const unsigned int nhat);

        Conserved  calc_stateSR2D(double rho, double vx,
                                  double vy, double pressure);

        Conserved    calc_hll_state(
                                const Conserved   &left_state,
                                const Conserved   &right_state,
                                const Flux        &left_flux,
                                const Flux        &right_flux,
                                const Primitives  &left_prims,
                                const Primitives  &right_prims,
                                unsigned int nhat);

        Conserved calc_intermed_statesSR2D( const Primitives &prims,
                                            const Conserved &state,
                                            double a,
                                            double aStar,
                                            double pStar,
                                            int nhat);

        Flux      calc_hllc_flux(
                                const Conserved    &left_state,
                                const Conserved    &right_state,
                                const Flux         &left_flux,
                                const Flux         &right_flux,
                                const Primitives   &left_prims,
                                const Primitives   &right_prims,
                                const unsigned int nhat);

        Flux calc_Flux(double rho, double vx, 
                            double vy, double pressure, 
                            bool x_direction);

        Flux   calc_hll_flux(
                        const Conserved    &left_state,
                        const Conserved    &right_state,
                        const Flux         &left_flux,
                        const Flux         &right_flux,
                        const Primitives   &left_prims,
                        const Primitives   &right_prims,
                        const unsigned int nhat);

        Conserved  u_dot(unsigned int ii, unsigned int jj);

        ConserveData  u_dot2D(const ConserveData  &cons_state);


        double adapt_dt(const PrimitiveData &prims);

        std::vector<std::vector<double> >   simulate2D( const std::vector<double> lorentz_gamma,
                                                        const std::vector<std::vector<double> > sources,
                                                        float tstart,
                                                        float tend, 
                                                        double dt,
                                                        double theta,
                                                        double chkpt_interval,
                                                        bool first_order,  
                                                        bool periodic,  
                                                        bool linspace,
                                                        bool hllc
                                                        );
                                                                
    };
}

#endif 