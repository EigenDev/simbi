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

        /* Define Data Structures for the Fluid Properties. */
        struct Conserved
        {
            Conserved() {}
            ~Conserved() {}
            double D, S1, S2, tau;

            Conserved(double D, double S1, double S2, double tau) : D(D), S1(S1), S2(S2), tau(tau) {}  
            Conserved(const Conserved &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau)    {}  
            Conserved operator + (const Conserved &p)  const { return Conserved(D+p.D, S1+p.S1, S2+p.S2, tau+p.tau); }  
            Conserved operator - (const Conserved &p)  const { return Conserved(D-p.D, S1-p.S1, S2-p.S2, tau-p.tau); }  
            Conserved operator * (const double c)      const { return Conserved(D*c, S1*c, S2*c, tau*c ); }
            Conserved operator / (const double c)      const { return Conserved(D/c, S1/c, S2/c, tau/c ); }

            double momentum(int nhat)
            {
                return (nhat == 1 ? S1 : S2);
            }
        };

        struct Flux
        {
            Flux() {}
            ~Flux() {}
            double D, S1, S2, tau;

            Flux(double D, double S1, double S2, double tau) : D(D), S1(S1), S2(S2), tau(tau) {}  
            Flux(const Flux &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau)    {}  
            Flux operator + (const Flux &p)  const { return Flux(D+p.D, S1+p.S1, S2+p.S2, tau+p.tau); }  
            Flux operator - (const Flux &p)  const { return Flux(D-p.D, S1-p.S1, S2+p.S2, tau+p.tau); }  
            Flux operator * (double c)       const { return Flux(D*c,   S1*c, S2*c, tau*c  ); }

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
            ConserveData() {}
            ~ConserveData() {}
            std::vector<double> D, S1, S2, tau;

            ConserveData(std::vector<double>  &D,  std::vector<double>  &S1, 
                          std::vector<double>  &S2, std::vector<double>  &tau) : D(D), S1(S1), S2(S2), tau(tau) {} 

            ConserveData(const ConserveData &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau){} 
            Conserved operator[] (int i) const {return Conserved(D[i], S1[i], S2[i], tau[i]); }

            void swap( ConserveData &c) {
                this->D.swap(c.D); 
                this->S1.swap(c.S1);
                this->S2.swap(c.S2); 
                this->tau.swap(c.tau); 
                };
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
        double CFL, dt, decay_const;
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
                                const Conserved   &left_flux,
                                const Conserved   &right_flux,
                                const Primitives  &left_prims,
                                const Primitives  &right_prims,
                                unsigned int nhat);

        Conserved calc_intermed_statesSR2D( const Primitives &prims,
                                            const Conserved &state,
                                            double a,
                                            double aStar,
                                            double pStar,
                                            int nhat);

        Conserved      calc_hllc_flux(
                                const Conserved    &left_state,
                                const Conserved    &right_state,
                                const Conserved         &left_flux,
                                const Conserved         &right_flux,
                                const Primitives   &left_prims,
                                const Primitives   &right_prims,
                                const unsigned int nhat);

        Conserved calc_Flux(double rho, double vx, 
                            double vy, double pressure, 
                            unsigned int nhat);

        Conserved   calc_hll_flux(
                        const Conserved    &left_state,
                        const Conserved    &right_state,
                        const Conserved    &left_flux,
                        const Conserved    &right_flux,
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