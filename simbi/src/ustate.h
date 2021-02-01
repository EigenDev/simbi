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

#define _NX 0
#define _NY 0

namespace states {
    class Ustate {
        public: 
            std::vector< std::vector<double> > state, cons_state; 
            std::vector<double> r;
            float theta, gamma, tend, dt;
            bool first_order, periodic, linspace;
            double CFL;
            std::string coord_system;
            Ustate();
            Ustate(std:: vector <std::vector <double> > state, float gamma, double CFL,
                    std::vector<double> r, std::string coord_system);
            ~Ustate();
            std::vector < std::vector<double > > cons2prim1D(
                std::vector < std::vector<double > > cons_state);

            std::vector<std::vector<double> > u_dot1D(std::vector<std::vector<double> > &cons_state, bool first_order,
                bool periodic, float theta, bool linspace, bool hllc);

            std::vector<std::vector<double> > simulate1D(float tend, float dt, float theta, 
                                                            bool first_order, bool periodic, bool linspace, bool hllc);
            double adapt_dt(std::vector<std::vector<double> > &cons_state, 
                                    std::vector<double> &r, bool linspace, bool first_order);
            

    };

    class UstateSR {
        public: 
            std::vector< std::vector<double> > state, cons_state; 
            std::vector<double> r;
            float theta, gamma, tend, dt;
            bool first_order, periodic, linspace;
            double CFL;
            std::string coord_system;
            UstateSR();
            UstateSR(std:: vector <std::vector <double> > state, float gamma, double CFL,
                    std::vector<double> r, std::string coord_system);
            ~UstateSR();
            std::vector < std::vector<double > > cons2prim1D(
                std::vector < std::vector<double > > &cons_state, std::vector<double> &lorentz_gamma);

            std::vector<std::vector<double> > u_dot1D(std::vector<std::vector<double> > &cons_state,
                                                        std::vector<double> &lorentz_gamma, 
                                                        std::vector<std::vector<double> > &sources,
                                                        bool first_order,
                                                        bool periodic, float theta, bool linspace, bool hllc);

            std::vector<std::vector<double> > simulate1D(std::vector<double> &lorentz_gamma, 
                                                            std::vector<std::vector<double> > &sources,
                                                            float tend, float dt, float theta, 
                                                            bool first_order, bool periodic, bool linspace, bool hllc);
            double adapt_dt(std::vector<std::vector<double> > &prims, 
                                     bool linspace, bool first_order, bool periodic);
            

    };

    class Ustate2D {
        public:
        std::vector<std::vector<std::vector<double> > > state2D, cons_state2D;
        float theta, gamma, tend;
        bool first_order, periodic;
        double CFL, dt;
        std::string coord_system;
        std::vector<double> x1, x2;
        Ustate2D();
        Ustate2D(std::vector< std::vector< std::vector<double> > > state2D, float gamma, std::vector<double> x1,
                            std::vector<double> x2, double CFL, std::string coord_system);
        ~Ustate2D();
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

    class UstateSR2D {
        public:

        /* Define Data Structures for the Fluid Properties. */
        struct Conserved
        {
            double D;
            double S1;
            double S2;
            double tau;
            Conserved() : S1(0) {}
            ~Conserved() {}
            double momentum(int nhat)
            {
                if (nhat == 1.){
                    return S1;
                } else {
                    return S2;
                }
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
                if (nhat == 1.){
                    return S1;
                } else {
                    return S2;
                }
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
        ConserveData u_now; 
        std::vector<std::vector<double> > state2D, sources;
        float theta, gamma, tend;
        bool first_order, periodic, hllc, linspace;
        double CFL, dt;
        int NX, NY, nzones, n;
        std::string coord_system;
        std::vector<double> x1, x2, sourceD, source_S1, source_S2, source_tau, pressure_guess;


        /* Methods */
        UstateSR2D();
        UstateSR2D(std::vector<std::vector<double> > state2D, int NX, int NY, float gamma, std::vector<double> x1,
                            std::vector<double> x2,
                            double CFL, std::string coord_system);
        ~UstateSR2D();

        Primitives cons2primSR(float gamma, Conserved  &u_state,
                                 double lorentz_gamma,
                                 std::tuple<int, int>(coordinates));

        PrimitiveData cons2prim2D(
            const ConserveData &cons_state2D,
            const std::vector<double> &lorentz_gamma);

        Eigenvals  calc_Eigenvals(float gamma, Primitives &prims_l,
                                      Primitives &prims_r,
                                      unsigned int nhat);

        Conserved  calc_stateSR2D(float gamma, 
                                    double rho, double vx,
                                    double vy, double pressure);

        Conserved    calc_hll_state(float gamma,
                                Conserved  &left_state,
                                Conserved  &right_state,
                                Flux      &left_flux,
                                Flux      &right_flux,
                                Primitives    &left_prims,
                                Primitives    &right_prims,
                                unsigned int nhat);

        Conserved calc_intermed_statesSR2D(  Primitives &prims,
                                        Conserved &state,
                                        double a,
                                        double aStar,
                                        double pStar,
                                        int nhat);

        Flux      calc_hllc_flux(float gamma,
                                Conserved &left_state,
                                Conserved &right_state,
                                Flux     &left_flux,
                                Flux     &right_flux,
                                Primitives   &left_prims,
                                Primitives   &right_prims,
                                int nhat);

        Flux calc_Flux(float gamma, double rho, double vx, 
                                        double vy, double pressure, 
                                        bool x_direction);

        Flux   calc_hll_flux(float gamma,
                        Conserved &left_state,
                        Conserved &right_state,
                        Flux     &left_flux,
                        Flux     &right_flux,
                        Primitives   &left_prims,
                        Primitives   &right_prims,
                        unsigned int nhat);

        Conserved  u_dot( 
            const ConserveData  &cons_state,
            const std::vector<double>  &lorentz_gamma,
            std::tuple<int , int> coordinates);

        ConserveData  u_dot2D( 
            const ConserveData  &cons_state,
            const std::vector<double>  &lorentz_gamma,
            bool first_order,
            bool periodic, bool linspace, bool hllc, float theta);


        double adapt_dt(const PrimitiveData &prims,
                                   bool linspace, bool first_order);

        std::vector<std::vector<double> >   simulate2D(const std::vector<double> lorentz_gamma,
                              const std::vector<std::vector<double> > sources,
                              float tend,bool first_order,  bool periodic,  
                              bool linspace,
                              bool hllc,
                              double dt);
                                                                
    };
}

#endif 