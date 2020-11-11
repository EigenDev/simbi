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
#include <iostream>

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
                bool periodic, float theta, bool linspace);

            std::vector<std::vector<double> > simulate1D(float tend, float dt, float theta, 
                                                            bool first_order, bool periodic, bool linspace);
            long double adapt_dt(std::vector<std::vector<double> > &cons_state, 
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
                                                        bool periodic, float theta, bool linspace);

            std::vector<std::vector<double> > simulate1D(std::vector<double> &lorentz_gamma, 
                                                            std::vector<std::vector<double> > &sources,
                                                            float tend, float dt, float theta, 
                                                            bool first_order, bool periodic, bool linspace);
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
            std::vector<std::vector<std::vector<double> > > cons_state2D);

        std::vector<std::vector<std::vector<double> > > u_dot2D(float gamma, 
            std::vector<std::vector<std::vector<double> > >  cons_state,
            bool periodic, float theta, bool linspace);

        double adapt_dt(std::vector<std::vector<std::vector<double> > >  &cons_state,
                                    bool linspace);

        std::vector<std::vector<std::vector<double> > > simulate2D(float tend, bool periodic, double dt, bool linspace);
    };

    class UstateSR2D {
        public:
        std::vector<std::vector<std::vector<double> > > state2D, cons_state2D;
        float theta, gamma, tend;
        bool first_order, periodic;
        double CFL, dt;
        std::string coord_system;
        std::vector<double> x1, x2;
        UstateSR2D();
        UstateSR2D(std::vector< std::vector< std::vector<double> > > state2D, float gamma, std::vector<double> x1,
                            std::vector<double> x2,
                            double CFL, std::string coord_system);
        ~UstateSR2D();
        std::vector<double> cons2prim(float gamma, std::vector<double> &u_state, double lorent_gamma);
        std::vector <std::vector < std::vector<double > > > cons2prim2D(
            std::vector<std::vector<std::vector<double> > > &cons_state2D,
            std::vector<std::vector<double> > &lorentz_gamma);

        std::vector<std::vector<std::vector<double> > > u_dot2D( 
            std::vector<std::vector<std::vector<double> > >  &cons_state,
            std::vector<std::vector<double> > &lorentz_gamma,
            std::vector<std::vector<std::vector<double> > > & sources,
            bool first_order,
            bool periodic, float theta, bool linspace);


        double adapt_dt(std::vector<std::vector<std::vector<double> > >  &prims,
                                   bool linspace, bool first_order);

        std::vector<std::vector<std::vector<double> > > simulate2D(std::vector<std::vector<double> > &lorentz_gamma,
                                                                std::vector<std::vector<std::vector<double> > > & sources,
                                                                float tend,bool first_order,  bool periodic, double dt, bool linspace);
                                                                
    };
}

#endif 