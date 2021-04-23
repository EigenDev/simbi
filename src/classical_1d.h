/* 
* Interface between python construction of the 1D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/

#include <vector>
#include <string>


#ifndef CLASSICAL_1D_H
#define CLASSICAL_1D_H

namespace simbi {
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
}

#endif