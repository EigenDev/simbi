/* 
* Interface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#include <vector>
#include <string>


#ifndef CLASSICAL_2D_H
#define CLASSICAL_2D_H

namespace simbi {
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
}

#endif