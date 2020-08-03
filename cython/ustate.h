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
            float theta, gamma, tend;
            bool first_order, periodic;
            Ustate();
            Ustate(std:: vector <std::vector <double> > state, float gamma);
            ~Ustate();
            std::vector < std::vector<double > > cons2prim1D(
                std::vector < std::vector<double > > cons_state);

            std::vector<std::vector<double> > u_dot1D(float gamma, 
                std::vector<std::vector<double> > cons_state, bool first_order,
                bool periodic, float theta);

            std::vector<std::vector<double> > simulate1D(float tend, bool first_order, 
                                                            bool periodic);
            

    };

    class Ustate2D {
        public:
        std::vector<std::vector<std::vector<double> > > state2D, cons_state2D;
        float theta, gamma, tend;
        bool first_order, periodic;
        Ustate2D();
        Ustate2D(std::vector< std::vector< std::vector<double> > > state2D, float gamma);
        ~Ustate2D();
        std::vector <std::vector < std::vector<double > > > cons2prim2D(
            std::vector<std::vector<std::vector<double> > > cons_state2D);

        std::vector<std::vector<std::vector<double> > > u_dot2D(float gamma, 
            std::vector<std::vector<std::vector<double> > >  cons_state,
            bool periodic, float theta);

        std::vector<std::vector<std::vector<double> > > simulate2D(float tend, bool periodic);
    };
}

#endif 