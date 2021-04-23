/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "classical_2d.h" 
#include "helper_functions.h"
#include <cmath>
#include <map>
#include <algorithm>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace simbi;
using namespace chrono;

// Default Constructor 
Newtonian2D::Newtonian2D () {}

// Overloaded Constructor
Newtonian2D::Newtonian2D(vector<vector< vector<double> > > u_state2D, float Gamma, vector<double> X1, 
                    vector<double> X2, double cfl, string Coord_system = "cartesian")
{
    state2D = u_state2D;
    gamma = Gamma;
    x1 = X1;
    x2 = X2;
    CFL = cfl;
    coord_system = Coord_system;
}

// Destructor 
Newtonian2D::~Newtonian2D() {}


//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
vector<double>  cons2prim(float gamma, vector<double>  &u_state, string direction = "x"){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, momx, momy;
    vector<double> prims(4);
    double vx, vy, vtot, pressure;

    rho = u_state[0];
    momx = u_state[1];
    momy = u_state[2];
    energy = u_state[3];

    vx = momx/rho;
    vy = momy/rho;

    vtot = sqrt(vx*vx + vy*vy);

    pressure = calc_pressure(gamma, rho, energy, vtot);
    
    prims[0] = rho;
    prims[1] = pressure;
    prims[2] = vx;
    prims[3] = vy;

    return prims;
};

vector<vector< vector<double> > > Newtonian2D::cons2prim2D(vector<vector< vector<double> > > &u_state2D){
    /**
     * Return a 1 + 2D matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, momx, momy;
    double vx,vy, pressure, vtot;
     
    int n_vars = u_state2D.size();
    int ny_gridpts = u_state2D[0].size();
    int nx_gridpts = u_state2D[0][0].size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (ny_gridpts, vector<double>(nx_gridpts)));
   

    for (int jj=0; jj < ny_gridpts; jj++){
        for(int ii=0; ii< nx_gridpts; ii++){
            rho = u_state2D[0][jj][ii];       // Density
            momx = u_state2D[1][jj][ii];      // X-Momentum
            momy = u_state2D[2][jj][ii];      // Y-Momentum
            energy = u_state2D[3][jj][ii];    // Energy

            vx = momx/rho;
            vy = momy/rho;

            vtot = sqrt( vx*vx + vy*vy );

            pressure = calc_pressure(gamma, rho, energy, vtot);

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = pressure;
            prims[2][jj][ii] = vx;
            prims[3][jj][ii] = vy;
            

        }
    }
    

    return prims;
};

vector<vector< vector<double> > > Newtonian2D::cons2prim2D(vector<vector< vector<double> > > &u_state2D,
                                                        int xcoord, int ycoord){
    /**
     * Return a 1 + 2D subspace matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, momx, momy;
    double vx,vy, pressure, vtot;
     
    int n_vars = u_state2D.size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (5, vector<double>(5)));
   

    for (int jj=ycoord - 2; jj < ycoord + 3; jj++){
        for(int ii=xcoord - 2; ii< xcoord + 3; ii++){
            rho = u_state2D[0][jj][ii];       // Density
            momx = u_state2D[1][jj][ii];      // X-Momentum
            momy = u_state2D[2][jj][ii];      // Y-Momentum
            energy = u_state2D[3][jj][ii];    // Energy

            vx = momx/rho;
            vy = momy/rho;

            vtot = sqrt( vx*vx + vy*vy );

            pressure = calc_pressure(gamma, rho, energy, vtot);

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = pressure;
            prims[2][jj][ii] = vx;
            prims[3][jj][ii] = vy;
            

        }
    }
    

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------

map<string, map<string, double > > calc_eigenvals(float gamma, vector<double> &left_state,
                                            vector<double> &right_state, string direction = "x",
                                            bool contact = false, float dummy=1)
{
    // Initialize your important variables
    double vx_r, vx_l, vy_l, vy_r, p_r, p_l, cs_r, cs_l, vtot_l, vtot_r, pStar, aStar; 
    double rho_l, rho_r,  momx_l, momx_r, momy_l, momy_r, energy_l, energy_r;
    double cbar, z, aL, aR, qL, qR, rhoBar;
    map<string, map<string, double > > lambda;
    string default_direction = "x";
    string coord_system = "cartesian";

    // Separate the left and right state components 
    rho_l = left_state[0];
    momx_l = left_state[1];
    momy_l = left_state[2];
    energy_l = left_state[3];

    rho_r = right_state[0];
    momx_r = right_state[1];
    momy_r = right_state[2];
    energy_r = right_state[3];

    vx_l = momx_l/rho_l;
    vx_r = momx_r/rho_r;

    vy_l = momy_l/rho_l;
    vy_r = momy_r/rho_r;
        
    vtot_l = sqrt( vx_l*vx_l + vy_l*vy_l );
    vtot_r = sqrt( vx_r*vx_r + vy_r*vy_r );

    
    p_r = calc_pressure(gamma, rho_r, energy_r, vtot_r);
    p_l = calc_pressure(gamma, rho_l, energy_l, vtot_l);
    

    cs_r = calc_sound_speed(gamma, rho_r, p_r);
    cs_l = calc_sound_speed(gamma, rho_l, p_l);

    // Populate the lambda dictionary
    if (direction == default_direction){
        // Calculate the mean velocities of sound and fluid
        cbar = 0.5*(cs_l + cs_r);
        rhoBar = 0.5*(rho_l + rho_r);
        z = (gamma - 1.)/(2.*gamma);
        double num = cs_l + cs_r - ( gamma-1.)/2 *(vx_r - vx_l);
        double denom = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
        double p_term = num/denom;
        pStar = pow(p_term, (1./z));

        if (pStar <= p_l){
            qL = 1.;
        } else {
            qL = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));
        }

        if (pStar <= p_r){
            qR = 1.;
        } else {
            qR = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));
        }
        aL = vx_l - qL*cs_l;
        aR = vx_r + qR*cs_r;

        aStar = ( (p_r - p_l + rho_l*vx_l*(aL - vx_l) - rho_r*vx_r*(aR - vx_r))/
                     (rho_l*(aL - vx_l) - rho_r*(aR - vx_r) ) );


        // cout << "PStar in x: "<< pStar << endl;
        // cin.get();
        // cout << pStar<< endl;
        // cin.get();

        // cout << "Vbar: " << vbar << endl;
        // cout << "Other: " <<  0.5*(p_l - p_r)/(rhoBar*cbar) << endl;
        

        if (contact){
            double qBar = 1.0;
            lambda["left"]["plus"] = vx_l + cs_l; 
            lambda["left"]["minus"] = vx_l - cs_l; 
            lambda["right"]["plus"] = vx_r + cs_r; 
            lambda["right"]["minus"] = vx_r - cs_r; 

            lambda["signal"]["aL"] = vx_l - qL*cs_l; 
            lambda["signal"]["aR"] = vx_r + qR*cs_r; 
            lambda["signal"]["aStar"] = aStar;
            lambda["signal"]["pStar"] = pStar;

        } else {
            lambda["left"]["plus"] = vx_l + cs_l; 
            lambda["left"]["minus"] = vx_l - cs_l; 
            lambda["right"]["plus"] = vx_r + cs_r; 
            lambda["right"]["minus"] = vx_r - cs_r;  

        }
    } else {
        cbar = 0.5*(cs_l + cs_r);
        rhoBar = 0.5*(rho_l + rho_r);
        z = (gamma - 1.)/(2.*gamma);
        double num = cs_l + cs_r - ( gamma-1.)/2 *(vy_r - vy_l);
        double denom = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
        double p_term = num/denom;
        pStar = pow(p_term, (1./z));

        if (pStar <= p_l){
            qL = 1.;
        } else {
            qL = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));
        }

        if (pStar <= p_r){
            qR = 1.;
        } else {
            qR = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));
        }

        aL = vy_l - qL*cs_l;
        aR = vy_r + qR*cs_r;

        aStar = ( (p_r - p_l + rho_l*vy_l*(aL - vy_l) - rho_r*vy_r*(aR - vy_r))/
                     (rho_l*(aL - vy_l) - rho_r*(aR - vy_r) ) );


        // cout << "PStar in y: "<< pStar << endl;
        // cin.get();

        // cout << "Vbar: " << vbar << endl;
        // cout << "Other: " <<  0.5*(p_l - p_r)/(rhoBar*cbar) << endl;
        

        if (contact){
            lambda["left"]["plus"] = vy_l + cs_l; 
            lambda["left"]["minus"] = vy_l - cs_l; 
            lambda["right"]["plus"] = vy_r + cs_r; 
            lambda["right"]["minus"] = vy_r - cs_r; 

            lambda["signal"]["aL"] = aL; 
            lambda["signal"]["aR"] = aR; 
            lambda["signal"]["aStar"] = aStar;
            lambda["signal"]["pStar"] = pStar;

        } else {
            lambda["left"]["plus"] = vy_l + cs_l; 
            lambda["left"]["minus"] = vy_l - cs_l; 
            lambda["right"]["plus"] = vy_r + cs_r; 
            lambda["right"]["minus"] = vy_r - cs_r;  

        }
    }
        
    return lambda;

    
    
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state tensor for computation. 
// It is being doing pointwise in this case as opposed to over
// the entire array since we are in c++
 vector<double>  calc_state2D(float gamma, double rho, double pressure, double vx, double vy)
 {
    double vtot = sqrt( vx*vx + vy*vy );
    vector<double>  cons_state(4);
    double energy = calc_energy(gamma, rho, pressure, vtot);
    
    
    
    cons_state[0] = rho; 
    cons_state[1] = rho*vx;
    cons_state[2] = rho*vy;
    cons_state[3] = energy;
        
        
    return cons_state;
};

vector<double> calc_intermed_state2D(vector<double> &prims,
                                vector<double> &state,
                                double a,
                                double aStar,
                                double pStar,
                                int index=1)
{
    double rho, energy, pressure, v1, v2, m1, m2, cofactor;
    double rhoStar, m1star, m2star, v2star, eStar;
    map<string, map<string, double > > lambda; 
    vector<double> hll_states(4);

    pressure = prims[1];
    v1 = prims[2];
    v2 = prims[3];
    rho = state[0];
    m1 = state[1];
    m2 = state[2];
    energy = state[3];

    if (index == 1){
        rhoStar = ( (a - v1)/(a - aStar))*rho;
        m1star = (1./(a-aStar))*(m1*(a - v1) - pressure + pStar);
        m2star = (a - v1)/(a - aStar)*m2;
        eStar = (1./(a-aStar))*(energy*(a - v1) + pStar*aStar - pressure*v1);

        // Toro's
        // cofactor = rho*((a - v1)/(a - aStar));
        // rhoStar = cofactor;
        // m1star = cofactor*aStar;
        // m2star = cofactor*v2;
        // eStar = cofactor * (energy/rho + (aStar - v1)*(aStar + pressure/(rho*(a - v1))));
    } else {
        rhoStar = ((a - v2)/(a - aStar))*rho;
        m1star = ((a - v2)/(a - aStar))*m1; 
        m2star = (1./(a - aStar))*(m2*(a - v2) - pressure + pStar);
        eStar = (1./(a - aStar))*(energy*(a - v2) + pStar*aStar - pressure*v2);

        // Toro's
        // cofactor = rho*((a - v2)/(a - aStar));
        // rhoStar = cofactor;
        // m1star = cofactor*v1;
        // m2star = cofactor*aStar;
        // eStar = cofactor * (energy/rho + (aStar - v2)*(aStar + pressure/(rho*(a - v2))));
    }
        
    // Try Toro's
    // m1star = rho*( (a - v1)/(a - aStar))*aStar;
    // eStar = rho*( (a - v1)/(a - aStar) ) * (energy/rho + (aStar - v1)*(aStar + pressure/(rho*(a - v1))));
    hll_states[0] = rhoStar;
    hll_states[1] = m1star;
    hll_states[2] = m2star;
    hll_states[3] = eStar;
    
    return hll_states;
}

vector<double> calc_hll_state2D(float gamma,
                                vector<double> &left_state,
                                vector<double> &right_state,
                                vector<double> &left_flux,
                                vector<double> &right_flux,
                                string direction = "x")
{
    double aL, aR;
    double D, S1, S2, tau;
    double Dstar, S1star, S2star, tauStar;
    map<string, map<string, double > > lambda; 
    vector<double> hll_states(4);

    lambda = calc_eigenvals(gamma, left_state, right_state, direction, true);

    aL = lambda["signal"]["aL"];
    aR = lambda["signal"]["aR"];

    hll_states[0] = ( aR*right_state[0] - aL*left_state[0] 
                        - right_flux[0] + left_flux[0])/(aR - aL);

    hll_states[1] = ( aR*right_state[1] - aL*left_state[1] 
                        - right_flux[1] + left_flux[1])/(aR - aL);

    hll_states[2] = ( aR*right_state[2] - aL*left_state[2] 
                        - right_flux[2] + left_flux[2])/(aR - aL);

    hll_states[3] = ( aR*right_state[3] - aL*left_state[3] 
                        - right_flux[3] + left_flux[3])/(aR - aL);



    return hll_states;

}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
double Newtonian2D::adapt_dt(vector<vector<vector<double> > > &u_state,
                        bool linspace=true){

    double r_left, r_right, left_cell, right_cell, upper_cell, lower_cell;
    double dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg;
    double delta_logr, log_rLeft, log_rRight, min_dt, cfl_dt;
    int shift_i, shift_j, x1physical_grid, x2physical_grid;

    // Get the primitive vector 
    vector<vector<vector<double> > >  prims = cons2prim2D(u_state);

    int x1grid_size = prims[0][0].size();
    int x2grid_size = prims[0].size();

    min_dt = 0;
    // Find the minimum timestep over all i
    x1physical_grid = x1grid_size - 4;
    x2physical_grid = x2grid_size - 4;


    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < x2physical_grid; jj ++){
        for (int ii = 0; ii < x1physical_grid; ii++){
            shift_i = ii + 2;
            shift_j = jj + 2;

            // Find the left and right cell centers in one direction
            if (ii - 1 < 0){
                left_cell = x1[ii];
                right_cell = x1[ii + 1];
            }
            else if (ii + 1 > x1physical_grid - 1){
                right_cell = x1[ii];
                left_cell = x1[ii - 1];
            } else {
                right_cell = x1[ii + 1];
                left_cell = x1[ii];
            }

            if (jj - 1 < 0){
                lower_cell = x2[jj];
                upper_cell = x2[jj + 1];
            }
            else if (jj + 1 > x2physical_grid - 1){
                upper_cell = x2[jj];
                lower_cell = x2[jj - 1];
            } else {
                upper_cell = x2[jj + 1];
                lower_cell = x2[jj];
            }

            // Check if using linearly-spaced grid or logspace
            if (linspace){
                r_right = 0.5*(right_cell + x1[ii]);
                r_left = 0.5*(x1[ii] + left_cell);

                if (coord_system == "cartesian"){
                    x2_right = 0.5*(upper_cell + x2[jj]);
                    x2_left = 0.5*(lower_cell + x2[jj]);
                } else {
                    x2_right = atan2 ( sin(upper_cell) + sin(x2[jj]), cos(upper_cell) + cos(x2[jj]) );
                    x2_left = atan2( sin(lower_cell) + sin(x2[jj]), cos(lower_cell) + cos(x2[jj]) );      
                }

            } else {
                // r_right = right_cell; //0.5*(right_cell + x1[ii]);
                // r_left = left_cell; //0.5*(x1[ii] + left_cell);

                log_rLeft = log10(x1[0]) + ii*delta_logr;
                log_rRight = log_rLeft + delta_logr;
                r_right = pow(10, log_rRight);
                r_left = pow(10, log_rLeft);

                if (coord_system == "cartesian"){
                    x2_right = 0.5*(upper_cell + x2[jj]);
                    x2_left = 0.5*(lower_cell + x2[jj]);
                } else {
                    x2_right = atan2 ( sin(upper_cell) + sin(x2[jj]), cos(upper_cell) + cos(x2[jj]) );
                    x2_left = atan2( sin(lower_cell) + sin(x2[jj]), cos(lower_cell) + cos(x2[jj]) );      
                }
            }

            dx1 = r_right - r_left;
            dx2 = x2_right - x2_left;
            rho = prims[0][shift_j][shift_i];
            v1 = prims[2][shift_j][shift_i];
            v2 = prims[3][shift_j][shift_i];
            pressure = prims[1][shift_j][shift_i];
            cs = calc_sound_speed(gamma, rho, pressure);

            if (coord_system == "cartesian"){
                
                cfl_dt = min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

            } else {
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                cfl_dt = min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), volAvg*dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

            }

            
            if ((ii > 0) | (jj > 0) ){
                min_dt = min(min_dt, cfl_dt);
            }
            else {
                min_dt = cfl_dt;
            }
        }
        
    }

    return CFL*min_dt;
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------


// Get the 2D Flux array (4,1). Either return F or G depending on directional flag
vector<double> calc_flux(float gamma, double rho, double pressure, 
                                        double vx, double vy, bool x_direction=true){
    
    // The Flux Tensor
    vector<double> flux(4);

     // The Flux components
    double momx, momy, convect_xy, energy_dens, zeta;

    double vtot = sqrt(vx*vx + vy*vy );
    double energy = calc_energy(gamma, rho, pressure, vtot);

    


    // Check if we're calculating the x-direction flux. If not, calculate the y-direction
    if (x_direction){
        momx = rho*vx;
        convect_xy = rho*vx*vy;
        energy_dens = rho*vx*vx + pressure;
        zeta = (energy + pressure)*vx;

        flux[0] = momx;
        flux[1] = energy_dens;
        flux[2] = convect_xy;
        flux[3] = zeta;
           
        return flux;
    } else {
        momy = rho*vy;
        convect_xy = rho*vx*vy;
        energy_dens = rho*vy*vy + pressure;
        zeta = (energy + pressure)*vy;

        flux[0] = momy;
        flux[1] = convect_xy;
        flux[2] = energy_dens;
        flux[3] = zeta;
           
        return flux;
    }
    
};


vector<double> calc_hll_flux(float gamma, vector<double> &left_state,
                                        vector<double> &right_state,
                                        vector<double> &left_flux,
                                        vector<double> &right_flux,
                                        string direction = "x",
                                        bool hllc = false)
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(4);
    double alpha_plus, alpha_minus;  

    if (hllc){
        lambda = calc_eigenvals(gamma, left_state, right_state, direction, true);
        double aL, aR;
        aL = lambda["signal"]["aL"];
        aR = lambda["signal"]["aR"];
        // Calculate /pm alphas
        alpha_plus = max(0.0, aR);
        alpha_minus = max(0.0 ,-aL);

    } else {
        lambda = calc_eigenvals(gamma, left_state, right_state, direction);
        // Calculate /pm alphas
        alpha_plus = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
        alpha_minus = findMax(0 , -lambda["left"]["minus"], -lambda["right"]["minus"]);
    }

    // Compute the HLL Flux component-wise
    hll_flux[0] = ( alpha_plus*left_flux[0] + alpha_minus*right_flux[0]
                            - alpha_minus*alpha_plus*(right_state[0] - left_state[0] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[1] = ( alpha_plus*left_flux[1] + alpha_minus*right_flux[1]
                            - alpha_minus*alpha_plus*(right_state[1] - left_state[1] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[2] = ( alpha_plus*left_flux[2] + alpha_minus*right_flux[2]
                            - alpha_minus*alpha_plus*(right_state[2] - left_state[2]) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[3] = ( alpha_plus*left_flux[3] + alpha_minus*right_flux[3]
                            - alpha_minus*alpha_plus*(right_state[3] - left_state[3]) )  /
                            (alpha_minus + alpha_plus);

    return hll_flux;
};

vector<double> calc_hllc_flux2D(float gamma,
                                vector<double> &left_state,
                                vector<double> &right_state,
                                vector<double> &left_flux,
                                vector<double> &right_flux,
                                vector<double> &left_prims,
                                vector<double> &right_prims,
                                string direction = "x",
                                int idx = 1
                                )
{
    map<string, map<string, double > > lambda; 
    vector<double> interflux_left(4);
    vector<double> interflux_right(4);
    vector<double> interstate_left(4);
    vector<double> interstate_right(4);
    vector<double> hll_flux(4);
    vector<double> hll_state(4);
    vector<double> hllc_flux(4);
    // vector<double> hll_prims(4);
    double aL, aR, aStar, alpha_plus, alpha_minus, pStar;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state, direction, true);

    aL = lambda["signal"]["aL"];
    aR = lambda["signal"]["aR"];
    aStar = lambda["signal"]["aStar"];
    pStar = lambda["signal"]["pStar"];

    interstate_left = calc_intermed_state2D(left_prims, left_state,
                                                aL, aStar, pStar, idx);

    interstate_right = calc_intermed_state2D(right_prims, right_state,
                                                aR, aStar, pStar, idx);


    // Compute the intermediate left flux
    interflux_left[0]  = left_flux[0] + aL*(interstate_left[0] - left_state[0]);
    interflux_left[1]  = left_flux[1] + aL*(interstate_left[1] - left_state[1]);
    interflux_left[2]  = left_flux[2] + aL*(interstate_left[2] - left_state[2]);
    interflux_left[3]  = left_flux[3] + aL*(interstate_left[3] - left_state[3]);

    // Compute the intermediate right flux
    interflux_right[0] = right_flux[0] + aR*(interstate_right[0] - right_state[0]);
    interflux_right[1] = right_flux[1] + aR*(interstate_right[1] - right_state[1]);
    interflux_right[2] = right_flux[2] + aR*(interstate_right[2] - right_state[2]);
    interflux_right[3] = right_flux[3] + aR*(interstate_right[3] - right_state[3]);

    if (0.0 <= aL){
        return left_flux;
    } else if (aL <= 0.0 && 0.0 <= aStar){
        return interflux_left;
    } else if (aStar <= 0.0 && 0.0 <= aR){
        return interflux_right;
    } else {
        return right_flux;
    }
    
};

//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

vector<double> Newtonian2D::u_dot2D1(float gamma, 
                                        vector<vector<vector<double> > > &u_state,
                                        vector<vector<vector<double> > > &sources, 
                                        int ii,
                                        int jj,
                                        bool periodic = false, float theta = 1.5, 
                                        bool  linspace=true,
                                        bool hllc = false)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate, xcenter, ycenter;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    string default_coordinates = "cartesian";

    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;
    xcenter = xphysical_grid/2 + 2; 
    ycenter = yphysical_grid/2 + 2;

    vector<double> L(n_vars);

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    // Define the source terms
    vector<vector<double> > sourceRho = sources[0];
    vector<vector<double> > sourceM1 = sources[1];
    vector<vector<double> > sourceM2 = sources[2];
    vector<vector<double> > sourceE = sources[3];
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));

    prims = cons2prim2D(u_state);

    /**
    for (int jj = j_start; jj < j_bound; jj++){
        for (int ii = i_start; ii < i_bound; ii++){
            ycoordinate = jj - 2;
            xcoordinate = ii - 2;

            // i+1/2
            ux_l[0] = u_state[0][jj][ii];
            ux_l[1] = u_state[1][jj][ii];
            ux_l[2] = u_state[2][jj][ii];
            ux_l[3] = u_state[3][jj][ii];

            ux_r[0] = u_state[0][jj][ii + 1];
            ux_r[1] = u_state[1][jj][ii + 1];
            ux_r[2] = u_state[2][jj][ii + 1];
            ux_r[3] = u_state[3][jj][ii + 1];

            // j+1/2
            uy_l[0] = u_state[0][jj][ii];
            uy_l[1] = u_state[1][jj][ii];
            uy_l[2] = u_state[2][jj][ii];
            uy_l[3] = u_state[3][jj][ii];

            uy_r[0] = u_state[0][jj + 1][ii];
            uy_r[1] = u_state[1][jj + 1][ii];
            uy_r[2] = u_state[2][jj + 1][ii];
            uy_r[3] = u_state[3][jj + 1][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);
            
            f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l[0] = u_state[0][jj][ii - 1];
            ux_l[1] = u_state[1][jj][ii - 1];
            ux_l[2] = u_state[2][jj][ii - 1];
            ux_l[3] = u_state[3][jj][ii - 1];

            ux_r[0] = u_state[0][jj][ii];
            ux_r[1] = u_state[1][jj][ii];
            ux_r[2] = u_state[2][jj][ii];
            ux_r[3] = u_state[3][jj][ii];

            // j-1/2
            uy_l[0] = u_state[0][jj - 1][ii];
            uy_l[1] = u_state[1][jj - 1][ii];
            uy_l[2] = u_state[2][jj - 1][ii];
            uy_l[3] = u_state[3][jj - 1][ii];

            uy_r[0] = u_state[0][jj][ii];
            uy_r[1] = u_state[1][jj][ii];
            uy_r[2] = u_state[2][jj][ii];
            uy_r[3] = u_state[3][jj][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);

            f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        if (periodic){
            /* Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables */

            // X Coordinate
            xleft_most[0] = roll(prims[0][jj], ii - 2);
            xleft_mid[0] = roll(prims[0][jj], ii - 1);
            center[0] = prims[0][jj][ii];
            xright_mid[0] = roll(prims[0][jj], ii + 1);
            xright_most[0] = roll(prims[0][jj], ii + 2);

            xleft_most[1] = roll(prims[1][jj], ii - 2);
            xleft_mid[1] = roll(prims[1][jj], ii - 1);
            center[1] = prims[1][jj][ii];
            xright_mid[1] = roll(prims[1][jj], ii + 1);
            xright_most[1] = roll(prims[1][jj], ii + 2);

            xleft_most[2] = roll(prims[2][jj], ii - 2);
            xleft_mid[2] = roll(prims[2][jj], ii - 1);
            center[2] = prims[2][jj][ii];
            xright_mid[2] = roll(prims[2][jj], ii + 1);
            xright_most[2] = roll(prims[2][jj], ii + 2);

            xleft_most[3] = roll(prims[3][jj], ii - 2);
            xleft_mid[3] = roll(prims[3][jj], ii - 1);
            center[3] = prims[3][jj][ii];
            xright_mid[3] = roll(prims[3][jj], ii + 1);
            xright_most[3] = roll(prims[3][jj], ii + 2);

            yleft_most[0] = roll(prims[0], ii, jj - 2);
            yleft_mid[0] = roll(prims[0], ii, jj - 1);
            yright_mid[0] = roll(prims[0], ii, jj + 1);
            yright_most[0] = roll(prims[0], ii, jj + 2);

            yleft_most[1] = roll(prims[1], ii, jj - 2);
            yleft_mid[1] = roll(prims[1], ii, jj - 1);
            yright_mid[1] = roll(prims[1], ii, jj + 1);
            yright_most[1] = roll(prims[1], ii, jj + 2);

            yleft_most[2] = roll(prims[2],ii, jj - 2);
            yleft_mid[2] = roll(prims[2], ii, jj - 1);
            yright_mid[2] = roll(prims[2], ii, jj + 1);
            yright_most[2] = roll(prims[2], ii, jj + 2);

            yleft_most[3] = roll(prims[3], ii, jj - 2);
            yleft_mid[3] = roll(prims[3], ii, jj - 1);
            yright_mid[3] = roll(prims[3], ii, jj + 1);
            yright_most[3] = roll(prims[3], ii, jj + 2);


        } else {
            // Adjust for beginning input of L vector
            xcoordinate = ii - 2;
            ycoordinate = jj - 2;

            // Coordinate X
            xleft_most[0] = prims[0][jj][ii - 2];
            xleft_mid[0] = prims[0][jj][ii - 1];
            center[0] = prims[0][jj][ii];
            xright_mid[0] = prims[0][jj][ii + 1];
            xright_most[0] = prims[0][jj][ii + 2];

            xleft_most[1] = prims[1][jj][ii - 2];
            xleft_mid[1] = prims[1][jj][ii - 1];
            center[1] = prims[1][jj][ii];
            xright_mid[1] = prims[1][jj][ii + 1];
            xright_most[1] = prims[1][jj][ii + 2];

            xleft_most[2] = prims[2][jj][ii - 2];
            xleft_mid[2] = prims[2][jj][ii - 1];
            center[2] = prims[2][jj][ii];
            xright_mid[2] = prims[2][jj][ii + 1];
            xright_most[2] = prims[2][jj][ii + 2];

            xleft_most[3] = prims[3][jj][ii - 2];
            xleft_mid[3] = prims[3][jj][ii - 1];
            center[3] = prims[3][jj][ii];
            xright_mid[3] = prims[3][jj][ii + 1];
            xright_most[3] = prims[3][jj][ii + 2];

            // Coordinate Y
            yleft_most[0] = prims[0][jj - 2][ii];
            yleft_mid[0] = prims[0][jj - 1][ii];
            yright_mid[0] = prims[0][jj + 1][ii];
            yright_most[0] = prims[0][jj + 2][ii];

            yleft_most[1] = prims[1][jj - 2][ii];
            yleft_mid[1] = prims[1][jj - 1][ii];
            yright_mid[1] = prims[1][jj + 1][ii];
            yright_most[1] = prims[1][jj + 2][ii];

            yleft_most[2] = prims[2][jj - 2][ii];
            yleft_mid[2] = prims[2][jj - 1][ii];
            yright_mid[2] = prims[2][jj + 1][ii];
            yright_most[2] = prims[2][jj + 2][ii];

            yleft_most[3] = prims[3][jj - 2][ii];
            yleft_mid[3] = prims[3][jj - 1][ii];
            yright_mid[3] = prims[3][jj + 1][ii];
            yright_most[3] = prims[3][jj + 2][ii];

        }

        // Reconstructed left X primitives vector at the i+1/2 interface
        xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                            0.5*(xright_mid[0] - xleft_mid[0]),
                                            theta*(xright_mid[0] - center[0]));

        
        xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                            0.5*(xright_mid[1] - xleft_mid[1]),
                                            theta*(xright_mid[1] - center[1]));

        xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                            0.5*(xright_mid[2] - xleft_mid[2]),
                                            theta*(xright_mid[2] - center[2]));

        xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                            0.5*(xright_mid[3] - xleft_mid[3]),
                                            theta*(xright_mid[3] - center[3]));

        // Reconstructed right primitives vector in x
        xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                            0.5*(xright_most[0] - center[0]),
                                            theta*(xright_most[0] - xright_mid[0]));

        xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                            0.5*(xright_most[1] - center[1]),
                                            theta*(xright_most[1] - xright_mid[1]));

        xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                            0.5*(xright_most[2] - center[2]),
                                            theta*(xright_most[2] - xright_mid[2]));

        xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                            0.5*(xright_most[3] - center[3]),
                                            theta*(xright_most[3] - xright_mid[3]));

        
        // Reconstructed right primitives vector in y-direction at j+1/2 interfce
        yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                            0.5*(yright_mid[0] - yleft_mid[0]),
                                            theta*(yright_mid[0] - center[0]));

        yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                            0.5*(yright_mid[1] - yleft_mid[1]),
                                            theta*(yright_mid[1] - center[1]));

        yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                            0.5*(yright_mid[2] - yleft_mid[2]),
                                            theta*(yright_mid[2] - center[2]));

        yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                            0.5*(yright_mid[3] - yleft_mid[3]),
                                            theta*(yright_mid[3] - center[3]));
        

        yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                            0.5*(yright_most[0] - center[0]),
                                            theta*(yright_most[0] - yright_mid[0]));

        yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                            0.5*(yright_most[1] - center[1]),
                                            theta*(yright_most[1] - yright_mid[1]));

        yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                            0.5*(yright_most[2] - center[2]),
                                            theta*(yright_most[2] - yright_mid[2]));

        yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                            0.5*(yright_most[3] - center[3]),
                                            theta*(yright_most[3] - yright_mid[3]));

        
        // Calculate the left and right states using the reconstructed PLM primitives
        ux_l = calc_state2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        ux_r = calc_state2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
        uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

        f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
        g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

        if (hllc){
            f1 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
            g1 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
        } else {
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
        }


        // Do the same thing, but for the left side interface [i - 1/2]

        // Left side primitives in x
        xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                0.5*(center[0] - xleft_most[0]),
                                                theta*(center[0] - xleft_mid[0]));

        xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                0.5*(center[1] -xleft_most[1]),
                                                theta*(center[1] - xleft_mid[1]));
        
        xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                0.5*(center[2] - xleft_most[2]),
                                                theta*(center[2] - xleft_mid[2]));
        
        xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                0.5*(center[3] - xleft_most[3]),
                                                theta*(center[3] - xleft_mid[3]));

            
        // Right side primitives in x
        xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                            0.5*(xright_mid[0] - xleft_mid[0]),
                                            theta*(xright_mid[0] - center[0]));

        xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                            0.5*(xright_mid[1] - xleft_mid[1]),
                                            theta*(xright_mid[1] - center[1]));

        xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                            0.5*(xright_mid[2] - xleft_mid[2]),
                                            theta*(xright_mid[2] - center[2]));

        xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                            0.5*(xright_mid[3] - xleft_mid[3]),
                                            theta*(xright_mid[3] - center[3]));


        // Left side primitives in y
        yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                0.5*(center[0] - yleft_most[0]),
                                                theta*(center[0] - yleft_mid[0]));

        yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                0.5*(center[1] -yleft_most[1]),
                                                theta*(center[1] - yleft_mid[1]));
        
        yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                0.5*(center[2] - yleft_most[2]),
                                                theta*(center[2] - yleft_mid[2]));
        
        yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                0.5*(center[3] - yleft_most[3]),
                                                theta*(center[3] - yleft_mid[3]));

            
        // Right side primitives in y
        yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                            0.5*(yright_mid[0] - yleft_mid[0]),
                                            theta*(yright_mid[0] - center[0]));

        yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                            0.5*(yright_mid[1] - yleft_mid[1]),
                                            theta*(yright_mid[1] - center[1]));

        yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                            0.5*(yright_mid[2] - yleft_mid[2]),
                                            theta*(yright_mid[2] - center[2]));

        yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                            0.5*(yright_mid[3] - yleft_mid[3]),
                                            theta*(yright_mid[3] - center[3])); 
        
    

        // Calculate the left and right states using the reconstructed PLM primitives
        ux_l = calc_state2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        ux_r = calc_state2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
        uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

        f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
        g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

        if (hllc){
            f2 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
            g2 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
        } else {
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
        }

        
        

        L[0] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy 
                                                + sourceRho[ycoordinate][xcoordinate];
        L[1] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy 
                                                    + sourceM1[ycoordinate][xcoordinate];
        L[2] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy
                                                    + sourceM2[ycoordinate][xcoordinate];
        L[3] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy 
                                                    + sourceE[ycoordinate][xcoordinate];


        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, r_avg, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr, dtheta; 

        if (periodic){
            rcoordinate = ii;
            tcoordinate = jj;

            // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

            // X Coordinate
            xleft_most[0] = roll(prims[0][ii], ii - 2);
            xleft_mid[0] = roll(prims[0][ii], ii - 1);
            center[0] = prims[0][ii][jj];
            xright_mid[0] = roll(prims[0][ii], ii + 1);
            xright_most[0] = roll(prims[0][ii], ii + 2);

            xleft_most[1] = roll(prims[1][ii], ii - 2);
            xleft_mid[1] = roll(prims[1][ii], ii - 1);
            center[1] = prims[1][ii][jj];
            xright_mid[1] = roll(prims[1][ii], ii + 1);
            xright_most[1] = roll(prims[1][ii], ii + 2);

            xleft_most[2] = roll(prims[2][ii], ii - 2);
            xleft_mid[2] = roll(prims[2][ii], ii - 1);
            center[2] = prims[2][ii][jj];
            xright_mid[2] = roll(prims[2][ii], ii + 1);
            xright_most[2] = roll(prims[2][ii], ii + 2);

            xleft_most[3] = roll(prims[3][ii], ii - 2);
            xleft_mid[3] = roll(prims[3][ii], ii - 1);
            center[3] = prims[3][ii][jj];
            xright_mid[3] = roll(prims[3][ii], ii + 1);
            xright_most[3] = roll(prims[3][ii], ii + 2);

            yleft_most[0] = roll(prims[0], ii, jj - 2);
            yleft_mid[0] = roll(prims[0], ii, jj - 1);
            yright_mid[0] = roll(prims[0], ii, jj + 1);
            yright_most[0] = roll(prims[0], ii, jj + 2);

            yleft_most[1] = roll(prims[1], ii, jj - 2);
            yleft_mid[1] = roll(prims[1], ii, jj - 1);
            yright_mid[1] = roll(prims[1], ii, jj + 1);
            yright_most[1] = roll(prims[1], ii, jj + 2);

            yleft_most[2] = roll(prims[2],ii, jj - 2);
            yleft_mid[2] = roll(prims[2], ii, jj - 1);
            yright_mid[2] = roll(prims[2], ii, jj + 1);
            yright_most[2] = roll(prims[2], ii, jj + 2);

            yleft_most[3] = roll(prims[3], ii, jj - 2);
            yleft_mid[3] = roll(prims[3], ii, jj - 1);
            yright_mid[3] = roll(prims[3], ii, jj + 1);
            yright_most[3] = roll(prims[3], ii, jj + 2);

        } else {
            // Adjust for beginning input of L vector
            rcoordinate = ii - 2;
            tcoordinate = jj - 2;

            // Coordinate X
            xleft_most[0] = prims[0][jj][ii - 2];
            xleft_mid[0] = prims[0][jj][ii - 1];
            center[0] = prims[0][jj][ii];
            xright_mid[0] = prims[0][jj][ii + 1];
            xright_most[0] = prims[0][jj][ii + 2];

            xleft_most[1] = prims[1][jj][ii - 2];
            xleft_mid[1] = prims[1][jj][ii - 1];
            center[1] = prims[1][jj][ii];
            xright_mid[1] = prims[1][jj][ii + 1];
            xright_most[1] = prims[1][jj][ii + 2];

            xleft_most[2] = prims[2][jj][ii - 2];
            xleft_mid[2] = prims[2][jj][ii - 1];
            center[2] = prims[2][jj][ii];
            xright_mid[2] = prims[2][jj][ii + 1];
            xright_most[2] = prims[2][jj][ii + 2];

            xleft_most[3] = prims[3][jj][ii - 2];
            xleft_mid[3] = prims[3][jj][ii - 1];
            center[3] = prims[3][jj][ii];
            xright_mid[3] = prims[3][jj][ii + 1];
            xright_most[3] = prims[3][jj][ii + 2];

            // Coordinate Y
            yleft_most[0] = prims[0][jj - 2][ii];
            yleft_mid[0] = prims[0][jj - 1][ii];
            yright_mid[0] = prims[0][jj + 1][ii];
            yright_most[0] = prims[0][jj + 2][ii];

            yleft_most[1] = prims[1][jj - 2][ii];
            yleft_mid[1] = prims[1][jj - 1][ii];
            yright_mid[1] = prims[1][jj + 1][ii];
            yright_most[1] = prims[1][jj + 2][ii];

            yleft_most[2] = prims[2][jj - 2][ii];
            yleft_mid[2] = prims[2][jj - 1][ii];
            yright_mid[2] = prims[2][jj + 1][ii];
            yright_most[2] = prims[2][jj + 2][ii];

            yleft_most[3] = prims[3][jj - 2][ii];
            yleft_mid[3] = prims[3][jj - 1][ii];
            yright_mid[3] = prims[3][jj + 1][ii];
            yright_most[3] = prims[3][jj + 2][ii];

        }
        
        // Reconstructed left X primitives vector at the i+1/2 interface
        xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                            0.5*(xright_mid[0] - xleft_mid[0]),
                                            theta*(xright_mid[0] - center[0]));

        
        xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                            0.5*(xright_mid[1] - xleft_mid[1]),
                                            theta*(xright_mid[1] - center[1]));

        xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                            0.5*(xright_mid[2] - xleft_mid[2]),
                                            theta*(xright_mid[2] - center[2]));

        xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                            0.5*(xright_mid[3] - xleft_mid[3]),
                                            theta*(xright_mid[3] - center[3]));

        // Reconstructed right primitives vector in x
        xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                            0.5*(xright_most[0] - center[0]),
                                            theta*(xright_most[0] - xright_mid[0]));

        xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                            0.5*(xright_most[1] - center[1]),
                                            theta*(xright_most[1] - xright_mid[1]));

        xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                            0.5*(xright_most[2] - center[2]),
                                            theta*(xright_most[2] - xright_mid[2]));

        xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                            0.5*(xright_most[3] - center[3]),
                                            theta*(xright_most[3] - xright_mid[3]));

        
        // Reconstructed right primitives vector in y-direction at j+1/2 interfce
        yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                            0.5*(yright_mid[0] - yleft_mid[0]),
                                            theta*(yright_mid[0] - center[0]));

        yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                            0.5*(yright_mid[1] - yleft_mid[1]),
                                            theta*(yright_mid[1] - center[1]));

        yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                            0.5*(yright_mid[2] - yleft_mid[2]),
                                            theta*(yright_mid[2] - center[2]));

        yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                            0.5*(yright_mid[3] - yleft_mid[3]),
                                            theta*(yright_mid[3] - center[3]));
        

        yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                            0.5*(yright_most[0] - center[0]),
                                            theta*(yright_most[0] - yright_mid[0]));

        yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                            0.5*(yright_most[1] - center[1]),
                                            theta*(yright_most[1] - yright_mid[1]));

        yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                            0.5*(yright_most[2] - center[2]),
                                            theta*(yright_most[2] - yright_mid[2]));

        yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                            0.5*(yright_most[3] - center[3]),
                                            theta*(yright_most[3] - yright_mid[3]));

        // Calculate the left and right states using the reconstructed PLM primitives
        ux_l = calc_state2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        ux_r = calc_state2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
        uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

        f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
        g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

        if (hllc){
            f1 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
            g1 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
        } else {
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
        }

        // Do the same thing, but for the left side interface [i - 1/2]

        // Left side primitives in x
        xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                0.5*(center[0] - xleft_most[0]),
                                                theta*(center[0] - xleft_mid[0]));

        xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                0.5*(center[1] -xleft_most[1]),
                                                theta*(center[1] - xleft_mid[1]));
        
        xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                0.5*(center[2] - xleft_most[2]),
                                                theta*(center[2] - xleft_mid[2]));
        
        xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                0.5*(center[3] - xleft_most[3]),
                                                theta*(center[3] - xleft_mid[3]));

            
        // Right side primitives in x
        xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                            0.5*(xright_mid[0] - xleft_mid[0]),
                                            theta*(xright_mid[0] - center[0]));

        xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                            0.5*(xright_mid[1] - xleft_mid[1]),
                                            theta*(xright_mid[1] - center[1]));

        xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                            0.5*(xright_mid[2] - xleft_mid[2]),
                                            theta*(xright_mid[2] - center[2]));

        xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                            0.5*(xright_mid[3] - xleft_mid[3]),
                                            theta*(xright_mid[3] - center[3]));


        // Left side primitives in y
        yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                0.5*(center[0] - yleft_most[0]),
                                                theta*(center[0] - yleft_mid[0]));

        yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                0.5*(center[1] -yleft_most[1]),
                                                theta*(center[1] - yleft_mid[1]));
        
        yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                0.5*(center[2] - yleft_most[2]),
                                                theta*(center[2] - yleft_mid[2]));
        
        yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                0.5*(center[3] - yleft_most[3]),
                                                theta*(center[3] - yleft_mid[3]));

            
        // Right side primitives in y
        yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                            0.5*(yright_mid[0] - yleft_mid[0]),
                                            theta*(yright_mid[0] - center[0]));

        yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                            0.5*(yright_mid[1] - yleft_mid[1]),
                                            theta*(yright_mid[1] - center[1]));

        yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                            0.5*(yright_mid[2] - yleft_mid[2]),
                                            theta*(yright_mid[2] - center[2]));

        yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                            0.5*(yright_mid[3] - yleft_mid[3]),
                                            theta*(yright_mid[3] - center[3])); 
        
    

        // Calculate the left and right states using the reconstructed PLM primitives
        ux_l = calc_state2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        ux_r = calc_state2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
        uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

        f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
        f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

        g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
        g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

        if (hllc){
            f2 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
            g2 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
        } else {
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
        }

        // f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
        // g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");

        if (linspace){
            right_cell = x1[rcoordinate + 1];
            left_cell = x1[rcoordinate - 1];
            upper_cell = x2[tcoordinate + 1];
            lower_cell = x2[tcoordinate - 1];

            // cout << "Theta Coordinate: " << tcoordinate << endl;
            // cout << "R Coordinate: " << rcoordinate << endl;
            
            // Outflow the left/right boundaries
            if (rcoordinate - 1 < 0){
                left_cell = x1[rcoordinate];

            } else if (rcoordinate == xphysical_grid - 1){
                right_cell = x1[rcoordinate];

            }

            if (tcoordinate - 1 < 0){

                lower_cell = x2[tcoordinate];
                
            }  else if(tcoordinate == yphysical_grid - 1){
                upper_cell = x2[tcoordinate];
            }

            r_right = 0.5*(right_cell + x1[rcoordinate]);
            r_left = 0.5*(x1[rcoordinate] + left_cell);

            theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                        cos(upper_cell) + cos(x2[tcoordinate]) );

            theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                        cos(lower_cell) + cos(x2[tcoordinate]) );

        } else {
            right_cell = x1[rcoordinate + 1];
            left_cell = x1[rcoordinate - 1];
            upper_cell = x2[tcoordinate + 1];
            lower_cell = x2[tcoordinate - 1];

            // cout << "Theta Coordinate: " << tcoordinate << endl;
            // cout << "R Coordinate: " << rcoordinate << endl;
            
            // Outflow the left/right boundaries
            if (rcoordinate - 1 < 0){
                left_cell = x1[rcoordinate];

            } else if (rcoordinate == xphysical_grid - 1){
                right_cell = x1[rcoordinate];

            }

            log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
            log_rRight = log_rLeft + delta_logr;
            r_right = pow(10, log_rRight);
            r_left = pow(10, log_rLeft);

            // cout << rcoordinate << endl;
            // cout << delta_logr << endl;
            // cout << "Initial r_right: " << r_right << endl;
            // cout << "numpy r_right" << right_cell << endl;
            // cout << "Initial r_left: " << r_left << endl;
            // cout << "numpy r_left: " << left_cell << endl;
            // r_right = right_cell; // 0.5*(right_cell + x1[rcoordinate]);
            // r_left = left_cell; //0.5*(x1[rcoordinate] + left_cell);

            // cout << "r_r: " << r_right << endl;
            // cout << "r_l: " << r_left << endl;
            // cin.get();
            upper_cell = x2[tcoordinate + 1];
            lower_cell = x2[tcoordinate - 1];
            
            // Outflow the left/right boundaries
            if (tcoordinate - 1 < 0){
                lower_cell = x2[tcoordinate];

            } else if(tcoordinate == yphysical_grid - 1){
                upper_cell = x2[tcoordinate];
            }

            theta_right = 0.5*(upper_cell + x2[tcoordinate]);
            theta_left = 0.5*(x2[tcoordinate] + lower_cell);
        }

        dr = r_right - r_left;
        dtheta = theta_right - theta_left;
        rhoc = center[0];
        pc = center[1];
        uc = center[2];
        vc = center[3];

        
        
        r_avg = 0.5*(r_right + r_left);
        ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
        // Compute the surface areas
        right_rsurface = r_right*r_right;
        left_rsurface = r_left*r_left;
        upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
        lower_tsurface = sin(theta_left); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
        volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
        deltaV1 = pow(volAvg, 2)*dr;
        deltaV2 = volAvg*sin(ang_avg)*dtheta; 

        L[0] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                            - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2
                                            + sourceRho[tcoordinate][rcoordinate];

        L[1] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                            - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                            + rhoc*vc*vc/volAvg + 2*pc/volAvg
                                            + sourceM1[tcoordinate][rcoordinate];

        L[2] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                            - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                            -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg)))
                                            + sourceM2[tcoordinate][rcoordinate];

        L[3] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                            - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2
                                            + sourceE[tcoordinate][rcoordinate];
        

        return L;
        
    }
    

};

vector<vector<vector<double> > > Newtonian2D::u_dot2D(float gamma, 
                                        vector<vector<vector<double> > > &u_state,
                                        vector<vector<vector<double> > > &sources, 
                                        bool periodic = false, float theta = 1.5, bool  linspace=true,
                                        bool hllc = false)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate, xcenter, ycenter;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid, yphysical_grid;
    string default_coordinates = "cartesian";

    if (periodic){
        xphysical_grid = xgrid_size;
        yphysical_grid = ygrid_size;
    } else {
        xphysical_grid = xgrid_size - 4;
        yphysical_grid = ygrid_size - 4;
    }

    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;
    xcenter = xphysical_grid/2 + 2; 
    ycenter = yphysical_grid/2 + 2;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    // Define the source terms
    vector<vector<double> > sourceRho = sources[0];
    vector<vector<double> > sourceM1 = sources[1];
    vector<vector<double> > sourceM2 = sources[2];
    vector<vector<double> > sourceE = sources[3];
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    prims = cons2prim2D(u_state);
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        int true_nxpts = xgrid_size - 2;
        int true_nypts = ygrid_size - 2;
        i_start = 2;
        i_bound = true_nxpts;
        j_start = 2;
        j_bound = true_nypts;
    }

    
    /**
    for (int jj = j_start; jj < j_bound; jj++){
        for (int ii = i_start; ii < i_bound; ii++){
            ycoordinate = jj - 2;
            xcoordinate = ii - 2;

            // i+1/2
            ux_l[0] = u_state[0][jj][ii];
            ux_l[1] = u_state[1][jj][ii];
            ux_l[2] = u_state[2][jj][ii];
            ux_l[3] = u_state[3][jj][ii];

            ux_r[0] = u_state[0][jj][ii + 1];
            ux_r[1] = u_state[1][jj][ii + 1];
            ux_r[2] = u_state[2][jj][ii + 1];
            ux_r[3] = u_state[3][jj][ii + 1];

            // j+1/2
            uy_l[0] = u_state[0][jj][ii];
            uy_l[1] = u_state[1][jj][ii];
            uy_l[2] = u_state[2][jj][ii];
            uy_l[3] = u_state[3][jj][ii];

            uy_r[0] = u_state[0][jj + 1][ii];
            uy_r[1] = u_state[1][jj + 1][ii];
            uy_r[2] = u_state[2][jj + 1][ii];
            uy_r[3] = u_state[3][jj + 1][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);
            
            f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l[0] = u_state[0][jj][ii - 1];
            ux_l[1] = u_state[1][jj][ii - 1];
            ux_l[2] = u_state[2][jj][ii - 1];
            ux_l[3] = u_state[3][jj][ii - 1];

            ux_r[0] = u_state[0][jj][ii];
            ux_r[1] = u_state[1][jj][ii];
            ux_r[2] = u_state[2][jj][ii];
            ux_r[3] = u_state[3][jj][ii];

            // j-1/2
            uy_l[0] = u_state[0][jj - 1][ii];
            uy_l[1] = u_state[1][jj - 1][ii];
            uy_l[2] = u_state[2][jj - 1][ii];
            uy_l[3] = u_state[3][jj - 1][ii];

            uy_r[0] = u_state[0][jj][ii];
            uy_r[1] = u_state[1][jj][ii];
            uy_r[2] = u_state[2][jj][ii];
            uy_r[3] = u_state[3][jj][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);

            f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        // cout << "Am Cartesian" << endl;
        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    // X Coordinate
                    xleft_most[0] = roll(prims[0][jj], ii - 2);
                    xleft_mid[0] = roll(prims[0][jj], ii - 1);
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = roll(prims[0][jj], ii + 1);
                    xright_most[0] = roll(prims[0][jj], ii + 2);

                    xleft_most[1] = roll(prims[1][jj], ii - 2);
                    xleft_mid[1] = roll(prims[1][jj], ii - 1);
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = roll(prims[1][jj], ii + 1);
                    xright_most[1] = roll(prims[1][jj], ii + 2);

                    xleft_most[2] = roll(prims[2][jj], ii - 2);
                    xleft_mid[2] = roll(prims[2][jj], ii - 1);
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = roll(prims[2][jj], ii + 1);
                    xright_most[2] = roll(prims[2][jj], ii + 2);

                    xleft_most[3] = roll(prims[3][jj], ii - 2);
                    xleft_mid[3] = roll(prims[3][jj], ii - 1);
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = roll(prims[3][jj], ii + 1);
                    xright_most[3] = roll(prims[3][jj], ii + 2);

                    yleft_most[0] = roll(prims[0], ii, jj - 2);
                    yleft_mid[0] = roll(prims[0], ii, jj - 1);
                    yright_mid[0] = roll(prims[0], ii, jj + 1);
                    yright_most[0] = roll(prims[0], ii, jj + 2);

                    yleft_most[1] = roll(prims[1], ii, jj - 2);
                    yleft_mid[1] = roll(prims[1], ii, jj - 1);
                    yright_mid[1] = roll(prims[1], ii, jj + 1);
                    yright_most[1] = roll(prims[1], ii, jj + 2);

                    yleft_most[2] = roll(prims[2],ii, jj - 2);
                    yleft_mid[2] = roll(prims[2], ii, jj - 1);
                    yright_mid[2] = roll(prims[2], ii, jj + 1);
                    yright_most[2] = roll(prims[2], ii, jj + 2);

                    yleft_most[3] = roll(prims[3], ii, jj - 2);
                    yleft_mid[3] = roll(prims[3], ii, jj - 1);
                    yright_mid[3] = roll(prims[3], ii, jj + 1);
                    yright_most[3] = roll(prims[3], ii, jj + 2);

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }

                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_state2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_state2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                if (hllc){
                    f1 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
                    g1 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
                } else {
                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
                }

                

                // Do the same thing, but for the left side interface [i - 1/2]


                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_state2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_state2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);


                if (hllc){
                    f2 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
                    g2 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
                } else {
                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
                }


                L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy 
                                                        + sourceRho[ycoordinate][xcoordinate];
                L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy 
                                                            + sourceM1[ycoordinate][xcoordinate];
                L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy
                                                            + sourceM2[ycoordinate][xcoordinate];
                L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy 
                                                            + sourceE[ycoordinate][xcoordinate];
            }

        }
        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, r_avg, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr, dtheta; 

        double pi = 2*acos(0.0);

        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][jj], ii - 2);
                    xleft_mid[0] = roll(prims[0][jj], ii - 1);
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = roll(prims[0][jj], ii + 1);
                    xright_most[0] = roll(prims[0][jj], ii + 2);

                    xleft_most[1] = roll(prims[1][jj], ii - 2);
                    xleft_mid[1] = roll(prims[1][jj], ii - 1);
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = roll(prims[1][jj], ii + 1);
                    xright_most[1] = roll(prims[1][jj], ii + 2);

                    xleft_most[2] = roll(prims[2][jj], ii - 2);
                    xleft_mid[2] = roll(prims[2][jj], ii - 1);
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = roll(prims[2][jj], ii + 1);
                    xright_most[2] = roll(prims[2][jj], ii + 2);

                    xleft_most[3] = roll(prims[3][jj], ii - 2);
                    xleft_mid[3] = roll(prims[3][jj], ii - 1);
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = roll(prims[3][jj], ii + 1);
                    xright_most[3] = roll(prims[3][jj], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    yleft_most[0] = roll(prims[0], ii, jj - 2);
                    yleft_mid[0] = roll(prims[0], ii, jj - 1);
                    yright_mid[0] = roll(prims[0], ii, jj + 1);
                    yright_most[0] = roll(prims[0], ii, jj + 2);

                    yleft_most[1] = roll(prims[1], ii, jj - 2);
                    yleft_mid[1] = roll(prims[1], ii, jj - 1);
                    yright_mid[1] = roll(prims[1], ii, jj + 1);
                    yright_most[1] = roll(prims[1], ii, jj + 2);

                    yleft_most[2] = roll(prims[2],ii, jj - 2);
                    yleft_mid[2] = roll(prims[2], ii, jj - 1);
                    yright_mid[2] = roll(prims[2], ii, jj + 1);
                    yright_most[2] = roll(prims[2], ii, jj + 2);

                    yleft_most[3] = roll(prims[3], ii, jj - 2);
                    yleft_mid[3] = roll(prims[3], ii, jj - 1);
                    yright_mid[3] = roll(prims[3], ii, jj + 1);
                    yright_most[3] = roll(prims[3], ii, jj + 2);
 

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_state2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_state2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                if (hllc){
                    f1 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
                    g1 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
                } else {
                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
                }
                // f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                // g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");


                /**
                if ((tcoordinate == 0) && (rcoordinate == 0)){
                    cout << ii << endl;
                    cout << jj << endl;
                    cout << "Rho  (L): " << xprims_l[0] << endl;
                    cout << "Rho  (R): " << xprims_r[0] << endl;
                    cout << "P  (L): " << xprims_l[1] << endl;
                    cout << "P  (R): " << xprims_r[1] << endl;
                    cout << "V_r  (L): " << xprims_l[2] << endl;
                    cout << "V_r  (R): " << xprims_l[2] << endl;
                    cout << "V_t  (L): " << xprims_l[3] << endl;
                    cout << "V_t  (R): " << xprims_l[3] << endl;
                    cout << " " << endl;
                    cout << "Rho  (L): " << yprims_l[0] << endl;
                    cout << "Rho  (R): " << yprims_r[0] << endl;
                    cout << "P  (L): " << yprims_l[1] << endl;
                    cout << "P  (R): " << yprims_r[1] << endl;
                    cout << "V_r  (L): " << yprims_l[2] << endl;
                    cout << "V_r  (R): " << yprims_r[2] << endl;
                    cout << "V_t  (L): " << yprims_l[3] << endl;
                    cout << "V_t  (R): " << yprims_r[3] << endl;
                    string h;
                    cin >> h;

                }
                */
                
                
                



                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_state2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_state2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_flux(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_flux(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_flux(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_flux(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                if (hllc){
                    f2 = calc_hllc_flux2D(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, "x", 1);
                    g2 = calc_hllc_flux2D(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, "y", 2);
                } else {
                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");
                }

                // f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, "x");
                // g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, "y");

                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    if (tcoordinate - 1 < 0){

                        lower_cell = x2[tcoordinate];
                        
                    }  else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    r_right = 0.5*(right_cell + x1[rcoordinate]);
                    r_left = 0.5*(x1[rcoordinate] + left_cell);

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                } else {
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_right = pow(10, log_rRight);
                    r_left = pow(10, log_rLeft);

                    // cout << rcoordinate << endl;
                    // cout << delta_logr << endl;
                    // cout << "Initial r_right: " << r_right << endl;
                    // cout << "numpy r_right" << right_cell << endl;
                    // cout << "Initial r_left: " << r_left << endl;
                    // cout << "numpy r_left: " << left_cell << endl;
                    // r_right = right_cell; // 0.5*(right_cell + x1[rcoordinate]);
                    // r_left = left_cell; //0.5*(x1[rcoordinate] + left_cell);

                    // cout << "r_r: " << r_right << endl;
                    // cout << "r_l: " << r_left << endl;
                    // cin.get();
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        lower_cell = x2[tcoordinate];

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = 0.5*(upper_cell + x2[tcoordinate]);
                    theta_left = 0.5*(x2[tcoordinate] + lower_cell);
                }

                dr = r_right - r_left;
                dtheta = theta_right - theta_left;
                rhoc = center[0];
                pc = center[1];
                uc = center[2];
                vc = center[3];

                
                
                r_avg = 0.5*(r_right + r_left);
                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = sin(theta_left); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = volAvg*sin(ang_avg)*dtheta; 

                // cout << "Theta [i + 1/2]: " << theta_right << endl;
                // cout << "Theta[i - 1/2]: " << theta_left << endl;
                // cout << "Ang Avg: " << ang_avg << endl;
                // // string a;
                // // cin >> a;
                // cout << "Ang Avg: " << ang_avg << endl;
                // string a;
                // cin >> a;

                
                /**
                cout << "Sub: " << (cos(theta_left) - cos(theta_right)) << endl;
                cout << "VolAvg:" << volAvg << endl;
                cout << "F1: "<< f1[0] << endl;
                cout << "F2: "<< f2[0] << endl;
                cout << "G1: "<< g1[0] << endl;
                cout << "G2: "<< g2[0] << endl;
                string a;
                cin >> a;
                */

                L[0][tcoordinate][rcoordinate] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                                    - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2
                                                    + sourceRho[tcoordinate][rcoordinate];

                L[1][tcoordinate][rcoordinate] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                                    - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg
                                                    + sourceM1[tcoordinate][rcoordinate];

                L[2][tcoordinate][rcoordinate] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                                    - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg)))
                                                    + sourceM2[tcoordinate][rcoordinate];

                L[3][tcoordinate][rcoordinate] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                                    - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2
                                                    + sourceE[tcoordinate][rcoordinate];
                
            }

        }

        return L;
        
    }
    

};



//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
vector<vector<vector<double> > > Newtonian2D::simulate2D(vector<vector<vector<double> > > &sources, 
                                                        float tend = 0.1, bool periodic = false, 
                                                        double dt = 1.e-4, bool linspace=true,
                                                        bool hllc = false){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int total_zones = xgrid_size*ygrid_size;
    float t = 0;
    int i_real, j_real, xphysical_grid, yphysical_grid;;

    if (periodic){
        xphysical_grid = xgrid_size;
        yphysical_grid = ygrid_size;
    } else {
        xphysical_grid = xgrid_size - 4;
        yphysical_grid = ygrid_size - 4;
    }

    vector<vector<vector<double> > > u_p(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));
    
    vector<vector<vector<double> > > udot1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // vector<double> udot1(n_vars);
    // vector<double> udot2(n_vars);
    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    
    
    while (t <= tend){
        /* Compute the loop execution time */
        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        // if (!periodic){
        //     if (t == 0){
        //         config_ghosts2D(u, xgrid_size, ygrid_size, false);
        //     }
        // }
        
        /**
        cout << " " << endl;
        cout << "U[1]: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u[1][jj][ii] << ", ";
            }
            cout << endl;
        }
        cin.get();
        */

        udot = u_dot2D(gamma, u, sources, periodic, theta, linspace, hllc);

        /* Perform Higher Order RK3 */
        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            if (periodic){
                j_real = jj;
            } else {
                j_real = jj + 2;
            }
            
            for (int ii = 0; ii < xphysical_grid; ii++){
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 2;
                }
                u1[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                u1[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                u1[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                u1[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
            }
        }

        /**
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[1][jj][ii] << ", ";
            }
            cout << endl;
        }
        cin.get();
        */

        // if (!periodic){
        //     config_ghosts2D(u1, xgrid_size, ygrid_size, false);
        // }
        

        udot1 = u_dot2D(gamma, u1, sources, periodic, theta, linspace, hllc);
        

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            if (periodic){
                j_real = jj;
            } else {
                j_real = jj + 2;
            }
            for (int ii = 0; ii < xphysical_grid; ii++){
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 2;
                }

                u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

            }
            

        }
        
        // if (!periodic){
        //     config_ghosts2D(u2, xgrid_size, ygrid_size, false);
        // }
        
        udot2 = u_dot2D(gamma, u2, sources, periodic, theta, linspace, hllc);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            if (periodic){
                j_real = jj;
            } else {
                j_real =  jj + 2;
            }
            for (int ii = 0; ii < xphysical_grid; ii++){
                if (periodic){
                    i_real = ii;
                } else {
                    i_real =  ii + 2;
                }
                u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        
        // if (!periodic){
        //     config_ghosts2D(u_p, xgrid_size, ygrid_size, false);
        // }
        
        
        if (t > 0){
            dt = adapt_dt(u_p);
        }
        
        /* Compute the loop execution time */
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        cout << fixed << setprecision(7);
        cout << "\r" << "dt: " << setw(5) << dt 
             << "\t" << "t: " << setw(5) << t 
             << "\t" <<  "Zones: " << total_zones
             << "\t" << "Execution Time: " <<
            time_span.count() << "sec" << flush;

        // Swap the arrays
        u.swap(u_p);
        
        t += dt;

    }

    cout << "\n " << endl;
    return u;

 };