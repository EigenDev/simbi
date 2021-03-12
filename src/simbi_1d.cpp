/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "ustate.h" 
#include "helper_functions.h"
#include <cmath>
#include <map>
#include <algorithm>

using namespace std;
using namespace hydro;


// Default Constructor 
Newtonian1D::Newtonian1D () {}

// Overloaded Constructor
Newtonian1D::Newtonian1D(vector< vector<double> > u_state, float Gamma, double cfl, vector<double> R = {0},
                string Coord_system = "cartesian")
{
    state = u_state;
    gamma = Gamma;
    r = R;
    coord_system = Coord_system;
    CFL = cfl;

}

// Destructor 
Newtonian1D::~Newtonian1D() {}

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
vector<double>  cons2prim(float gamma, vector<double>  u_state){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

        double rho, energy, mom;
        vector<double> prims(3);
        double v, pressure;

        rho = u_state[0];
        mom = u_state[1];
        energy = u_state[2];

        v = mom/rho;
        pressure = calc_pressure(gamma, rho, energy, v);
        
        prims[0] = rho;
        prims[1] = pressure;
        prims[2] = v;

        return prims;
        
    
    
};

vector<vector<double> > Newtonian1D::cons2prim1D(vector<vector<double> > u_state){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, mom;
    double v, pressure;
    int n_vars = u_state.size();
    int n_gridpts = u_state[0].size();

    vector < vector<double> > prims(n_vars, vector<double> (n_gridpts, 0));
    
    
    for (int ii=0; ii < n_gridpts; ii++){

        rho = u_state[0][ii];
        mom = u_state[1][ii];
        energy = u_state[2][ii];

        v = mom/rho;
        pressure = calc_pressure(gamma, rho, energy, v);
        
        prims[0][ii] = rho;
        prims[1][ii] = pressure;
        prims[2][ii] = v;
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------


map<string, map<string, double > > calc_eigenvals(float gamma, vector<double> left_state,
                                            vector<double> right_state)
{
    
    // Initialize your important variables
    double v_r, v_l, p_r, p_l, cs_r, cs_l, aStar, cbar, pStar, aL, aR; 
    double rho_l, rho_r,  mom_l, mom_r, energy_l, energy_r, rhoBar, z, qL, qR;
    map<string, map<string, double > > lambda;
    

    // Separate the left and right state components
    rho_l = left_state[0];
    mom_l = left_state[1];
    energy_l = left_state[2];

    rho_r = right_state[0];
    mom_r = right_state[1];
    energy_r = right_state[2];
    
    v_l = mom_l/rho_l;
    v_r = mom_r/rho_r;
    
    p_r = calc_pressure(gamma, rho_r, energy_r, v_r);
    p_l = calc_pressure(gamma, rho_l, energy_l ,v_l);

    cs_r = calc_sound_speed(gamma, rho_r, p_r);
    cs_l = calc_sound_speed(gamma, rho_l, p_l);

    cbar = 0.5*(cs_l + cs_r);
    rhoBar = 0.5*(rho_l + rho_r);
    pStar = 0.5*(p_l + p_r) + 0.5*(v_l - v_r)*cbar*rhoBar;
    aStar = 0.5*(v_l + v_r) + 0.5*(p_l - p_r)/(rhoBar*cbar);

    // Steps to Compute HLLC as described in Toro et al. 2019
    z = (gamma - 1.)/(2.*gamma);
    double num = cs_l + cs_r - ( gamma-1.)/2 *(v_r - v_l);
    double denom = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
    double p_term = num/denom;
    pStar = pow(p_term, (1./z));
    // cout << "pL: " << p_l << endl;
    // cout << "pR: " << p_r << endl;
    // cout << "p_term: " << pow(p_term , (1./z)) << endl;
    // cout << "denom: " << cs_l/pow(p_l,z) + cs_r/pow(p_r, z) << endl;
    // cout << "num: " << cs_l + cs_r - ( gamma-1.)/2 *(v_r - v_l) << endl;
    // cout << "ratio: " << num/denom << endl;
    // cout << "Vel term: " << ( gamma-1.)/2 *(v_r - v_l) << endl;
    // cin.get();
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
    aL = v_l - qL*cs_l;
    aR = v_r + qR*cs_r;

    aStar = ( (p_r - p_l + rho_l*v_l*(aL - v_l) - rho_r*v_r*(aR - v_r))/
                     (rho_l*(aL - v_l) - rho_r*(aR - v_r) ) );

    // Populate the Dictionary
    lambda["left"]["plus"] = v_l + cs_l; 
    lambda["left"]["minus"] = v_l - cs_l; 
    lambda["right"]["plus"] = v_r + cs_r; 
    lambda["right"]["minus"] = v_r - cs_r; 

    lambda["signal"]["pStar"] = pStar;
    lambda["signal"]["aStar"] = aStar;

    lambda["signal"]["aL"] = aL;
    lambda["signal"]["aR"] = aR;

    return lambda;
};

// Adapt the CFL conditonal timestep
double Newtonian1D::adapt_dt(vector<vector<double> > &u_state, vector<double> &r, bool linspace=true, bool first_order=true){

    double r_left, r_right, left_cell, right_cell, dr, cs;
    long double delta_logr, log_rLeft, log_rRight, min_dt, cfl_dt;
    int shift_i, physical_grid;

    // Get the primitive vector 
    vector<vector<double> >  prims = cons2prim1D(u_state);

    int grid_size = prims[0].size();

    min_dt = 0;
    // Find the minimum timestep over all i
    if (first_order){
        physical_grid = grid_size - 2;
    } else {
        physical_grid = grid_size - 4;
    }

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < physical_grid; ii++){
        if (first_order){
            shift_i = ii + 1;
        } else {
            shift_i = ii + 2;
        }

        if (ii - 1 < 0){
            left_cell = r[ii];
            right_cell = r[ii + 1];
        }
        else if (ii + 1 > physical_grid - 1){
            right_cell = r[ii];
            left_cell = r[ii - 1];
        } 
        else {
            right_cell = r[ii + 1];
            left_cell = r[ii];
        }

        // Check if using linearly-spaced grid or logspace
        if (linspace){
            r_right = 0.5*(right_cell + r[ii]);
            r_left = 0.5*(r[ii] + left_cell);
        } else {
            delta_logr = (log(r[physical_grid - 1]) - log(r[0]))/physical_grid;
            log_rLeft = log(r[0]) + ii*delta_logr;
            log_rRight = log_rLeft + delta_logr;
            r_left = exp(log_rLeft);
            r_right = exp(log_rRight);
        }

        dr = r_right - r_left;

        
        cs = calc_sound_speed(gamma, prims[0][shift_i], prims[1][shift_i]);

        cfl_dt = dr/(max(abs(prims[2][shift_i] + cs), abs(prims[2][shift_i] - cs)));

        if (ii > 0){
            min_dt = min(min_dt, cfl_dt);
        }
        else {
            min_dt = cfl_dt;
        }
    }

    return CFL*min_dt;
};

//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state tensor for computation. Used for Higher Order Reconstruction
vector<double>  calc_state1D(float gamma, double rho, double pressure, double v)
{
    // int n_vars = 3;
    vector<double> cons_state(3);

    // cons_state.resize(n_vars, vector<double> (1, -1));
    
    double energy = calc_energy(gamma, rho, pressure, v);
    
    cons_state[0] = rho; 
    cons_state[1] = rho*v;
    cons_state[2] = energy;

    return cons_state;
};

vector<double> calc_intermed_state1D(vector<double> prims,
                                vector<double> state,
                                double a,
                                double aStar,
                                double pStar,
                                int index=1)
{
    double rho, energy, pressure, v1, m1;
    double rhoStar, m1star, eStar;
    map<string, map<string, double > > lambda; 
    vector<double> hll_states(3);

    pressure = prims[1];
    v1 = prims[2];
    rho = state[0];
    m1 = state[1];
    energy = state[2];


    rhoStar = ( (a - v1)/(a - aStar))*rho;
    m1star = (1./(a-aStar))*(m1*(a - v1) - pressure + pStar);
    eStar = (1./(a-aStar))*(energy*(a - v1) + pStar*aStar - pressure*v1);

    // Try Toro's
    // m1star = rho*( (a - v1)/(a - aStar))*aStar;
    // eStar = rho*( (a - v1)/(a - aStar) ) * (energy/rho + (aStar - v1)*(aStar + pressure/(rho*(a - v1))));
    hll_states[0] = rhoStar;
    hll_states[1] = m1star;
    hll_states[2] = eStar;
    
    return hll_states;
}

vector<double> calc_hll_state1D(float gamma,
                                vector<double> &left_state,
                                vector<double> &right_state,
                                vector<double> &left_flux,
                                vector<double> &right_flux,
                                vector<double> &left_prims,
                                vector<double> &right_prims)
{
    double aL, aR;
    map<string, map<string, double > > lambda; 
    vector<double> hll_states(3);

    lambda = calc_eigenvals(gamma, left_state, right_state);

    aL = lambda["signal"]["aL"];
    aR = lambda["signal"]["aR"];

    hll_states[0] = ( aR*right_state[0] - aL*left_state[0] 
                        - right_flux[0] + left_flux[0])/(aR - aL);

    hll_states[1] = ( aR*right_state[1] - aL*left_state[1] 
                        - right_flux[1] + left_flux[1])/(aR - aL);

    hll_states[2] = ( aR*right_state[2] - aL*left_state[2] 
                        - right_flux[2] + left_flux[2])/(aR - aL);


    return hll_states;

}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
vector<double> calc_flux1D(float gamma, double rho, double pressure, double v){
    
    // The Flux Tensor
    vector<double> flux(3);

    // The Flux components
    double mom, energy_dens, zeta;
    

    // flux.resize(n_vars, vector<double> (grid_size, -1));
    double energy = calc_energy(gamma, rho, pressure, v);
    
    mom = rho*v;
    energy_dens = rho*v*v + pressure;
    zeta = (energy + pressure)*v;
    flux[0] = mom;
    flux[1] = energy_dens;
    flux[2] = zeta;

    return flux;
};

vector<double> calc_hll_flux1D(float gamma, vector<double> left_state,
                                        vector<double> right_state,
                                        vector<double> left_flux,
                                        vector<double> right_flux,
                                        bool hllc = false)
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(3);
    double alpha_plus, alpha_minus, aL, aR;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state);

    // cout << "Lambda Size: " <<  lambda["left"]["plus"].size() << endl;
    // Crudely form the arrays to search for the maximums
    alpha_plus = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
    alpha_minus = findMax(0 , - lambda["left"]["minus"], - lambda["right"]["minus"]);
    
    if (hllc){
        aR = lambda["signal"]["aR"];
        aL = lambda["signal"]["aL"];
        alpha_plus = max(0.0, aR);
        alpha_minus = max(0.0, - aL);
    }
    

    // Compute the HLL Flux component-wise
    hll_flux[0] = ( ( alpha_plus*left_flux[0] + alpha_minus*right_flux[0]
                            - alpha_minus*alpha_plus*(right_state[0] - left_state[0] ) )  /
                            (alpha_minus + alpha_plus) );

    hll_flux[1] = ( alpha_plus*left_flux[1] + alpha_minus*right_flux[1]
                            - alpha_minus*alpha_plus*(right_state[1] - left_state[1] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[2] = ( alpha_plus*left_flux[2] + alpha_minus*right_flux[2]
                            - alpha_minus*alpha_plus*(right_state[2] - left_state[2]) )  /
                            (alpha_minus + alpha_plus);


    return hll_flux;
};

vector<double> calc_hllc_flux1D(float gamma,
                                vector<double> &left_state,
                                vector<double> &right_state,
                                vector<double> &left_flux,
                                vector<double> &right_flux,
                                vector<double> &left_prims,
                                vector<double> &right_prims,
                                int index = 1
                                )
{
    map<string, map<string, double > > lambda; 
    vector<double> interflux_left(3);
    vector<double> interflux_right(3);
    vector<double> interstate_left(3);
    vector<double> interstate_right(3);
    vector<double> hll_flux(3);
    vector<double> hll_state(3);
    vector<double> hllc_flux(3);
    double aL, aR, aStar, pStar;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state);

    aL = lambda["signal"]["aL"]; //findMin(0, lambda["right"]["minus"], lambda["left"]["minus"]);
    aR = lambda["signal"]["aR"]; //findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);

    // aL *= -1.;
    aStar = lambda["signal"]["aStar"];
    pStar = lambda["signal"]["pStar"];

    // cout << "aStar1: " << aStar << endl;
    // aStar = 0.5*(aL + aR);

    // cout << aStar << endl;

    // cout << "aL: " << aL << endl;
    // cout << "aR: " << aR << endl;
    // cout << "aStar: " << aStar << endl;
    // cout << "pStar: " << pStar << endl;
    // cin.get();
    // aStar = 1.;

    hll_flux = calc_hll_flux1D(gamma, left_state, right_state, left_flux,
                                    right_flux, true);

    hll_state = calc_hll_state1D(gamma, left_state, right_state, left_flux,
                                     right_flux, left_prims, right_prims);


    // hll_prims = cons2primSR(gamma, hll_state, )
    // cout << "aStar: " << aStar << endl;
    // Calculate the left and right intermediate states
    // cout << "RHO HLLC: " << hll_state[0] << endl;
    // cout << "Mom HLLC: " << hll_state[1] << endl;
    // cout << "E HLLC: " << hll_state[2] << endl;

    // aStar = calc_intermed_wave(hll_state[2], hll_flux[2], hll_flux[1], hll_state[1]);
    // aStar = 0;

    // pStar = calc_intermed_pressure(aL, aStar, left_state[0], left_state[1], left_prims[2], left_prims[1]);

    // cout << "aL: " << aL << endl;
    // cout << "aR: " << aR << endl;
    // cout << "aStar: "<<  aStar << endl;
    // cout << "New Pstar:" << pStar << endl;
    // cin.get();

    interstate_left = calc_intermed_state1D(left_prims, left_state,
                                                aL, aStar, pStar, index);

    interstate_right = calc_intermed_state1D(right_prims, right_state,
                                                aR, aStar, pStar, index);
    

    // Compute the intermediate left flux
    interflux_left[0]  = left_flux[0] + aL*(interstate_left[0] - left_state[0]);
    interflux_left[1]  = left_flux[1] + aL*(interstate_left[1] - left_state[1]);
    interflux_left[2]  = left_flux[2] + aL*(interstate_left[2] - left_state[2]);

    // Compute the intermediate right flux
    interflux_right[0] = right_flux[0] + aR*(interstate_right[0] - right_state[0]);
    interflux_right[1] = right_flux[1] + aR*(interstate_right[1] - right_state[1]);
    interflux_right[2] = right_flux[2] + aR*(interstate_right[2] - right_state[2]);
    

    // cout << interflux_left[0] << endl;
    // cout << interflux_right[0] << endl;
    // cout << "aR: " << aR << endl;
    // cout << "aL: " << aL << endl;
    // cout << "aS: " << aStar << endl;
    // cin.get();

    
    if (0.0 <= aL){
    return left_flux;
    }  else if (aL <= 0.0 && 0.0 <= aStar){
        return interflux_left;
    } else if (aStar <= 0.0 && 0.0 <= aR){
        return interflux_right;
    } else {
        return right_flux;
    }
    
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

vector<vector<double> > Newtonian1D::u_dot1D(vector<vector<double> > &u_state, bool first_order=true, bool periodic = false, float theta = 1.5,
                                        bool linspace = true, bool hllc=false){
    int i_start, i_bound, coordinate;
    int grid_size = u_state[0].size();
    int n_vars = u_state.size();
    string default_coordinates = "cartesian";
    
    
    vector<double> u_l(n_vars), u_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double> prims_l(n_vars), prims_r(n_vars), f1(n_vars), f2(n_vars);
    
    
    if (first_order){
        int physical_grid;
        if (periodic){
            physical_grid = grid_size;
        } else {
            physical_grid = grid_size - 2;
        }
        vector<vector<double> > L(n_vars, vector<double> (physical_grid, 0));

        double dx = (r[physical_grid - 1] - r[0])/physical_grid;
        if (periodic){
            i_start = 0;
            i_bound = grid_size;
        } else{
            int true_npts = grid_size - 1;
            i_start = 1;
            i_bound = true_npts;
        }

        //==============================================
        //              CARTESIAN
        //==============================================
        if (coord_system == default_coordinates) {
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = roll(u_state[0], ii + 1);
                    u_r[1] = roll(u_state[1], ii + 1);
                    u_r[2] = roll(u_state[2], ii + 1);

                } else {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = u_state[0][ii + 1];
                    u_r[1] = u_state[1][ii + 1];
                    u_r[2] = u_state[2][ii + 1];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);
                
                double rho_l = prims_l[0];
                double rho_r = prims_r[0];

                double p_l = prims_l[1];
                double p_r = prims_r[1];

                double v_l = prims_l[2];
                double v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                if (hllc){
                    f1 = calc_hllc_flux1D(gamma, u_l, u_r, f_l, f_r,
                    prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);
                }
                

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l[0] = roll(u_state[0], ii - 1);
                    u_l[1] = roll(u_state[1], ii - 1);
                    u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);

                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i-1/2 interface
                if (hllc){
                    f2 = calc_hllc_flux1D(gamma, u_l, u_r, f_l, f_r,
                    prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);
                }
                // f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                L[0][coordinate] = - (f1[0] - f2[0])/dx;
                L[1][coordinate] = - (f1[1] - f2[1])/dx;
                L[2][coordinate] = - (f1[2] - f2[2])/dx;

        }
            
        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log(r[physical_grid - 1]) - log(r[0]))/physical_grid;

            double dr; 

            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = roll(u_state[0], ii + 1);
                    u_r[1] = roll(u_state[1], ii + 1);
                    u_r[2] = roll(u_state[2], ii + 1);

                } else {
                    // Shift the index for C++ [0] indexing
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    

                    u_r[0] = u_state[0][ii + 1];
                    u_r[1] = u_state[1][ii + 1];
                    u_r[2] = u_state[2][ii + 1];
                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);
                
                double rho_l = prims_l[0];
                double rho_r = prims_r[0];

                double p_l = prims_l[1];
                double p_r = prims_r[1];

                double v_l = prims_l[2];
                double v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Get the central pressure
                pc = prims_l[1];

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l[0] = roll(u_state[0], ii - 1);
                    u_l[1] = roll(u_state[1], ii - 1);
                    u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);

                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                if (linspace){
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell + r[coordinate]);
                    r_left = 0.5*(r[coordinate] + left_cell);

                } else {
                    log_rLeft = log(r[0]) + coordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = exp(log_rLeft);
                    r_right = exp(log_rRight);
                }

                dr = r_right - r_left;
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                L[0][coordinate] = - (r_right*r_right*f1[0] - r_left*r_left*f2[0] )/(volAvg*volAvg*dr);
                L[1][coordinate] = - (r_right*r_right*f1[1] - r_left*r_left*f2[1] )/(volAvg*volAvg*dr) + 2*pc/volAvg;
                L[2][coordinate] = - (r_right*r_right*f1[2] - r_left*r_left*f2[2] )/(volAvg*volAvg*dr);

            }
            
            
        }

        
        return L;
    } else {
        int physical_grid;
        if (periodic){
            physical_grid = grid_size;
        } else {
            physical_grid = grid_size - 4;
        }

        double dx = (r[physical_grid - 1] - r[0])/physical_grid;
        
        // Calculate the primitives for the entire state
        vector<vector<double> > prims(n_vars, vector<double>(grid_size, 0));
        vector<double> left_most(n_vars), left_mid(n_vars), center(n_vars);
        vector<double> right_mid(n_vars), right_most(n_vars);
        vector<vector<double> > L(n_vars, vector<double> (physical_grid, 0));


        prims = cons2prim1D(u_state);

        // The periodic BC doesn't require ghost cells. Shift the index
        // to the beginning since all of he.
        if (periodic){ 
            i_start = 0;
            i_bound = grid_size;
        } else {
            int true_npts = grid_size - 2;
            i_start = 2;
            i_bound = true_npts;
        }

        if (coord_system == default_coordinates){
            //==============================================
            //                  CARTESIAN
            //==============================================
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    coordinate = ii;
                    left_most[0] = roll(prims[0], ii - 2);
                    left_mid[0] = roll(prims[0], ii - 1);
                    center[0] = prims[0][ii];
                    right_mid[0] = roll(prims[0], ii + 1);
                    right_most[0] = roll(prims[0], ii + 2);

                    left_most[1] = roll(prims[1], ii - 2);
                    left_mid[1] = roll(prims[1], ii - 1);
                    center[1] = prims[1][ii];
                    right_mid[1] = roll(prims[1], ii + 1);
                    right_most[1] = roll(prims[1], ii + 2);

                    left_most[2] = roll(prims[2], ii - 2);
                    left_mid[2] = roll(prims[2], ii - 1);
                    center[2] = prims[2][ii];
                    right_mid[2] = roll(prims[2], ii + 1);
                    right_most[2] = roll(prims[2], ii + 2);
                    
                ;

                } else {
                    coordinate = ii - 2;
                    left_most[0] = prims[0][ii - 2];
                    left_mid[0] = prims[0][ii - 1];
                    center[0] = prims[0][ii];
                    right_mid[0] = prims[0][ii + 1];
                    right_most[0] = prims[0][ii + 2];

                    left_most[1] = prims[1][ii - 2];
                    left_mid[1] = prims[1][ii - 1];
                    center[1] = prims[1][ii];
                    right_mid[1] = prims[1][ii + 1];
                    right_most[1] = prims[1][ii + 2];

                    left_most[2] = prims[2][ii - 2];
                    left_mid[2] = prims[2][ii - 1];
                    center[2] = prims[2][ii];
                    right_mid[2] = prims[2][ii + 1];
                    right_most[2] = prims[2][ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Reconstructed right primitives vector
                prims_r[0] = right_mid[0] - 0.5*minmod(theta*(right_mid[0] - center[0]),
                                                    0.5*(right_most[0] - center[0]),
                                                    theta*(right_most[0] - right_mid[0]));

                prims_r[1] = right_mid[1] - 0.5*minmod(theta*(right_mid[1] - center[1]),
                                                    0.5*(right_most[1] - center[1]),
                                                    theta*(right_most[1] - right_mid[1]));

                prims_r[2] = right_mid[2] - 0.5*minmod(theta*(right_mid[2] - center[2]),
                                                    0.5*(right_most[2] - center[2]),
                                                    theta*(right_most[2] - right_mid[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                if (hllc){
                    f1 = calc_hllc_flux1D(gamma, u_l, u_r, f_l, f_r,
                    prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);
                }
                // f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l[0] = left_mid[0] + 0.5*minmod(theta*(left_mid[0] - left_most[0]),
                                                        0.5*(center[0] -left_most[0]),
                                                        theta*(center[0] - left_mid[0]));

                prims_l[1] = left_mid[1] + 0.5*minmod(theta*(left_mid[1] - left_most[1]),
                                                        0.5*(center[1] -left_most[1]),
                                                        theta*(center[1] - left_mid[1]));
                
                prims_l[2] = left_mid[2] + 0.5*minmod(theta*(left_mid[2] - left_most[2]),
                                                        0.5*(center[2] -left_most[2]),
                                                        theta*(center[2] - left_mid[2]));


                    
                prims_r[0] = center[0] - 0.5*minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_r[1] = center[1] - 0.5*minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_r[2] = center[2] - 0.5*minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                if (hllc){
                    f2 = calc_hllc_flux1D(gamma, u_l, u_r, f_l, f_r,
                    prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);
                }

                // f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                L[0][coordinate] = - (f1[0] - f2[0])/dx;
                L[1][coordinate] = - (f1[1] - f2[1])/dx;
                L[2][coordinate] = - (f1[2] - f2[2])/dx;
            }            
                                                                                                                         
            

        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log(r[physical_grid - 1]) - log(r[0]))/physical_grid;

            double dr;
            for (int ii=i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most[0] = roll(prims[0], ii - 2);
                    left_mid[0] = roll(prims[0], ii - 1);
                    center[0] = prims[0][ii];
                    right_mid[0] = roll(prims[0], ii + 1);
                    right_most[0] = roll(prims[0], ii + 2);

                    left_most[1] = roll(prims[1], ii - 2);
                    left_mid[1] = roll(prims[1], ii - 1);
                    center[1] = prims[1][ii];
                    right_mid[1] = roll(prims[1], ii + 1);
                    right_most[1] = roll(prims[1], ii + 2);

                    left_most[2] = roll(prims[2], ii - 2);
                    left_mid[2] = roll(prims[2], ii - 1);
                    center[2] = prims[2][ii];
                    right_mid[2] = roll(prims[2], ii + 1);
                    right_most[2] = roll(prims[2], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    coordinate = ii - 2;
                    left_most[0] = prims[0][ii - 2];
                    left_mid[0] = prims[0][ii - 1];
                    center[0] = prims[0][ii];
                    right_mid[0] = prims[0][ii + 1];
                    right_most[0] = prims[0][ii + 2];

                    left_most[1] = prims[1][ii - 2];
                    left_mid[1] = prims[1][ii - 1];
                    center[1] = prims[1][ii];
                    right_mid[1] = prims[1][ii + 1];
                    right_most[1] = prims[1][ii + 2];

                    left_most[2] = prims[2][ii - 2];
                    left_mid[2] = prims[2][ii - 1];
                    center[2] = prims[2][ii];
                    right_mid[2] = prims[2][ii + 1];
                    right_most[2] = prims[2][ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Reconstructed right primitives vector
                prims_r[0] = right_mid[0] - 0.5*minmod(theta*(right_mid[0] - center[0]),
                                                    0.5*(right_most[0] - center[0]),
                                                    theta*(right_most[0] - right_mid[0]));

                prims_r[1] = right_mid[1] - 0.5*minmod(theta*(right_mid[1] - center[1]),
                                                    0.5*(right_most[1] - center[1]),
                                                    theta*(right_most[1] - right_mid[1]));

                prims_r[2] = right_mid[2] - 0.5*minmod(theta*(right_mid[2] - center[2]),
                                                    0.5*(right_most[2] - center[2]),
                                                    theta*(right_most[2] - right_mid[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l[0] = left_mid[0] + 0.5 *minmod(theta*(left_mid[0] - left_most[0]),
                                                        0.5*(center[0] -left_most[0]),
                                                        theta*(center[0] - left_mid[0]));

                prims_l[1] = left_mid[1] + 0.5 *minmod(theta*(left_mid[1] - left_most[1]),
                                                        0.5*(center[1] -left_most[1]),
                                                        theta*(center[1] - left_mid[1]));
                
                prims_l[2] = left_mid[2] + 0.5 *minmod(theta*(left_mid[2] - left_most[2]),
                                                        0.5*(center[2] -left_most[2]),
                                                        theta*(center[2] - left_mid[2]));


                    
                prims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                //Get Central Pressure
                pc = center[1];

                if (linspace){
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell + r[coordinate]);
                    r_left = 0.5*(r[coordinate] + left_cell);

                } else {
                    log_rLeft = log(r[0]) + coordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = exp(log_rLeft);
                    r_right = exp(log_rRight);
                }
                
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );
                dr = r_right - r_left;

                L[0][coordinate] = - (r_right*r_right*f1[0] - r_left*r_left*f2[0] )/(volAvg*volAvg*dr);
                L[1][coordinate] = - (r_right*r_right*f1[1] - r_left*r_left*f2[1] )/(volAvg*volAvg*dr) + 2*pc/volAvg;
                L[2][coordinate] = - (r_right*r_right*f1[2] - r_left*r_left*f2[2] )/(volAvg*volAvg*dr);

            }
        
        }

        return L; 
    }
    
};


 vector<vector<double> > Newtonian1D::simulate1D(float tend = 0.1, float dt = 1.e-4, float theta=1.5,
                                            bool first_order = true, bool periodic = false, bool linspace = true,
                                            bool hllc=false){

    // Define the swap vector for the integrated state
    int grid_size = state[0].size();
    int n_vars = state.size();
    int i_real;

    vector<vector<double> > u_p(n_vars, vector<double>(grid_size, 0));
    vector<vector<double> > u(n_vars, vector<double>(grid_size, 0)); 
    vector<vector<double> > s(n_vars, vector<double>(grid_size, 0));
    float t = 0;

    // Copy the state array into real & profile variables
    u = state;
    u_p = u;

    if (first_order){
        int physical_grid;
        if (periodic){
            physical_grid = grid_size;
        } else {
            physical_grid = grid_size - 2;
        }
        vector<vector<double> > udot(n_vars, vector<double>(physical_grid, 0));

        // cout << "E Init: " << u[2][1] << endl;
        while (t < tend){
            // if (t == 0){
            //     u[1][0] = - u[1][1];
            // }

            // Compute the REAL udot array, purging the ghost cells.
            udot = u_dot1D(u, true, periodic, theta, linspace, hllc);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 1;
                }
                u_p[0][i_real] = u[0][i_real] + dt*udot[0][ii];
                u_p[1][i_real] = u[1][i_real] + dt*udot[1][ii];
                u_p[2][i_real] = u[2][i_real] + dt*udot[2][ii];

            }

            // Readjust the ghost cells at i-1,i+1 if not periodic
            // if (periodic == false){
            //     config_ghosts1D(u_p, physical_grid);
            // }
            

            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(u_p, r, linspace, first_order);
            }
            
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;

        }   

    } else {
        int physical_grid;
        if (periodic){
            physical_grid = grid_size;
        } else {
            physical_grid = grid_size - 4;
        }
        
        vector<vector<double> > u1(n_vars, vector<double>(grid_size, 0));
        vector<vector<double> > u2(n_vars, vector<double>(grid_size, 0));

        vector<vector<double> > udot(n_vars, vector<double>(physical_grid, 0));
        vector<vector<double> > udot1(n_vars, vector<double>(physical_grid, 0));
        vector<vector<double> > udot2(n_vars, vector<double>(physical_grid, 0));

        u1 = u;
        u2 = u;
        while (t < tend){
            // Compute the REAL udot array, purging the ghost cells.
            // if (t == 0){
            //     u[1][0] = - u[1][2];
            //     u[1][1] = - u[1][2];
            // }

            udot = u_dot1D(u, false, periodic, theta, linspace, hllc);
            
            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 2;
                }
                u1[0][i_real] = u[0][i_real] + dt*udot[0][ii];
                u1[1][i_real] = u[1][i_real] + dt*udot[1][ii];
                u1[2][i_real] = u[2][i_real] + dt*udot[2][ii];

            }
            
            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            // if (periodic == false){
            //     config_ghosts1D(u1, physical_grid, false);
            // }
            
            udot1 = u_dot1D(u1, false, periodic, theta, linspace, hllc);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 2;
                }
                u2[0][i_real] = 0.75*u[0][i_real] + 0.25*u1[0][i_real] + 0.25*dt*udot1[0][ii];
                u2[1][i_real] = 0.75*u[1][i_real] + 0.25*u1[1][i_real] + 0.25*dt*udot1[1][ii];
                u2[2][i_real] = 0.75*u[2][i_real] + 0.25*u1[2][i_real] + 0.25*dt*udot1[2][ii];

            }

            
            // if (periodic == false){
            //     config_ghosts1D(u2, physical_grid, false);
            // }
            
            udot2 = u_dot1D(u2, false, periodic, theta, linspace, hllc);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                if (periodic){
                    i_real = ii;
                } else {
                    i_real = ii + 2;
                }
                u_p[0][i_real] = (1.0/3.0)*u[0][i_real] + (2.0/3.0)*u2[0][i_real] + (2.0/3.0)*dt*udot2[0][ii];
                u_p[1][i_real] = (1.0/3.0)*u[1][i_real] + (2.0/3.0)*u2[1][i_real] + (2.0/3.0)*dt*udot2[1][ii];
                u_p[2][i_real] = (1.0/3.0)*u[2][i_real] + (2.0/3.0)*u2[2][i_real] + (2.0/3.0)*dt*udot2[2][ii];
                
            }


            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            // if (periodic == false){
            //     config_ghosts1D(u_p, physical_grid, false);
            // }
            

            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(u_p, r, linspace, first_order);
            }
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;

        }  

    }

    // write_data(u, tend, "sod");
    return u;

 };
