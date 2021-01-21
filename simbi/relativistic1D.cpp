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
#include <iomanip>

using namespace std;
using namespace states;


// Default Constructor 
UstateSR::UstateSR () {}

// Overloaded Constructor
UstateSR::UstateSR(vector< vector<double> > u_state, float Gamma, double cfl, vector<double> R = {0},
                string Coord_system = "cartesian")
{
    state = u_state;
    gamma = Gamma;
    r = R;
    coord_system = Coord_system;
    CFL = cfl;

}

// Destructor 
UstateSR::~UstateSR() {}

//================================================
//              DATA STRUCTURES
//================================================
struct primitives {
    double rho;
    double v;
    double p;
};

struct conserveSR1D {
    double D;
    double S;
    double tau;
};

struct fluxSR1D {
    double D;
    double S;
    double tau;
};

struct eigenvals
{
    double aL;
    double aR;
};

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
vector<double>  cons2prim(float gamma, vector<double>  &u_state, double lorentz_gamma){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    vector<double> prims(3);

    double D = u_state[0];
    double S = u_state[1];
    double tau = u_state[2];

    double pmin = abs(abs(S) - tau - D);

    // cout << "Pmin: " << pmin << endl;
    
    double pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, lorentz_gamma, gamma, S);

    // cout << "NR Pressure: " << pressure << endl;

    double v = S/(tau + pressure + D);

    double Wnew = calc_lorentz_gamma(v);

    

    double rho = D/Wnew;

    prims[0] = rho;
    prims[1] = pressure;
    prims[2] = v;

    return prims;
    
    
};

vector<vector<double> > UstateSR::cons2prim1D(vector<vector<double> > &u_state, vector<double> &lorentz_gamma){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, S, D, tau, pmin;
    double v, pressure, W;
    int n_vars = u_state.size();
    int n_gridpts = u_state[0].size();

    vector < vector<double> > prims(n_vars, vector<double> (n_gridpts, 0));
    
    for (int ii=0; ii < n_gridpts; ii++){
        D = u_state[0][ii];
        S = u_state[1][ii];
        tau = u_state[2][ii];
        W = lorentz_gamma[ii];

        pmin = abs( abs(S) - tau - D);

        pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, W, gamma, S);

        v = S/(tau + D + pressure);

        W = calc_lorentz_gamma(v);

        rho = D/W;

        prims[0][ii] = rho;
        prims[1][ii] = pressure;
        prims[2][ii] = v;
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
map<string, map<string, double > > calc_eigenvals(float gamma, vector<double> &prims_l,
                                            vector<double> &prims_r, int dummy=0)
{
    
    // Initialize your important variables
    double v_r, v_l, p_r, p_l, cs_r, cs_l, D_l, D_r, tau_l, tau_r; 
    double rho_l, rho_r, vbar ,cbar, h_l ,h_r, W_l, W_r, aL, aR;
    double sL,sR, lammy_l, lammy_r, lam_l, lam_r;
    map<string, map<string, double > > lambda;
    
    // vector<double> prims_l, prims_r;


    // Separate the left and right state components
    rho_l = prims_l[0];
    p_l = prims_l[1];
    v_l = prims_l[2];
    h_l = calc_enthalpy(gamma, rho_l, p_l);
    W_l = calc_lorentz_gamma(v_l);
    D_l = W_l*rho_l;
    tau_l = rho_l*h_l*W_l*W_l - p_l - rho_l*W_l;



    rho_r = prims_r[0];
    p_r = prims_r[1];
    v_r = prims_r[2];
    h_r = calc_enthalpy(gamma, rho_r, p_r);
    W_r = calc_lorentz_gamma(v_r);
    D_r = W_r*rho_r;
    tau_r = rho_r*h_r*W_r*W_r- p_r - rho_r*W_r;
   
    // if (dummy = 1){
    //     tau_l = tau_l + D_l;
    //     tau_r = tau_r + D_r;
    // }

    cs_r = calc_rel_sound_speed(p_r, D_r, tau_r, W_r, gamma);
    cs_l = calc_rel_sound_speed(p_l, D_l, tau_l, W_l, gamma);

    vbar = 0.5*(v_l + v_r);
    cbar = 0.5*(cs_l + cs_r);
    aL = (vbar - cbar)/(1 - vbar*cbar );
    aR = (vbar + cbar)/(1 + vbar*cbar );

    
    // Populate the Dictionary
    lambda["left"]["plus"] = (v_l + cs_l)/(1 + v_l*cs_l); 
    lambda["left"]["minus"] = (v_l - cs_l)/(1 - v_l*cs_l); 
    lambda["right"]["plus"] = (v_r + cs_r)/(1 + v_r*cs_r); 
    lambda["right"]["minus"] = (v_r - cs_r)/(1 - v_r*cs_r); 

    // Approximate the spreading of the characteristics in the case of a 
    // rarefaction wave according to Schneider et al. (1992)
    lambda["signal"]["aL"] = min(aL, (v_l - cs_l)/(1 - v_l*cs_l));
    lambda["signal"]["aR"] = max(aR, (v_r + cs_r)/(1 + v_r*cs_r));

    // cout << "aL Mine: " << lambda["signal"]["aL"] << endl;
    // cout << "aR Mine: " << lambda["signal"]["aR"] << endl;
    // cs_l = sqrt(gamma*p_l/(h_l*rho_l));
    // cs_r = sqrt(gamma*p_r/(h_r*rho_r));

    sL = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
    sR = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
    lammy_l = (v_l - sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    lammy_r = (v_r - sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);
    lam_l = (v_l + sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    lam_r = (v_r + sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);

    aL = min(lammy_l, lammy_r);
    aR = max(lam_l, lam_r);

    lambda["signal"]["aL"] = aL;
    lambda["signal"]["aR"] = aR;
    // cout << "aL: Mignone" << aL << endl;
    // cout << "aR: Mignone" << aR << endl;
    // cin.get();

    
    // cout << "Left-PLus: " << lambda["left"]["plus"] << endl;
    // cout << "Left-Minus: " << lambda["left"]["minus"] << endl;
    // cout << "Right-PLus: " << lambda["right"]["plus"] << endl;
    // cout << "Right-Minus: " << lambda["right"]["minus"] << endl;

    // cin.get();

    return lambda;
};

// Adapt the CFL conditonal timestep
double UstateSR::adapt_dt(vector<vector<double> > &prims,
                                bool linspace=true, bool first_order=true, bool periodic=false){

    double r_left, r_right, left_cell, right_cell, dr, cs;
    double delta_logr, log_rLeft, log_rRight, min_dt, cfl_dt;
    double D, tau, h, rho, p, v, vPLus, vMinus, W;
    int shift_i, physical_grid;

    // Get the primitive vector 
    // vector<vector<double> >  prims = cons2prim1D(u_state, lorentz_gamma);

    int grid_size = prims[0].size();

    min_dt = 0;
    // Find the minimum timestep over all i
    if (periodic){
        physical_grid = grid_size;
    } else{
        if (first_order){
            physical_grid = grid_size - 2;
        } else {
            physical_grid = grid_size - 4;
        }

    }
    

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < physical_grid; ii++){
        if (periodic){
            shift_i = ii;
        } else {
            if (first_order){
                shift_i = ii + 1;
            } else {
                shift_i = ii + 2;
            }

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
            // delta_logr = (log10(r[physical_grid - 1]) - log10(r[0]))/physical_grid;
            // log_rLeft = log10(r[0]) + ii*delta_logr;
            // log_rRight = log_rLeft + delta_logr;

            r_left = right_cell; //+ r[ii]);
            r_right = left_cell; //+ r[ii]);
        }

        dr = r_right - r_left;
        rho = prims[0][shift_i];
        p = prims[1][shift_i];
        v = prims[2][shift_i];
        h = calc_enthalpy(gamma, rho, p);
        W = calc_lorentz_gamma(v);

        D = rho*W;
        tau = rho*h*W*W - p - rho*W;

        
        cs = calc_rel_sound_speed(p, D, tau, W, gamma);


        vPLus = (v + cs)/(1 + v*cs);
        vMinus = (v - cs)/(1 - v*cs);

        cfl_dt = dr/(max(abs(vPLus), abs(vMinus)));

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
vector<double>  calc_state(float gamma, double rho, double pressure, double v)
{
    // int n_vars = 3;
    vector<double> cons_state(3);

    // cons_state.resize(n_vars, vector<double> (1, -1));
    double D, S,  tau, W, h;

    h = calc_enthalpy(gamma, rho, pressure);
    W = calc_lorentz_gamma(v);

    D = rho*W;
    S = rho*h*W*W*v;
    tau = rho*h*W*W - pressure - rho*W;
    
    cons_state[0] = D; 
    cons_state[1] = S;
    cons_state[2] = tau;

    return cons_state;
};

vector<double> calc_hll_state1DSR(float gamma,
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

    lambda = calc_eigenvals(gamma, left_prims, right_prims, 1);

    aL = lambda["signal"]["aL"];
    aR = lambda["signal"]["aR"];

    // aL = findMin(0.0, lambda["left"]["minus"], lambda["right"]["minus"]);
    // aR = findMax(0.0, lambda["left"]["plus"], lambda["right"]["plus"]);

    hll_states[0] = ( aR*right_state[0] - aL*left_state[0] 
                        - right_flux[0] + left_flux[0])/(aR - aL);

    hll_states[1] = ( aR*right_state[1] - aL*left_state[1] 
                        - right_flux[1] + left_flux[1])/(aR - aL);

    hll_states[2] = ( aR*right_state[2] - aL*left_state[2] 
                        - right_flux[2] + left_flux[2])/(aR - aL);


    return hll_states;

}

vector<double> calc_intermed_state(vector<double> prims,
                                vector<double> state,
                                double a,
                                double aStar,
                                double pStar,
                                int index=1)
{
    double pressure, v1, S1, D, tau;
    double DStar, S1star, tauStar;
    map<string, map<string, double > > lambda; 
    vector<double> hll_states(3);

    pressure = prims[1];
    v1 = prims[2];

    D = state[0];
    S1 = state[1];
    tau = state[2];

    DStar = ( (a - v1)/(a - aStar))*D;
    S1star = (1./(a-aStar))*(S1*(a - v1) - pressure + pStar);
    tauStar = (1./(a-aStar))*(tau*(a - v1) + pStar*aStar - (pressure)*v1);

    hll_states[0] = DStar;
    hll_states[1] = S1star;
    hll_states[2] = tauStar;
    
    return hll_states;
}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
vector<double> calc_flux(double gamma, double rho, double pressure, double v){
    
    // The Flux Tensor
    vector<double> flux(3);

    // The Flux components
    double mom, energy_dens, zeta, D, S, tau, h, W;

    W = calc_lorentz_gamma(v);
    h = calc_enthalpy(gamma, rho, pressure);
    D = rho*W;
    S = rho*h*W*W*v;
    tau = rho*h*W*W - pressure - W*rho;
    
    mom = D*v;
    energy_dens = S*v + pressure;
    zeta = (tau + pressure)*v;
    flux[0] = mom;
    flux[1] = energy_dens;
    flux[2] = zeta;

    return flux;
};

vector<double> calc_hll_flux(float gamma, vector<double> &left_prims, 
                                        vector<double> &right_prims,
                                        vector<double> &left_state,
                                        vector<double> &right_state,
                                        vector<double> &left_flux,
                                        vector<double> &right_flux,
                                        int dummy = 0)
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(3);
    double alpha_plus, alpha_minus, aR, aL;  
    
    lambda = calc_eigenvals(gamma, left_prims, right_prims, dummy);

    // Grab the necessary wave speeds
    aR = lambda["signal"]["aR"];
    aL = lambda["signal"]["aL"];

    alpha_minus = max(0.0, - aL);
    alpha_plus = max(0.0, aR);

    // alpha_minus = findMax(0.0, - lambda["left"]["minus"], - lambda["right"]["minus"]);
    // alpha_plus = findMax(0.0, lambda["left"]["plus"], lambda["right"]["plus"]);

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

vector<double> calc_hllc_fluxSR1D(float gamma,
                                vector<double> &left_state,
                                vector<double> &right_state,
                                vector<double> &left_flux,
                                vector<double> &right_flux,
                                vector<double> &left_prims,
                                vector<double> &right_prims,
                                double lorentz_gamma,
                                int index = 1
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
    double aL, aR, aStar, pStar;  
    
    lambda = calc_eigenvals(gamma, left_prims, right_prims, 1);

    aL = lambda["signal"]["aL"];
    aR = lambda["signal"]["aR"];

    // aR = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
    // aL = findMin(0 , lambda["left"]["minus"], lambda["right"]["minus"]);

    // cout << "aStar1: " << aStar << endl;
    // aStar = 0.5*(aL + aR);

    // cout << aStar << endl;

    // cout << "aL: " << aL << endl;
    // cout << "aR: " << aR << endl;
    // cout << "aSta2r: " << aStar << endl;
    // cout << "pStar: " << pStar << endl;
    // cin.get();
    // aStar = 1.;

    hll_flux = calc_hll_flux(gamma, left_prims, right_prims, left_state, right_state, left_flux,
                                    right_flux, 1);

    hll_state = calc_hll_state1DSR(gamma, left_state, right_state, left_flux,
                                     right_flux, left_prims, right_prims);

    double E_mass = hll_state[2] + hll_state[0];    
    double S_hll = hll_state[1];
    double F_S = hll_flux[1];
    double F_E = hll_flux[2] + hll_flux[0];
    
    aStar = calc_intermed_wave(E_mass, S_hll, F_S, F_E);

    pStar = -F_E*aStar + F_S;
    // cout << "aL: " << aL << endl;
    // cout << "aR: " << aR << endl;
    // cout << "aStar: " << aStar << endl;
    // cout << "Pstar:" << pStar << endl;
    // cin.get();
    interstate_left = calc_intermed_state(left_prims, left_state,
                                                aL, aStar, pStar, index);

    interstate_right = calc_intermed_state(right_prims, right_state,
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
    }  else if (aL < 0.0 && 0.0 <= aStar){
        return interflux_left;
    } else if (aStar < 0.0 && 0.0 <= aR){
        return interflux_right;
    } else {
        return right_flux;
    }
    
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

vector<vector<double> > UstateSR::u_dot1D(vector<vector<double> > &u_state, vector<double> &lorentz_gamma,
                                            vector<vector<double> > &sources,
                                            bool first_order=true, 
                                            bool periodic = false, 
                                            float theta = 1.5,
                                            bool linspace = true,
                                            bool hllc=false){
    int i_start, i_bound, coordinate;
    int grid_size = u_state[0].size();
    int n_vars = u_state.size();
    double sourceD, sourceR, source0;
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
            double rho_l, rho_r, v_l, v_r, p_l, p_r, W_l, W_r;
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

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii];
                    W_r = roll(lorentz_gamma, ii + 1);

                } else {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = u_state[0][ii + 1];
                    u_r[1] = u_state[1][ii + 1];
                    u_r[2] = u_state[2][ii + 1];

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii];
                    W_r = lorentz_gamma[ii + 1];

                }

                prims_l = cons2prim(gamma, u_l, W_l);
                prims_r = cons2prim(gamma, u_r, W_r);
                
                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux(gamma, rho_l, p_l, v_l);
                f_r = calc_flux(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                if (hllc){
                    f1 = calc_hllc_fluxSR1D(gamma, u_l, u_r, f_l, f_r, prims_l, prims_r, lorentz_gamma[ii]);
                } else {
                    f1 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l[0] = roll(u_state[0], ii - 1);
                    u_l[1] = roll(u_state[1], ii - 1);
                    u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                    // Grab the Lorentz Factors
                    W_l = roll(lorentz_gamma, ii - 1);
                    W_r = lorentz_gamma[ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii - 1];
                    W_r = lorentz_gamma[ii];

                }

                prims_l = cons2prim(gamma, u_l, W_l);
                prims_r = cons2prim(gamma, u_r, W_r);


                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux(gamma, rho_l, p_l, v_l);
                f_r = calc_flux(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i-1/2 interface
                if (hllc){
                    f2 = calc_hllc_fluxSR1D(gamma, u_l, u_r, f_l, f_r, prims_l, prims_r, lorentz_gamma[ii]);
                } else {
                    f2 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                // f2 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);

                sourceD = sources[0][coordinate];
                sourceR = sources[1][coordinate];
                source0 = sources[2][coordinate];

                L[0][coordinate] = - (f1[0] - f2[0])/dx + sourceD;
                L[1][coordinate] = - (f1[1] - f2[1])/dx + sourceR;
                L[2][coordinate] = - (f1[2] - f2[2])/dx + source0;

        }
            
        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight, rho_l, rho_r;
            double v_l, v_r, p_l, p_r, W_l, W_r;

            // double delta_logr = (log10(r[physical_grid - 1]) - log10(r[0]))/physical_grid;

            double dr = 0; 
            vector<vector<double> > prims = cons2prim1D(u_state, lorentz_gamma);
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

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii];
                    W_r = roll(lorentz_gamma, ii + 1);

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

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii];
                    W_r = lorentz_gamma[ii + 1];
                }

                prims_l = cons2prim(gamma, u_l, W_l);
                prims_r = cons2prim(gamma, u_r, W_r);
                
                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux(gamma, rho_l, p_l, v_l);
                f_r = calc_flux(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);

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

                    // Grab the Lorentz Factors
                    W_l = roll(lorentz_gamma, ii - 1);
                    W_r = lorentz_gamma[ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                    // Grab the Lorentz Factors
                    W_l = lorentz_gamma[ii - 1];
                    W_r = lorentz_gamma[ii];

                }

                
                // cout << "D (L): " << u_l[0] << endl;
                // cout << "S (L): " << u_l[1] << endl;
                // cout << "Tau (L): " << u_l[2] << endl;
                

                prims_l = cons2prim(gamma, u_l, W_l);
                prims_r = cons2prim(gamma, u_r, W_r);

                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux(gamma, rho_l, p_l, v_l);
                f_r = calc_flux(gamma, rho_r, p_r, v_r);

                
                // cout << "Rho L: " << prims_l[0] << endl;
                // cout << "P L: " << prims_l[1] << endl;
                // cout << "V L: " << prims_l[2] << endl;
                // string b;
                // cin >> b;
                

                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);

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
                    // log_rLeft = log10(r[0]) + coordinate*delta_logr;
                    // log_rRight = log_rLeft + delta_logr;
                    // r_left = pow(10, log_rLeft);
                    // r_right = pow(10, log_rRight);
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell +r[coordinate]);
                    r_left = 0.5*(left_cell+ r[coordinate]);

                }
                
                /**
                cout << "F1: " << f1[0] << endl;
                cout << "F2: " << f2[0] << endl;
                string a;
                cin >> a;
                */

                dr = r_right - r_left;
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                

                sourceD = sources[0][coordinate];
                sourceR = sources[1][coordinate];
                source0 = sources[2][coordinate];

                L[0][coordinate] = - (r_right*r_right*f1[0] - r_left*r_left*f2[0] )/(volAvg*volAvg*dr) + sourceD;
                L[1][coordinate] = - (r_right*r_right*f1[1] - r_left*r_left*f2[1] )/(volAvg*volAvg*dr) + 2*pc/volAvg + sourceR;
                L[2][coordinate] = - (r_right*r_right*f1[2] - r_left*r_left*f2[2] )/(volAvg*volAvg*dr) + source0;

                
                // cout << "D Dot: " << L[0][coordinate] << endl;
                // cout << "S Dot: " << L[1][coordinate] << endl;
                // cout << "Tau Dot: " << L[2][coordinate] << endl;
                // cout << "dr: " << dr << endl;
                // cout << "volAvg: " << volAvg << endl;
                // cin.get();
                
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
        double W_l, W_r;
        
        // Calculate the primitives for the entire state
        vector<vector<double> > prims(n_vars, vector<double>(grid_size, 0));
        vector<double> left_most(n_vars), left_mid(n_vars), center(n_vars);
        vector<double> right_mid(n_vars), right_most(n_vars);
        vector<vector<double> > L(n_vars, vector<double> (physical_grid, 0));


        prims = cons2prim1D(u_state, lorentz_gamma);

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
                u_l = calc_state(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux(gamma, prims_r[0], prims_r[1], prims_r[2]);

                W_l = calc_lorentz_gamma(prims_l[2]);
                W_r = calc_lorentz_gamma(prims_r[2]);

                if (hllc){
                    f1 = calc_hllc_fluxSR1D(gamma, u_l, u_r, f_l, f_r, prims_l, prims_r, lorentz_gamma[ii]);
                } else {
                    f1 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                // f1 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);

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
                u_l = calc_state(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux(gamma, prims_r[0], prims_r[1], prims_r[2]);

                W_l = calc_lorentz_gamma(prims_l[2]);
                W_r = calc_lorentz_gamma(prims_r[2]);

                if (hllc){
                    f2 = calc_hllc_fluxSR1D(gamma, u_l, u_r, f_l, f_r, prims_l, prims_r, lorentz_gamma[ii]);
                } else {
                    f2 = calc_hll_flux(gamma, prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                

                // cout << "F1: " << f1[0] << endl;
                // cout << "F2: " << f2[0] << endl;
                // string a;
                // cin >>a;

                sourceD = sources[0][coordinate];
                sourceR = sources[1][coordinate];
                source0 = sources[2][coordinate];


                L[0][coordinate] = - (f1[0] - f2[0])/dx  + sourceD;
                L[1][coordinate] = - (f1[1] - f2[1])/dx  + sourceR;
                L[2][coordinate] = - (f1[2] - f2[2])/dx  + source0;

                /**
                cout << "D Dot: " << L[0][coordinate] << endl;
                cout << "S Dot: " << L[1][coordinate] << endl;
                cout << "Tau Dot: " << L[2][coordinate] << endl;
                cin.get();
                */
            }            
                                                                                                                         
            

        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log10(r[physical_grid - 1]) - log10(r[0]))/physical_grid;

            double dr = 0;
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
                u_l = calc_state(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux(gamma, prims_r[0], prims_r[1], prims_r[2]);

                W_l = calc_lorentz_gamma(prims_l[2]);
                W_r = calc_lorentz_gamma(prims_r[2]);

                f1 = calc_hll_flux(gamma,prims_l, prims_r,  u_l, u_r, f_l, f_r);

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
                u_l = calc_state(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux(gamma, prims_r[0], prims_r[1], prims_r[2]);

                W_l = calc_lorentz_gamma(prims_l[2]);
                W_r = calc_lorentz_gamma(prims_r[2]);

                f2 = calc_hll_flux(gamma, prims_l, prims_r,  u_l, u_r, f_l, f_r);

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
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell + r[coordinate]);
                    r_left = 0.5*(left_cell + r[coordinate]); 
                }
                
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );
                dr = r_right - r_left;

                sourceD = sources[0][coordinate];
                sourceR = sources[1][coordinate];
                source0 = sources[2][coordinate];

                L[0][coordinate] = - (r_right*r_right*f1[0] - r_left*r_left*f2[0] )/(volAvg*volAvg*dr) + sourceD;
                L[1][coordinate] = - (r_right*r_right*f1[1] - r_left*r_left*f2[1] )/(volAvg*volAvg*dr) + 2*pc/volAvg + sourceR;
                L[2][coordinate] = - (r_right*r_right*f1[2] - r_left*r_left*f2[2] )/(volAvg*volAvg*dr) + source0;

            }
        
        }

        return L; 
    }
    
};


 vector<vector<double> > UstateSR::simulate1D(vector<double> &lorentz_gamma, vector<vector<double> > &sources,
                                            float tend = 0.1, float dt = 1.e-4, float theta=1.5,
                                            bool first_order = true, bool periodic = false, bool linspace = true,
                                            bool hllc = false){

    // Define the swap vector for the integrated state
    int grid_size = state[0].size();
    int n_vars = state.size();
    int i_real;

    vector<vector<double> > u_p(n_vars, vector<double>(grid_size, 0));
    vector<vector<double> > u(n_vars, vector<double>(grid_size, 0)); 
    vector<vector<double> > prims(n_vars, vector<double>(grid_size, 0));
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
            if (t == 0){
                config_ghosts1D(u, grid_size);
            }

            // Compute the REAL udot array, purging the ghost cells.
            udot = u_dot1D(u, lorentz_gamma, sources, true, periodic, theta, linspace, hllc);

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
            if (periodic == false){
                config_ghosts1D(u_p, grid_size);
            }

            prims = cons2prim1D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[2]);

            

            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(prims, linspace, first_order, periodic);
            }
            if (isnan(dt)){
                break;
            }

            // cin.get();

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
        vector<vector<double> > prims(n_vars, vector<double>(grid_size, 0));

        vector<vector<double> > udot(n_vars, vector<double>(physical_grid, 0));
        vector<vector<double> > udot1(n_vars, vector<double>(physical_grid, 0));
        vector<vector<double> > udot2(n_vars, vector<double>(physical_grid, 0));


        u1 = u;
        u2 = u;
        u_p = u;
        while (t < tend){
            // Compute the REAL udot array, purging the ghost cells.
            if (t == 0){
                config_ghosts1D(u, grid_size, first_order);
            }

            // cout << "Blandford McKee" << endl;
            udot = u_dot1D(u, lorentz_gamma, sources, false, periodic, theta, linspace, hllc);
            
            // for (int ii = 0; ii < grid_size; ii++){
            //     cout << u[0][ii] << ", " << endl;
            // }
            // cin.get();
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
            if (periodic == false){
                config_ghosts1D(u1, grid_size, false);
            }

            prims = cons2prim1D(u1, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[2]);
            
            // dt = adapt_dt(prims, linspace, first_order, periodic);

            
            udot1 = u_dot1D(u1, lorentz_gamma, sources, false, periodic, theta, linspace, hllc);

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

            
            if (periodic == false){
                config_ghosts1D(u2, grid_size, false);
            }

            prims = cons2prim1D(u2, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[2]);
            
            udot2 = u_dot1D(u2, lorentz_gamma, sources, false, periodic, theta, linspace, hllc);

            // dt = adapt_dt(prims, linspace, first_order, periodic); 

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
            if (periodic == false){
                config_ghosts1D(u_p, grid_size, false);
            }

            // for (int ii =0; ii < grid_size; ii++){
            //     cout << u_p[0][ii] << endl;
            // }
            // cin.get();
            prims = cons2prim1D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[2]);
            
            // dt = adapt_dt(prims, linspace, false, false);
            
            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(prims, linspace, first_order, periodic);
            }
            
            

            // Swap the arrays
            u.swap(u_p);

            
            cout << "\r" << "dt: " << dt << " " << "t: " << t << flush;
            t += dt;

            

            // cout << t << endl;

        }  

    }
    prims = cons2prim1D(u, lorentz_gamma);

    return prims;

 };
