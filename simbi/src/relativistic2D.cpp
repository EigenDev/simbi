/* 
* C++ Library to perform 2D hydro calculations
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
#include <chrono>

using namespace std;
using namespace states;
using namespace chrono;

// Default Constructor 
UstateSR2D::UstateSR2D () {}

// Overloaded Constructor
UstateSR2D::UstateSR2D(vector<vector< vector<double> > > u_state2D, float Gamma, vector<double> X1, 
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
UstateSR2D::~UstateSR2D() {}


// Get the 2-Dimensional (4, 1) tensor for computation. 
// It is being doing pointwise in this case as opposed to over
// the entire array since we are in c++
struct conserveSR2D
{
    double D;
    double S1;
    double S2;
    double tau;
    conserveSR2D() : S1(0) {}
    ~conserveSR2D() {}
    double momentum(int nhat)
    {
        if (nhat == 1.){
            return S1;
        } else {
            return S2;
        }
    }
};

/* Define a similar struct for the corresponding fluxes */
struct fluxSR2D
{
    double D;
    double S1;
    double S2;
    double tau;

    fluxSR2D() : S1(0) {}
    ~fluxSR2D() {}
    double momentum(int nhat)
    {
        if (nhat == 1.){
            return S1;
        } else {
            return S2;
        }
    }
};

struct primitives {
    double rho;
    double v1;
    double v2;
    double p;
};

struct eigenvals{
    double aL;
    double aR;
};

//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
primitives cons2primSR(float gamma, conserveSR2D  &u_state, double lorentz_gamma){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    primitives prims;
    double D   = u_state.D;
    double S1  = u_state.S1;
    double S2  = u_state.S2;
    double tau = u_state.tau;
    
    double S = sqrt(S1*S1 + S2*S2);
    
    double pmin = abs(S - tau - D);

    double pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, lorentz_gamma, gamma, S);

    double v1 = S1/(tau + pressure + D);

    double v2 = S2/(tau + pressure + D);

    double vtot = sqrt(v1*v1 + v2*v2);

    double Wnew = 1./sqrt(1 - vtot*vtot);

    double rho = D/Wnew;

    // p0 = pressure;

    prims.rho = rho;
    prims.v1  = v1;
    prims.v2  = v2;
    prims.p   = pressure;

    return prims;
    
};


vector<vector< vector<double> > > UstateSR2D::cons2prim2D(vector<vector< vector<double> > > &u_state2D,
                                                            vector<vector<double> > &lorentz_gamma){
    /**
     * Return a 2D matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    double rho, S1,S2, S, D, tau, pmin, tol;
    double pressure, W;
    double v1, v2, vtot;
     
    int n_vars = u_state2D.size();
    int ny_gridpts = u_state2D[0].size();
    int nx_gridpts = u_state2D[0][0].size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (ny_gridpts, vector<double>(nx_gridpts)));
   
    double epsilon, D_0, S1_0, S2_0, tau_0;
    double S0, Sn, D_n, S1_n, S2_n, tau_n, pn, p0;
    
    for (int jj=0; jj < ny_gridpts; jj++){
        for(int ii=0; ii< nx_gridpts; ii++){
            D   =  u_state2D[0][jj][ii];      // Relativistic Density
            S1  =  u_state2D[1][jj][ii];      // X1-Momentum Denity
            S2  =  u_state2D[2][jj][ii];      // x2-Momentum Density
            tau =  u_state2D[3][jj][ii];      // Energy Density
            W   =  lorentz_gamma[jj][ii]; 

            S = sqrt(S1*S1 + S2*S2);

            pmin = abs(S - tau - D);

            tol = 1.e-6; //D*1.e-12;

            pressure = newton_raphson(pmin, pressure_func, dfdp, tol, D, tau, W, gamma, S);

            v1 = S1/(tau + D + pressure);

            v2 = S2/(tau + D + pressure);

            vtot = sqrt( v1*v1 + v2*v2 );

            W = 1./sqrt(1 - vtot*vtot);

            rho = D/W;

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = v1;
            prims[2][jj][ii] = v2;
            prims[3][jj][ii] = pressure;
            

        }
    }
    

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
eigenvals calc_eigenvals(float gamma, primitives &prims_l,
                                      primitives &prims_r,
                                      unsigned int nhat = 1)
{

    // Initialize your important variables
    double v1_r, v1_l, v2_l, v2_r, p_r, p_l, cs_r, cs_l, vtot_l, vtot_r, D_r, D_l ,tau_r, tau_l; 
    double rho_l, rho_r, W_l, W_r, h_l, h_r, v, qL, qR, pStar, rhoBar, vStar;
    double sL,sR, lamLp, lamRp, lamLm, lamRm;
    eigenvals lambda;

    // Separate the left and right primitives
    rho_l = prims_l.rho;
    p_l   = prims_l.p;
    v1_l  = prims_l.v1;
    v2_l  = prims_l.v2;
    v     = sqrt(v1_l*v1_l + v2_l*v2_l);
    W_l   = 1./sqrt(1 - v*v);
    h_l   = 1. + gamma*p_l/(rho_l*(gamma - 1));

    rho_r = prims_r.rho;
    p_r   = prims_r.p;
    v1_r  = prims_r.v1;
    v2_r  = prims_r.v2;
    v     = sqrt(v1_r*v1_r + v2_r*v2_r);
    W_r   = 1./sqrt(1 - v*v);
    h_r   = 1. + gamma*p_r/(rho_r*(gamma - 1));

    
    D_l = W_l*rho_l;
    D_r = W_r*rho_r;
    tau_l = rho_l*h_l*W_l*W_l - p_l - rho_l*W_l;
    tau_r = rho_r*h_r*W_r*W_r - p_r - rho_r*W_r;

    cs_r = calc_rel_sound_speed(p_r, D_r, tau_r, W_r, gamma);
    cs_l = calc_rel_sound_speed(p_l, D_l, tau_l, W_l, gamma);
 
    switch (nhat){
        case 1:
            // Calc the wave speeds based on Mignone and Bodo (2005)
            sL    = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
            sR    = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
            lamLm = (v1_l - sqrt(sL*(1 - v1_l*v1_l + sL)))/(1 + sL);
            lamRm = (v1_r - sqrt(sR*(1 - v1_r*v1_r + sR)))/(1 + sR);
            lamRp = (v1_l + sqrt(sL*(1 - v1_l*v1_l + sL)))/(1 + sL);
            lamLp = (v1_r + sqrt(sR*(1 - v1_r*v1_r + sR)))/(1 + sR);

            lambda.aL = min(lamLm, lamRm); 
            lambda.aR = max(lamLp, lamRp); 

            break;
        case 2:
            // Calc the wave speeds based on Mignone and Bodo (2005)
            sL    = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
            sR    = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
            lamLm = (v2_l - sqrt(sL*(1 - v2_l*v2_l + sL)))/(1 + sL);
            lamRm = (v2_r - sqrt(sR*(1 - v2_r*v2_r + sR)))/(1 + sR);
            lamRp = (v2_l + sqrt(sL*(1 - v2_l*v2_l + sL)))/(1 + sL);
            lamLp = (v2_r + sqrt(sR*(1 - v2_r*v2_r + sR)))/(1 + sR);

            lambda.aL = min(lamLm, lamRm); 
            lambda.aR = max(lamLp, lamRp); 

            break; 

    }
        
    return lambda;

    
    
};



//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

conserveSR2D calc_stateSR2D(float gamma, double rho, double vx, double vy, double pressure)
{
    conserveSR2D state;
    double D, S1, S2, tau, h, vtot, lorentz_gamma;

    vtot = sqrt(vx*vx + vy*vy);
    lorentz_gamma = 1./sqrt(1 - vtot*vtot);

    h         = 1. + gamma*pressure/(rho*(gamma - 1.)); 
    state.D   = rho*lorentz_gamma; 
    state.S1  = rho*h*lorentz_gamma*lorentz_gamma*vx;
    state.S2  = rho*h*lorentz_gamma*lorentz_gamma*vy;
    state.tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;
    
    return state;

};

conserveSR2D calc_hll_state(float gamma,
                                conserveSR2D  &left_state,
                                conserveSR2D  &right_state,
                                fluxSR2D      &left_flux,
                                fluxSR2D      &right_flux,
                                primitives    &left_prims,
                                primitives    &right_prims,
                                unsigned int nhat)
{
    double aL, aR;
    eigenvals lambda; 
    conserveSR2D hll_states;

    lambda = calc_eigenvals(gamma, left_prims, right_prims, nhat);

    aL = lambda.aL;
    aR = lambda.aR;

    hll_states.D = ( aR*right_state.D - aL*left_state.D 
                        - right_flux.D + left_flux.D)/(aR - aL);

    hll_states.S1 = ( aR*right_state.S1 - aL*left_state.S1 
                        - right_flux.S1 + left_flux.S1)/(aR - aL);

    hll_states.S2 = ( aR*right_state.S2 - aL*left_state.S2
                        - right_flux.S2 + left_flux.S2)/(aR - aL);

    hll_states.tau = ( aR*right_state.tau - aL*left_state.tau
                        - right_flux.tau + left_flux.tau)/(aR - aL);


    return hll_states;

}

conserveSR2D calc_intermed_statesSR2D(  primitives &prims,
                                        conserveSR2D &state,
                                        double a,
                                        double aStar,
                                        double pStar,
                                        int nhat = 1)
{
    double rho, energy, pressure, v1, v2, cofactor;
    double D, S1, S2, tau;
    double Dstar, S1star, S2star, tauStar, Estar, E;
    eigenvals lambda; 
    conserveSR2D starStates;

    rho      = prims.rho;
    pressure = prims.p;
    v1       = prims.v1;
    v2       = prims.v2;

    D   = state.D;
    S1  = state.S1;
    S2  = state.S2;
    tau = state.tau;
    E   = tau + D;

    
    switch (nhat){
        case 1:
            cofactor = 1./(a - aStar); 
            Dstar    = cofactor * (a - v1)*D;
            S1star   = cofactor * (S1*(a - v1) - pressure + pStar);
            S2star   = cofactor * (a - v1)*S2;
            Estar    = cofactor * (E*(a - v1) + pStar*aStar - pressure*v1);
            tauStar  = Estar - Dstar;

            break;
        case 2:
            cofactor = 1./(a - aStar); 
            Dstar    = cofactor * (a - v2) * D;
            S1star   = cofactor * (a - v2) * S1; 
            S2star   = cofactor * (S2*(a - v2) - pressure + pStar);
            Estar    = cofactor * (E*(a - v2) + pStar*aStar - pressure*v2);
            tauStar  = Estar - Dstar;

            break;
        
    }
    
    starStates.D = Dstar;
    starStates.S1 = S1star;
    starStates.S2 = S2star;
    starStates.tau = tauStar;

    return starStates;
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
double UstateSR2D::adapt_dt(vector<vector<vector<double> > > &prims,
                        bool linspace=true, bool firs_order=true){

    double r_left, r_right, left_cell, right_cell, upper_cell, lower_cell;
    double dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg, h;
    double delta_logr, log_rLeft, log_rRight, min_dt, cfl_dt, vtot, D, tau, W;
    int shift_i, shift_j, x1physical_grid, x2physical_grid;
    double plus_v1, plus_v2, minus_v1, minus_v2, pmin_dt;

    int x1grid_size = prims[0][0].size();
    int x2grid_size = prims[0].size();

    min_dt = 0;
    // Find the minimum timestep over all i
    if (firs_order){
        x1physical_grid = x1grid_size - 2;
        x2physical_grid = x2grid_size - 2;
        
    } else {
        x1physical_grid = x1grid_size - 4;
        x2physical_grid = x2grid_size - 4;
    }


    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < x2physical_grid; jj ++){
        for (int ii = 0; ii < x1physical_grid; ii++){
            if (firs_order){
                shift_i = ii + 1;
                shift_j = jj + 1;
            } else {
                shift_i = ii + 2;
                shift_j = jj + 2;
            }
            

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
                // delta_logr = (log10(x1[x1physical_grid - 1]) - log10(x1[0]))/(x1physical_grid );
                // log_rLeft = log10(x1[0]) + ii*delta_logr;
                // log_rRight = log_rLeft + delta_logr;
                // r_left = pow(10, log_rLeft);
                // r_right = pow(10, log_rRight);
                if (coord_system == "cartesian"){
                    x2_right = 0.5*(upper_cell + x2[jj]);
                    x2_left = 0.5*(lower_cell + x2[jj]);
                } else {
                    x2_right = atan2 ( sin(upper_cell) + sin(x2[jj]), cos(upper_cell) + cos(x2[jj]) );
                    x2_left = atan2( sin(lower_cell) + sin(x2[jj]), cos(lower_cell) + cos(x2[jj]) ); 
    
                    // x2_right = 0.5*(upper_cell + x2[jj]);
                    // x2_left = 0.5*(lower_cell + x2[jj]);
  
                }

                // cout << "R_right: " << r_right << endl;
                // cout << "R_left: " << r_left << endl;
                r_right = sqrt(right_cell * x1[ii]);
                r_left  = sqrt(left_cell  * x1[ii]);
                // cout << "R_right: " << r_right << endl;
                // cout << "R_left: " << r_left << endl;
                // cin.get();
            }

            dx1      = r_right - r_left;
            dx2      = x2_right - x2_left;
            rho      = prims[0][shift_j][shift_i];
            v1       = prims[1][shift_j][shift_i];
            v2       = prims[2][shift_j][shift_i];
            pressure = prims[3][shift_j][shift_i];

            vtot = sqrt(v1*v1 + v2*v2);
            W    = 1./sqrt(1 - vtot*vtot);
            D    = rho*W;
            h    = 1. + gamma*pressure/(rho*(gamma - 1.));
            tau  = rho*h*W*W - pressure - rho*W;

            cs = calc_rel_sound_speed(pressure, D, tau, W, gamma);

            plus_v1 = (v1 + cs)/(1 + v1*cs);
            plus_v2 = (v2 + cs)/(1 + v2*cs);

            minus_v1 = (v1 - cs)/(1 - v1*cs);
            minus_v2 = (v2 - cs)/(1 - v2*cs);

            if (coord_system == "cartesian"){
                
                cfl_dt = min( dx1/(max(abs(plus_v1), abs(minus_v1))), dx2/(max(abs(plus_v2), abs(minus_v2))) );

            } else {
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );
                // cout << r_right << endl;
                // cout << r_left << endl;
                // cout << dx1 << endl;
                // cout << volAvg*dx2 << endl;
                // cout << dx1/(volAvg*dx2) << endl;
                // cin.get();

                cfl_dt = min( dx1/(max(abs(plus_v1), abs(minus_v1))), volAvg*dx2/(max(abs(plus_v2), abs(minus_v2))) );

            }

            
            if ((ii > 0) || (jj > 0) ){
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
fluxSR2D calc_fluxSR2D(float gamma, double rho, double vx, 
                                        double vy, double pressure, 
                                        bool x_direction=true){
    
    // The Flux Tensor
    fluxSR2D flux;

     // The Flux components
    double h, D, S1, S2, convect_12, tau, zeta;
    double mom1, mom2, energy_dens;

    double vtot = sqrt(vx*vx + vy*vy );
    double lorentz_gamma = calc_lorentz_gamma(vtot);

    h   = calc_enthalpy(gamma, rho, pressure);
    D   = rho*lorentz_gamma;
    S1  = rho*lorentz_gamma*lorentz_gamma*h*vx;
    S2  = rho*lorentz_gamma*lorentz_gamma*h*vy;
    tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;

    


    // Check if we're calculating the x-direction flux. If not, calculate the y-direction
    if (x_direction){
        mom1        = D * vx;
        convect_12  = S2 * vx;
        energy_dens = S1 * vx + pressure;
        zeta        = (tau + pressure) * vx;

        flux.D   = mom1;
        flux.S1  = energy_dens;
        flux.S2  = convect_12;
        flux.tau = zeta;
           
        return flux;

    } else {
        mom2 = D*vy;
        convect_12 = S1*vy;
        energy_dens = S2*vy + pressure;
        zeta = (tau + pressure)*vy;

        flux.D   = mom2;
        flux.S1  = convect_12;
        flux.S2  = energy_dens;
        flux.tau = zeta;
           
        return flux;
    }
    
};


fluxSR2D calc_hll_flux(float gamma,
                                conserveSR2D &left_state,
                                conserveSR2D &right_state,
                                fluxSR2D     &left_flux,
                                fluxSR2D     &right_flux,
                                primitives   &left_prims,
                                primitives   &right_prims,
                                unsigned int nhat
                                )
{
    eigenvals lambda; 
    fluxSR2D  hll_flux;
    double aL, aR, aLminus, aRplus;  
    
    lambda = calc_eigenvals(gamma, left_prims, right_prims, nhat);

    aL = lambda.aL;
    aR = lambda.aR;

    // Calculate /pm alphas
    aLminus = max(0.0 , - aL);
    aRplus  = max(0.0 ,   aR);

    // Compute the HLL Flux component-wise
    hll_flux.D = ( aRplus*left_flux.D + aLminus*right_flux.D
                            - aRplus*aLminus*(right_state.D - left_state.D ) )  /
                            (aRplus + aLminus);

    hll_flux.S1 = ( aRplus*left_flux.S1 + aLminus*right_flux.S1
                            - aRplus*aLminus*(right_state.S1 - left_state.S1 ) )  /
                            (aRplus + aLminus);

    hll_flux.S2 = ( aRplus*left_flux.S2 + aLminus*right_flux.S2
                            - aRplus*aLminus*(right_state.S2 - left_state.S2) )  /
                            (aRplus + aLminus);

    hll_flux.tau = ( aRplus*left_flux.tau + aLminus*right_flux.tau
                            - aRplus*aLminus*(right_state.tau - left_state.tau) )  /
                            (aRplus + aLminus);

    return hll_flux;
};


fluxSR2D calc_hllc_flux(float gamma,
                                conserveSR2D &left_state,
                                conserveSR2D &right_state,
                                fluxSR2D     &left_flux,
                                fluxSR2D     &right_flux,
                                primitives   &left_prims,
                                primitives   &right_prims,
                                int nhat = 1
                                )
{
    eigenvals lambda; 
    fluxSR2D interflux_left, interflux_right, hllc_flux, hll_flux;
    conserveSR2D interstate_left, interstate_right, hll_state;

    double aL, aR, aStar, pStar; 
    double fe, fs, e, s; 
    double aLminus, aRplus;

    lambda = calc_eigenvals(gamma, left_prims, right_prims, nhat);

    aL = lambda.aL;
    aR = lambda.aR;

    aLminus = max(0.0, - aL);
    aRplus  = max(0.0,   aR);
    /**
    hll_flux  = calc_hll_flux(gamma, left_state, right_state, left_flux,
                                    right_flux, left_prims, right_prims, direction);

    hll_state = calc_hll_state(gamma, left_state, right_state, left_flux,
                                     right_flux, left_prims, right_prims, direction);
    */
   

    hll_state.D = ( aR*right_state.D - aL*left_state.D 
                        - right_flux.D + left_flux.D)/(aR - aL);

    hll_state.S1 = ( aR*right_state.S1 - aL*left_state.S1 
                        - right_flux.S1 + left_flux.S1)/(aR - aL);

    hll_state.S2 = ( aR*right_state.S2 - aL*left_state.S2
                        - right_flux.S2 + left_flux.S2)/(aR - aL);

    hll_state.tau = ( aR*right_state.tau - aL*left_state.tau
                        - right_flux.tau + left_flux.tau)/(aR - aL);


    hll_flux.D = ( aRplus*left_flux.D + aLminus*right_flux.D
                            - aRplus*aLminus*(right_state.D - left_state.D ) )  /
                            (aRplus + aLminus);

    hll_flux.S1 = ( aRplus*left_flux.S1 + aLminus*right_flux.S1
                            - aRplus*aLminus*(right_state.S1 - left_state.S1 ) )  /
                            (aRplus + aLminus);

    hll_flux.S2 = ( aRplus*left_flux.S2 + aLminus*right_flux.S2
                            - aRplus*aLminus*(right_state.S2 - left_state.S2) )  /
                            (aRplus + aLminus);

    hll_flux.tau = ( aRplus*left_flux.tau + aLminus*right_flux.tau
                            - aRplus*aLminus*(right_state.tau - left_state.tau) )  /
                            (aRplus + aLminus);

    e  = hll_state.tau + hll_state.D;
    s  = hll_state.momentum(nhat);
    fe = hll_flux.tau + hll_flux.D;
    fs = hll_flux.momentum(nhat);


    aStar = calc_intermed_wave(e, s, fs, fe);
    pStar = -fe * aStar + fs;

    interstate_left = calc_intermed_statesSR2D(left_prims, left_state,
                                                aL, aStar, pStar, nhat);

    interstate_right = calc_intermed_statesSR2D(right_prims, right_state,
                                                aR, aStar, pStar, nhat);



    // Compute the intermediate left flux
    interflux_left.D    = left_flux.D   + aL*(interstate_left.D   - left_state.D   );
    interflux_left.S1   = left_flux.S1  + aL*(interstate_left.S1  - left_state.S1  );
    interflux_left.S2   = left_flux.S2  + aL*(interstate_left.S2  - left_state.S2  );
    interflux_left.tau  = left_flux.tau + aL*(interstate_left.tau - left_state.tau );

    // Compute the intermediate right flux
    interflux_right.D   = right_flux.D   + aR*(interstate_right.D   - right_state.D   );
    interflux_right.S1  = right_flux.S1  + aR*(interstate_right.S1  - right_state.S1  );
    interflux_right.S2  = right_flux.S2  + aR*(interstate_right.S2  - right_state.S2  );
    interflux_right.tau = right_flux.tau + aR*(interstate_right.tau - right_state.tau );

    
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



//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

vector<vector<vector<double> > > UstateSR2D::u_dot2D(vector<vector<vector<double> > > &u_state, 
                                        vector<vector<double> > &lorentz_gamma,
                                        vector<vector<vector<double> > > &sources,
                                        bool first_order = true,
                                        bool periodic = false,
                                        bool linspace=true,
                                        bool hllc = false,
                                        float theta = 1.5)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    int xphysical_grid, yphysical_grid;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();

    if (first_order){
        xphysical_grid = xgrid_size - 2;
        yphysical_grid = ygrid_size - 2;
    } else {
        xphysical_grid = xgrid_size - 4;
        yphysical_grid = ygrid_size - 4;
    }
    
    string default_coordinates = "cartesian";

    double Wx_l, Wx_r, Wy_l, Wy_r, vx_l, vx_r, vy_l, vy_r;
    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    conserveSR2D ux_l, ux_r, uy_l, uy_r; 
    fluxSR2D     f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    primitives   xprims_l, xprims_r, yprims_l, yprims_r;

    
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    primitives xleft_most, xleft_mid, xright_mid, xright_most;
    primitives yleft_most, yleft_mid, yright_mid, yright_most;
    primitives center;

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));


    // Define the source terms
    vector<vector<double> > sourceD = sources[0];
    vector<vector<double> > source_S1 = sources[1];
    vector<vector<double> > source_S2 = sources[2];
    vector<vector<double> > source_tau = sources[3];

    
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        if (first_order){
            int true_nxpts = xgrid_size - 1;
            int true_nypts = ygrid_size - 1;
            i_start = 1;
            i_bound = true_nxpts;
            j_start = 1;
            j_bound = true_nypts;

        } else {
            int true_nxpts = xgrid_size - 2;
            int true_nypts = ygrid_size - 2;
            i_start = 2;
            i_bound = true_nxpts;
            j_start = 2;
            j_bound = true_nypts;
        }
    }
    
    

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        if (first_order){
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l.D   = u_state[0][jj][ii];
                    ux_l.S1  = u_state[1][jj][ii];
                    ux_l.S2  = u_state[2][jj][ii];
                    ux_l.tau = u_state[3][jj][ii];

                    ux_r.D   = u_state[0][jj][ii + 1];
                    ux_r.S1  = u_state[1][jj][ii + 1];
                    ux_r.S2  = u_state[2][jj][ii + 1];
                    ux_r.tau = u_state[3][jj][ii + 1];

                    // j+1/2
                    uy_l.D   = u_state[0][jj][ii];
                    uy_l.S1  = u_state[1][jj][ii];
                    uy_l.S2  = u_state[2][jj][ii];
                    uy_l.tau = u_state[3][jj][ii];

                    uy_r.D   = u_state[0][jj + 1][ii];
                    uy_r.S1  = u_state[1][jj + 1][ii];
                    uy_r.S2  = u_state[2][jj + 1][ii];
                    uy_r.tau = u_state[3][jj + 1][ii];

                    Wx_l = lorentz_gamma[jj][ii];
                    Wx_r = lorentz_gamma[jj][ii + 1];
                    Wy_l = lorentz_gamma[jj][ii];
                    Wy_r = lorentz_gamma[jj + 1][ii];

                    xprims_l = cons2primSR(gamma, ux_l, Wx_l);
                    xprims_r = cons2primSR(gamma, ux_r, Wx_r);
                    yprims_l = cons2primSR(gamma, uy_l, Wy_l);
                    yprims_r = cons2primSR(gamma, uy_r, Wy_r);
                    
                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l.D   = u_state[0][jj][ii - 1];
                    ux_l.S1  = u_state[1][jj][ii - 1];
                    ux_l.S2  = u_state[2][jj][ii - 1];
                    ux_l.tau = u_state[3][jj][ii - 1];

                    ux_r.D   = u_state[0][jj][ii];
                    ux_r.S1  = u_state[1][jj][ii];
                    ux_r.S2  = u_state[2][jj][ii];
                    ux_r.tau = u_state[3][jj][ii];

                    // j-1/2
                    uy_l.D   = u_state[0][jj - 1][ii];
                    uy_l.S1  = u_state[1][jj - 1][ii];
                    uy_l.S2  = u_state[2][jj - 1][ii];
                    uy_l.tau = u_state[3][jj - 1][ii];

                    uy_r.D   = u_state[0][jj][ii];
                    uy_r.S1  = u_state[1][jj][ii];
                    uy_r.S2  = u_state[2][jj][ii];
                    uy_r.tau = u_state[3][jj][ii];

                    Wx_l = lorentz_gamma[jj][ii - 1];
                    Wx_r = lorentz_gamma[jj][ii];
                    Wy_l = lorentz_gamma[jj - 1][ii];
                    Wy_r = lorentz_gamma[jj][ii];

                    xprims_l = cons2primSR(gamma, ux_l, Wx_l);
                    xprims_r = cons2primSR(gamma, ux_r, Wx_r);
                    yprims_l = cons2primSR(gamma, uy_l, Wy_l);
                    yprims_r = cons2primSR(gamma, uy_r, Wy_r);

                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    

                    L[0][ycoordinate][xcoordinate] = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
                    L[1][ycoordinate][xcoordinate] = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
                    L[2][ycoordinate][xcoordinate] = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
                    L[3][ycoordinate][xcoordinate] = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;

                }
            }

            return L;

        } else {
            prims = cons2prim2D(u_state, lorentz_gamma);
            // cout << "Am Cartesian" << endl;
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    if (periodic){
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Poplate this later */

                    } else {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;
                        ycoordinate = jj - 2;

                        // Coordinate X
                        xleft_most.rho  = prims[0][jj][ii - 2];
                        xleft_mid.rho   = prims[0][jj][ii - 1];
                        center.rho      = prims[0][jj][ii];
                        xright_mid.rho  = prims[0][jj][ii + 1];
                        xright_most.rho = prims[0][jj][ii + 2];

                        xleft_most.v1  = prims[1][jj][ii - 2];
                        xleft_mid.v1   = prims[1][jj][ii - 1];
                        center.v1      = prims[1][jj][ii];
                        xright_mid.v1  = prims[1][jj][ii + 1];
                        xright_most.v1 = prims[1][jj][ii + 2];

                        xleft_most.v2  = prims[2][jj][ii - 2];
                        xleft_mid.v2   = prims[2][jj][ii - 1];
                        center.v2      = prims[2][jj][ii];
                        xright_mid.v2  = prims[2][jj][ii + 1];
                        xright_most.v2 = prims[2][jj][ii + 2];

                        xleft_most.p  = prims[3][jj][ii - 2];
                        xleft_mid.p   = prims[3][jj][ii - 1];
                        center.p      = prims[3][jj][ii];
                        xright_mid.p  = prims[3][jj][ii + 1];
                        xright_most.p = prims[3][jj][ii + 2];

                        // Coordinate Y
                        yleft_most.rho   = prims[0][jj - 2][ii];
                        yleft_mid.rho    = prims[0][jj - 1][ii];
                        yright_mid.rho   = prims[0][jj + 1][ii];
                        yright_most.rho  = prims[0][jj + 2][ii];

                        yleft_most.v1   = prims[1][jj - 2][ii];
                        yleft_mid.v1    = prims[1][jj - 1][ii];
                        yright_mid.v1   = prims[1][jj + 1][ii];
                        yright_most.v1  = prims[1][jj + 2][ii];

                        yleft_most.v2   = prims[2][jj - 2][ii];
                        yleft_mid.v2    = prims[2][jj - 1][ii];
                        yright_mid.v2   = prims[2][jj + 1][ii];
                        yright_most.v2  = prims[2][jj + 2][ii];

                        yleft_most.p   = prims[3][jj - 2][ii];
                        yleft_mid.p    = prims[3][jj - 1][ii];
                        yright_mid.p   = prims[3][jj + 1][ii];
                        yright_most.p  = prims[3][jj + 2][ii];

                    }
                    
                    // Reconstructed left X primitives vector at the i+1/2 interface
                    xprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - xleft_mid.rho),
                                                        0.5*(xright_mid.rho - xleft_mid.rho),
                                                        theta*(xright_mid.rho - center.rho));

                    
                    xprims_l.v1 = center.v1 + 0.5*minmod(theta*(center.v1 - xleft_mid.v1),
                                                        0.5*(xright_mid.v1 - xleft_mid.v1),
                                                        theta*(xright_mid.v1 - center.v1));

                    xprims_l.v2 = center.v2 + 0.5*minmod(theta*(center.v2 - xleft_mid.v2),
                                                        0.5*(xright_mid.v2 - xleft_mid.v2),
                                                        theta*(xright_mid.v2 - center.v2));

                    xprims_l.p = center.p + 0.5*minmod(theta*(center.p - xleft_mid.p),
                                                        0.5*(xright_mid.p - xleft_mid.p),
                                                        theta*(xright_mid.p - center.p));

                    // Reconstructed right primitives vector in x
                    xprims_r.rho = xright_mid.rho - 0.5*minmod(theta*(xright_mid.rho - center.rho),
                                                        0.5*(xright_most.rho - center.rho),
                                                        theta*(xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 - 0.5*minmod(theta*(xright_mid.v1 - center.v1),
                                                        0.5*(xright_most.v1 - center.v1),
                                                        theta*(xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 - 0.5*minmod(theta*(xright_mid.v2 - center.v2),
                                                        0.5*(xright_most.v2 - center.v2),
                                                        theta*(xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p - 0.5*minmod(theta*(xright_mid.p - center.p),
                                                        0.5*(xright_most.p - center.p),
                                                        theta*(xright_most.p - xright_mid.p));

                    
                    // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                    yprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - yleft_mid.rho),
                                                        0.5*(yright_mid.rho - yleft_mid.rho),
                                                        theta*(yright_mid.rho - center.rho));

                    yprims_l.v1 = center.v1 + 0.5*minmod(theta*(center.v1 - yleft_mid.v1),
                                                        0.5*(yright_mid.v1 - yleft_mid.v1),
                                                        theta*(yright_mid.v1 - center.v1));

                    yprims_l.v2 = center.v2 + 0.5*minmod(theta*(center.v2 - yleft_mid.v2),
                                                        0.5*(yright_mid.v2 - yleft_mid.v2),
                                                        theta*(yright_mid.v2 - center.v2));

                    yprims_l.p = center.p + 0.5*minmod(theta*(center.p - yleft_mid.p),
                                                        0.5*(yright_mid.p - yleft_mid.p),
                                                        theta*(yright_mid.p - center.p));
                    

                    yprims_r.rho = yright_mid.rho - 0.5*minmod(theta*(yright_mid.rho - center.rho),
                                                        0.5*(yright_most.rho - center.rho),
                                                        theta*(yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 - 0.5*minmod(theta*(yright_mid.v1 - center.v1),
                                                        0.5*(yright_most.v1 - center.v1),
                                                        theta*(yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 - 0.5*minmod(theta*(yright_mid.v2 - center.v2),
                                                        0.5*(yright_most.v2 - center.v2),
                                                        theta*(yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p - 0.5*minmod(theta*(yright_mid.p - center.p),
                                                        0.5*(yright_most.p - center.p),
                                                        theta*(yright_most.p - yright_mid.p));

                
                    
                    // Calculate the left and right states using the reconstructed PLM primitives
                    ux_l = calc_stateSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r,yprims_l, yprims_r,  2);
                    




                    // Left side primitives in x
                    xprims_l.rho = xleft_mid.rho + 0.5 *minmod(theta*(xleft_mid.rho - xleft_most.rho),
                                                            0.5*(center.rho - xleft_most.rho),
                                                            theta*(center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 + 0.5 *minmod(theta*(xleft_mid.v1 - xleft_most.v1),
                                                            0.5*(center.v1 -xleft_most.v1),
                                                            theta*(center.v1 - xleft_mid.v1));
                    
                    xprims_l.v2 = xleft_mid.v2 + 0.5 *minmod(theta*(xleft_mid.v2 - xleft_most.v2),
                                                            0.5*(center.v2 - xleft_most.v2),
                                                            theta*(center.v2 - xleft_mid.v2));
                    
                    xprims_l.p = xleft_mid.p + 0.5 *minmod(theta*(xleft_mid.p - xleft_most.p),
                                                            0.5*(center.p - xleft_most.p),
                                                            theta*(center.p - xleft_mid.p));

                        
                    // Right side primitives in x
                    xprims_r.rho = center.rho - 0.5 *minmod(theta*(center.rho - xleft_mid.rho),
                                                        0.5*(xright_mid.rho - xleft_mid.rho),
                                                        theta*(xright_mid.rho - center.rho));

                    xprims_r.v1 = center.v1 - 0.5 *minmod(theta*(center.v1 - xleft_mid.v1),
                                                        0.5*(xright_mid.v1 - xleft_mid.v1),
                                                        theta*(xright_mid.v1 - center.v1));

                    xprims_r.v2 = center.v2 - 0.5 *minmod(theta*(center.v2 - xleft_mid.v2),
                                                        0.5*(xright_mid.v2 - xleft_mid.v2),
                                                        theta*(xright_mid.v2 - center.v2));

                    xprims_r.p = center.p - 0.5 *minmod(theta*(center.p - xleft_mid.p),
                                                        0.5*(xright_mid.p - xleft_mid.p),
                                                        theta*(xright_mid.p - center.p));


                    // Left side primitives in y
                    yprims_l.rho = yleft_mid.rho + 0.5 *minmod(theta*(yleft_mid.rho - yleft_most.rho),
                                                            0.5*(center.rho - yleft_most.rho),
                                                            theta*(center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 + 0.5 *minmod(theta*(yleft_mid.v1 - yleft_most.v1),
                                                            0.5*(center.v1 -yleft_most.v1),
                                                            theta*(center.v1 - yleft_mid.v1));
                    
                    yprims_l.v2 = yleft_mid.v2 + 0.5 *minmod(theta*(yleft_mid.v2 - yleft_most.v2),
                                                            0.5*(center.v2 - yleft_most.v2),
                                                            theta*(center.v2 - yleft_mid.v2));
                    
                    yprims_l.p = yleft_mid.p + 0.5 *minmod(theta*(yleft_mid.p - yleft_most.p),
                                                            0.5*(center.p - yleft_most.p),
                                                            theta*(center.p - yleft_mid.p));

                        
                    // Right side primitives in y
                    yprims_r.rho = center.rho - 0.5 *minmod(theta*(center.rho - yleft_mid.rho),
                                                        0.5*(yright_mid.rho - yleft_mid.rho),
                                                        theta*(yright_mid.rho - center.rho));

                    yprims_r.v1 = center.v1 - 0.5 *minmod(theta*(center.v1 - yleft_mid.v1),
                                                        0.5*(yright_mid.v1 - yleft_mid.v1),
                                                        theta*(yright_mid.v1 - center.v1));

                    yprims_r.v2 = center.v2 - 0.5 *minmod(theta*(center.v2 - yleft_mid.v2),
                                                        0.5*(yright_mid.v2 - yleft_mid.v2),
                                                        theta*(yright_mid.v2 - center.v2));

                    yprims_r.p = center.p  - 0.5 *minmod(theta*(center.p - yleft_mid.p),
                                                        0.5*(yright_mid.p - yleft_mid.p),
                                                        theta*(yright_mid.p - center.p)); 
                    
                

                    // Calculate the left and right states using the reconstructed PLM primitives
                    ux_l = calc_stateSR2D(gamma,xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(gamma,xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    

                    
                    

                    L[0][ycoordinate][xcoordinate] = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
                    L[1][ycoordinate][xcoordinate] = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
                    L[2][ycoordinate][xcoordinate] = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
                    L[3][ycoordinate][xcoordinate] = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;
                    
                }

            }

            return L;

        }

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr; 

        if (first_order){
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    tcoordinate = jj - 1;
                    rcoordinate = ii - 1;

                    // i+1/2
                    ux_l.D   = u_state[0][jj][ii];
                    ux_l.S1  = u_state[1][jj][ii];
                    ux_l.S2  = u_state[2][jj][ii];
                    ux_l.tau = u_state[3][jj][ii];

                    ux_r.D   = u_state[0][jj][ii + 1];
                    ux_r.S1  = u_state[1][jj][ii + 1];
                    ux_r.S2  = u_state[2][jj][ii + 1];
                    ux_r.tau = u_state[3][jj][ii + 1];

                    // j+1/2
                    uy_l.D    = u_state[0][jj][ii];
                    uy_l.S1   = u_state[1][jj][ii];
                    uy_l.S2   = u_state[2][jj][ii];
                    uy_l.tau  = u_state[3][jj][ii];

                    uy_r.D = u_state[0][jj + 1][ii];
                    uy_r.S1 = u_state[1][jj + 1][ii];
                    uy_r.S2 = u_state[2][jj + 1][ii];
                    uy_r.tau = u_state[3][jj + 1][ii];

                    Wx_l = lorentz_gamma[jj][ii];
                    Wx_r = lorentz_gamma[jj][ii + 1];
                    Wy_l = lorentz_gamma[jj][ii];
                    Wy_r = lorentz_gamma[jj + 1][ii];

                    xprims_l = cons2primSR(gamma, ux_l, Wx_l);
                    xprims_r = cons2primSR(gamma, ux_r, Wx_r);
                    yprims_l = cons2primSR(gamma, uy_l, Wy_l);
                    yprims_r = cons2primSR(gamma, uy_r, Wy_r);

                    rhoc = xprims_l.rho;
                    pc   = xprims_l.p;
                    uc   = xprims_l.v1;
                    vc   = xprims_l.v2;
                    
                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l.D    = u_state[0][jj][ii - 1];
                    ux_l.S1   = u_state[1][jj][ii - 1];
                    ux_l.S2   = u_state[2][jj][ii - 1];
                    ux_l.tau  = u_state[3][jj][ii - 1];

                    ux_r.D    = u_state[0][jj][ii];
                    ux_r.S1   = u_state[1][jj][ii];
                    ux_r.S2   = u_state[2][jj][ii];
                    ux_r.tau  = u_state[3][jj][ii];

                    // j-1/2
                    uy_l.D    = u_state[0][jj - 1][ii];
                    uy_l.S1   = u_state[1][jj - 1][ii];
                    uy_l.S2   = u_state[2][jj - 1][ii];
                    uy_l.tau  = u_state[3][jj - 1][ii];

                    uy_r.D    = u_state[0][jj][ii];
                    uy_r.S1   = u_state[1][jj][ii];
                    uy_r.S2   = u_state[2][jj][ii];
                    uy_r.tau  = u_state[3][jj][ii];

                    Wx_l = lorentz_gamma[jj][ii - 1];
                    Wx_r = lorentz_gamma[jj][ii];
                    Wy_l = lorentz_gamma[jj - 1][ii];
                    Wy_r = lorentz_gamma[jj][ii];

                    xprims_l = cons2primSR(gamma, ux_l, Wx_l);
                    xprims_r = cons2primSR(gamma, ux_r, Wx_r);
                    yprims_l = cons2primSR(gamma, uy_l, Wy_l);
                    yprims_r = cons2primSR(gamma, uy_r, Wy_r);

                    f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    if (linspace){
                        right_cell = x1[rcoordinate + 1];
                        left_cell  = x1[rcoordinate - 1];
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

                        theta_right = 0.5*(upper_cell + x2[tcoordinate]);
                        theta_left = 0.5*(lower_cell + x2[tcoordinate]);

                } else {
                    // log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    // log_rRight = log_rLeft + delta_logr;
                    // r_left = pow(10, log_rLeft);
                    // r_right = pow(10, log_rRight);
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];
                    }

                    r_right = sqrt(right_cell * x1[rcoordinate]);
                    r_left  = sqrt(left_cell  * x1[rcoordinate]);

                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        lower_cell = x2[tcoordinate];

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                    // theta_right = 0.5*(upper_cell + x2[jj]);
                    // theta_left = 0.5*(lower_cell + x2[jj]);
                }

                dr = r_right - r_left;
                
                
                ang_avg = atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = sin(theta_left); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); //deltaV1*(cos(theta_left) - cos(theta_right)); 

                    

                L[0][tcoordinate][rcoordinate] = - (right_rsurface*f1.D - left_rsurface*f2.D)/deltaV1 
                                                    - (upper_tsurface*g1.D - lower_tsurface*g2.D)/deltaV2 + sourceD[tcoordinate][rcoordinate];

                L[1][tcoordinate][rcoordinate] = - (right_rsurface*f1.S1 - left_rsurface*f2.S1)/deltaV1 
                                                    - (upper_tsurface*g1.S1 - lower_tsurface*g2.S1)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[tcoordinate][rcoordinate];

                L[2][tcoordinate][rcoordinate] = - (right_rsurface*f1.S2 - left_rsurface*f2.S2)/deltaV1 
                                                    - (upper_tsurface*g1.S2 - lower_tsurface*g2.S2)/deltaV2
                                                    -(rhoc*vc*uc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg) ) ) + source_S2[tcoordinate][rcoordinate];

                L[3][tcoordinate][rcoordinate] = - (right_rsurface*f1.tau - left_rsurface*f2.tau)/deltaV1 
                                                    - (upper_tsurface*g1.tau - lower_tsurface*g2.tau)/deltaV2 + source_tau[tcoordinate][rcoordinate];

                }
            }

            return L;

        } else {
            prims = cons2prim2D(u_state, lorentz_gamma);
            // cout << "High Order Spherical" << endl;
            for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most.rho = roll(prims[0][ii], ii - 2);
                    xleft_mid.rho = roll(prims[0][ii], ii - 1);
                    center.rho = prims[0][ii][jj];
                    xright_mid.rho = roll(prims[0][ii], ii + 1);
                    xright_most.rho = roll(prims[0][ii], ii + 2);

                    xleft_most.v1 = roll(prims[1][ii], ii - 2);
                    xleft_mid.v1 = roll(prims[1][ii], ii - 1);
                    center.v1 = prims[1][ii][jj];
                    xright_mid.v1 = roll(prims[1][ii], ii + 1);
                    xright_most.v1 = roll(prims[1][ii], ii + 2);

                    xleft_most.v2 = roll(prims[2][ii], ii - 2);
                    xleft_mid.v2 = roll(prims[2][ii], ii - 1);
                    center.v2 = prims[2][ii][jj];
                    xright_mid.v2 = roll(prims[2][ii], ii + 1);
                    xright_most.v2 = roll(prims[2][ii], ii + 2);

                    xleft_most.p = roll(prims[3][ii], ii - 2);
                    xleft_mid.p = roll(prims[3][ii], ii - 1);
                    center.p = prims[3][ii][jj];
                    xright_mid.p = roll(prims[3][ii], ii + 1);
                    xright_most.p = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most.rho = roll(rho_transpose[ii], ii - 2);
                    yleft_mid.rho = roll(rho_transpose[ii], ii - 1);
                    yright_mid.rho = roll(rho_transpose[ii], ii + 1);
                    yright_most.rho = roll(rho_transpose[ii], ii + 2);

                    yleft_most.v1 = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid.v1 = roll(pressure_transpose[ii], ii - 1);
                    yright_mid.v1 = roll(pressure_transpose[ii], ii + 1);
                    yright_most.v1 = roll(pressure_transpose[ii], ii + 2);

                    yleft_most.v2 = roll(vx_transpose[ii], ii - 2);
                    yleft_mid.v2 = roll(vx_transpose[ii], ii - 1);
                    yright_mid.v2 = roll(vx_transpose[ii], ii + 1);
                    yright_most.v2 = roll(vx_transpose[ii], ii + 2);

                    yleft_most.p = roll(vy_transpose[ii], ii - 2);
                    yleft_mid.p = roll(vy_transpose[ii], ii - 1);
                    yright_mid.p = roll(vy_transpose[ii], ii + 1);
                    yright_most.p = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most.rho  = prims[0][jj][ii - 2];
                    xleft_mid.rho   = prims[0][jj][ii - 1];
                    center.rho      = prims[0][jj][ii];
                    xright_mid.rho  = prims[0][jj][ii + 1];
                    xright_most.rho = prims[0][jj][ii + 2];

                    xleft_most.v1   = prims[1][jj][ii - 2];
                    xleft_mid.v1    = prims[1][jj][ii - 1];
                    center.v1       = prims[1][jj][ii];
                    xright_mid.v1   = prims[1][jj][ii + 1];
                    xright_most.v1  = prims[1][jj][ii + 2];

                    xleft_most.v2   = prims[2][jj][ii - 2];
                    xleft_mid.v2    = prims[2][jj][ii - 1];
                    center.v2       = prims[2][jj][ii];
                    xright_mid.v2   = prims[2][jj][ii + 1];
                    xright_most.v2  = prims[2][jj][ii + 2];

                    xleft_most.p    = prims[3][jj][ii - 2];
                    xleft_mid.p     = prims[3][jj][ii - 1];
                    center.p        = prims[3][jj][ii];
                    xright_mid.p    = prims[3][jj][ii + 1];
                    xright_most.p   = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most.rho  = prims[0][jj - 2][ii];
                    yleft_mid.rho   = prims[0][jj - 1][ii];
                    yright_mid.rho  = prims[0][jj + 1][ii];
                    yright_most.rho = prims[0][jj + 2][ii];

                    yleft_most.v1   = prims[1][jj - 2][ii];
                    yleft_mid.v1    = prims[1][jj - 1][ii];
                    yright_mid.v1   = prims[1][jj + 1][ii];
                    yright_most.v1  = prims[1][jj + 2][ii];

                    yleft_most.v2   = prims[2][jj - 2][ii];
                    yleft_mid.v2    = prims[2][jj - 1][ii];
                    yright_mid.v2   = prims[2][jj + 1][ii];
                    yright_most.v2  = prims[2][jj + 2][ii];

                    yleft_most.p    = prims[3][jj - 2][ii];
                    yleft_mid.p     = prims[3][jj - 1][ii];
                    yright_mid.p    = prims[3][jj + 1][ii];
                    yright_most.p   = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - xleft_mid.rho),
                                                    0.5*(xright_mid.rho - xleft_mid.rho),
                                                    theta*(xright_mid.rho - center.rho));

                
                xprims_l.v1 = center.v1 + 0.5*minmod(theta*(center.v1 - xleft_mid.v1),
                                                    0.5*(xright_mid.v1 - xleft_mid.v1),
                                                    theta*(xright_mid.v1 - center.v1));

                xprims_l.v2 = center.v2 + 0.5*minmod(theta*(center.v2 - xleft_mid.v2),
                                                    0.5*(xright_mid.v2 - xleft_mid.v2),
                                                    theta*(xright_mid.v2 - center.v2));

                xprims_l.p = center.p + 0.5*minmod(theta*(center.p - xleft_mid.p),
                                                    0.5*(xright_mid.p - xleft_mid.p),
                                                    theta*(xright_mid.p - center.p));

                // Reconstructed right primitives vector in x
                xprims_r.rho = xright_mid.rho - 0.5*minmod(theta*(xright_mid.rho - center.rho),
                                                    0.5*(xright_most.rho - center.rho),
                                                    theta*(xright_most.rho - xright_mid.rho));

                xprims_r.v1 = xright_mid.v1 - 0.5*minmod(theta*(xright_mid.v1 - center.v1),
                                                    0.5*(xright_most.v1 - center.v1),
                                                    theta*(xright_most.v1 - xright_mid.v1));

                xprims_r.v2 = xright_mid.v2 - 0.5*minmod(theta*(xright_mid.v2 - center.v2),
                                                    0.5*(xright_most.v2 - center.v2),
                                                    theta*(xright_most.v2 - xright_mid.v2));

                xprims_r.p = xright_mid.p - 0.5*minmod(theta*(xright_mid.p - center.p),
                                                    0.5*(xright_most.p - center.p),
                                                    theta*(xright_most.p - xright_mid.p));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - yleft_mid.rho),
                                                    0.5*(yright_mid.rho - yleft_mid.rho),
                                                    theta*(yright_mid.rho - center.rho));

                yprims_l.v1 = center.v1 + 0.5*minmod(theta*(center.v1 - yleft_mid.v1),
                                                    0.5*(yright_mid.v1 - yleft_mid.v1),
                                                    theta*(yright_mid.v1 - center.v1));

                yprims_l.v2 = center.v2 + 0.5*minmod(theta*(center.v2 - yleft_mid.v2),
                                                    0.5*(yright_mid.v2 - yleft_mid.v2),
                                                    theta*(yright_mid.v2 - center.v2));

                yprims_l.p = center.p + 0.5*minmod(theta*(center.p - yleft_mid.p),
                                                    0.5*(yright_mid.p - yleft_mid.p),
                                                    theta*(yright_mid.p - center.p));
                

                yprims_r.rho = yright_mid.rho - 0.5*minmod(theta*(yright_mid.rho - center.rho),
                                                    0.5*(yright_most.rho - center.rho),
                                                    theta*(yright_most.rho - yright_mid.rho));

                yprims_r.v1 = yright_mid.v1 - 0.5*minmod(theta*(yright_mid.v1 - center.v1),
                                                    0.5*(yright_most.v1 - center.v1),
                                                    theta*(yright_most.v1 - yright_mid.v1));

                yprims_r.v2 = yright_mid.v2 - 0.5*minmod(theta*(yright_mid.v2 - center.v2),
                                                    0.5*(yright_most.v2 - center.v2),
                                                    theta*(yright_most.v2 - yright_mid.v2));

                yprims_r.p = yright_mid.p - 0.5*minmod(theta*(yright_mid.p - center.p),
                                                    0.5*(yright_most.p - center.p),
                                                    theta*(yright_most.p - yright_mid.p));
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                ux_r = calc_stateSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                uy_l = calc_stateSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                uy_r = calc_stateSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                if (hllc){
                    f1 = calc_hllc_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hllc_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                }
                
                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l.rho = xleft_mid.rho + 0.5 *minmod(theta*(xleft_mid.rho - xleft_most.rho),
                                                        0.5*(center.rho - xleft_most.rho),
                                                        theta*(center.rho - xleft_mid.rho));

                xprims_l.v1 = xleft_mid.v1 + 0.5 *minmod(theta*(xleft_mid.v1 - xleft_most.v1),
                                                        0.5*(center.v1 -xleft_most.v1),
                                                        theta*(center.v1 - xleft_mid.v1));
                
                xprims_l.v2 = xleft_mid.v2 + 0.5 *minmod(theta*(xleft_mid.v2 - xleft_most.v2),
                                                        0.5*(center.v2 - xleft_most.v2),
                                                        theta*(center.v2 - xleft_mid.v2));
                
                xprims_l.p = xleft_mid.p + 0.5 *minmod(theta*(xleft_mid.p - xleft_most.p),
                                                        0.5*(center.p - xleft_most.p),
                                                        theta*(center.p - xleft_mid.p));

                    
                // Right side primitives in x
                xprims_r.rho = center.rho - 0.5 *minmod(theta*(center.rho - xleft_mid.rho),
                                                    0.5*(xright_mid.rho - xleft_mid.rho),
                                                    theta*(xright_mid.rho - center.rho));

                xprims_r.v1 = center.v1 - 0.5 *minmod(theta*(center.v1 - xleft_mid.v1),
                                                    0.5*(xright_mid.v1 - xleft_mid.v1),
                                                    theta*(xright_mid.v1 - center.v1));

                xprims_r.v2 = center.v2 - 0.5 *minmod(theta*(center.v2 - xleft_mid.v2),
                                                    0.5*(xright_mid.v2 - xleft_mid.v2),
                                                    theta*(xright_mid.v2 - center.v2));

                xprims_r.p = center.p - 0.5 *minmod(theta*(center.p - xleft_mid.p),
                                                    0.5*(xright_mid.p - xleft_mid.p),
                                                    theta*(xright_mid.p - center.p));


                // Left side primitives in y
                yprims_l.rho = yleft_mid.rho + 0.5 *minmod(theta*(yleft_mid.rho - yleft_most.rho),
                                                        0.5*(center.rho - yleft_most.rho),
                                                        theta*(center.rho - yleft_mid.rho));

                yprims_l.v1 = yleft_mid.v1 + 0.5 *minmod(theta*(yleft_mid.v1 - yleft_most.v1),
                                                        0.5*(center.v1 -yleft_most.v1),
                                                        theta*(center.v1 - yleft_mid.v1));
                
                yprims_l.v2 = yleft_mid.v2 + 0.5 *minmod(theta*(yleft_mid.v2 - yleft_most.v2),
                                                        0.5*(center.v2 - yleft_most.v2),
                                                        theta*(center.v2 - yleft_mid.v2));
                
                yprims_l.p = yleft_mid.p + 0.5 *minmod(theta*(yleft_mid.p - yleft_most.p),
                                                        0.5*(center.p - yleft_most.p),
                                                        theta*(center.p - yleft_mid.p));

                    
                // Right side primitives in y
                yprims_r.rho = center.rho - 0.5 *minmod(theta*(center.rho - yleft_mid.rho),
                                                    0.5*(yright_mid.rho - yleft_mid.rho),
                                                    theta*(yright_mid.rho - center.rho));

                yprims_r.v1 = center.v1 - 0.5 *minmod(theta*(center.v1 - yleft_mid.v1),
                                                    0.5*(yright_mid.v1 - yleft_mid.v1),
                                                    theta*(yright_mid.v1 - center.v1));

                yprims_r.v2 = center.v2 - 0.5 *minmod(theta*(center.v2 - yleft_mid.v2),
                                                    0.5*(yright_mid.v2 - yleft_mid.v2),
                                                    theta*(yright_mid.v2 - center.v2));

                yprims_r.p = center.p  - 0.5 *minmod(theta*(center.p - yleft_mid.p),
                                                    0.5*(yright_mid.p - yleft_mid.p),
                                                    theta*(yright_mid.p - center.p)); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                ux_r = calc_stateSR2D(gamma,xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                uy_l = calc_stateSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                uy_r = calc_stateSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                f_l = calc_fluxSR2D(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                f_r = calc_fluxSR2D(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                g_l = calc_fluxSR2D(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                g_r = calc_fluxSR2D(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);
                
                
                if (hllc){
                    f2 = calc_hllc_flux(gamma, ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hllc_flux(gamma, uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r,xprims_l,xprims_r,  1);
                    g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                }
                
                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell  = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
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
                    // log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    // log_rRight = log_rLeft + delta_logr;
                    // r_left = pow(10, log_rLeft);
                    // r_right = pow(10, log_rRight);
                    right_cell = x1[rcoordinate + 1];
                    left_cell  = x1[rcoordinate - 1];

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];
                    }

                    r_right = sqrt(right_cell * x1[rcoordinate]); //+ x1[rcoordinate]);
                    r_left  = sqrt(left_cell  * x1[rcoordinate]); // + left_cell);

                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        lower_cell = x2[tcoordinate];

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                }

                dr = r_right - r_left;
                rhoc = center.rho;
                pc   = center.p;
                uc   = center.v1;
                vc   = center.v2;

                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );

                // Compute the surface areas
                right_rsurface = r_right * r_right ;
                left_rsurface  = r_left  * r_left  ;
                upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = sin(theta_left) ; //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );

                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); 

                L[0][tcoordinate][rcoordinate] = - (f1.D*right_rsurface - f2.D*left_rsurface)/deltaV1
                                                    - (g1.D*upper_tsurface - g2.D*lower_tsurface)/deltaV2 + sourceD[tcoordinate][rcoordinate];

                L[1][tcoordinate][rcoordinate] = - (f1.S1*right_rsurface - f2.S1*left_rsurface)/deltaV1
                                                    - (g1.S1*upper_tsurface - g2.S1*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[tcoordinate][rcoordinate];

                L[2][tcoordinate][rcoordinate] = - (f1.S2*right_rsurface - f2.S2*left_rsurface)/deltaV1
                                                    - (g1.S2*upper_tsurface - g2.S2*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg))) + source_S2[tcoordinate][rcoordinate];

                L[3][tcoordinate][rcoordinate] = - (f1.tau*right_rsurface - f2.tau*left_rsurface)/deltaV1
                                                    - (g1.tau*upper_tsurface - g2.tau*lower_tsurface)/deltaV2 + source_tau[tcoordinate][rcoordinate];

            }

        }

        return L;

        }
        
    }
    

};


//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
vector<vector<vector<double> > > UstateSR2D::simulate2D(vector<vector<double> > &lorentz_gamma, 
                                                        vector<vector<vector<double> > > &sources,
                                                        float tend = 0.1, 
                                                        bool first_order = true, bool periodic = false,
                                                        bool linspace=true, bool hllc=false,
                                                        double dt = 1.e-4){

    // Define the swap vector for the integrated state
    int xphysical_grid, yphysical_grid;
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int total_zones = xgrid_size*ygrid_size;
    int n_vars = state2D.size();
    float t = 0;

    if (first_order){
        xphysical_grid = xgrid_size - 2;
        yphysical_grid = ygrid_size - 2;
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

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    

    if (first_order){
        while (t < tend){
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            udot = u_dot2D(u, lorentz_gamma, sources, true, periodic, linspace, hllc, theta);

            for (int jj = 0; jj < yphysical_grid; jj++){
                // Get the non-ghost index 
                int j_real = jj + 1;
                for (int ii = 0; ii < xphysical_grid; ii++){
                    int i_real = ii + 1; 

                    u_p[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                    u_p[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                    u_p[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                    u_p[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
                }
            }

            config_ghosts2D(u_p, xgrid_size, ygrid_size, true);
            prims = cons2prim2D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[1], prims[2]);
            if (t > 0){
                dt = adapt_dt(prims, linspace);
            }

            u.swap(u_p);
            
            t += dt;
            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);


            cout << fixed << setprecision(3)
            << scientific;
            cout << "\r" << "dt: " << setw(5) << dt 
            << "\t" << "t: " << setw(5) << t 
            << "\t" << "Zones per sec: " << total_zones/time_span.count()
            << flush;


        }

    } else {
           while (t < tend){
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            if (t == 0){
                config_ghosts2D(u, xgrid_size, ygrid_size, false);
            }

            udot = u_dot2D(u, lorentz_gamma, sources, first_order,  periodic, linspace, hllc, theta);

            for (int jj = 0; jj < yphysical_grid; jj++){
                // Get the non-ghost index 
                int j_real = jj + 2;
                for (int ii = 0; ii < xphysical_grid; ii++){
                    int i_real = ii + 2; 
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
                    cout << u1[3][jj][ii] << ", ";
                }
                cout << endl;
            }
            cin.get();
            */
            
            config_ghosts2D(u1, xgrid_size, ygrid_size, false);
            prims = cons2prim2D(u1, lorentz_gamma);

            /**
            cout << " " << endl;
            cout << "Prims: " << endl;
            for (int jj=0; jj <ygrid_size; jj++){
                for (int ii=0; ii < xgrid_size; ii++){
                    cout << prims[0][jj][ii] << ", ";
                }
                cout << endl;
            }
            cin.get();
            */
        
            
            lorentz_gamma = calc_lorentz_gamma(prims[1], prims[2]);
            udot1 = u_dot2D(u1, lorentz_gamma, sources, first_order, periodic, linspace, hllc, theta);

            
            // cout << " " << endl;
            // cout << "UDot1: " << endl;
            // for (int jj=0; jj <yphysical_grid; jj++){
            //     for (int ii=0; ii < xphysical_grid; ii++){
            //         cout << prims[3][jj][ii] << ", ";
            //         //cout << udot1[0][jj][ii] << ", ";
            //     }
            //     cout << endl;
            // }
            // string p;
            // cin >> p;
        
            for (int jj = 0; jj < yphysical_grid; jj++){
                // Get the non-ghost index 
                int j_real =  jj + 2;
                for (int ii = 0; ii < xphysical_grid; ii++){
                    int i_real = ii + 2;
                    u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                    u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                    u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                    u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

                }

            }

            /**
            cout << " " << endl;
            cout << "U2: " << endl;
            for (int jj=0; jj <ygrid_size; jj++){
                for (int ii=0; ii < xgrid_size; ii++){
                    cout << u2[0][jj][ii] << ", ";
                }
                cout << endl;
            }
            cin.get();
            */
            
            
            config_ghosts2D(u2, xgrid_size, ygrid_size, false);

            prims = cons2prim2D(u2, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[1], prims[2]);

            
            udot2 = u_dot2D(u2, lorentz_gamma, sources, first_order, periodic, linspace, hllc, theta);

            // for (int jj=0; jj <ygrid_size; jj++){
            //     for (int ii=0; ii < xgrid_size; ii++){
            //         // cout << u[0][jj][ii] << endl;
            //         if (isnan(u2[0][jj][ii])){ break; }
            //         if (isnan(u2[1][jj][ii])){ break; }
            //         if (isnan(u2[2][jj][ii])){ break; }
            //         if (isnan(u2[3][jj][ii])){ break; }
            //     }
            //     // cout << endl;
            // }


            for (int jj = 0; jj < yphysical_grid; jj++){
                // Get the non-ghost index 
                int j_real =  jj + 2;
                for (int ii = 0; ii < xphysical_grid; ii++){
                    int i_real = ii + 2;
                    u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                    u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                    u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                    u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

                }

            }
            
            config_ghosts2D(u_p, xgrid_size, ygrid_size, false);


            prims = cons2prim2D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims[1], prims[2]);
            

            dt = adapt_dt(prims, linspace, false);
            

            if (isnan(dt)){
                break;
            }
            

            /**
            cout << " " << endl;
            cout << "UP: " << endl;
            for (int jj=0; jj <ygrid_size; jj++){
                for (int ii=0; ii < xgrid_size; ii++){
                    cout << u_p[1][jj][ii]/1.e-10 << ", ";
                }
                cout << endl;
            }
            string b;
            cin >> b;
            */
            
            
            
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;
            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            cout << fixed << setprecision(3)
            << scientific;
            cout << "\r" << "dt: " << setw(5) << dt 
            << "\t" << "t: " << setw(5) << t 
            << "\t" << "Zones per sec: " << total_zones/time_span.count()
            << flush;


            

        }
        
    }
    cout << "\n " << endl;
    prims = cons2prim2D(u, lorentz_gamma);

    return prims;

 };
