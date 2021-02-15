/* 
* C++ Source to perform 2D SRHD Calculations
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
UstateSR2D::UstateSR2D(vector<vector<double>> state2D, int nx, int ny, float gamma, vector<double> x1, 
                    vector<double> x2, double Cfl, string coord_system = "cartesian")
{
    auto nzones = state2D[0].size();
    
    this->NX = nx;
    this->NY = ny;
    this->nzones = nzones;
    this->state2D = state2D;
    this->gamma   = gamma;
    this->x1      = x1;
    this->x2      = x2;
    this->CFL     = Cfl;
    this->coord_system = coord_system;
}

// Destructor 
UstateSR2D::~UstateSR2D() {}

/* Define typedefs because I am lazy */
typedef vector<vector<double>> twoVec; 
typedef UstateSR2D::PrimitiveData PrimitiveArray;
typedef UstateSR2D::ConserveData ConserveArray;
typedef UstateSR2D::Primitives Primitives;
typedef UstateSR2D::Flux Flux;
typedef UstateSR2D::Conserved Conserved;
typedef UstateSR2D::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
Primitives UstateSR2D::cons2primSR(Conserved  &u_state, 
                                    double lorentz_gamma, tuple<int, int>(coordinates)){
    /**
     * Return a vector containing the primitive
     * variables density, pressure, and
     * velocity components
     */

    int ii = get<0>(coordinates);
    int jj = get<1>(coordinates);
    Primitives prims;
    double D   = u_state.D;
    double S1  = u_state.S1;
    double S2  = u_state.S2;
    double tau = u_state.tau;
    
    double S = sqrt(S1*S1 + S2*S2);
    
    double pmin = abs(S - tau - D) ? n == 0: pressure_guess[ii + NX * jj];

    double pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, lorentz_gamma, gamma, S);

    double v1 = S1/(tau + pressure + D);

    double v2 = S2/(tau + pressure + D);

    double vtot = sqrt(v1*v1 + v2*v2);

    double Wnew = 1./sqrt(1 - vtot*vtot);

    double rho = D/Wnew;

    prims.rho = rho;
    prims.v1  = v1;
    prims.v2  = v2;
    prims.p   = pressure;

    return prims;
    
};


PrimitiveArray UstateSR2D::cons2prim2D(const ConserveArray &u_state2D, const vector<double> &lorentz_gamma){
    /**
     * Return a 2D matrix containing the primitive
     * variables density , pressure, and
     * three-velocity 
     */

    double rho, S1, S2, S, D, tau, pmin, tol;
    double pressure, W;
    double v1, v2, vtot;

    PrimitiveArray prims;
    prims.rho.reserve(nzones);
    prims.v1.reserve(nzones);
    prims.v2.reserve(nzones);
    prims.p.reserve(nzones);
    for (int jj=0; jj < NY; jj ++){
        for(int ii=0; ii< NX; ii ++){
            D   =  u_state2D.D  [ii + NX * jj];      // Relativistic Density
            S1  =  u_state2D.S1 [ii + NX * jj];      // X1-Momentum Denity
            S2  =  u_state2D.S2 [ii + NX * jj];      // X2-Momentum Density
            tau =  u_state2D.tau[ii + NX * jj];      // Energy Density
            W   =  lorentz_gamma[ii + NX * jj]; 

            S = sqrt(S1*S1 + S2*S2);

            pmin = abs(S - tau - D) ? n == 0: pressure_guess[ii + NX * jj];

            tol = 1.e-6; //D*1.e-12;

            pressure = newton_raphson(pmin, pressure_func, dfdp, tol, D, tau, W, gamma, S);

            v1 = S1/(tau + D + pressure);

            v2 = S2/(tau + D + pressure);

            W = 1./sqrt(1 - v1*v1 + v2*v2);

            rho = D/W;

            
            /* TODO: Is this faster than the [] operator? */
            // prims.rho.emplace_back(rho);
            // prims.v1.emplace_back(v1);
            // prims.v2.emplace_back(v2);
            // prims.p.emplace_back(pressure);

            prims.rho.emplace_back(rho);
            prims.v1.emplace_back(v1);
            prims.v2.emplace_back(v2);
            prims.p.emplace_back(pressure);
                
        }
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
Eigenvals UstateSR2D::calc_Eigenvals(Primitives &prims_l,
                                      Primitives &prims_r,
                                      unsigned int nhat = 1)
{

    // Initialize your important variables
    double v1_r, v1_l, v2_l, v2_r, p_r, p_l, cs_r, cs_l, D_r, D_l ,tau_r, tau_l; 
    double rho_l, rho_r, W_l, W_r, h_l, h_r, v, pStar;
    double sL,sR, lamLp, lamRp, lamLm, lamRm;
    Eigenvals lambda;

    // Separate the left and right Primitives
    rho_l = prims_l.rho;
    p_l   = prims_l.p;
    v1_l  = prims_l.v1;
    v2_l  = prims_l.v2;
    // W_l   = 1./sqrt(1 - v1_l*v1_l + v2_l*v2_l);
    h_l   = 1. + gamma*p_l/(rho_l*(gamma - 1));

    rho_r = prims_r.rho;
    p_r   = prims_r.p;
    v1_r  = prims_r.v1;
    v2_r  = prims_r.v2;
    // W_r   = 1./sqrt(1 - v1_r*v1_r + v2_r*v2_r);
    h_r   = 1. + gamma*p_r/(rho_r*(gamma - 1));

    
    //D_l = W_l*rho_l;
    //D_r = W_r*rho_r;
    //tau_l = rho_l*h_l*W_l*W_l - p_l - rho_l*W_l;
    //tau_r = rho_r*h_r*W_r*W_r - p_r - rho_r*W_r;

    cs_r = sqrt(gamma * p_r/(h_r*rho_r)); //calc_rel_sound_speed(p_r, D_r, tau_r, W_r, gamma);
    cs_l = sqrt(gamma * p_l/(h_l*rho_l)); //calc_rel_sound_speed(p_l, D_l, tau_l, W_l, gamma);
 
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

Conserved UstateSR2D::calc_stateSR2D(double rho, double vx,
                                     double vy, double pressure)
{
    Conserved state;
    double h, lorentz_gamma;

    lorentz_gamma = 1./sqrt(1 - vx*vx + vy*vy);

    h         = 1. + gamma*pressure/(rho*(gamma - 1.)); 
    state.D   = rho*lorentz_gamma; 
    state.S1  = rho*h*lorentz_gamma*lorentz_gamma*vx;
    state.S2  = rho*h*lorentz_gamma*lorentz_gamma*vy;
    state.tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;
    
    return state;

};

Conserved UstateSR2D::calc_hll_state(
                                Conserved  &left_state,
                                Conserved  &right_state,
                                Flux      &left_flux,
                                Flux      &right_flux,
                                Primitives    &left_prims,
                                Primitives    &right_prims,
                                unsigned int nhat)
{
    double aL, aR;
    Eigenvals lambda; 
    Conserved hll_states;

    lambda = calc_Eigenvals(left_prims, right_prims, nhat);

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

Conserved UstateSR2D::calc_intermed_statesSR2D(  Primitives &prims,
                                        Conserved &state,
                                        double a,
                                        double aStar,
                                        double pStar,
                                        int nhat = 1)
{
    double rho, pressure, v1, v2, cofactor;
    double D, S1, S2, tau;
    double Dstar, S1star, S2star, tauStar, Estar, E;
    Conserved starStates;

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
    
    starStates.D   = Dstar;
    starStates.S1  = S1star;
    starStates.S2  = S2star;
    starStates.tau = tauStar;

    return starStates;
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
double UstateSR2D::adapt_dt(const PrimitiveArray &prims,
                        bool linspace=true, bool firs_order=true){

    double r_left, r_right, left_cell, right_cell, upper_cell, lower_cell;
    double dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg, h;
    double delta_logr, min_dt, cfl_dt, D, tau, W;
    int shift_i, shift_j;
    double plus_v1, plus_v2, minus_v1, minus_v2;

    min_dt = 0;
    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < yphysical_grid; jj ++){
        shift_j = jj + idx_active;
        for (int ii = 0; ii < xphysical_grid; ii++){
           
            shift_i = ii + idx_active;

            // Find the left and right cell centers in one direction
            if (ii - 1 < 0){
                left_cell = x1[ii];
                right_cell = x1[ii + 1];
            }
            else if (ii + 1 > xphysical_grid - 1){
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
            else if (jj + 1 > yphysical_grid - 1){
                upper_cell = x2[jj];
                lower_cell = x2[jj - 1];
            } else {
                upper_cell = x2[jj + 1];
                lower_cell = x2[jj];
            }

            // Check if using linearly-spaced grid or logspace
            if (linspace){
                r_right = 0.5*(right_cell + x1[ii]);
                r_left  = 0.5*(x1[ii] + left_cell);

            } else {

                r_right = sqrt(right_cell * x1[ii]); //0.5 * (right_cell + x1[ii]);
                r_left  = sqrt(left_cell  * x1[ii]); //0.5 * (left_cell  + x1[ii]);

            }

            x2_right = 0.5 * (upper_cell + x2[jj]);
            x2_left  = 0.5 * (lower_cell + x2[jj]);

            dx1      = r_right - r_left;
            dx2      = x2_right - x2_left;
            rho      = prims.rho[shift_i + NX * shift_j];
            v1       = prims.v1 [shift_i + NX * shift_j];
            v2       = prims.v2 [shift_i + NX * shift_j];
            pressure = prims.p  [shift_i + NX * shift_j];

            // W    = 1./sqrt(1 - v1*v1 + v2*v2);
            // D    = rho*W;
            h    = 1. + gamma*pressure/(rho*(gamma - 1.));
            // tau  = rho*h*W*W - pressure - rho*W;

            cs = sqrt(gamma * pressure/(rho * h)); //calc_rel_sound_speed(pressure, D, tau, W, gamma);

            plus_v1 = (v1 + cs)/(1 + v1*cs);
            plus_v2 = (v2 + cs)/(1 + v2*cs);

            minus_v1 = (v1 - cs)/(1 - v1*cs);
            minus_v2 = (v2 - cs)/(1 - v2*cs);

            if (coord_system == "cartesian"){
                
                cfl_dt = min( dx1/(max(abs(plus_v1), abs(minus_v1))), dx2/(max(abs(plus_v2), abs(minus_v2))) );

            } else {
                // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                volAvg = 0.75*( ( r_right * r_right * r_right * r_right - r_left * r_left * r_left * r_left ) / 
                                    ( r_right * r_right * r_right - r_left * r_left * r_left) );
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
Flux UstateSR2D::calc_Flux(double rho, double vx, 
                                double vy, double pressure, 
                                bool x_direction=true){
    
    // The Flux Tensor
    Flux flux;

     // The Flux components
    double h, D, S1, S2, convect_12, tau, zeta;
    double mom1, mom2, energy_dens;

    double lorentz_gamma = 1./sqrt(1. - vx*vx + vy*vy );

    h   = 1. + gamma*pressure/(rho*(gamma - 1));
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


Flux UstateSR2D::calc_hll_flux(
                        Conserved &left_state,
                        Conserved &right_state,
                        Flux     &left_flux,
                        Flux     &right_flux,
                        Primitives   &left_prims,
                        Primitives   &right_prims,
                        unsigned int nhat)
{
    Eigenvals lambda; 
    Flux  hll_flux;
    double aL, aR, aLminus, aRplus;  
    
    lambda = calc_Eigenvals(left_prims, right_prims, nhat);

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


Flux UstateSR2D::calc_hllc_flux(
                                Conserved &left_state,
                                Conserved &right_state,
                                Flux     &left_flux,
                                Flux     &right_flux,
                                Primitives   &left_prims,
                                Primitives   &right_prims,
                                int nhat = 1)
{
    Eigenvals lambda; 
    Flux interflux_left, interflux_right, hllc_flux, hll_flux;
    Conserved interstate_left, interstate_right, hll_state;

    double aL, aR, aStar, pStar; 
    double fe, fs, e, s, a, b, c, quad; 
    double aLminus, aRplus;
    double rho, pressure, v1, v2, cofactor;
    double D, S1, S2, tau;
    double Dstar, S1star, S2star, tauStar, Estar, E;

    lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    aL = lambda.aL;
    aR = lambda.aR;

    aLminus = max(0.0, - aL);
    aRplus  = max(0.0,   aR);

    /* Calculate the HLL Intermediate State and Flux */
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

    /* Mignone & Bodo subtract off the rest mass density */
    e  = hll_state.tau + hll_state.D;
    s  = hll_state.momentum(nhat);
    fe = hll_flux.tau + hll_flux.D;
    fs = hll_flux.momentum(nhat);

    a = fe;
    b = -(fs + e);
    c = s;
    quad = quad = - 0.5 * (b + sign(b)*sqrt(b * b - 4 * a * c));
    aStar = c/quad;
    pStar = -fe * aStar + fs;

    /* Compute the L/R Star State */
    switch (nhat)
    {
    case 1:
        // Left Star State in x-direction of coordinate lattice
        rho      = left_prims.rho;
        pressure = left_prims.p;
        v1       = left_prims.v1;
        v2       = left_prims.v2;

        D   = left_state.D;
        S1  = left_state.S1;
        S2  = left_state.S2;
        tau = left_state.tau;
        E   = tau + D;

        cofactor = 1./(aL - aStar); 
        Dstar    = cofactor * (aL - v1)*D;
        S1star   = cofactor * (S1*(aL - v1) - pressure + pStar);
        S2star   = cofactor * (aL - v1)*S2;
        Estar    = cofactor * (E*(aL - v1) + pStar*aStar - pressure*v1);
        tauStar  = Estar - Dstar;

        interstate_left.D   = Dstar;
        interstate_left.S1  = S1star;
        interstate_left.S2  = S2star;
        interstate_left.tau = tauStar;

        // Right Star State
        rho      = right_prims.rho;
        pressure = right_prims.p;
        v1       = right_prims.v1;
        v2       = right_prims.v2;

        D   = right_state.D;
        S1  = right_state.S1;
        S2  = right_state.S2;
        tau = right_state.tau;
        E   = tau + D;

        cofactor = 1./(aR - aStar); 
        Dstar    = cofactor * (aR - v1)*D;
        S1star   = cofactor * (S1*(aR - v1) - pressure + pStar);
        S2star   = cofactor * (aR - v1)*S2;
        Estar    = cofactor * (E*(aR - v1) + pStar*aStar - pressure*v1);
        tauStar  = Estar - Dstar;

        interstate_right.D   = Dstar;
        interstate_right.S1  = S1star;
        interstate_right.S2  = S2star;
        interstate_right.tau = tauStar;
        break;
    
    case 2: // Start States in y-direction in the coordinate lattice
        rho      = left_prims.rho;
        pressure = left_prims.p;
        v1       = left_prims.v1;
        v2       = left_prims.v2;

        D   = left_state.D;
        S1  = left_state.S1;
        S2  = left_state.S2;
        tau = left_state.tau;
        E   = tau + D;

        cofactor = 1./(aL - aStar); 
        Dstar    = cofactor * (aL - v2) * D;
        S1star   = cofactor * (aL - v2) * S1; 
        S2star   = cofactor * (S2*(aL - v2) - pressure + pStar);
        Estar    = cofactor * (E*(aL - v2) + pStar*aStar - pressure*v2);
        tauStar  = Estar - Dstar;

        interstate_left.D   = Dstar;
        interstate_left.S1  = S1star;
        interstate_left.S2  = S2star;
        interstate_left.tau = tauStar;

        // Right Star State
        rho      = right_prims.rho;
        pressure = right_prims.p;
        v1       = right_prims.v1;
        v2       = right_prims.v2;

        D   = right_state.D;
        S1  = right_state.S1;
        S2  = right_state.S2;
        tau = right_state.tau;
        E   = tau + D;

        cofactor = 1./(aR - aStar); 
        Dstar    = cofactor * (aR - v2) * D;
        S1star   = cofactor * (aR - v2) * S1; 
        S2star   = cofactor * (S2*(aR - v2) - pressure + pStar);
        Estar    = cofactor * (E*(aR - v2) + pStar*aStar - pressure*v2);
        tauStar  = Estar - Dstar;
        
        interstate_right.D   = Dstar;
        interstate_right.S1  = S1star;
        interstate_right.S2  = S2star;
        interstate_right.tau = tauStar;

        break;

    }

    if (0.0 <= aL){
        return left_flux;
    }  else if (aL <= 0.0 && 0.0 <= aStar){
        // Compute the intermediate left flux
        interflux_left.D    = left_flux.D   + aL*(interstate_left.D   - left_state.D   );
        interflux_left.S1   = left_flux.S1  + aL*(interstate_left.S1  - left_state.S1  );
        interflux_left.S2   = left_flux.S2  + aL*(interstate_left.S2  - left_state.S2  );
        interflux_left.tau  = left_flux.tau + aL*(interstate_left.tau - left_state.tau );

        return interflux_left;
    } else if (aStar <= 0.0 && 0.0 <= aR){

        // Compute the intermediate right flux
        interflux_right.D   = right_flux.D   + aR*(interstate_right.D   - right_state.D   );
        interflux_right.S1  = right_flux.S1  + aR*(interstate_right.S1  - right_state.S1  );
        interflux_right.S2  = right_flux.S2  + aR*(interstate_right.S2  - right_state.S2  );
        interflux_right.tau = right_flux.tau + aR*(interstate_right.tau - right_state.tau );

    
        return interflux_right;
    } else {
        return right_flux;
    }
    
};



//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

ConserveArray UstateSR2D::u_dot2D(const ConserveArray &u_state)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    
    

    
    ConserveArray L;
    L.D.reserve(active_zones);
    L.S1.reserve(active_zones);
    L.S2.reserve(active_zones);
    L.tau.reserve(active_zones);

    // L.D.resize(active_zones);
    // L.S1.resize(active_zones);
    // L.S2.resize(active_zones);
    // L.tau.resize(active_zones);

    Conserved ux_l, ux_r, uy_l, uy_r; 
    Flux     f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    Primitives   xprims_l, xprims_r, yprims_l, yprims_r;

    Primitives xleft_most, xleft_mid, xright_mid, xright_most;
    Primitives yleft_most, yleft_mid, yright_mid, yright_most;
    Primitives center;

    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    i_start = j_start = idx_active; 
    i_bound = x_bound;
    j_bound = y_bound;
    

    if (coord_system == "cartesian"){
        double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
        double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;
        if (first_order){
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l.D   = u_state.D[ii + NX * jj];
                    ux_l.S1  = u_state.S1[ii + NX * jj];
                    ux_l.S2  = u_state.S2[ii + NX * jj];
                    ux_l.tau = u_state.tau[ii + NX * jj];

                    ux_r.D   = u_state.D[(ii + 1) + NX * jj];
                    ux_r.S1  = u_state.S1[(ii + 1) + NX * jj];
                    ux_r.S2  = u_state.S2[(ii + 1) + NX * jj];
                    ux_r.tau = u_state.tau[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l.D   = u_state.D[ii + NX * jj];
                    uy_l.S1  = u_state.S1[ii + NX * jj];
                    uy_l.S2  = u_state.S2[ii + NX * jj];
                    uy_l.tau = u_state.tau[ii + NX * jj];

                    uy_r.D   = u_state.D[(ii + 1) + NX * jj];
                    uy_r.S1  = u_state.S1[(ii + 1) + NX * jj];
                    uy_r.S2  = u_state.S2[(ii + 1) + NX * jj];
                    uy_r.tau = u_state.tau[(ii + 1) + NX * jj];

                    xprims_l.rho = prims.rho[ii + jj * NX]; 
                    xprims_l.v1  = prims.v1 [ii + jj * NX];
                    xprims_l.v2  = prims.v2 [ii + jj * NX];
                    xprims_l.p   = prims.p  [ii + jj * NX];

                    xprims_r.rho = prims.rho[(ii + 1) + jj * NX]; 
                    xprims_r.v1  = prims.v1 [(ii + 1) + jj * NX];
                    xprims_r.v2  = prims.v2 [(ii + 1) + jj * NX];
                    xprims_r.p   = prims.p  [(ii + 1) + jj * NX];

                    yprims_l.rho = prims.rho[ii + jj * NX]; 
                    yprims_l.v1  = prims.v1 [ii + jj * NX];
                    yprims_l.v2  = prims.v2 [ii + jj * NX];
                    yprims_l.p   = prims.p  [ii + jj * NX];

                    yprims_r.rho = prims.rho[ii + (jj + 1) * NX]; 
                    yprims_r.v1  = prims.v1 [ii + (jj + 1) * NX];
                    yprims_r.v2  = prims.v2 [ii + (jj + 1) * NX];
                    yprims_r.p   = prims.p  [ii + (jj + 1) * NX];
                    
                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l.D   = u_state.D[(ii - 1) + NX * jj];
                    ux_l.S1  = u_state.S1[(ii - 1) + NX * jj];
                    ux_l.S2  = u_state.S2[(ii - 1) + NX * jj];
                    ux_l.tau = u_state.tau[(ii - 1) + NX * jj];

                    ux_r.D   = u_state.D[ii + NX * jj];
                    ux_r.S1  = u_state.S1[ii + NX * jj];
                    ux_r.S2  = u_state.S2[ii + NX * jj];
                    ux_r.tau = u_state.tau[ii + NX * jj];

                    // j-1/2
                    uy_l.D   = u_state.D[(ii - 1) + NX * jj];
                    uy_l.S1  = u_state.S1[(ii - 1) + NX * jj];
                    uy_l.S2  = u_state.S2[(ii - 1) + NX * jj];
                    uy_l.tau = u_state.tau[(ii - 1) + NX * jj];

                    uy_r.D   = u_state.D[ii + NX * jj];
                    uy_r.S1  = u_state.S1[ii + NX * jj];
                    uy_r.S2  = u_state.S2[ii + NX * jj];
                    uy_r.tau = u_state.tau[ii + NX * jj];

                    xprims_l.rho = prims.rho[(ii - 1) + jj * NX]; 
                    xprims_l.v1  = prims.v1 [(ii - 1) + jj * NX];
                    xprims_l.v2  = prims.v2 [(ii - 1) + jj * NX];
                    xprims_l.p   = prims.p  [(ii - 1) + jj * NX];

                    xprims_r.rho = prims.rho[ii + jj * NX]; 
                    xprims_r.v1  = prims.v1 [ii + jj * NX];
                    xprims_r.v2  = prims.v2 [ii + jj * NX];
                    xprims_r.p   = prims.p  [ii + jj * NX];

                    yprims_l.rho = prims.rho[ii + (jj - 1) * NX]; 
                    yprims_l.v1  = prims.v1 [ii + (jj - 1) * NX];
                    yprims_l.v2  = prims.v2 [ii + (jj - 1) * NX];
                    yprims_l.p   = prims.p  [ii + (jj - 1) * NX];

                    yprims_r.rho = prims.rho[ii + jj * NX]; 
                    yprims_r.v1  = prims.v1 [ii + jj * NX];
                    yprims_r.v2  = prims.v2 [ii + jj * NX];
                    yprims_r.p   = prims.p  [ii + jj * NX];

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i+1/2 interface
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    

                    L.D[xcoordinate + xphysical_grid*ycoordinate] = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
                    L.S1[xcoordinate + xphysical_grid*ycoordinate] = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
                    L.S2[xcoordinate + xphysical_grid*ycoordinate] = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
                    L.tau[xcoordinate + xphysical_grid*ycoordinate] = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;

                }
            }

            return L;

        } else {
            // prims = cons2prim2D(u_state, lorentz_gamma);
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
                        xleft_most.rho  = prims.rho[(ii - 2) + NX * jj];
                        xleft_mid.rho   = prims.rho[(ii - 1) + NX * jj];
                        center.rho      = prims.rho[ii + NX * jj];
                        xright_mid.rho  = prims.rho[(ii + 1) + NX * jj];
                        xright_most.rho = prims.rho[(ii + 2) + NX * jj];

                        xleft_most.v1  = prims.v1[(ii - 2) + NX*jj];
                        xleft_mid.v1   = prims.v1[(ii - 1) + NX * jj];
                        center.v1      = prims.v1[ii + NX * jj];
                        xright_mid.v1  = prims.v1[(ii + 1) + NX * jj];
                        xright_most.v1 = prims.v1[(ii + 2) + NX * jj];

                        xleft_most.v2  = prims.v2[(ii - 2) + NX*jj];
                        xleft_mid.v2   = prims.v2[(ii - 1) + NX * jj];
                        center.v2      = prims.v2[ii + NX * jj];
                        xright_mid.v2  = prims.v2[(ii + 1) + NX * jj];
                        xright_most.v2 = prims.v2[(ii + 2) + NX * jj];

                        xleft_most.p  = prims.p[(ii - 2) + NX*jj];
                        xleft_mid.p   = prims.p[(ii - 1) + NX * jj];
                        center.p      = prims.p[ii + NX * jj];
                        xright_mid.p  = prims.p[(ii + 1) + NX * jj];
                        xright_most.p = prims.p[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most.rho   = prims.rho[ii + NX * (jj - 2)];
                        yleft_mid.rho    = prims.rho[ii + NX * (jj - 1)];
                        yright_mid.rho   = prims.rho[ii + NX * (jj + 1)];
                        yright_most.rho  = prims.rho[ii + NX * (jj + 2)];

                        yleft_most.v1   = prims.v1[ii + NX * (jj - 2)];
                        yleft_mid.v1    = prims.v1[ii + NX * (jj - 1)];
                        yright_mid.v1   = prims.v1[ii + NX * (jj + 1)];
                        yright_most.v1  = prims.v1[ii + NX * (jj + 2)];

                        yleft_most.v2   = prims.v2[ii + NX * (jj - 2)];
                        yleft_mid.v2    = prims.v2[ii + NX * (jj - 1)];
                        yright_mid.v2   = prims.v2[ii + NX * (jj + 1)];
                        yright_most.v2  = prims.v2[ii + NX * (jj + 2)];

                        yleft_most.p   = prims.p[ii + NX * (jj - 2)];
                        yleft_mid.p    = prims.p[ii + NX * (jj - 1)];
                        yright_mid.p   = prims.p[ii + NX * (jj + 1)];
                        yright_most.p  = prims.p[ii + NX * (jj + 2)];

                    }
                    
                    // Reconstructed left X Primitives vector at the i+1/2 interface
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

                    // Reconstructed right Primitives vector in x
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

                    
                    // Reconstructed right Primitives vector in y-direction at j+1/2 interfce
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

                
                    
                    // Calculate the left and right states using the reconstructed PLM Primitives
                    ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r,  2);
                    




                    // Left side Primitives in x
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

                        
                    // Right side Primitives in x
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


                    // Left side Primitives in y
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

                        
                    // Right side Primitives in y
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
                    
                

                    // Calculate the left and right states using the reconstructed PLM Primitives
                    ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    

                    
                    

                    L.D[xcoordinate + xphysical_grid*ycoordinate] = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
                    L.S1[xcoordinate + xphysical_grid*ycoordinate] = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
                    L.S2[xcoordinate + xphysical_grid*ycoordinate] = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
                    L.tau[xcoordinate + xphysical_grid*ycoordinate] = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;
                    
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
        // double log_rLeft, log_rRight;
        double theta_right, theta_left, ycoordinate, xcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        // double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/(xphysical_grid - 1);

        double dr; 

        if (first_order){
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l.D   = u_state.D[ii + NX * jj];
                    ux_l.S1  = u_state.S1[ii + NX * jj];
                    ux_l.S2  = u_state.S2[ii + NX * jj];
                    ux_l.tau = u_state.tau[ii + NX * jj];

                    ux_r.D   = u_state.D[(ii + 1) + NX * jj];
                    ux_r.S1  = u_state.S1[(ii + 1) + NX * jj];
                    ux_r.S2  = u_state.S2[(ii + 1) + NX * jj];
                    ux_r.tau = u_state.tau[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l.D    = u_state.D[ii + NX * jj];
                    uy_l.S1   = u_state.S1[ii + NX * jj];
                    uy_l.S2   = u_state.S2[ii + NX * jj];
                    uy_l.tau  = u_state.tau[ii + NX * jj];

                    uy_r.D    = u_state.D[ii + NX * (jj + 1)];
                    uy_r.S1   = u_state.S1[ii + NX * (jj + 1)];
                    uy_r.S2   = u_state.S2[ii + NX * (jj + 1)];
                    uy_r.tau  = u_state.tau[ii + NX * (jj + 1)];

                    xprims_l.rho = prims.rho[ii + jj * NX]; 
                    xprims_l.v1  = prims.v1 [ii + jj * NX];
                    xprims_l.v2  = prims.v2 [ii + jj * NX];
                    xprims_l.p   = prims.p  [ii + jj * NX];

                    xprims_r.rho = prims.rho[(ii + 1) + jj * NX]; 
                    xprims_r.v1  = prims.v1 [(ii + 1) + jj * NX];
                    xprims_r.v2  = prims.v2 [(ii + 1) + jj * NX];
                    xprims_r.p   = prims.p  [(ii + 1) + jj * NX];

                    yprims_l.rho = prims.rho[ii + jj * NX]; 
                    yprims_l.v1  = prims.v1 [ii + jj * NX];
                    yprims_l.v2  = prims.v2 [ii + jj * NX];
                    yprims_l.p   = prims.p  [ii + jj * NX];

                    yprims_r.rho = prims.rho[ii + (jj + 1.) * NX]; 
                    yprims_r.v1  = prims.v1 [ii + (jj + 1.) * NX];
                    yprims_r.v2  = prims.v2 [ii + (jj + 1.) * NX];
                    yprims_r.p   = prims.p  [ii + (jj + 1.) * NX];

                    rhoc = xprims_l.rho;
                    pc   = xprims_l.p;
                    uc   = xprims_l.v1;
                    vc   = xprims_l.v2;
                    
                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i,j +1/2 interface
                    if (hllc){
                        f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    } else {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    }
                    
                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l.D    = u_state.D[(ii - 1) + NX * jj];
                    ux_l.S1   = u_state.S1[(ii - 1) + NX * jj];
                    ux_l.S2   = u_state.S2[(ii - 1) + NX * jj];
                    ux_l.tau  = u_state.tau[(ii - 1) + NX * jj];

                    ux_r.D    = u_state.D[ii + NX * jj];
                    ux_r.S1   = u_state.S1[ii + NX * jj];
                    ux_r.S2   = u_state.S2[ii + NX * jj];
                    ux_r.tau  = u_state.tau[ii + NX * jj];

                    // j-1/2
                    uy_l.D    = u_state.D[ii + NX * (jj - 1)];
                    uy_l.S1   = u_state.S1[ii + NX * (jj - 1)];
                    uy_l.S2   = u_state.S2[ii + NX * (jj - 1)];
                    uy_l.tau  = u_state.tau[ii + NX * (jj - 1)];

                    uy_r.D    = u_state.D[ii + NX * jj];
                    uy_r.S1   = u_state.S1[ii + NX * jj];
                    uy_r.S2   = u_state.S2[ii + NX * jj];
                    uy_r.tau  = u_state.tau[ii + NX * jj];

                    xprims_l.rho = prims.rho[(ii - 1) + jj * NX]; 
                    xprims_l.v1  = prims.v1 [(ii - 1) + jj * NX];
                    xprims_l.v2  = prims.v2 [(ii - 1) + jj * NX];
                    xprims_l.p   = prims.p  [(ii - 1) + jj * NX];

                    xprims_r.rho = prims.rho[ii + jj * NX]; 
                    xprims_r.v1  = prims.v1 [ii + jj * NX];
                    xprims_r.v2  = prims.v2 [ii + jj * NX];
                    xprims_r.p   = prims.p  [ii + jj * NX];

                    yprims_l.rho = prims.rho[ii + (jj - 1) * NX]; 
                    yprims_l.v1  = prims.v1 [ii + (jj - 1) * NX];
                    yprims_l.v2  = prims.v2 [ii + (jj - 1) * NX];
                    yprims_l.p   = prims.p  [ii + (jj - 1) * NX];

                    yprims_r.rho = prims.rho[ii + jj * NX]; 
                    yprims_r.v1  = prims.v1 [ii + jj * NX];
                    yprims_r.v2  = prims.v2 [ii + jj * NX];
                    yprims_r.p   = prims.p  [ii + jj * NX];

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    // Calc HLL Flux at i,j - 1/2 interface
                    if (hllc){
                        f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    } else {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    }

                    if (linspace){
                        right_cell = x1[xcoordinate + 1];
                        left_cell  = x1[xcoordinate - 1];
                        upper_cell = x2[ycoordinate + 1];
                        lower_cell = x2[ycoordinate - 1];

                        // cout << "Theta Coordinate: " << ycoordinate << endl;
                        // cout << "R Coordinate: " << xcoordinate << endl;
                        
                        // Outflow the left/right boundaries
                        if (xcoordinate - 1 < 0){
                            left_cell = x1[xcoordinate];

                        } else if (xcoordinate == xphysical_grid - 1){
                            right_cell = x1[xcoordinate];

                        }

                        if (ycoordinate - 1 < 0){
                            lower_cell = x2[ycoordinate];
                        }  else if(ycoordinate == yphysical_grid - 1){
                            upper_cell = x2[ycoordinate];
                        }

                        
                        r_right = 0.5*(right_cell + x1[xcoordinate]);
                        r_left  = 0.5*(x1[xcoordinate] + left_cell);

                        theta_right = 0.5*(upper_cell + x2[ycoordinate]);
                        theta_left = 0.5*(lower_cell + x2[ycoordinate]);

                } else {
                    // log_rLeft = log10(x1[0]) + xcoordinate*delta_logr;
                    // log_rRight = log_rLeft + delta_logr;
                    // r_left = pow(10, log_rLeft);
                    // r_right = pow(10, log_rRight);
                    right_cell = x1[xcoordinate + 1];
                    left_cell  = x1[xcoordinate - 1];

                    upper_cell = x2[ycoordinate + 1];
                    lower_cell = x2[ycoordinate - 1];
                    
                    if (xcoordinate - 1 < 0){
                        left_cell = x1[xcoordinate];

                    } else if (xcoordinate == xphysical_grid - 1){
                        right_cell = x1[xcoordinate];
                    }

                    r_right = sqrt(right_cell * x1[xcoordinate]);
                    r_left  = sqrt(left_cell  * x1[xcoordinate]);

                    // Outflow the left/right boundaries
                    if (ycoordinate - 1 < 0){
                        lower_cell = x2[ycoordinate];

                    } else if(ycoordinate == yphysical_grid - 1){
                        upper_cell = x2[ycoordinate];
                    }

                    theta_right = 0.5 * (upper_cell + x2[ycoordinate]);
                    theta_left  = 0.5 * (lower_cell + x2[ycoordinate]);
                }

                dr = r_right - r_left;
                
                
                ang_avg = 0.5 *(theta_right + theta_left); //atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = sin(theta_left); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*( (r_right * r_right * r_right * r_right - r_left * r_left * r_left * r_left) / 
                                        (r_right * r_right * r_right -  r_left * r_left * r_left) );

                deltaV1 = volAvg * volAvg * dr;
                deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); //deltaV1*(cos(theta_left) - cos(theta_right)); 
                    
                L.D[xcoordinate + xphysical_grid*ycoordinate] = - (right_rsurface*f1.D - left_rsurface*f2.D)/deltaV1 
                                                    - (upper_tsurface*g1.D - lower_tsurface*g2.D)/deltaV2 + sourceD[xcoordinate + xphysical_grid*ycoordinate];

                L.S1[xcoordinate + xphysical_grid*ycoordinate] = - (right_rsurface*f1.S1 - left_rsurface*f2.S1)/deltaV1 
                                                    - (upper_tsurface*g1.S1 - lower_tsurface*g2.S1)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[xcoordinate + xphysical_grid*ycoordinate];

                L.S2[xcoordinate + xphysical_grid*ycoordinate] = - (right_rsurface*f1.S2 - left_rsurface*f2.S2)/deltaV1 
                                                    - (upper_tsurface*g1.S2 - lower_tsurface*g2.S2)/deltaV2
                                                    -(rhoc*vc*uc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg) ) ) + source_S2[xcoordinate + xphysical_grid*ycoordinate];

                L.tau[xcoordinate + xphysical_grid*ycoordinate] = - (right_rsurface*f1.tau - left_rsurface*f2.tau)/deltaV1 
                                                    - (upper_tsurface*g1.tau - lower_tsurface*g2.tau)/deltaV2 + source_tau[xcoordinate + xphysical_grid*ycoordinate];

            

                }

                
            }


            return L;

        } else {
            MinMod fslope;
            PrimData prods;
            toWritePrim(&prims, &prods);
            fslope.NX           = NX;
            fslope.theta        = theta;
            fslope.prims        = prods;
            fslope.active_zones = active_zones;
            fslope.j_bound      = j_bound;
            fslope.i_bound      = i_bound;
            // fslope.compute(1);
            for (int jj = j_start; jj < j_bound; jj++){
                for (int ii = i_start; ii < i_bound; ii++){
                    if (periodic){
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Fix this */

                    } else {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;
                        ycoordinate = jj - 2;

                        // Coordinate X
                        xleft_most.rho  = prims.rho[(ii - 2) + NX * jj];
                        xleft_mid.rho   = prims.rho[(ii - 1) + NX * jj];
                        center.rho      = prims.rho[ii + NX * jj];
                        xright_mid.rho  = prims.rho[(ii + 1) + NX * jj];
                        xright_most.rho = prims.rho[(ii + 2) + NX * jj];

                        xleft_most.v1   = prims.v1[(ii - 2) + NX * jj];
                        xleft_mid.v1    = prims.v1[(ii - 1) + NX * jj];
                        center.v1       = prims.v1[ii + NX * jj];
                        xright_mid.v1   = prims.v1[(ii + 1) + NX * jj];
                        xright_most.v1  = prims.v1[(ii + 2) + NX * jj];

                        xleft_most.v2   = prims.v2[(ii - 2) + NX * jj];
                        xleft_mid.v2    = prims.v2[(ii - 1) + NX * jj];
                        center.v2       = prims.v2[ii + NX * jj];
                        xright_mid.v2   = prims.v2[(ii + 1) + NX * jj];
                        xright_most.v2  = prims.v2[(ii + 2) + NX * jj];

                        xleft_most.p    = prims.p[(ii - 2) + NX * jj];
                        xleft_mid.p     = prims.p[(ii - 1) + NX * jj];
                        center.p        = prims.p[ii + NX * jj];
                        xright_mid.p    = prims.p[(ii + 1) + NX * jj];
                        xright_most.p   = prims.p[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most.rho  = prims.rho[ii + NX * (jj - 2)];
                        yleft_mid.rho   = prims.rho[ii + NX * (jj - 1)];
                        yright_mid.rho  = prims.rho[ii + NX * (jj + 1)];
                        yright_most.rho = prims.rho[ii + NX * (jj + 2)];

                        yleft_most.v1   = prims.v1[ii + NX * (jj - 2)];
                        yleft_mid.v1    = prims.v1[ii + NX * (jj - 1)];
                        yright_mid.v1   = prims.v1[ii + NX * (jj + 1)];
                        yright_most.v1  = prims.v1[ii + NX * (jj + 2)];

                        yleft_most.v2   = prims.v2[ii + NX * (jj - 2)];
                        yleft_mid.v2    = prims.v2[ii + NX * (jj - 1)];
                        yright_mid.v2   = prims.v2[ii + NX * (jj + 1)];
                        yright_most.v2  = prims.v2[ii + NX * (jj - 2)];

                        yleft_most.p    = prims.p[ii + NX * (jj - 2)];
                        yleft_mid.p     = prims.p[ii + NX * (jj - 1)];
                        yright_mid.p    = prims.p[ii + NX * (jj + 1)];
                        yright_most.p   = prims.p[ii + NX * (jj + 2)];

                    }
                    
                    // Reconstructed left X Primitives vector at the i+1/2 interface
                    // toWritePrim(&prims, &prods);
                    // fslope = minmod(prods, theta, 1, ii, jj, NX);
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

                    // Reconstructed right Primitives vector in x
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

                    
                    // Reconstructed right Primitives vector in y-direction at j+1/2 interfce
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
                    
                    // Calculate the left and right states using the reconstructed PLM Primitives
                    ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

                    if (hllc){
                        f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    } else {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    }
                    
                    // Do the same thing, but for the left side interface [i - 1/2]

                    // Left side Primitives in x
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


                    // Right side Primitives in x
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


                    // Left side Primitives in y
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

                        
                    // Right side Primitives in y
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
                    
                

                    // Calculate the left and right states using the reconstructed PLM Primitives
                    ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                    uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                    f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                    f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                    g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                    g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);
                    
                    
                    if (hllc){
                        f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    } else {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l,xprims_r,  1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                    }
                    
                    if (linspace){
                        right_cell = x1[xcoordinate + 1];
                        left_cell  = x1[xcoordinate - 1];
                        upper_cell = x2[ycoordinate + 1];
                        lower_cell = x2[ycoordinate - 1];
                        
                        // Outflow the left/right boundaries
                        if (xcoordinate - 1 < 0){
                            left_cell = x1[xcoordinate];

                        } else if (xcoordinate == xphysical_grid - 1){
                            right_cell = x1[xcoordinate];

                        }

                        if (ycoordinate - 1 < 0){
                            lower_cell = x2[ycoordinate];
                        }  else if(ycoordinate == yphysical_grid - 1){
                            upper_cell = x2[ycoordinate];
                        }

                        
                        r_right = 0.5*(right_cell + x1[xcoordinate]);
                        r_left = 0.5*(x1[xcoordinate] + left_cell);

                        theta_right = 0.5 * (upper_cell + x2[ycoordinate]);
                        theta_left  = 0.5 * (lower_cell + x2[ycoordinate]);

                    } else {
                        
                        right_cell = x1[xcoordinate + 1];
                        left_cell  = x1[xcoordinate - 1];

                        upper_cell = x2[ycoordinate + 1];
                        lower_cell = x2[ycoordinate - 1];
                        
                        if (xcoordinate - 1 < 0){
                            left_cell = x1[xcoordinate];

                        } else if (xcoordinate == xphysical_grid - 1){
                            right_cell = x1[xcoordinate];
                        }

                        r_right = sqrt(right_cell * x1[xcoordinate]); //sqrt(right_cell * x1[xcoordinate]); //+ x1[xcoordinate]);
                        r_left  = sqrt(left_cell  * x1[xcoordinate]); //sqrt(left_cell  * x1[xcoordinate]); // + left_cell);

                        // Outflow the left/right boundaries
                        if (ycoordinate - 1 < 0){
                            lower_cell = x2[ycoordinate];

                        } else if(ycoordinate == yphysical_grid - 1){
                            upper_cell = x2[ycoordinate];
                        }

                        theta_right = 0.5 * (upper_cell + x2[ycoordinate]);
                        theta_left  = 0.5 * (lower_cell + x2[ycoordinate]);

                    }   

                    dr   = r_right - r_left;
                    rhoc = center.rho;
                    pc   = center.p;
                    uc   = center.v1;
                    vc   = center.v2;

                    ang_avg = 0.5 *(theta_right + theta_left); //atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );

                    // Compute the surface areas
                    right_rsurface = r_right * r_right ;
                    left_rsurface  = r_left  * r_left  ;
                    upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                    lower_tsurface = sin(theta_left) ; //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                    volAvg = 0.75*( (r_right * r_right * r_right * r_right - r_left * r_left * r_left * r_left) / 
                                            (r_right * r_right * r_right -  r_left * r_left * r_left) );

                    deltaV1 = volAvg * volAvg * dr;
                    deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); 


                    L.D.emplace_back(- (f1.D*right_rsurface - f2.D*left_rsurface)/deltaV1
                                                        - (g1.D*upper_tsurface - g2.D*lower_tsurface)/deltaV2 + sourceD[xcoordinate + xphysical_grid*ycoordinate]);

                    L.S1.emplace_back( - (f1.S1*right_rsurface - f2.S1*left_rsurface)/deltaV1
                                                        - (g1.S1*upper_tsurface - g2.S1*lower_tsurface)/deltaV2 
                                                        + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[xcoordinate + xphysical_grid*ycoordinate]);

                    L.S2.emplace_back( - (f1.S2*right_rsurface - f2.S2*left_rsurface)/deltaV1
                                                        - (g1.S2*upper_tsurface - g2.S2*lower_tsurface)/deltaV2
                                                        -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg))) + source_S2[xcoordinate + xphysical_grid*ycoordinate] );

                    L.tau.emplace_back(- (f1.tau*right_rsurface - f2.tau*left_rsurface)/deltaV1
                                                        - (g1.tau*upper_tsurface - g2.tau*lower_tsurface)/deltaV2 + source_tau[xcoordinate + xphysical_grid*ycoordinate] );

                }

            }

        return L;

        }
        
    }
    

};

Conserved UstateSR2D::u_dot(unsigned int ii, unsigned int jj)
{
    int  xcoordinate, ycoordinate;
    
    Conserved L;
    Conserved ux_l, ux_r, uy_l, uy_r; 
    Flux     f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    Primitives   xprims_l, xprims_r, yprims_l, yprims_r;

    Primitives xleft_most, xleft_mid, xright_mid, xright_most;
    Primitives yleft_most, yleft_mid, yright_mid, yright_most;
    Primitives center;
    
    if (coord_system == "cartesian"){
        double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
        double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;
        if (first_order){

            // i+1/2
            ux_l.D   = u_state.D[ii + NX * jj];
            ux_l.S1  = u_state.S1[ii + NX * jj];
            ux_l.S2  = u_state.S2[ii + NX * jj];
            ux_l.tau = u_state.tau[ii + NX * jj];

            ux_r.D   = u_state.D[(ii + 1) + NX * jj];
            ux_r.S1  = u_state.S1[(ii + 1) + NX * jj];
            ux_r.S2  = u_state.S2[(ii + 1) + NX * jj];
            ux_r.tau = u_state.tau[(ii + 1) + NX * jj];

            // j+1/2
            uy_l.D   = u_state.D[ii + NX * jj];
            uy_l.S1  = u_state.S1[ii + NX * jj];
            uy_l.S2  = u_state.S2[ii + NX * jj];
            uy_l.tau = u_state.tau[ii + NX * jj];

            uy_r.D   = u_state.D[(ii + 1) + NX * jj];
            uy_r.S1  = u_state.S1[(ii + 1) + NX * jj];
            uy_r.S2  = u_state.S2[(ii + 1) + NX * jj];
            uy_r.tau = u_state.tau[(ii + 1) + NX * jj];

            xprims_l.rho = prims.rho[ii + jj * NX]; 
            xprims_l.v1  = prims.v1 [ii + jj * NX];
            xprims_l.v2  = prims.v2 [ii + jj * NX];
            xprims_l.p   = prims.p  [ii + jj * NX];

            xprims_r.rho = prims.rho[(ii + 1) + jj * NX]; 
            xprims_r.v1  = prims.v1 [(ii + 1) + jj * NX];
            xprims_r.v2  = prims.v2 [(ii + 1) + jj * NX];
            xprims_r.p   = prims.p  [(ii + 1) + jj * NX];

            yprims_l.rho = prims.rho[ii + jj * NX]; 
            yprims_l.v1  = prims.v1 [ii + jj * NX];
            yprims_l.v2  = prims.v2 [ii + jj * NX];
            yprims_l.p   = prims.p  [ii + jj * NX];

            yprims_r.rho = prims.rho[ii + (jj + 1.) * NX]; 
            yprims_r.v1  = prims.v1 [ii + (jj + 1.) * NX];
            yprims_r.v2  = prims.v2 [ii + (jj + 1.) * NX];
            yprims_r.p   = prims.p  [ii + (jj + 1.) * NX];

            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l.D   = u_state.D[(ii - 1) + NX * jj];
            ux_l.S1  = u_state.S1[(ii - 1) + NX * jj];
            ux_l.S2  = u_state.S2[(ii - 1) + NX * jj];
            ux_l.tau = u_state.tau[(ii - 1) + NX * jj];

            ux_r.D   = u_state.D[ii + NX * jj];
            ux_r.S1  = u_state.S1[ii + NX * jj];
            ux_r.S2  = u_state.S2[ii + NX * jj];
            ux_r.tau = u_state.tau[ii + NX * jj];

            // j-1/2
            uy_l.D   = u_state.D[(ii - 1) + NX * jj];
            uy_l.S1  = u_state.S1[(ii - 1) + NX * jj];
            uy_l.S2  = u_state.S2[(ii - 1) + NX * jj];
            uy_l.tau = u_state.tau[(ii - 1) + NX * jj];

            uy_r.D   = u_state.D[ii + NX * jj];
            uy_r.S1  = u_state.S1[ii + NX * jj];
            uy_r.S2  = u_state.S2[ii + NX * jj];
            uy_r.tau = u_state.tau[ii + NX * jj];

            xprims_l.rho = prims.rho[(ii - 1) + jj * NX]; 
            xprims_l.v1  = prims.v1 [(ii - 1) + jj * NX];
            xprims_l.v2  = prims.v2 [(ii - 1) + jj * NX];
            xprims_l.p   = prims.p  [(ii - 1) + jj * NX];

            xprims_r.rho = prims.rho[ii + jj * NX]; 
            xprims_r.v1  = prims.v1 [ii + jj * NX];
            xprims_r.v2  = prims.v2 [ii + jj * NX];
            xprims_r.p   = prims.p  [ii + jj * NX];

            yprims_l.rho = prims.rho[ii + (jj - 1) * NX]; 
            yprims_l.v1  = prims.v1 [ii + (jj - 1) * NX];
            yprims_l.v2  = prims.v2 [ii + (jj - 1) * NX];
            yprims_l.p   = prims.p  [ii + (jj - 1) * NX];

            yprims_r.rho = prims.rho[ii + jj * NX]; 
            yprims_r.v1  = prims.v1 [ii + jj * NX];
            yprims_r.v2  = prims.v2 [ii + jj * NX];
            yprims_r.p   = prims.p  [ii + jj * NX];

            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            

            L.D   = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
            L.S1  = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
            L.S2  = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
            L.tau = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;

            
        

            return L;

        } else {
            
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
                    xleft_most.rho  = prims.rho[(ii - 2) + NX * jj];
                    xleft_mid.rho   = prims.rho[(ii - 1) + NX * jj];
                    center.rho      = prims.rho[ii + NX * jj];
                    xright_mid.rho  = prims.rho[(ii + 1) + NX * jj];
                    xright_most.rho = prims.rho[(ii + 2) + NX * jj];

                    xleft_most.v1  = prims.v1[(ii - 2) + NX*jj];
                    xleft_mid.v1   = prims.v1[(ii - 1) + NX * jj];
                    center.v1      = prims.v1[ii + NX * jj];
                    xright_mid.v1  = prims.v1[(ii + 1) + NX * jj];
                    xright_most.v1 = prims.v1[(ii + 2) + NX * jj];

                    xleft_most.v2  = prims.v2[(ii - 2) + NX*jj];
                    xleft_mid.v2   = prims.v2[(ii - 1) + NX * jj];
                    center.v2      = prims.v2[ii + NX * jj];
                    xright_mid.v2  = prims.v2[(ii + 1) + NX * jj];
                    xright_most.v2 = prims.v2[(ii + 2) + NX * jj];

                    xleft_most.p  = prims.p[(ii - 2) + NX*jj];
                    xleft_mid.p   = prims.p[(ii - 1) + NX * jj];
                    center.p      = prims.p[ii + NX * jj];
                    xright_mid.p  = prims.p[(ii + 1) + NX * jj];
                    xright_most.p = prims.p[(ii + 2) + NX * jj];

                    // Coordinate Y
                    yleft_most.rho   = prims.rho[ii + NX * (jj - 2)];
                    yleft_mid.rho    = prims.rho[ii + NX * (jj - 1)];
                    yright_mid.rho   = prims.rho[ii + NX * (jj + 1)];
                    yright_most.rho  = prims.rho[ii + NX * (jj + 2)];

                    yleft_most.v1   = prims.v1[ii + NX * (jj - 2)];
                    yleft_mid.v1    = prims.v1[ii + NX * (jj - 1)];
                    yright_mid.v1   = prims.v1[ii + NX * (jj + 1)];
                    yright_most.v1  = prims.v1[ii + NX * (jj + 2)];

                    yleft_most.v2   = prims.v2[ii + NX * (jj - 2)];
                    yleft_mid.v2    = prims.v2[ii + NX * (jj - 1)];
                    yright_mid.v2   = prims.v2[ii + NX * (jj + 1)];
                    yright_most.v2  = prims.v2[ii + NX * (jj + 2)];

                    yleft_most.p   = prims.p[ii + NX * (jj - 2)];
                    yleft_mid.p    = prims.p[ii + NX * (jj - 1)];
                    yright_mid.p   = prims.p[ii + NX * (jj + 1)];
                    yright_most.p  = prims.p[ii + NX * (jj + 2)];

                }
                
                // Reconstructed left X Primitives vector at the i+1/2 interface
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

                // Reconstructed right Primitives vector in x
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

                
                // Reconstructed right Primitives vector in y-direction at j+1/2 interfce
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

            
                
                // Calculate the left and right states using the reconstructed PLM Primitives
                ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r,  2);
                




                // Left side Primitives in x
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

                    
                // Right side Primitives in x
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


                // Left side Primitives in y
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

                    
                // Right side Primitives in y
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
                
            

                // Calculate the left and right states using the reconstructed PLM Primitives
                ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
                uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

                f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
                f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

                g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
                g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);


                f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
                
                L.D   = - (f1.D - f2.D)/dx - (g1.D - g2.D)/dy;
                L.S1  = - (f1.S1 - f2.S1)/dx - (g1.S1 - g2.S1)/dy;
                L.S2  = - (f1.S2 - f2.S2)/dx - (g1.S2 - g2.S2)/dy;
                L.tau = - (f1.tau - f2.tau)/dx - (g1.tau - g2.tau)/dy;
                
            

            return L;

        }

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, ycoordinate, xcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/(xphysical_grid - 1);

        double dr; 

        if (first_order){
            ycoordinate = jj - 1;
            xcoordinate = ii - 1;

            // i+1/2
            ux_l.D   = u_state.D[ii + NX * jj];
            ux_l.S1  = u_state.S1[ii + NX * jj];
            ux_l.S2  = u_state.S2[ii + NX * jj];
            ux_l.tau = u_state.tau[ii + NX * jj];

            ux_r.D   = u_state.D[(ii + 1) + NX * jj];
            ux_r.S1  = u_state.S1[(ii + 1) + NX * jj];
            ux_r.S2  = u_state.S2[(ii + 1) + NX * jj];
            ux_r.tau = u_state.tau[(ii + 1) + NX * jj];

            // j+1/2
            uy_l.D    = u_state.D[ii + NX * jj];
            uy_l.S1   = u_state.S1[ii + NX * jj];
            uy_l.S2   = u_state.S2[ii + NX * jj];
            uy_l.tau  = u_state.tau[ii + NX * jj];

            uy_r.D    = u_state.D[ii + NX * (jj + 1)];
            uy_r.S1   = u_state.S1[ii + NX * (jj + 1)];
            uy_r.S2   = u_state.S2[ii + NX * (jj + 1)];
            uy_r.tau  = u_state.tau[ii + NX * (jj + 1)];

            xprims_l.rho = prims.rho[ii + jj * NX]; 
            xprims_l.v1  = prims.v1 [ii + jj * NX];
            xprims_l.v2  = prims.v2 [ii + jj * NX];
            xprims_l.p   = prims.p  [ii + jj * NX];

            xprims_r.rho = prims.rho[(ii + 1) + jj * NX]; 
            xprims_r.v1  = prims.v1 [(ii + 1) + jj * NX];
            xprims_r.v2  = prims.v2 [(ii + 1) + jj * NX];
            xprims_r.p   = prims.p  [(ii + 1) + jj * NX];

            yprims_l.rho = prims.rho[ii + jj * NX]; 
            yprims_l.v1  = prims.v1 [ii + jj * NX];
            yprims_l.v2  = prims.v2 [ii + jj * NX];
            yprims_l.p   = prims.p  [ii + jj * NX];

            yprims_r.rho = prims.rho[ii + (jj + 1.) * NX]; 
            yprims_r.v1  = prims.v1 [ii + (jj + 1.) * NX];
            yprims_r.v2  = prims.v2 [ii + (jj + 1.) * NX];
            yprims_r.p   = prims.p  [ii + (jj + 1.) * NX];

            rhoc = xprims_l.rho;
            pc   = xprims_l.p;
            uc   = xprims_l.v1;
            vc   = xprims_l.v2;
            
            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            // Calc HLLE/C Flux at i+1/2 interface
            if (hllc){
                f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
            
            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l.D    = u_state.D[(ii - 1) + NX * jj];
            ux_l.S1   = u_state.S1[(ii - 1) + NX * jj];
            ux_l.S2   = u_state.S2[(ii - 1) + NX * jj];
            ux_l.tau  = u_state.tau[(ii - 1) + NX * jj];

            ux_r.D    = u_state.D[ii + NX * jj];
            ux_r.S1   = u_state.S1[ii + NX * jj];
            ux_r.S2   = u_state.S2[ii + NX * jj];
            ux_r.tau  = u_state.tau[ii + NX * jj];

            // j-1/2
            uy_l.D    = u_state.D[ii + NX * (jj - 1)];
            uy_l.S1   = u_state.S1[ii + NX * (jj - 1)];
            uy_l.S2   = u_state.S2[ii + NX * (jj - 1)];
            uy_l.tau  = u_state.tau[ii + NX * (jj - 1)];

            uy_r.D    = u_state.D[ii + NX * jj];
            uy_r.S1   = u_state.S1[ii + NX * jj];
            uy_r.S2   = u_state.S2[ii + NX * jj];
            uy_r.tau  = u_state.tau[ii + NX * jj];

            xprims_l.rho = prims.rho[(ii - 1) + jj * NX];
            xprims_l.v1  = prims.v1 [(ii - 1) + jj * NX];
            xprims_l.v2  = prims.v2 [(ii - 1) + jj * NX];
            xprims_l.p   = prims.p  [(ii - 1) + jj * NX];

            xprims_r.rho = prims.rho[ii + jj * NX]; 
            xprims_r.v1  = prims.v1 [ii + jj * NX];
            xprims_r.v2  = prims.v2 [ii + jj * NX];
            xprims_r.p   = prims.p  [ii + jj * NX];

            yprims_l.rho = prims.rho[ii + (jj - 1) * NX]; 
            yprims_l.v1  = prims.v1 [ii + (jj - 1) * NX];
            yprims_l.v2  = prims.v2 [ii + (jj - 1) * NX];
            yprims_l.p   = prims.p  [ii + (jj - 1) * NX];

            yprims_r.rho = prims.rho[ii + jj * NX]; 
            yprims_r.v1  = prims.v1 [ii + jj * NX];
            yprims_r.v2  = prims.v2 [ii + jj * NX];
            yprims_r.p   = prims.p  [ii + jj * NX];

            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            // Calc HLL Flux at i+1/2 interface
            if (hllc){
                f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
            
 
            if (linspace){
                right_cell = x1[xcoordinate + 1];
                left_cell  = x1[xcoordinate - 1];
                upper_cell = x2[ycoordinate + 1];
                lower_cell = x2[ycoordinate - 1];

                // Outflow the left/right boundaries
                if (xcoordinate - 1 < 0){
                    left_cell = x1[xcoordinate];

                } else if (xcoordinate == xphysical_grid - 1){
                    right_cell = x1[xcoordinate];

                }

                if (ycoordinate - 1 < 0){
                    lower_cell = x2[ycoordinate];
                }  else if(ycoordinate == yphysical_grid - 1){
                    upper_cell = x2[ycoordinate];
                }

                
                r_right = 0.5*(right_cell + x1[xcoordinate]);
                r_left  = 0.5*(x1[xcoordinate] + left_cell);

                theta_right = atan2( sin(upper_cell) + sin(x2[ycoordinate]) , 
                                            cos(upper_cell) + cos(x2[ycoordinate]) );

                theta_left = atan2( sin(lower_cell) + sin(x2[ycoordinate]), 
                                            cos(lower_cell) + cos(x2[ycoordinate]) );

                theta_right = 0.5*(upper_cell + x2[ycoordinate]);
                theta_left = 0.5*(lower_cell + x2[ycoordinate]);

        } else {
            right_cell = x1[xcoordinate + 1];
            left_cell  = x1[xcoordinate - 1];

            upper_cell = x2[ycoordinate + 1];
            lower_cell = x2[ycoordinate - 1];
            
            if (xcoordinate - 1 < 0){
                left_cell = x1[xcoordinate];

            } else if (xcoordinate == xphysical_grid - 1){
                right_cell = x1[xcoordinate];
            }

            r_right = sqrt(right_cell * x1[xcoordinate]);
            r_left  = sqrt(left_cell  * x1[xcoordinate]);


            // Outflow the left/right boundaries
            if (ycoordinate - 1 < 0){
                lower_cell = x2[ycoordinate];

            } else if(ycoordinate == yphysical_grid - 1){
                upper_cell = x2[ycoordinate];
            }

            theta_right = 0.5 * (upper_cell + x2[ycoordinate]);
            theta_left  = 0.5 * (lower_cell + x2[ycoordinate]);
        }

        dr = r_right - r_left;
        
        
        ang_avg = 0.5 *(theta_right + theta_left); //atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
        // Compute the surface areas
        right_rsurface = r_right*r_right;
        left_rsurface = r_left*r_left;
        upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
        lower_tsurface = sin(theta_left); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
        volAvg = 0.75*( (r_right * r_right * r_right * r_right - r_left * r_left * r_left * r_left) / 
                                (r_right * r_right * r_right -  r_left * r_left * r_left) );

        deltaV1 = volAvg * volAvg * dr;
        deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); //deltaV1*(cos(theta_left) - cos(theta_right)); 

        L.D   = - (right_rsurface*f1.D - left_rsurface*f2.D)/deltaV1 
                                            - (upper_tsurface*g1.D - lower_tsurface*g2.D)/deltaV2 + sourceD[xcoordinate + xphysical_grid*ycoordinate];

        L.S1  = - (right_rsurface*f1.S1 - left_rsurface*f2.S1)/deltaV1 
                                            - (upper_tsurface*g1.S1 - lower_tsurface*g2.S1)/deltaV2 
                                            + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[xcoordinate + xphysical_grid*ycoordinate];

        L.S2  = - (right_rsurface*f1.S2 - left_rsurface*f2.S2)/deltaV1 
                                            - (upper_tsurface*g1.S2 - lower_tsurface*g2.S2)/deltaV2
                                            -(rhoc*vc*uc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg) ) ) + source_S2[xcoordinate + xphysical_grid*ycoordinate];

        L.tau = - (right_rsurface*f1.tau - left_rsurface*f2.tau)/deltaV1 
                                            - (upper_tsurface*g1.tau - lower_tsurface*g2.tau)/deltaV2 + source_tau[xcoordinate + xphysical_grid*ycoordinate];

        


        return L;

        } else {

            if (periodic){
                xcoordinate = ii;
                ycoordinate = jj;

                // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                /* TODO: Fix this */

            } else {
                // Adjust for beginning input of L vector
                xcoordinate = ii - 2;
                ycoordinate = jj - 2;

                // Coordinate X
                xleft_most.rho  = prims.rho[(ii - 2) + NX * jj];
                xleft_mid.rho   = prims.rho[(ii - 1) + NX * jj];
                center.rho      = prims.rho[ii + NX * jj];
                xright_mid.rho  = prims.rho[(ii + 1) + NX * jj];
                xright_most.rho = prims.rho[(ii + 2) + NX * jj];

                xleft_most.v1   = prims.v1[(ii - 2) + NX * jj];
                xleft_mid.v1    = prims.v1[(ii - 1) + NX * jj];
                center.v1       = prims.v1[ii + NX * jj];
                xright_mid.v1   = prims.v1[(ii + 1) + NX * jj];
                xright_most.v1  = prims.v1[(ii + 2) + NX * jj];

                xleft_most.v2   = prims.v2[(ii - 2) + NX * jj];
                xleft_mid.v2    = prims.v2[(ii - 1) + NX * jj];
                center.v2       = prims.v2[ii + NX * jj];
                xright_mid.v2   = prims.v2[(ii + 1) + NX * jj];
                xright_most.v2  = prims.v2[(ii + 2) + NX * jj];

                xleft_most.p    = prims.p[(ii - 2) + NX * jj];
                xleft_mid.p     = prims.p[(ii - 1) + NX * jj];
                center.p        = prims.p[ii + NX * jj];
                xright_mid.p    = prims.p[(ii + 1) + NX * jj];
                xright_most.p   = prims.p[(ii + 2) + NX * jj];

                // Coordinate Y
                yleft_most.rho  = prims.rho[ii + NX * (jj - 2)];
                yleft_mid.rho   = prims.rho[ii + NX * (jj - 1)];
                yright_mid.rho  = prims.rho[ii + NX * (jj + 1)];
                yright_most.rho = prims.rho[ii + NX * (jj + 2)];

                yleft_most.v1   = prims.v1[ii + NX * (jj - 2)];
                yleft_mid.v1    = prims.v1[ii + NX * (jj - 1)];
                yright_mid.v1   = prims.v1[ii + NX * (jj + 1)];
                yright_most.v1  = prims.v1[ii + NX * (jj + 2)];

                yleft_most.v2   = prims.v2[ii + NX * (jj - 2)];
                yleft_mid.v2    = prims.v2[ii + NX * (jj - 1)];
                yright_mid.v2   = prims.v2[ii + NX * (jj + 1)];
                yright_most.v2  = prims.v2[ii + NX * (jj - 2)];

                yleft_most.p    = prims.p[ii + NX * (jj - 2)];
                yleft_mid.p     = prims.p[ii + NX * (jj - 1)];
                yright_mid.p    = prims.p[ii + NX * (jj + 1)];
                yright_most.p   = prims.p[ii + NX * (jj + 2)];

            }
            
            // Reconstructed left X Primitives vector at the i+1/2 interface
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

            // Reconstructed right Primitives vector in x
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

            
            // Reconstructed right Primitives vector in y-direction at j+1/2 interfce
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
            
            // Calculate the left and right states using the reconstructed PLM Primitives
            ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
            uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            if (hllc){
                f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l, xprims_r, 1);
                g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
            }
            
            // Do the same thing, but for the left side interface [i - 1/2]

            // Left side Primitives in x
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

                
            // Right side Primitives in x
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


            // Left side Primitives in y
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

                
            // Right side Primitives in y
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
            
        

            // Calculate the left and right states using the reconstructed PLM Primitives
            ux_l = calc_stateSR2D(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            ux_r = calc_stateSR2D(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            uy_l = calc_stateSR2D(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p);
            uy_r = calc_stateSR2D(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p);

            f_l = calc_Flux(xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_Flux(xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_Flux(yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_Flux(yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);
            
            
            if (hllc){
                f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r,xprims_l,xprims_r,  1);
                g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r,yprims_l, yprims_r, 2);
            }
            
            if (linspace){
                right_cell = x1[xcoordinate + 1];
                left_cell  = x1[xcoordinate - 1];
                upper_cell = x2[ycoordinate + 1];
                lower_cell = x2[ycoordinate - 1];
                
                // Outflow the left/right boundaries
                if (xcoordinate - 1 < 0){
                    left_cell = x1[xcoordinate];

                } else if (xcoordinate == xphysical_grid - 1){
                    right_cell = x1[xcoordinate];

                }

                if (ycoordinate - 1 < 0){
                    lower_cell = x2[ycoordinate];
                }  else if(ycoordinate == yphysical_grid - 1){
                    upper_cell = x2[ycoordinate];
                }

                
                r_right = 0.5*(right_cell + x1[xcoordinate]);
                r_left = 0.5*(x1[xcoordinate] + left_cell);

                theta_right = atan2( sin(upper_cell) + sin(x2[ycoordinate]) , 
                                            cos(upper_cell) + cos(x2[ycoordinate]) );

                theta_left = atan2( sin(lower_cell) + sin(x2[ycoordinate]), 
                                            cos(lower_cell) + cos(x2[ycoordinate]) );

            } else {
                // log_rLeft = log10(x1[0]) + xcoordinate*delta_logr;
                // log_rRight = log_rLeft + delta_logr;
                // r_left = pow(10, log_rLeft);
                // r_right = pow(10, log_rRight);

                right_cell = x1[xcoordinate + 1];
                left_cell  = x1[xcoordinate - 1];

                upper_cell = x2[ycoordinate + 1];
                lower_cell = x2[ycoordinate - 1];
                
                if (xcoordinate - 1 < 0){
                    left_cell = x1[xcoordinate];

                } else if (xcoordinate == xphysical_grid - 1){
                    right_cell = x1[xcoordinate];
                }

                r_right = sqrt(right_cell * x1[xcoordinate]); //sqrt(right_cell * x1[xcoordinate]); //+ x1[xcoordinate]);
                r_left  = sqrt(left_cell  *  x1[xcoordinate]); //sqrt(left_cell  * x1[xcoordinate]); // + left_cell);

                // Outflow the left/right boundaries
                if (ycoordinate - 1 < 0){
                    lower_cell = x2[ycoordinate];

                } else if(ycoordinate == yphysical_grid - 1){
                    upper_cell = x2[ycoordinate];
                }

                // theta_right = atan2( sin(upper_cell) + sin(x2[ycoordinate]) , 
                //                             cos(upper_cell) + cos(x2[ycoordinate]) );

                // theta_left = atan2( sin(lower_cell) + sin(x2[ycoordinate]), 
                //                             cos(lower_cell) + cos(x2[ycoordinate]) );

                theta_right = 0.5 * (upper_cell + x2[ycoordinate]);
                theta_left  = 0.5 * (lower_cell + x2[ycoordinate]);

            }   

            dr   = r_right - r_left;
            rhoc = center.rho;
            pc   = center.p;
            uc   = center.v1;
            vc   = center.v2;

            ang_avg = 0.5 *(theta_right + theta_left); //atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );

            // Compute the surface areas
            right_rsurface = r_right * r_right ;
            left_rsurface  = r_left  * r_left  ;
            upper_tsurface = sin(theta_right); //0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
            lower_tsurface = sin(theta_left) ; //0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
            volAvg = 0.75*( (r_right * r_right * r_right * r_right - r_left * r_left * r_left * r_left) / 
                                    (r_right * r_right * r_right -  r_left * r_left * r_left) );

            deltaV1 = volAvg * volAvg * dr;
            deltaV2 = volAvg * sin(ang_avg)*(theta_right - theta_left); 

            L.D   = - (f1.D*right_rsurface - f2.D*left_rsurface)/deltaV1
                                                - (g1.D*upper_tsurface - g2.D*lower_tsurface)/deltaV2 + sourceD[xcoordinate + xphysical_grid*ycoordinate];

            L.S1  = - (f1.S1*right_rsurface - f2.S1*left_rsurface)/deltaV1
                                                - (g1.S1*upper_tsurface - g2.S1*lower_tsurface)/deltaV2 
                                                + rhoc*vc*vc/volAvg + 2*pc/volAvg + source_S1[xcoordinate + xphysical_grid*ycoordinate];

            L.S2  = - (f1.S2*right_rsurface - f2.S2*left_rsurface)/deltaV1
                                                - (g1.S2*upper_tsurface - g2.S2*lower_tsurface)/deltaV2
                                                -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg))) + source_S2[xcoordinate + xphysical_grid*ycoordinate];

            L.tau = - (f1.tau*right_rsurface - f2.tau*left_rsurface)/deltaV1
                                                - (g1.tau*upper_tsurface - g2.tau*lower_tsurface)/deltaV2 + source_tau[xcoordinate + xphysical_grid*ycoordinate];


        return L;

        }
        
    }
    

};


//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
twoVec UstateSR2D::simulate2D(vector<double> lorentz_gamma, 
                                    const vector<vector<double> > sources,
                                    float tend = 0.1, 
                                    bool first_order = true, bool periodic = false,
                                    bool linspace=true, bool hllc=false,
                                    double dt = 1.e-4){

    
    int i_real, j_real;
    string tnow, tchunk, tstep;
    int total_zones = NX * NY;
    double t0 = 0;
    double t_interval = 0.1;
    double s_interval = 0.1;
    float t = 0;
    string filename;

    this->sources       = sources;
    this->first_order   = first_order;
    this->periodic      = periodic;
    this->hllc          = hllc;
    this->linspace      = linspace;
    this->theta         = theta;
    this->lorentz_gamma = lorentz_gamma;

    if (first_order){
        this->xphysical_grid = NX - 2;
        this->yphysical_grid = NY - 2;
        this->idx_active = 1;
        this->x_bound = NX - 1;
        this->y_bound = NY - 1;
    } else {
        this->xphysical_grid = NX - 4;
        this->yphysical_grid = NY - 4;
        this->idx_active = 2;
        this->x_bound = NX - 2;
        this->y_bound = NY - 2;
    }

    this->active_zones = xphysical_grid * yphysical_grid;

    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.xmax = x1[xphysical_grid - 1];
    setup.xmin = x1[0];
    setup.ymax = x2[yphysical_grid - 1];
    setup.ymin = x2[0];
    setup.NX   = NX;
    setup.NY   = NY;

    ConserveArray u, u1, u2, udot, udot1, u_p, state;
    PrimData prods;
    u.D.reserve(nzones);
    u.S1.reserve(nzones);
    u.S2.reserve(nzones);
    u.tau.reserve(nzones);

    u1.D.reserve(nzones);
    u1.S1.reserve(nzones);
    u1.S2.reserve(nzones);
    u1.tau.reserve(nzones);

    u2.D.reserve(nzones);
    u2.S1.reserve(nzones);
    u2.S2.reserve(nzones);
    u2.tau.reserve(nzones);

    udot.D.reserve(active_zones);
    udot.S1.reserve(active_zones);
    udot.S2.reserve(active_zones);
    udot.tau.reserve(active_zones);

    u_p.D.reserve(nzones);
    u_p.S1.reserve(nzones);
    u_p.S2.reserve(nzones);
    u_p.tau.reserve(nzones);

    prims.rho.resize(nzones);
    prims.v1.resize(nzones);
    prims.v2.resize(nzones);
    prims.p.reserve(nzones);

    // Define the source terms
    sourceD    = sources[0];
    source_S1  = sources[1];
    source_S2  = sources[2];
    source_tau = sources[3];

    // Copy the state array into real & profile variables
    u.D   = state2D[0];
    u.S1  = state2D[1];
    u.S2  = state2D[2];
    u.tau = state2D[3];

    u_p = u;
    u1  = u; 
    u2  = u;

    Conserved L;
    n = 0;

    block_size = 4;

    // Initialize the primitives for the initial conditions
    prims = cons2prim2D(u, lorentz_gamma);
    n++;
    if (first_order){
        while (t < tend){
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // u_state = u;
            udot = u_dot2D(u);

            for (int jj = 0; jj < yphysical_grid; jj ++){
                for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 1; j_real = jj + 1;
                        // L = u_dot(i_real, j_real);
                        u_p.D  [i_real + NX * j_real]   = u.D  [i_real + NX * j_real] + dt * udot.D  [ii + xphysical_grid * jj]; 
                        u_p.S1 [i_real + NX * j_real]   = u.S1 [i_real + NX * j_real] + dt * udot.S1 [ii + xphysical_grid * jj]; 
                        u_p.S2 [i_real + NX * j_real]   = u.S2 [i_real + NX * j_real] + dt * udot.S2 [ii + xphysical_grid * jj]; 
                        u_p.tau[i_real + NX * j_real]   = u.tau[i_real + NX * j_real] + dt * udot.tau[ii + xphysical_grid * jj]; 
                        
            
                }
            }

            config_ghosts2D(u_p, NX, NY, true);
            prims = cons2prim2D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims.v1, prims.v2, NX, NY);
            if (t > 0){
                dt = adapt_dt(prims, linspace);
            }

            u.D.swap(u_p.D  );
            u.S1.swap(u_p.S1);
            u.S2.swap(u_p.S2);
            u.tau.swap(u_p.tau);
            
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

            n++;
            pressure_guess = prims.p;

        }

    } else {
            tchunk = "000000";
            while (t < tend){
                /* Compute the loop execution time */
                high_resolution_clock::time_point t1 = high_resolution_clock::now();

                if (t == 0){
                    config_ghosts2D(u, NX, NY, false);
                }

                // u_state = u;
                udot = u_dot2D(u);

                for (int jj = 0; jj < yphysical_grid; jj ++){
                    for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 2; j_real = jj + 2;
                        // L = u_dot(i_real, j_real);
                        u1.D  [i_real + NX * j_real]   = u.D  [i_real + NX * j_real]  + dt * udot.D  [ii + xphysical_grid * jj]; 
                        u1.S1 [i_real + NX * j_real]   = u.S1 [i_real + NX * j_real]  + dt * udot.S1 [ii + xphysical_grid * jj]; 
                        u1.S2 [i_real + NX * j_real]   = u.S2 [i_real + NX * j_real]  + dt * udot.S2 [ii + xphysical_grid * jj]; 
                        u1.tau[i_real + NX * j_real]   = u.tau[i_real + NX * j_real]  + dt * udot.tau[ii + xphysical_grid * jj]; 
                }
            }
                
                
                config_ghosts2D(u1, NX, NY, false);
                prims = cons2prim2D(u1, lorentz_gamma);
                lorentz_gamma = calc_lorentz_gamma(prims.v1, prims.v2, NX, NY);

                
                udot = u_dot2D(u1);

                // u_state = u1;
                for (int jj = 0; jj < yphysical_grid; jj ++){
                    for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 2; j_real = jj + 2;
                        // L = u_dot(i_real, j_real);
                        u2.D  [i_real + NX * j_real] = 0.5 * u.D  [i_real + NX * j_real] + 0.5 * u1.D  [i_real + NX * j_real] + 0.5 * dt*udot.D  [ii + xphysical_grid * jj];
                        u2.S1 [i_real + NX * j_real] = 0.5 * u.S1 [i_real + NX * j_real] + 0.5 * u1.S1 [i_real + NX * j_real] + 0.5 * dt*udot.S1 [ii + xphysical_grid * jj];
                        u2.S2 [i_real + NX * j_real] = 0.5 * u.S2 [i_real + NX * j_real] + 0.5 * u1.S2 [i_real + NX * j_real] + 0.5 * dt*udot.S2 [ii + xphysical_grid * jj];
                        u2.tau[i_real + NX * j_real] = 0.5 * u.tau[i_real + NX * j_real] + 0.5 * u1.tau[i_real + NX * j_real] + 0.5 * dt*udot.tau[ii + xphysical_grid * jj];
                
                    }
                }
            
                config_ghosts2D(u2, NX, NY, false);

                prims = cons2prim2D(u2, lorentz_gamma);
                lorentz_gamma = calc_lorentz_gamma(prims.v1, prims.v2, NX, NY);

                
                // udot = u_dot2D(u2);
                // u_state = u2;
                // for (int jj = 0; jj < yphysical_grid; jj += block_size){
                //     for (int ii = 0; ii < xphysical_grid; ii += block_size){
                //         for (j2=jj; j2 < min(jj + block_size, yphysical_grid); j2++){
                //             for(i2 = ii; i2 < min(ii + block_size, xphysical_grid); i2 ++){
                //                 i_real = i2 + 2; j_real = j2 + 2;
                //                 // L = u_dot(i_real, j_real);
                //                 u_p.D  [i_real + NX * j_real] = (1.0/3.0)*u.D  [i_real + NX * j_real] + (2.0/3.0)*u2.D  [i_real + NX * j_real] + (2.0/3.0)*dt*udot.D  [i2 + xphysical_grid * j2];
                //                 u_p.S1 [i_real + NX * j_real] = (1.0/3.0)*u.S1 [i_real + NX * j_real] + (2.0/3.0)*u2.S1 [i_real + NX * j_real] + (2.0/3.0)*dt*udot.S1 [i2 + xphysical_grid * j2];
                //                 u_p.S2 [i_real + NX * j_real] = (1.0/3.0)*u.S2 [i_real + NX * j_real] + (2.0/3.0)*u2.S2 [i_real + NX * j_real] + (2.0/3.0)*dt*udot.S2 [i2 + xphysical_grid * j2];
                //                 u_p.tau[i_real + NX * j_real] = (1.0/3.0)*u.tau[i_real + NX * j_real] + (2.0/3.0)*u2.tau[i_real + NX * j_real] + (2.0/3.0)*dt*udot.tau[i2 + xphysical_grid * j2];
// 
                //             }
                //         }
                //         
                //     }
                // }
                
                // config_ghosts2D(u_p, NX, NY, false);
                // prims = cons2prim2D(u_p, lorentz_gamma);
                // lorentz_gamma = calc_lorentz_gamma(prims.v1, prims.v2, NX, NY);
                
                if (t > 0){
                    dt = adapt_dt(prims, linspace, false);
                }

                if (isnan(dt)){
                    break;
                }
                
                u.D.swap(u2.D  );
                u.S1.swap(u2.S1);
                u.S2.swap(u2.S2);
                u.tau.swap(u2.tau);
                
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

                n++;
                pressure_guess = prims.p;

                /* Write to a File every tenth of a second */
                if (t - t0 >= t_interval){
                    toWritePrim(&prims, &prods);
                    tnow = tchunk;
                    tstep = create_step_str(t_interval, tnow);
                    filename = string_format("%d.prods." + tstep + ".h5", NY);
                    setup.t  = t;
                    setup.dt = dt;
                    write_hdf5(filename, prods, setup);
                    t_interval += s_interval;
 
                }
                
            }
            
        }

    cout << "\n " << endl;
    
    prims = cons2prim2D(u, lorentz_gamma);

    static vector<vector<double> > solution(4, vector<double>(nzones)); 

    solution[0] = prims.rho;
    solution[1] = prims.v1;
    solution[2] = prims.v2;
    solution[3] = prims.p;

    return solution;

 };
