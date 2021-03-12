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
#include <chrono>

using namespace std;
using namespace hydro;
using namespace hydro1d;
using namespace chrono;


// Default Constructor 
SRHD::SRHD() {}

// Overloaded Constructor
SRHD::SRHD(vector< vector<double> > u_state, double gamma, double CFL, vector<double> r,
                string coord_system = "cartesian")
{
    this->state = u_state;
    this->gamma = gamma;
    this->r = r;
    this->coord_system = coord_system;
    this->CFL = CFL;

}

// Destructor 
SRHD::~SRHD() {}

//================================================
//              DATA STRUCTURES
//================================================

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
PrimitiveArray SRHD::cons2prim1D(const ConservedArray &u_state, vector<double> &lorentz_gamma){
    

    double rho, S, D, tau, pmin;
    double v, W, tol, f, g, peq;
    double eps, rhos, p, v2, et, c2;

    PrimitiveArray prims;
    prims.rho.resize(Nx);
    prims.v.resize(Nx);
    prims.p.resize(Nx);
    
    for (int ii = 0; ii < Nx; ii++){
        D   = u_state.D[ii];
        S   = u_state.S[ii];
        tau = u_state.tau[ii];

        peq = n != 0 ? pressure_guess[ii] : abs( abs(S) - tau - D);
       

        tol = D*1.e-12;

        do {
            p = peq;    
            et = tau + D + p;
            v2 = S*S/ (et * et);
            W = 1.0 / sqrt(1.0 - v2);
            rho = D/W;

            eps = (tau + (1.0 - W) * D + (1. - W * W) * p )/ (D * W);

            c2 = (gamma - 1.0)*eps/ (1 + gamma * eps);

            g = c2 * v2 - 1.0;
            f = (gamma - 1.0) * rho * eps - p;

            peq = p - f/g;

        }while(abs(peq - p) >= tol);

        v = S/(tau + D + peq);

        W = 1./sqrt(1 - v * v);

        lorentz_gamma[ii] = W;

        prims.rho[ii] = D/W;
        prims.v  [ii] = v;
        prims.p  [ii] = p;
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
Eigenvals SRHD::calc_eigenvals(const Primitive &prims_l, const Primitive &prims_r){
    
    // Initialize your important variables
    double v_r, v_l, p_r, p_l, cs_r, cs_l; 
    double rho_l, rho_r, h_l ,h_r, aL, aR;
    double sL, sR, minlam_l, minlam_r, pluslam_l, pluslam_r;
    double vbar, cbar;
    Eigenvals lambda;
    
    // Compute L/R Sound Speeds
    rho_l = prims_l.rho;
    p_l   = prims_l.p;
    v_l   = prims_l.v;
    h_l   = 1. + gamma * p_l/(rho_l * (gamma - 1.));
    cs_l  = sqrt(gamma * p_l /(rho_l * h_l));

    rho_r = prims_r.rho;
    p_r   = prims_r.p;
    v_r   = prims_r.v;
    h_r   = 1. + gamma * p_r/(rho_r * (gamma - 1.));
    cs_r  = sqrt(gamma * p_r /(rho_r * h_r));

    // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
    // vbar = 0.5 * (v_l + v_r);
    // cbar = 0.5 * (cs_r + cs_l);
    // double br = (vbar + cbar)/(1 + vbar*cbar);
    // double bl = (vbar - cbar)/(1 - vbar*cbar);

    // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
    sL          = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
    sR          = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
    minlam_l    = (v_l - sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    minlam_r    = (v_r - sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);
    pluslam_l   = (v_l + sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    pluslam_r   = (v_r + sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);

    lambda.aL = (minlam_l < minlam_r)   ? minlam_l : minlam_r;
    lambda.aR = (pluslam_l > pluslam_r) ? pluslam_l : pluslam_r;

    // lambda.aL = min(bl, (v_l - cs_l)/(1 - v_l*cs_l));
    // lambda.aR = max(br, (v_r + cs_r)/(1 + v_l*cs_l));

    return lambda;
};

// Adapt the CFL conditonal timestep
double SRHD::adapt_dt(PrimitiveArray &prims){

    double r_left, r_right, left_cell, right_cell, dr, cs;
    double min_dt, cfl_dt;
    double h, rho, p, v, vPLus, vMinus;

    min_dt = 0;

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < pgrid_size; ii++){

        left_cell  = (ii - 1 < 0 ) ? r[ii] : r[ii - 1];
        right_cell = (ii + 1 > pgrid_size -1) ? r[ii] : r[ii + 1];


        // Check if using linearly-spaced grid or logspace
        r_right = (linspace) ?  0.5*(right_cell + r[ii]) : sqrt(right_cell * r[ii]);
        r_left  = (linspace) ?  0.5*(left_cell  + r[ii]) : sqrt(left_cell  * r[ii]);
        

        dr = r_right - r_left;
        rho = prims.rho[ii + idx_shift];
        p   = prims.p  [ii + idx_shift];
        v   = prims.v  [ii + idx_shift];

        h   = 1. + gamma * p/(rho * (gamma - 1.));
        cs  = sqrt(gamma * p / (rho * h));

        vPLus  = (v + cs)/(1 + v*cs);
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
//              STATE ARRAY CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state array for computation. Used for Higher Order Reconstruction
Conserved SRHD::calc_state(double rho, double v, double pressure){

    Conserved state;
    double W, h;

    h = 1. + gamma * pressure/ (rho * (gamma - 1.)); 
    W = 1./sqrt(1 - v * v);

    state.D   = rho*W;
    state.S   = rho*h*W*W*v;
    state.tau = rho*h*W*W - pressure - rho*W;

    return state;
};

Conserved SRHD::calc_hll_state(
                                const Conserved &left_state,
                                const Conserved &right_state,
                                const Flux &left_flux,
                                const Flux &right_flux,
                                const Primitive &left_prims,
                                const Primitive &right_prims)
{
    double aL, aR;
    Conserved hll_states;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    hll_states.D = ( aR*right_state.D - aL*left_state.D 
                        - right_flux.D + left_flux.D)/(aR - aL);

    hll_states.S = ( aR*right_state.S - aL*left_state.S
                        - right_flux.S + left_flux.S)/(aR - aL);

    hll_states.tau = ( aR*right_state.tau - aL*left_state.tau 
                        - right_flux.tau + left_flux.tau)/(aR - aL);


    return hll_states;

}

Conserved SRHD::calc_intermed_state(const Primitive &prims,
                                    const Conserved &state,
                                    const double a,
                                    const double aStar,
                                    const double pStar)
{
    double pressure, v, S, D, tau, E, Estar;
    double DStar, Sstar, tauStar;
    Eigenvals lambda; 
    Conserved star_state;

    pressure = prims.p;
    v = prims.v;

    D   = state.D;
    S   = state.S;
    tau = state.tau;
    E   = tau + D;

    DStar   = ( (a - v)/(a - aStar))*D;
    Sstar  = (1./(a-aStar))*(S *(a - v) - pressure + pStar);
    Estar   = (1./(a-aStar))*(E *(a - v) + pStar*aStar - pressure*v);
    tauStar = Estar - DStar;

    star_state.D   = DStar;
    star_state.S   = Sstar;
    star_state.tau = tauStar;
    
    return star_state;
}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
Flux SRHD::calc_flux(double rho, double v, double pressure){
    
    Flux flux;

    // The Flux components
    double mom, energy_dens, zeta, D, S, tau, h, W;

    W   = 1./sqrt(1 - v * v);
    h   = 1. + gamma*pressure/(rho * (gamma - 1.));
    D   = rho*W;
    S   = rho*h*W*W*v;
    tau = rho*h*W*W - pressure - W*rho;
    
    mom = D*v;
    energy_dens = S*v + pressure;
    zeta = (tau + pressure)*v;

    flux.D   = mom;
    flux.S   = energy_dens;
    flux.tau = zeta;

    return flux;
};

Flux SRHD::calc_hll_flux(const Primitive &left_prims, 
                         const Primitive &right_prims,
                         const Conserved &left_state,
                         const Conserved &right_state,
                         const Flux &left_flux,
                         const Flux &right_flux)
{
    Flux hll_flux;
    double aLm, aRp;  
    
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    // Grab the necessary wave speeds
    double aR = lambda.aR;
    double aL = lambda.aL;

    aLm = (aL < 0.0 ) ? aL : 0.0;
    aRp = (aR > 0.0)  ? aR : 0.0;

    // Compute the HLL Flux component-wise
    hll_flux.D   = ( ( aRp*left_flux.D - aLm*right_flux.D
                            + aLm*aRp*(right_state.D - left_state.D ) )  /
                            (aRp - aLm) );

    hll_flux.S   = ( aRp*left_flux.S - aLm*right_flux.S
                            + aLm*aRp*(right_state.S - left_state.S ) )  /
                            (aRp - aLm);

    hll_flux.tau = ( aRp*left_flux.tau - aLm*right_flux.tau
                            + aLm*aRp*(right_state.tau - left_state.tau ) )  /
                            (aRp - aLm);


    return hll_flux;
};

Flux SRHD::calc_hllc_flux(const Conserved &left_state,
                            const Conserved &right_state,
                            const Flux &left_flux,
                            const Flux &right_flux,
                            const Primitive &left_prims,
                            const Primitive &right_prims){

    Flux interflux_left;
    Flux interflux_right;
    Flux hll_flux;

    Conserved interstate_left;
    Conserved interstate_right;
    Conserved hll_state;
    double aL, aR, aStar, pStar;  
    
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    hll_flux = calc_hll_flux(left_prims, right_prims, left_state, right_state, left_flux,
                                    right_flux);

    hll_state = calc_hll_state(left_state, right_state, left_flux,
                                     right_flux, left_prims, right_prims);

    double e  = hll_state.tau + hll_state.D;    
    double s  = hll_state.S;
    double fs = hll_flux.S;
    double fe = hll_flux.tau + hll_flux.D;
    
    aStar = calc_intermed_wave(e, s, fs, fe);

    pStar = - fe * aStar + fs;

    interstate_left = calc_intermed_state(left_prims, left_state,
                                                aL, aStar, pStar);

    interstate_right = calc_intermed_state(right_prims, right_state,
                                                aR, aStar, pStar);
    

    // Compute the intermediate left flux
    interflux_left.D    = left_flux.D   + aL*(interstate_left.D   - left_state.D);
    interflux_left.S    = left_flux.S   + aL*(interstate_left.S   - left_state.S);
    interflux_left.tau  = left_flux.tau + aL*(interstate_left.tau - left_state.tau);

    // Compute the intermediate right flux
    interflux_right.D   = right_flux.D   + aR*(interstate_right.D   - right_state.D);
    interflux_right.S   = right_flux.S   + aR*(interstate_right.S   - right_state.S);
    interflux_right.tau = right_flux.tau + aR*(interstate_right.tau - right_state.tau);

    
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

ConservedArray SRHD::u_dot1D(ConservedArray &u_state){

    int i_start, i_bound, coordinate;
    string default_coordinates = "cartesian";
    double left_cell, right_cell;

    Conserved u_l, u_r;
    Flux f_l, f_r, f1, f2; 
    Primitive prims_l, prims_r;
    
    
    if (first_order){

        ConservedArray L;
        L.D.resize(pgrid_size);
        L.S.resize(pgrid_size);
        L.tau.resize(pgrid_size);


        double dx = (r[pgrid_size - 1] - r[0])/pgrid_size;
        if (periodic){
            i_start = 0;
            i_bound = Nx;
        } else{
            i_start = 1;
            i_bound = Nx - 1;
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
                    u_l.D = u_state.D[ii];
                    u_l.S = u_state.S[ii];
                    u_l.D = u_state.tau[ii];

                    u_r.D = roll(u_state.D, ii + 1);
                    u_r.S = roll(u_state.S, ii + 1);
                    u_r.tau = roll(u_state.tau, ii + 1);

                } else {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l.D   = u_state.D[ii];
                    u_l.S   = u_state.S[ii];
                    u_l.tau = u_state.tau[ii];

                    u_r.D   = u_state.D[ii + 1];
                    u_r.S   = u_state.S[ii + 1];
                    u_r.tau = u_state.tau[ii + 1];

                }

                prims_l.rho = prims.rho[ii];
                prims_l.v   = prims.v[ii];
                prims_l.p   = prims.p[ii];

                prims_r.rho = prims.rho[ii + 1];
                prims_r.v   = prims.v[ii + 1];
                prims_r.p   = prims.p[ii + 1];

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i+1/2 interface
                if (hllc){
                    f1 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l.D = roll(u_state.D, ii - 1);
                    u_l.S = roll(u_state.S, ii - 1);
                    u_l.tau = roll(u_state.tau, ii - 1);
                    
                    u_r.D   = u_state.D[ii];
                    u_r.S   = u_state.S[ii];
                    u_r.tau = u_state.tau[ii];

                } else {
                    u_l.D = u_state.D[ii - 1];
                    u_l.S = u_state.S[ii - 1];
                    u_l.tau = u_state.tau[ii - 1];
                    
                    u_r.D = u_state.D[ii];
                    u_r.S = u_state.S[ii];
                    u_r.tau = u_state.tau[ii];

                }

                prims_l.rho = prims.rho[ii - 1];
                prims_l.v   = prims.v[ii - 1];
                prims_l.p   = prims.p[ii - 1];

                prims_r.rho = prims.rho[ii];
                prims_r.v   = prims.v[ii];
                prims_r.p   = prims.p[ii];

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i-1/2 interface
                if (hllc){
                    f2 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }


                L.D[coordinate]   = - (f1.D - f2.D)/dx     ;//+ sourceD[coordinate];
                L.S[coordinate]   = - (f1.S - f2.S)/dx     ;//+ sourceS[coordinate];
                L.tau[coordinate] = - (f1.tau - f2.tau)/dx ;//+ source0[coordinate];

        }
            
        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;

            double dr = 0; 

            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l.D   = u_state.D[ii];
                    u_l.S   = u_state.S[ii];
                    u_l.tau = u_state.tau[ii];

                    u_r.D   = roll(u_state.D, ii + 1);
                    u_r.S   = roll(u_state.S, ii + 1);
                    u_r.tau = roll(u_state.tau, ii + 1);

                } else {
                    // Shift the index for C++ [0] indexing
                    coordinate = ii - 1;

                    u_l.D = u_state.D[ii];
                    u_l.S = u_state.S[ii];
                    u_l.tau = u_state.tau[ii];

                    

                    u_r.D = u_state.D[ii + 1];
                    u_r.S = u_state.S[ii + 1];
                    u_r.tau = u_state.tau[ii + 1];
                }

                prims_l.rho = prims.rho[ii];
                prims_l.v   = prims.v[ii];
                prims_l.p   = prims.p[ii];

                prims_r.rho = prims.rho[ii + 1];
                prims_r.v   = prims.v[ii + 1];
                prims_r.p   = prims.p[ii + 1];

                f_l = calc_flux(prims_l.rho, prims_l.p, prims_l.v);
                f_r = calc_flux(prims_r.rho, prims_r.p, prims_r.v);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);

                // Get the central pressure
                pc = prims_l.p;

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l.D   = roll(u_state.D, ii - 1);
                    u_l.S   = roll(u_state.S, ii - 1);
                    u_l.tau = roll(u_state.tau, ii - 1);
                    
                    u_r.D   = u_state.D[ii];
                    u_r.S   = u_state.S[ii];
                    u_r.tau = u_state.tau[ii];

                } else {
                    u_l.D   = u_state.D[ii - 1];
                    u_l.S   = u_state.S[ii - 1];
                    u_l.tau = u_state.tau[ii - 1];
                    
                    u_r.D   = u_state.D[ii];
                    u_r.S   = u_state.S[ii];
                    u_r.tau = u_state.tau[ii];

                }

                prims_l.rho = prims.rho[ii - 1];
                prims_l.v   = prims.v  [ii - 1];
                prims_l.p   = prims.p  [ii - 1];

                prims_r.rho = prims.rho[ii];
                prims_r.v   = prims.v  [ii];
                prims_r.p   = prims.p  [ii];

                f_l = calc_flux(prims_l.rho, prims_l.p, prims_l.v);
                f_r = calc_flux(prims_r.rho, prims_r.p, prims_r.v);
                
                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);

                // Outflow the left/right boundaries
                left_cell = (coordinate - 1 < 0) ? r[coordinate] : r[coordinate - 1];
                right_cell = (coordinate == pgrid_size - 1) ? r[coordinate] : r[coordinate + 1];

                r_right = (linspace) ? 0.5*(right_cell + r[coordinate]) : sqrt(right_cell * r[coordinate]);
                r_left  = (linspace) ? 0.5*(left_cell  + r[coordinate]) : sqrt(left_cell  * r[coordinate]);

                dr = r_right - r_left;
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                L.D  [coordinate] = - (r_right*r_right*f1.D   - r_left*r_left*f2.D   )/(volAvg*volAvg*dr) + sourceD[coordinate];
                L.S  [coordinate] = - (r_right*r_right*f1.S   - r_left*r_left*f2.S   )/(volAvg*volAvg*dr) + 2*pc/volAvg + sourceS[coordinate];
                L.tau[coordinate] = - (r_right*r_right*f1.tau - r_left*r_left*f2.tau )/(volAvg*volAvg*dr) + source0[coordinate];
                
            }
            
            
        }

        return L;
    } else {
        double dx = (r[pgrid_size - 1] - r[0])/pgrid_size;

        Primitive left_most, left_mid, center;
        Primitive right_mid, right_most;
        ConservedArray L;
        L.D.resize(pgrid_size);
        L.S.resize(pgrid_size);
        L.tau.resize(pgrid_size);

        // The periodic BC doesn't require ghost cells. Shift the index
        // to the beginning since all of he.
        if (periodic){ 
            i_start = 0;
            i_bound = Nx;
        } else {
            i_start = 2;
            i_bound = Nx - 2;
        }

        if (coord_system == default_coordinates){
            //==============================================
            //                  CARTESIAN
            //==============================================
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    coordinate = ii;
                    left_most.rho = roll(prims.rho, ii - 2);
                    left_mid.rho = roll(prims.rho, ii - 1);
                    center.rho = prims.rho[ii];
                    right_mid.rho = roll(prims.rho, ii + 1);
                    right_most.rho = roll(prims.rho, ii + 2);

                    left_most.v = roll(prims.v, ii - 2);
                    left_mid.v = roll(prims.v, ii - 1);
                    center.v = prims.v[ii];
                    right_mid.v = roll(prims.v, ii + 1);
                    right_most.v = roll(prims.v, ii + 2);

                    left_most.p = roll(prims.p, ii - 2);
                    left_mid.p = roll(prims.p, ii - 1);
                    center.p = prims.p[ii];
                    right_mid.p = roll(prims.p, ii + 1);
                    right_most.p = roll(prims.p, ii + 2);
                    

                } else {
                    coordinate = ii - 2;
                    left_most.rho = prims.rho[ii - 2];
                    left_mid.rho = prims.rho[ii - 1];
                    center.rho = prims.rho[ii];
                    right_mid.rho = prims.rho[ii + 1];
                    right_most.rho = prims.rho[ii + 2];

                    left_most.v = prims.v[ii - 2];
                    left_mid.v = prims.v[ii - 1];
                    center.v = prims.v[ii];
                    right_mid.v = prims.v[ii + 1];
                    right_most.v = prims.v[ii + 2];

                    left_most.p = prims.p[ii - 2];
                    left_mid.p = prims.p[ii - 1];
                    center.p = prims.p[ii];
                    right_mid.p = prims.p[ii + 1];
                    right_most.p = prims.p[ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - left_mid.rho),
                                                0.5*(right_mid.rho - left_mid.rho),
                                                theta*(right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5*minmod(theta*(center.v - left_mid.v),
                                                 0.5*(right_mid.v - left_mid.v),
                                                 theta*(right_mid.v - center.v));

                prims_l.p = center.p + 0.5*minmod(theta*(center.p - left_mid.p),
                                                0.5*(right_mid.p - left_mid.p),
                                                theta*(right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho - 0.5*minmod(theta*(right_mid.rho - center.rho),
                                                 0.5*(right_most.rho - center.rho),
                                                 theta*(right_most.rho - right_mid.rho));

                prims_r.v = right_mid.v - 0.5*minmod(theta*(right_mid.v - center.v),
                                                  0.5*(right_most.v - center.v),
                                                  theta*(right_most.v - right_mid.v));

                prims_r.p = right_mid.p - 0.5*minmod(theta*(right_mid.p - center.p),
                                                0.5*(right_most.p - center.p),
                                                theta*(right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (hllc){
                    f1 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho = left_mid.rho + 0.5*minmod(theta*(left_mid.rho - left_most.rho),
                                                       0.5*(center.rho -left_most.rho),
                                                       theta*(center.rho - left_mid.rho));

                prims_l.v = left_mid.v + 0.5*minmod(theta*(left_mid.v - left_most.v),
                                                      0.5*(center.v -left_most.v),
                                                      theta*(center.v - left_mid.v));
                
                prims_l.p = left_mid.p + 0.5*minmod(theta*(left_mid.p - left_most.p),
                                                      0.5*(center.p -left_most.p),
                                                      theta*(center.p - left_mid.p));


                    
                prims_r.rho = center.rho - 0.5*minmod(theta*(center.rho - left_mid.rho),
                                                0.5*(right_mid.rho - left_mid.rho),
                                                theta*(right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5*minmod(theta*(center.v - left_mid.v),
                                                0.5*(right_mid.v - left_mid.v),
                                                theta*(right_mid.v - center.v));

                prims_r.p = center.p - 0.5*minmod(theta*(center.p - left_mid.p),
                                                 0.5*(right_mid.p - left_mid.p),
                                                 theta*(right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (hllc){
                    f2 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                
                
                L.D   [coordinate] = - (f1.D - f2.D)/dx      ; //+ sourceD[coordinate];
                L.S   [coordinate] = - (f1.S - f2.S)/dx      ; //+ sourceS[coordinate];
                L.tau [coordinate] = - (f1.tau - f2.tau)/dx  ; //+ source0[coordinate];
            }            
                                                                                                                         
            

        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log10(r[pgrid_size - 1]) - log10(r[0]))/pgrid_size;

            double dr = 0;
            for (int ii=i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most.rho = roll(prims.rho, ii - 2);
                    left_mid.rho = roll(prims.rho, ii - 1);
                    center.rho = prims.rho[ii];
                    right_mid.rho = roll(prims.rho, ii + 1);
                    right_most.rho = roll(prims.rho, ii + 2);

                    left_most.v = roll(prims.v, ii - 2);
                    left_mid.v = roll(prims.v, ii - 1);
                    center.v = prims.v[ii];
                    right_mid.v = roll(prims.v, ii + 1);
                    right_most.v = roll(prims.v, ii + 2);

                    left_most.p = roll(prims.p, ii - 2);
                    left_mid.p = roll(prims.p, ii - 1);
                    center.p = prims.p[ii];
                    right_mid.p = roll(prims.p, ii + 1);
                    right_most.p = roll(prims.p, ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    coordinate = ii - 2;
                    left_most.rho = prims.rho[ii - 2];
                    left_mid.rho = prims.rho[ii - 1];
                    center.rho = prims.rho[ii];
                    right_mid.rho = prims.rho[ii + 1];
                    right_most.rho = prims.rho[ii + 2];

                    left_most.v = prims.v[ii - 2];
                    left_mid.v = prims.v[ii - 1];
                    center.v = prims.v[ii];
                    right_mid.v = prims.v[ii + 1];
                    right_most.v = prims.v[ii + 2];

                    left_most.p = prims.p[ii - 2];
                    left_mid.p = prims.p[ii - 1];
                    center.p = prims.p[ii];
                    right_mid.p = prims.p[ii + 1];
                    right_most.p = prims.p[ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    theta*(right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5*minmod(theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    theta*(right_mid.v - center.v));

                prims_l.p = center.p + 0.5*minmod(theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    theta*(right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho - 0.5*minmod(theta*(right_mid.rho - center.rho),
                                                    0.5*(right_most.rho - center.rho),
                                                    theta*(right_most.rho - right_mid.rho));

                prims_r.v = right_mid.v - 0.5*minmod(theta*(right_mid.v - center.v),
                                                    0.5*(right_most.v - center.v),
                                                    theta*(right_most.v - right_mid.v));

                prims_r.p = right_mid.p - 0.5*minmod(theta*(right_mid.p - center.p),
                                                    0.5*(right_most.p - center.p),
                                                    theta*(right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                f1 = calc_hll_flux(prims_l, prims_r,  u_l, u_r, f_l, f_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho = left_mid.rho + 0.5 *minmod(theta*(left_mid.rho - left_most.rho),
                                                        0.5*(center.rho -left_most.rho),
                                                        theta*(center.rho - left_mid.rho));

                prims_l.v = left_mid.v + 0.5 *minmod(theta*(left_mid.v - left_most.v),
                                                        0.5*(center.v -left_most.v),
                                                        theta*(center.v - left_mid.v));
                
                prims_l.p = left_mid.p + 0.5 *minmod(theta*(left_mid.p - left_most.p),
                                                        0.5*(center.p -left_most.p),
                                                        theta*(center.p - left_mid.p));


                    
                prims_r.rho = center.rho - 0.5 *minmod(theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    theta*(right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5 *minmod(theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    theta*(right_mid.v - center.v));

                prims_r.p = center.p - 0.5 *minmod(theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    theta*(right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                f2 = calc_hll_flux(prims_l, prims_r,  u_l, u_r, f_l, f_r);

                //Get Central Pressure
                pc = center.p;

                // Outflow the left/right boundaries
                left_cell  = (coordinate - 1 < 0) ? r[coordinate] : r[coordinate - 1];
                right_cell = (coordinate == pgrid_size - 1) ? r[coordinate] : r[coordinate + 1];

                r_right = (linspace) ? 0.5*(right_cell + r[coordinate]) : sqrt(right_cell * r[coordinate]);
                r_left  = (linspace) ? 0.5*(left_cell  + r[coordinate]) : sqrt(left_cell  * r[coordinate]);
                
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );
                dr = r_right - r_left;


                L.D  [coordinate] = - (r_right*r_right*f1.D - r_left*r_left*f2.D )/(volAvg*volAvg*dr) + sourceD[coordinate];
                L.S  [coordinate] = - (r_right*r_right*f1.S - r_left*r_left*f2.S )/(volAvg*volAvg*dr) + 2*pc/volAvg + sourceS[coordinate];
                L.tau[coordinate] = - (r_right*r_right*f1.tau - r_left*r_left*f2.tau )/(volAvg*volAvg*dr) + source0[coordinate];

            }
        
        }

        return L; 
    }
    
};


 vector<vector<double> > SRHD::simulate1D(vector<double> &lorentz_gamma, vector<vector<double> > &sources,
                                            float tend = 0.1, float dt = 1.e-4, double theta=1.5,
                                            bool first_order = true, bool periodic = false, bool linspace = true,
                                            bool hllc = false){

    
    this->periodic = periodic;
    this->first_order = first_order;
    this->theta = theta;
    this->linspace = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this->sourceD = sources[0];
    this->sourceS = sources[1];
    this->source0 = sources[2];
    this->hllc    = hllc;
    
    // Define the swap vector for the integrated state
    this->Nx = lorentz_gamma.size();

    

    if (periodic){
        this->idx_shift = 0;
    } else {
        if (first_order){
            this->idx_shift = 1;
            this->pgrid_size = Nx - 2;
        }
        else {
            this->idx_shift = 2;
            this->pgrid_size = Nx - 4;
        }
    }
    int i_real;
    n = 0;
    ConservedArray u_p, u, u1, u2, udot;
    float t = 0;

    // Copy the state array into real & profile variables
    u.D   = state[0];
    u.S   = state[1];
    u.tau = state[2];
    u_p = u;

    prims = cons2prim1D(u, lorentz_gamma);
    pressure_guess = prims.p;
    n++;

    if (first_order){
        while (t < tend){
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            if (t == 0){
                config_ghosts1D(u, Nx);
            }

            // Compute the L(u).
            udot = u_dot1D(u);

            for (int ii = 0; ii < pgrid_size; ii++){
                i_real = ii + idx_shift;
                u_p.D  [i_real]   = u.D  [i_real]   + dt*udot.D[ii];
                u_p.S  [i_real]   = u.S  [i_real]   + dt*udot.S[ii];
                u_p.tau[i_real]   = u.tau[i_real] + dt*udot.tau[ii];

            }


            // Readjust the ghost cells at i-1,i+1 if not periodic
            if (periodic == false){
                config_ghosts1D(u_p, Nx);
            }

            prims = cons2prim1D(u_p, lorentz_gamma);
            pressure_guess = prims.p;

            

            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(prims);
            }
            if (isnan(dt)){
                break;
            }

            // Swap the arrays
            u.D.swap(u_p.D);
            u.S.swap(u_p.S);
            u.tau.swap(u_p.tau);
            
            
            t += dt;

            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            cout << fixed << setprecision(3) << scientific;
            cout << "\r" << "dt: " << setw(5) << dt 
            << "\t" << "t: " << setw(5) << t 
            << "\t" << "Zones per sec: " << Nx/time_span.count()
            << flush;
            

        }   

    } else {

        u1 = u;
        u2 = u;
        u_p = u;
        while (t < tend){
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            // Compute the REAL udot array, purging the ghost cells.
            if (t == 0){
                config_ghosts1D(u, Nx, first_order);
            }

            udot = u_dot1D(u);

            for (int ii = 0; ii < pgrid_size; ii++){
                i_real = ii + idx_shift;
                u1.D  [i_real] = u.D  [i_real] + dt*udot.D[ii];
                u1.S  [i_real] = u.S  [i_real] + dt*udot.S[ii];
                u1.tau[i_real] = u.tau[i_real] + dt*udot.tau[ii];

            }
            
            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            if (periodic == false){
                config_ghosts1D(u1, Nx, false);
            }

            prims = cons2prim1D(u1, lorentz_gamma);
            pressure_guess = prims.p;
            
            // udot = u_dot1D(u1);

            for (int ii = 0; ii < pgrid_size; ii++){
                i_real = ii + idx_shift;
                u2.D   [i_real]  = 0.5*u.D  [i_real] + 0.5*u1.D  [i_real] + 0.5 * dt * udot.D  [ii];
                u2.S   [i_real]  = 0.5*u.S  [i_real] + 0.5*u1.S  [i_real] + 0.5 * dt * udot.S  [ii];
                u2.tau [i_real]  = 0.5*u.tau[i_real] + 0.5*u1.tau[i_real] + 0.5 * dt * udot.tau[ii];

            }

            prims = cons2prim1D(u2, lorentz_gamma);
            pressure_guess = prims.p;

            
            if (periodic == false){
                config_ghosts1D(u2, Nx, false);
            }

            // Adjust the timestep 
            if (t > 0){
                dt = adapt_dt(prims);
            }
            
            // Swap the arrays
            u.D.swap(u2.D);
            u.S.swap(u2.S);
            u.tau.swap(u2.tau);

            
            cout << "\r" << "dt: " << dt << " " << "t: " << t << flush;
            t += dt;

            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            cout << fixed << setprecision(3) << scientific;
            cout << "\r" << "dt: " << setw(5) << dt 
            << "\t" << "t: " << setw(5) << t 
            << "\t" << "Zones per sec: " << Nx/time_span.count()
            << flush;

            n++;


        }  

    }
    cout << "\n";
    prims = cons2prim1D(u, lorentz_gamma);
    vector<vector<double> > final_prims(3, vector<double>(Nx, 0)); 
    final_prims[0] = prims.rho;
    final_prims[1] = prims.v;
    final_prims[2] = prims.p;
    

    return final_prims;

 };
