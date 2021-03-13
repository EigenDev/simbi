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
using namespace hydro;
using namespace chrono;

// Default Constructor 
SRHD2D::SRHD2D () {}

// Overloaded Constructor
SRHD2D::SRHD2D(vector<vector<double>> state2D, int nx, int ny, double gamma, vector<double> x1, 
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
SRHD2D::~SRHD2D() {}

/* Define typedefs because I am lazy */
typedef vector<vector<double>> twoVec; 
typedef SRHD2D::PrimitiveData PrimitiveArray;
typedef SRHD2D::ConserveData ConserveArray;
typedef SRHD2D::Primitives Primitives;
typedef SRHD2D::Flux Flux;
typedef SRHD2D::Conserved Conserved;
typedef SRHD2D::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

PrimitiveArray SRHD2D::cons2prim2D(const ConserveArray &u_state2D, vector<double> &lorentz_gamma){
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

    // Define Newton-Raphson Vars
    double ps, pb, h, den;
    double v, c2, f, g, p, peq;       
    double Ws, W2, rhos, eps;

    int iter = 0;
    int maximum_iteration = 50;
    for (int jj=0; jj < NY; jj ++){
        for(int ii=0; ii< NX; ii ++){
            D   =  u_state2D.D  [ii + NX * jj];      // Relativistic Density
            S1  =  u_state2D.S1 [ii + NX * jj];      // X1-Momentum Denity
            S2  =  u_state2D.S2 [ii + NX * jj];      // X2-Momentum Density
            tau =  u_state2D.tau[ii + NX * jj];      // Energy Density
            W   =  lorentz_gamma[ii + NX * jj];

            S = sqrt(S1*S1 + S2*S2);

            peq = n != 0.0 ? pressure_guess[ii + NX * jj] : abs(S - D - tau);

            tol = D*1.e-12;

            //--------- Iteratively Solve for Pressure using Newton-Raphson
            // Note: The NR scheme as been modified based on: 
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930 
            ps = peq;
            p  = peq;
            iter = 0;
            do {
                pb = p;
                p  = peq;

                den     = tau + p + D;
                v2      = S * S / (den * den);
                Ws      = 1.0 / sqrt(1.0 - v2);
                rhos    = D / Ws; 
                
                eps     = (tau + D * (1. - Ws) + (1. - Ws*Ws) * p )/(D * Ws);
                f       = (gamma - 1.0) * rhos * eps - p;
                
                h   = 1.0 + eps + 0.5 * (ps + p) / rhos;
                c2  = gamma * 0.5 * (ps + p) / (rhos * h);
                g   = c2 * v2 - 1.0;

                peq = p - f/g;

                eps = (tau + D * (1. - Ws) + (1. - Ws*Ws) * 0.5 * (pb + ps) )/(D * Ws);
                h   = 1.0 + eps + 0.5 * (pb + ps) / rhos;
                c2  = gamma * 0.5 * (ps + pb) / (rhos * h);
                g   = c2 * v2 - 1.0;

                ps = p - f/g;
                

                iter++;
                

                
                if (iter > maximum_iteration){
                    cout << "\n";
                    cout << "p: " << p << endl;
                    cout << "S: " << S << endl;
                    cout << "tau: " << tau << endl;
                    cout << "D: " << D << endl;
                    cout << "et: " << den << endl;
                    cout << "Ws: " << Ws << endl;
                    cout << "v2: " << v2 << endl;
                    cout << "W: " << W << endl;
                    cout << "n: " << n << endl;
                    cout << "\n Cons2Prim Cannot Converge" << endl;
                    exit(EXIT_FAILURE);
                }
                

            } while(abs(peq - p) >= tol);

            v1 = S1/(tau + D + peq);

            v2 = S2/(tau + D + peq);

            Ws = 1.0/sqrt(1.0 - (v1*v1 + v2*v2));
            
            // Update the Gamma factor
            lorentz_gamma[ii + NX * jj] = Ws;

            prims.rho.emplace_back(D/Ws);
            prims.v1.emplace_back(v1);
            prims.v2.emplace_back(v2);
            prims.p.emplace_back(peq);
            
                
        }
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
Eigenvals SRHD2D::calc_Eigenvals( const Primitives &prims_l,
                                      const Primitives &prims_r,
                                      unsigned int nhat = 1)
{
    Eigenvals lambda;

    // Separate the left and right Primitives
    double rho_l = prims_l.rho;
    double p_l   = prims_l.p;
    double v1_l  = prims_l.v1;
    double v2_l  = prims_l.v2;
    double h_l   = 1. + gamma*p_l/(rho_l*(gamma - 1));

    double rho_r = prims_r.rho;
    double p_r   = prims_r.p;
    double v1_r  = prims_r.v1;
    double v2_r  = prims_r.v2;
    double h_r   = 1. + gamma*p_r/(rho_r*(gamma - 1));


    double cs_r = sqrt(gamma * p_r/(h_r*rho_r)); 
    double cs_l = sqrt(gamma * p_l/(h_l*rho_l)); 
 
    switch (nhat){
        case 1:{
            // Calc the wave speeds based on Mignone and Bodo (2005)
            double sL    = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
            double sR    = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
            double lamLm = (v1_l - sqrt(sL*(1 - v1_l*v1_l + sL)))/(1 + sL);
            double lamRm = (v1_r - sqrt(sR*(1 - v1_r*v1_r + sR)))/(1 + sR);
            double lamRp = (v1_l + sqrt(sL*(1 - v1_l*v1_l + sL)))/(1 + sL);
            double lamLp = (v1_r + sqrt(sR*(1 - v1_r*v1_r + sR)))/(1 + sR);

            lambda.aL = lamLm < lamRm ? lamLm : lamRm;
            lambda.aR = lamLp > lamRp ? lamLp : lamRp; 

            break;
        }
        case 2:
            // Calc the wave speeds based on Mignone and Bodo (2005)
            double sL    = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
            double sR    = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
            double lamLm = (v2_l - sqrt(sL*(1 - v2_l*v2_l + sL)))/(1 + sL);
            double lamRm = (v2_r - sqrt(sR*(1 - v2_r*v2_r + sR)))/(1 + sR);
            double lamRp = (v2_l + sqrt(sL*(1 - v2_l*v2_l + sL)))/(1 + sL);
            double lamLp = (v2_r + sqrt(sR*(1 - v2_r*v2_r + sR)))/(1 + sR);

            lambda.aL = lamLm < lamRm ? lamLm : lamRm;
            lambda.aR = lamLp > lamRp ? lamLp : lamRp; 

            break; 

    }
        
    return lambda;

    
    
};



//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------

Conserved SRHD2D::calc_stateSR2D(double rho, double vx,
                                     double vy, double pressure)
{
    Conserved state;

    double lorentz_gamma = 1./ sqrt(1 - (vx*vx + vy*vy));
    double h             = 1. + gamma*pressure/(rho*(gamma - 1.)); 

    state.D   = rho*lorentz_gamma; 
    state.S1  = rho*h*lorentz_gamma*lorentz_gamma*vx;
    state.S2  = rho*h*lorentz_gamma*lorentz_gamma*vy;
    state.tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;
    
    return state;

};

Conserved SRHD2D::calc_hll_state(
                                const Conserved     &left_state,
                                const Conserved     &right_state,
                                const Flux          &left_flux,
                                const Flux          &right_flux,
                                const Primitives    &left_prims,
                                const Primitives    &right_prims,
                                unsigned int nhat)
{
    Conserved hll_states;

    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    double aL = lambda.aL;
    double aR = lambda.aR;

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

Conserved SRHD2D::calc_intermed_statesSR2D( const Primitives &prims,
                                                const Conserved &state,
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
double SRHD2D::adapt_dt(const PrimitiveArray &prims){

    double r_left, r_right, left_cell, right_cell, lower_cell, upper_cell;
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
            left_cell  = (ii - 1 < 0 ) ? x1[ii] : x1[ii - 1];
            right_cell = (ii == xphysical_grid - 1) ? x1[ii] : x1[ii + 1];
            upper_cell = (jj - 1 < 0 ) ? x2[jj] : x2[jj - 1];
            lower_cell = (jj == yphysical_grid - 1) ? x2[jj] : x2[jj + 1];
            

            // Check if using linearly-spaced grid or logspace
            if (linspace){
                r_right = 0.5*(right_cell + x1[ii]);
                r_left  = 0.5*(x1[ii] + left_cell);

            } else {
                r_right = sqrt(right_cell * x1[ii]);
                r_left  = sqrt(left_cell  * x1[ii]);

            }

            x2_right = 0.5 * (lower_cell + x2[jj]);
            x2_left  = 0.5 * (upper_cell + x2[jj]);

            dx1      = r_right - r_left;
            dx2      = x2_right - x2_left;
            rho      = prims.rho[shift_i + NX * shift_j];
            v1       = prims.v1 [shift_i + NX * shift_j];
            v2       = prims.v2 [shift_i + NX * shift_j];
            pressure = prims.p  [shift_i + NX * shift_j];

            h    = 1. + gamma*pressure/(rho*(gamma - 1.));
            cs   = sqrt(gamma * pressure/(rho * h));

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

                cfl_dt = min( dx1/(max(abs(plus_v1), abs(minus_v1))), volAvg*dx2/(max(abs(plus_v2), abs(minus_v2))) );

            }

            
            if ((ii > 0) || (jj > 0) ){
                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
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
Flux SRHD2D::calc_Flux(double rho, double vx, 
                                double vy, double pressure, 
                                bool x_direction=true){
    
    // The Flux Tensor
    Flux flux;

     // The Flux components
    double mom1, mom2, energy_dens, zeta, convect_12;

    double lorentz_gamma = 1./sqrt(1. - (vx*vx + vy*vy) );

    double h   = 1. + gamma*pressure/(rho*(gamma - 1));
    double D   = rho*lorentz_gamma;
    double S1  = rho*lorentz_gamma*lorentz_gamma*h*vx;
    double S2  = rho*lorentz_gamma*lorentz_gamma*h*vy;
    double tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;

    
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


Flux SRHD2D::calc_hll_flux(
                        const Conserved &left_state,
                        const Conserved &right_state,
                        const Flux     &left_flux,
                        const Flux     &right_flux,
                        const Primitives   &left_prims,
                        const Primitives   &right_prims,
                        unsigned int nhat)
{
    Flux  hll_flux;
    
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    double aL = lambda.aL;
    double aR = lambda.aR;

    // Calculate plus/minus alphas
    double aLminus = aL < 0.0 ? aL : 0.0;
    double aRplus  = aR > 0.0 ? aR : 0.0;

    // Compute the HLL Flux component-wise
    hll_flux.D = ( aRplus*left_flux.D - aLminus*right_flux.D
                            + aRplus*aLminus*(right_state.D - left_state.D ) )  /
                            (aRplus - aLminus);

    hll_flux.S1 = ( aRplus*left_flux.S1 - aLminus*right_flux.S1
                            + aRplus*aLminus*(right_state.S1 - left_state.S1 ) )  /
                            (aRplus - aLminus);

    hll_flux.S2 = ( aRplus*left_flux.S2 - aLminus*right_flux.S2
                            + aRplus*aLminus*(right_state.S2 - left_state.S2) )  /
                            (aRplus - aLminus);

    hll_flux.tau = ( aRplus*left_flux.tau - aLminus*right_flux.tau
                            + aRplus*aLminus*(right_state.tau - left_state.tau) )  /
                            (aRplus - aLminus);

    return hll_flux;
};


Flux SRHD2D::calc_hllc_flux(
                                const Conserved &left_state,
                                const Conserved &right_state,
                                const Flux     &left_flux,
                                const Flux     &right_flux,
                                const Primitives   &left_prims,
                                const Primitives   &right_prims,
                                const unsigned int nhat = 1)
{
    Flux hllc_flux, hll_flux;
    Conserved hll_state;

    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    double aL = lambda.aL;
    double aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if (0.0 <= aL){
        return left_flux;
    } else if (0.0 >= aR){
        return right_flux;
    }

    double aLminus = aL < 0.0 ? aL : 0.0;
    double aRplus  = aR > 0.0 ? aR : 0.0;

    //-------------------Calculate the HLL Intermediate State
    hll_state.D = ( aR*right_state.D - aL*left_state.D 
                        - right_flux.D + left_flux.D)/(aR - aL);

    hll_state.S1 = ( aR*right_state.S1 - aL*left_state.S1 
                        - right_flux.S1 + left_flux.S1)/(aR - aL);

    hll_state.S2 = ( aR*right_state.S2 - aL*left_state.S2
                        - right_flux.S2 + left_flux.S2)/(aR - aL);

    hll_state.tau = ( aR*right_state.tau - aL*left_state.tau
                        - right_flux.tau + left_flux.tau)/(aR - aL);


    //------------------Calculate the RHLLE Flux---------------
    hll_flux.D = ( aRplus*left_flux.D - aLminus*right_flux.D
                            + aRplus*aLminus*(right_state.D - left_state.D ) )  /
                            (aRplus - aLminus);

    hll_flux.S1 = ( aRplus*left_flux.S1 - aLminus*right_flux.S1
                            + aRplus*aLminus*(right_state.S1 - left_state.S1 ) )  /
                            (aRplus - aLminus);

    hll_flux.S2 = ( aRplus*left_flux.S2 - aLminus*right_flux.S2
                            + aRplus*aLminus*(right_state.S2 - left_state.S2) )  /
                            (aRplus - aLminus);

    hll_flux.tau = ( aRplus*left_flux.tau - aLminus*right_flux.tau
                            + aRplus*aLminus*(right_state.tau - left_state.tau) )  /
                            (aRplus - aLminus);

    //------ Mignone & Bodo subtract off the rest mass density
    double e  = hll_state.tau + hll_state.D;
    double s  = hll_state.momentum(nhat);
    double fe = hll_flux.tau + hll_flux.D;
    double fs = hll_flux.momentum(nhat);

    //------Calculate the contact wave velocity and pressure
    double a = fe;
    double b = -(fs + e);
    double c = s;
    double quad = -0.5*(b + sgn(b)*sqrt(b*b - 4*a*c));
    double aStar = c/quad;
    double pStar = -fe * aStar + fs;

    

    if ( -aL <= (aStar - aL)){
        Flux interflux_left;
        Conserved interstate_left;
        double pressure = left_prims.p;
        double v1       = left_prims.v1;
        double v2       = left_prims.v2;

        double D   = left_state.D;
        double S1  = left_state.S1;
        double S2  = left_state.S2;
        double tau = left_state.tau;
        double E   = tau + D;

        //--------------Compute the L Star State----------
        switch (nhat) {
            case 1:{
                // Left Star State in x-direction of coordinate lattice
                double cofactor = 1./(aL - aStar);
                double Dstar    = cofactor * (aL - v1)*D;
                double S1star   = cofactor * (S1*(aL - v1) - pressure + pStar);
                double S2star   = cofactor * (aL - v1)*S2;
                double Estar    = cofactor * (E*(aL - v1) + pStar*aStar - pressure*v1);
                double tauStar  = Estar - Dstar;

                interstate_left.D   = Dstar;
                interstate_left.S1  = S1star;
                interstate_left.S2  = S2star;
                interstate_left.tau = tauStar;
                break;
            }
            
            case 2: 
                // Start States in y-direction in the coordinate lattice
                double cofactor = 1./(aL - aStar);
                double Dstar    = cofactor * (aL - v2) * D;
                double S1star   = cofactor * (aL - v2) * S1; 
                double S2star   = cofactor * (S2*(aL - v2) - pressure + pStar);
                double Estar    = cofactor * (E*(aL - v2) + pStar*aStar - pressure*v2);
                double tauStar  = Estar - Dstar;

                interstate_left.D   = Dstar;
                interstate_left.S1  = S1star;
                interstate_left.S2  = S2star;
                interstate_left.tau = tauStar;
                break;

        }

        //---------Compute the L Star Flux
        interflux_left.D    = left_flux.D   + aL*(interstate_left.D   - left_state.D   );
        interflux_left.S1   = left_flux.S1  + aL*(interstate_left.S1  - left_state.S1  );
        interflux_left.S2   = left_flux.S2  + aL*(interstate_left.S2  - left_state.S2  );
        interflux_left.tau  = left_flux.tau + aL*(interstate_left.tau - left_state.tau );

        return interflux_left;

    } else if ( -aStar <= (aR - aStar)){
        Flux      interflux_right;
        Conserved interstate_right;
        // Left Star State in x-direction of coordinate lattice
        double pressure = right_prims.p;
        double v1       = right_prims.v1;
        double v2       = right_prims.v2;

        double D   = right_state.D;
        double S1  = right_state.S1;
        double S2  = right_state.S2;
        double tau = right_state.tau;
        double E   = tau + D;

        /* Compute the L/R Star State */
        switch (nhat) {
            case 1: {
                double cofactor = 1./(aR - aStar);
                double Dstar    = cofactor * (aR - v1)*D;
                double S1star   = cofactor * (S1*(aR - v1) - pressure + pStar);
                double S2star   = cofactor * (aR - v1)*S2;
                double Estar    = cofactor * (E*(aR - v1) + pStar*aStar - pressure*v1);
                double tauStar  = Estar - Dstar;

                interstate_right.D   = Dstar;
                interstate_right.S1  = S1star;
                interstate_right.S2  = S2star;
                interstate_right.tau = tauStar;
                break;
            }
            
            case 2: 
                // Start States in y-direction in the coordinate lattice
                double cofactor = 1./(aR - aStar);
                double Dstar    = cofactor * (aR - v2) * D;
                double S1star   = cofactor * (aR - v2) * S1; 
                double S2star   = cofactor * (S2*(aR - v2) - pressure + pStar);
                double Estar    = cofactor * (E*(aR - v2) + pStar*aStar - pressure*v2);
                double tauStar  = Estar - Dstar;

                interstate_right.D   = Dstar;
                interstate_right.S1  = S1star;
                interstate_right.S2  = S2star;
                interstate_right.tau = tauStar;
                break;

        }
        // Compute the intermediate right flux
        interflux_right.D   = right_flux.D   + aR*(interstate_right.D   - right_state.D   );
        interflux_right.S1  = right_flux.S1  + aR*(interstate_right.S1  - right_state.S1  );
        interflux_right.S2  = right_flux.S2  + aR*(interstate_right.S2  - right_state.S2  );
        interflux_right.tau = right_flux.tau + aR*(interstate_right.tau - right_state.tau );

    
        return interflux_right;
    }
    
};



//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

ConserveArray SRHD2D::u_dot2D(const ConserveArray &u_state)
{

    int xcoordinate, ycoordinate;
    
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
    int i_start = idx_active; 
    int j_start = idx_active;
    int i_bound = x_bound;
    int j_bound = y_bound;
    
    switch (geometry[coord_system]){
        case simulation::CARTESIAN: {
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

            } else {
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

            }

            break;
            }
        case simulation::SPHERICAL:
            //==============================================================================================
            //                                  SPHERICAL 
            //==============================================================================================
            double right_cell, left_cell, lower_cell, upper_cell, ang_avg; 
            double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
            double theta_right, theta_left, ycoordinate, xcoordinate;
            double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;
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
                            lower_cell = x2[ycoordinate + 1];
                            upper_cell = x2[ycoordinate - 1];

                            // Outflow the left/right boundaries
                            if (xcoordinate - 1 < 0){
                                left_cell = x1[xcoordinate];

                            } else if (xcoordinate == xphysical_grid - 1){
                                right_cell = x1[xcoordinate];

                            }

                            if (ycoordinate - 1 < 0){
                                upper_cell = x2[ycoordinate];
                            }  else if(ycoordinate == yphysical_grid - 1){
                                lower_cell = x2[ycoordinate];
                            }

                            
                            r_right = 0.5*(right_cell + x1[xcoordinate]);
                            r_left  = 0.5*(x1[xcoordinate] + left_cell);

                            theta_right = 0.5*(lower_cell + x2[ycoordinate]);
                            theta_left = 0.5*(upper_cell + x2[ycoordinate]);

                    } else {

                        right_cell = x1[xcoordinate + 1];
                        left_cell  = x1[xcoordinate - 1];

                        lower_cell = x2[ycoordinate + 1];
                        upper_cell = x2[ycoordinate - 1];
                        
                        if (xcoordinate - 1 < 0){
                            left_cell = x1[xcoordinate];

                        } else if (xcoordinate == xphysical_grid - 1){
                            right_cell = x1[xcoordinate];
                        }

                        r_right = sqrt(right_cell * x1[xcoordinate]);
                        r_left  = sqrt(left_cell  * x1[xcoordinate]);

                        // Outflow the left/right boundaries
                        if (ycoordinate - 1 < 0){
                            upper_cell = x2[ycoordinate];

                        } else if(ycoordinate == yphysical_grid - 1){
                            lower_cell = x2[ycoordinate];
                        }

                        theta_right = 0.5 * (lower_cell + x2[ycoordinate]);
                        theta_left  = 0.5 * (upper_cell + x2[ycoordinate]);
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

            } else {

                for (int jj = j_start; jj < j_bound; jj++){
                    for (int ii = i_start; ii < i_bound; ii++){
                        if (!periodic){
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

                        } else {
                            xcoordinate = ii;
                            ycoordinate = jj;

                            // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                            /* TODO: Fix this */

                        }
                        
                        // Reconstructed left X Primitives vector at the i+1/2 interface
                        xprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - xleft_mid.rho),
                                                            0.5*(xright_mid.rho - xleft_mid.rho),
                                                            theta*(xright_mid.rho - center.rho));

                        
                        xprims_l.v1 = center.v1   + 0.5*minmod(theta*(center.v1 - xleft_mid.v1),
                                                            0.5*(xright_mid.v1 - xleft_mid.v1),
                                                            theta*(xright_mid.v1 - center.v1));

                        xprims_l.v2 = center.v2   + 0.5*minmod(theta*(center.v2 - xleft_mid.v2),
                                                            0.5*(xright_mid.v2 - xleft_mid.v2),
                                                            theta*(xright_mid.v2 - center.v2));

                        xprims_l.p = center.p     + 0.5*minmod(theta*(center.p - xleft_mid.p),
                                                            0.5*(xright_mid.p - xleft_mid.p),
                                                            theta*(xright_mid.p - center.p));

                        // Reconstructed right Primitives vector in x
                        xprims_r.rho = xright_mid.rho - 0.5*minmod(theta*(xright_mid.rho - center.rho),
                                                            0.5*(xright_most.rho - center.rho),
                                                            theta*(xright_most.rho - xright_mid.rho));

                        xprims_r.v1 = xright_mid.v1   - 0.5*minmod(theta*(xright_mid.v1 - center.v1),
                                                            0.5*(xright_most.v1 - center.v1),
                                                            theta*(xright_most.v1 - xright_mid.v1));

                        xprims_r.v2 = xright_mid.v2   - 0.5*minmod(theta*(xright_mid.v2 - center.v2),
                                                            0.5*(xright_most.v2 - center.v2),
                                                            theta*(xright_most.v2 - xright_mid.v2));

                        xprims_r.p = xright_mid.p     - 0.5*minmod(theta*(xright_mid.p - center.p),
                                                            0.5*(xright_most.p - center.p),
                                                            theta*(xright_most.p - xright_mid.p));

                        
                        // Reconstructed right Primitives vector in y-direction at j+1/2 interfce
                        yprims_l.rho = center.rho + 0.5*minmod(theta*(center.rho - yleft_mid.rho),
                                                            0.5*(yright_mid.rho - yleft_mid.rho),
                                                            theta*(yright_mid.rho - center.rho));

                        yprims_l.v1  = center.v1  + 0.5*minmod(theta*(center.v1 - yleft_mid.v1),
                                                            0.5*(yright_mid.v1 - yleft_mid.v1),
                                                            theta*(yright_mid.v1 - center.v1));

                        yprims_l.v2  = center.v2  + 0.5*minmod(theta*(center.v2 - yleft_mid.v2),
                                                            0.5*(yright_mid.v2 - yleft_mid.v2),
                                                            theta*(yright_mid.v2 - center.v2));

                        yprims_l.p   = center.p   + 0.5*minmod(theta*(center.p - yleft_mid.p),
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

                        yprims_r.p  = yright_mid.p - 0.5*minmod(theta*(yright_mid.p - center.p),
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
                        
                        right_cell = (xcoordinate == xphysical_grid - 1) ? x1[xcoordinate] : x1[xcoordinate + 1];
                        left_cell  = (xcoordinate - 1 < 0) ? x1[xcoordinate]: x1[xcoordinate - 1];

                        lower_cell = (ycoordinate == yphysical_grid - 1) ? x2[ycoordinate] : x2[ycoordinate + 1];
                        upper_cell = (ycoordinate - 1 < 0) ? x2[ycoordinate] : x2[ycoordinate - 1];

                        theta_right = 0.5 * (lower_cell + x2[ycoordinate]);
                        theta_left  = 0.5 * (upper_cell + x2[ycoordinate]);

                        r_right = (linspace) ? 0.5*(right_cell + x1[xcoordinate]) : sqrt(right_cell * x1[xcoordinate]); 
                        r_left  = (linspace) ? 0.5*(left_cell  + x1[xcoordinate]) : sqrt(left_cell  * x1[xcoordinate]); 

                        dr   = r_right - r_left;
                        rhoc = center.rho;
                        pc   = center.p;
                        uc   = center.v1;
                        vc   = center.v2;

                        ang_avg = 0.5 *(theta_right + theta_left); 

                        // Compute the surface areas
                        right_rsurface = r_right * r_right ;
                        left_rsurface  = r_left  * r_left  ;
                        upper_tsurface = sin(theta_right);
                        lower_tsurface = sin(theta_left);
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

            }

            break;
    }
    
    return L;
};


//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
twoVec SRHD2D::simulate2D(vector<double> lorentz_gamma, 
                                    const vector<vector<double> > sources,
                                    float tstart = 0.,
                                    float tend = 0.1, 
                                    double dt = 1.e-4,
                                    double theta = 1.5,
                                    double chkpt_interval = 0.1,
                                    string data_directory = "data/",
                                    bool first_order = true, 
                                    bool periodic = false,
                                    bool linspace=true, 
                                    bool hllc=false){

    
    int i_real, j_real;
    string tnow, tchunk, tstep;
    int total_zones = NX * NY;

    double round_place = 1/chkpt_interval;
    double t = tstart;
    double t_interval = t == 0 ? 
                        floor(tstart * round_place + 0.5)/round_place : 
                        floor(tstart * round_place + 0.5)/round_place + chkpt_interval;

    string filename;

    this->sources       = sources;
    this->first_order   = first_order;
    this->periodic      = periodic;
    this->hllc          = hllc;
    this->linspace      = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this-> theta        = theta;

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

    //--------Config the System Enums
    config_system();

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
    prims.p.resize(nzones);

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

    // Set the primitives from the initial conditions and set the initial pressure guesses
    prims = cons2prim2D(u, lorentz_gamma);
    n++;
    pressure_guess = prims.p;

    if (t == 0){
        config_ghosts2D(u, NX, NY, false);
    }

    if (first_order){
        while (t < tend){
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            udot = u_dot2D(u);

            for (int jj = 0; jj < yphysical_grid; jj ++){
                for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 1; j_real = jj + 1;

                        u_p.D  [i_real + NX * j_real]   = u.D  [i_real + NX * j_real] + dt * udot.D  [ii + xphysical_grid * jj]; 
                        u_p.S1 [i_real + NX * j_real]   = u.S1 [i_real + NX * j_real] + dt * udot.S1 [ii + xphysical_grid * jj]; 
                        u_p.S2 [i_real + NX * j_real]   = u.S2 [i_real + NX * j_real] + dt * udot.S2 [ii + xphysical_grid * jj]; 
                        u_p.tau[i_real + NX * j_real]   = u.tau[i_real + NX * j_real] + dt * udot.tau[ii + xphysical_grid * jj]; 
                        
            
                }
            }

            config_ghosts2D(u_p, NX, NY, true);
            prims = cons2prim2D(u_p, lorentz_gamma);
            lorentz_gamma = calc_lorentz_gamma(prims.v1, prims.v2, NX, NY);

            u.D.swap(u_p.D  );
            u.S1.swap(u_p.S1);
            u.S2.swap(u_p.S2);
            u.tau.swap(u_p.tau);
            
            t += dt;
            dt = adapt_dt(prims);
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

                udot = u_dot2D(u);

                for (int jj = 0; jj < yphysical_grid; jj ++){
                    j_real = jj + 2;
                    for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 2;
                        u1.D  [i_real + NX * j_real]   = u.D  [i_real + NX * j_real]  + dt * udot.D  [ii + xphysical_grid * jj]; 
                        u1.S1 [i_real + NX * j_real]   = u.S1 [i_real + NX * j_real]  + dt * udot.S1 [ii + xphysical_grid * jj]; 
                        u1.S2 [i_real + NX * j_real]   = u.S2 [i_real + NX * j_real]  + dt * udot.S2 [ii + xphysical_grid * jj]; 
                        u1.tau[i_real + NX * j_real]   = u.tau[i_real + NX * j_real]  + dt * udot.tau[ii + xphysical_grid * jj]; 
                    }
                }
                
                
                config_ghosts2D(u1, NX, NY, false);
                prims = cons2prim2D(u1, lorentz_gamma);

                udot = u_dot2D(u1);

                for (int jj = 0; jj < yphysical_grid; jj ++){
                    j_real = jj + 2;
                    for (int ii = 0; ii < xphysical_grid; ii ++){
                        i_real = ii + 2;
                        u2.D  [i_real + NX * j_real] = 0.5 * u.D  [i_real + NX * j_real] + 0.5 * u1.D  [i_real + NX * j_real] + 0.5 * dt*udot.D  [ii + xphysical_grid * jj];
                        u2.S1 [i_real + NX * j_real] = 0.5 * u.S1 [i_real + NX * j_real] + 0.5 * u1.S1 [i_real + NX * j_real] + 0.5 * dt*udot.S1 [ii + xphysical_grid * jj];
                        u2.S2 [i_real + NX * j_real] = 0.5 * u.S2 [i_real + NX * j_real] + 0.5 * u1.S2 [i_real + NX * j_real] + 0.5 * dt*udot.S2 [ii + xphysical_grid * jj];
                        u2.tau[i_real + NX * j_real] = 0.5 * u.tau[i_real + NX * j_real] + 0.5 * u1.tau[i_real + NX * j_real] + 0.5 * dt*udot.tau[ii + xphysical_grid * jj];
                
                    }
                }
            
                config_ghosts2D(u2, NX, NY, false);
                prims = cons2prim2D(u2, lorentz_gamma);
                

                if (isnan(dt)){
                    break;
                }
                
                u.D.swap(u2.D  );
                u.S1.swap(u2.S1);
                u.S2.swap(u2.S2);
                u.tau.swap(u2.tau);
                
                t += dt;
                dt = adapt_dt(prims);
                /* Compute the loop execution time */
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

                cout << fixed << setprecision(3) << scientific;
                cout << "\r" << "dt: " << setw(5) << dt 
                << "\t" << "t: " << setw(5) << t 
                << "\t" << "Zones per sec: " << total_zones/time_span.count()
                << flush;

                n++;
                pressure_guess = prims.p;

                /* Write to a File every tenth of a second */
                if (t >= t_interval){
                    toWritePrim(&prims, &prods, 2);
                    tnow  = create_step_str(t_interval, tchunk);
                    filename = string_format("%d.chkpt." + tnow + ".h5", NY);
                    setup.t  = t;
                    setup.dt = dt;
                    write_hdf5(data_directory, filename, prods, setup, 2);
                    t_interval += chkpt_interval;
 
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
