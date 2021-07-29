/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "classical_2d.hpp" 
#include "helper_functions.hpp"
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
Newtonian2D::Newtonian2D(
    vector<vector<real> > init_state, 
    int NX,
    int NY,
    real gamma, 
    vector<real> x1, 
    vector<real> x2, 
    real CFL, 
    string coord_system = "cartesian")
:
    init_state(init_state),
    NX(NX),
    NY(NY),
    gamma(gamma),
    x1(x1),
    x2(x2),
    CFL(CFL),
    coord_system(coord_system)
{}

// Destructor 
Newtonian2D::~Newtonian2D() {}


// Typedefs because I'm lazy
typedef hydro2d::Conserved Conserved;
typedef hydro2d::Primitive Primitive;
typedef hydro2d::Eigenvals Eigenvals;
//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

/**
 * Return a 1 + 2D matrix containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
vector<Primitive> Newtonian2D::cons2prim(const vector<Conserved> &u_state)
{
    real rho, energy, momx, momy;
    real vx,vy, pressure;

    vector<Primitive> prims;
    prims.reserve(nzones);

    for(auto &u: u_state)
    {
        rho    = u.rho;       // Density
        momx   = u.m1;        // X-Momentum
        momy   = u.m2;        // Y-Momentum
        energy = u.e_dens;    // Energy

        vx       = momx/rho;
        vy       = momy/rho;
        pressure = (gamma - 1.0)*(energy - 0.5 * rho * (vx*vx + vy*vy));
        
        prims.push_back(Primitive{rho, vx ,vy, pressure});
            
    }
    
    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------

Eigenvals Newtonian2D::calc_eigenvals(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const int ehat)
{   
    switch (solver)
    {
    case simbi::Solver::HLLC:
        {
            const real vx_l = left_prims.v1;
            const real vx_r = right_prims.v1;
            const real vy_l = left_prims.v2;
            const real vy_r = right_prims.v2;

            const real p_r   = right_prims.p;
            const real p_l   = left_prims.p;
            const real rho_l = left_prims.rho;
            const real rho_r = right_prims.rho;

            const real cs_r = sqrt(gamma * p_r/rho_r);
            const real cs_l = sqrt(gamma * p_l/rho_l);

            switch (ehat)
            {
                case 1:
                    {
                        // Calculate the mean velocities of sound and fluid
                        const real cbar   = 0.5*(cs_l + cs_r);
                        const real rhoBar = 0.5*(rho_l + rho_r);
                        const real z      = (gamma - 1.)/(2.0*gamma);
                        const real num    = cs_l + cs_r - ( gamma-1.)/2. *(vx_r - vx_l);
                        const real denom  = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
                        const real p_term = num/denom;
                        const real pStar  = pow(p_term, (1./z));

                        const real qL = 
                            (pStar <= p_l) ? 1. : sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));

                        const real qR = 
                            (pStar <= p_r) ? 1. : sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));

                        const real aL = vx_l - qL*cs_l;
                        const real aR = vx_r + qR*cs_r;

                        const real aStar = 
                            ( (p_r - p_l + rho_l*vx_l*(aL - vx_l) - rho_r*vx_r*(aR - vx_r) )/ 
                                (rho_l*(aL - vx_l) - rho_r*(aR - vx_r) ) );

                        return Eigenvals(aL, aR, aStar, pStar);
                    }

                case 2:
                    {
                        const real cbar   = 0.5*(cs_l + cs_r);
                        const real rhoBar = 0.5*(rho_l + rho_r);
                        const real z      = (gamma - 1.)/(2.*gamma);
                        const real num    = cs_l + cs_r - ( gamma-1.)/2 *(vy_r - vy_l);
                        const real denom  = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
                        const real p_term = num/denom;
                        const real pStar  = pow(p_term, (1./z));

                        const real qL =
                            (pStar <= p_l) ? 1. : sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));

                        const real qR = 
                            (pStar <= p_r) ? 1. : sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));

                        const real aL = vy_l - qL*cs_l;
                        const real aR = vy_r + qR*cs_r;

                        const real aStar = 
                            ( (p_r - p_l + rho_l*vy_l*(aL - vy_l) - rho_r*vy_r*(aR - vy_r))/
                                    (rho_l*(aL - vy_l) - rho_r*(aR - vy_r) ) );
                        return Eigenvals(aL, aR, aStar, pStar);
                    }
                }
        }
    case simbi::Solver::HLLE:
        {
            switch (ehat)
            {
            case 1:
                {   
                    const real vx_l = left_prims.v1;
                    const real vx_r = right_prims.v1;
                    const real p_r   = right_prims.p;
                    const real p_l   = left_prims.p;
                    const real rho_l = left_prims.rho;
                    const real rho_r = right_prims.rho;
                    const real cs_r = sqrt(gamma * p_r/rho_r);
                    const real cs_l = sqrt(gamma * p_l/rho_l);

                    const real aL = std::min({(real)0.0, vx_l - cs_l, vx_r - cs_r});
                    const real aR = std::max({(real)0.0, vx_l + cs_l, vx_r + cs_r});
                    return Eigenvals(aL, aR);
                }
                
            
            case 2:
                {
                    const real vy_l  = left_prims.v2;
                    const real vy_r  = right_prims.v2;
                    const real p_r   = right_prims.p;
                    const real p_l   = left_prims.p;
                    const real rho_l = left_prims.rho;
                    const real rho_r = right_prims.rho;
                    const real cs_r = sqrt(gamma * p_r/rho_r);
                    const real cs_l = sqrt(gamma * p_l/rho_l);

                    const real aL = std::min({(real)0.0, vy_l - cs_l, vy_r - cs_r});
                    const real aR = std::max({(real)0.0, vy_l + cs_l, vy_r + cs_r});
                    return Eigenvals(aL, aR);
                }
            }
        }
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state tensor for computation. 
// It is being doing pointwise in this case as opposed to over
// the entire array since we are in c++
Conserved Newtonian2D::prims2cons(const Primitive &prims)
 {
    const real rho = prims.rho;
    const real vx  = prims.v1;
    const real vy  = prims.v2;
    const real pre = prims.p;
    const real et  = pre/(gamma - 1.0) + 0.5 * rho * (vx*vx + vy*vy);

    return Conserved{rho, rho*vx, rho*vy, et};
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
real Newtonian2D::adapt_dt(const vector<Primitive > &prims)
{
    real dx1, cs, dx2, rho, pressure, v1, v2, volAvg;
    real min_dt, cfl_dt;
    int shift_i, shift_j;

    min_dt = 0;
    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < yphysical_grid; jj++){
        dx2     = coord_lattice.dx2[jj];
        shift_j = jj + idx_shift;
        for (int ii = 0; ii < xphysical_grid; ii++){
            shift_i = ii + idx_shift;

            dx1 = coord_lattice.dx1[ii];

            rho      = prims[shift_j * NX + shift_i].rho;
            v1       = prims[shift_j * NX + shift_i].v1;
            v2       = prims[shift_j * NX + shift_i].v2;
            pressure = prims[shift_j * NX + shift_i].p;

            cs       = std::sqrt(gamma * pressure / rho );
            if (coord_system == "cartesian"){
                
                cfl_dt = 
                    min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

            } else {
                volAvg = coord_lattice.x1mean[ii];
                cfl_dt = 
                    min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), volAvg*dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

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
Conserved Newtonian2D::calc_flux(const Primitive &prims, const int ehat)
{
    const auto rho = prims.rho;
    const auto vx  = prims.v1;
    const auto vy  = prims.v2;
    const auto pre = prims.p;
    const auto et  = pre/(gamma - 1.0) + 0.5 * rho * (vx*vx + vy*vy);
    
    switch (ehat)
    {
    case 1:
        {
            const auto momx = rho*vx;
            const auto convect_xy = rho*vx*vy;
            const auto energy_dens = rho*vx*vx + pre;
            const auto zeta = (et + pre)*vx;

            return Conserved{momx, energy_dens, convect_xy, zeta};
        }
    case 2:
        {
            const auto momy        = rho*vy;
            const auto convect_xy  = rho*vx*vy;
            const auto energy_dens = rho*vy*vy + pre;
            const auto zeta        = (et + pre)*vy;

            return Conserved{momy, convect_xy, energy_dens, zeta};
        }
    }
};


Conserved Newtonian2D::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const int ehat)
                                        
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);
    real am = lambda.aL;
    real ap = lambda.aR;

    // Compute the HLL Flux 
    return  ( left_flux * ap - right_flux * am 
                + (right_state - left_state ) * am * ap )  /
                    (ap - am);
};

Conserved Newtonian2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const int ehat)
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);

    const real aL    = lambda.aL;
    const real aR    = lambda.aR;

    // Quick checks before moving on with rest of computation
    if (0.0 <= aL){
        return left_flux;
    } else if (0.0 >= aR){
        return right_flux;
    }

    const real aStar = lambda.aStar;
    const real pStar = lambda.pStar;

    if (-aL <= (aStar - aL) ){
        const auto pre      = left_prims.p;
        const auto v1       = left_prims.v1;
        const auto v2       = left_prims.v2;
        const auto rho      = left_prims.rho;
        const auto m1       = left_state.m1;
        const auto m2       = left_state.m2;
        const auto energy   = left_state.e_dens;
        const auto cofac    = 1./(aL - aStar);

        switch (ehat)
        {
        case 1:
            {
                const auto rhoStar = cofac * (aL - v1) * rho;
                const auto m1star  = cofac * (m1*(aL - v1) - pre + pStar);
                const auto m2star  = cofac * (aL - v1) * m2;
                const auto eStar   = cofac * (energy*(aL - v1) + pStar*aStar - pre*v1);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return left_flux + (starstate - left_state)*aL;
            }
        
        case 2:
                const auto rhoStar = cofac * (aL - v2) * rho;
                const auto m1star  = cofac * (aL - v2) * m1; 
                const auto m2star  = cofac * (m2 * (aL - v2) - pre + pStar);
                const auto eStar   = cofac * (energy*(aL - v2) + pStar*aStar - pre*v2);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return left_flux + (starstate - left_state)*aL;
        }

    } else {
        const auto pre      = right_prims.p;
        const auto v1       = right_prims.v1;
        const auto v2       = right_prims.v2;
        const auto rho      = right_prims.rho;
        const auto m1       = right_state.m1;
        const auto m2       = right_state.m2;
        const auto energy   = right_state.e_dens;
        const auto cofac    = 1./(aR - aStar);

        switch (ehat)
        {
        case 1:
            {
                const auto rhoStar = cofac * (aR - v1) * rho;
                const auto m1star  = cofac * (m1*(aR - v1) - pre + pStar);
                const auto m2star  = cofac * (aR - v1) * m2;
                const auto eStar   = cofac * (energy*(aR - v1) + pStar*aStar - pre*v1);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return right_flux + (starstate - right_state)*aR;
            }
        
        case 2:
                const auto rhoStar = cofac * (aR - v2) * rho;
                const auto m1star  = cofac * (aR - v2) * m1; 
                const auto m2star  = cofac * (m2 * (aR - v2) - pre + pStar);
                const auto eStar   = cofac * (energy*(aR - v2) + pStar*aStar - pre*v2);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return right_flux + (starstate - right_state)*aR;
        }


    }
    
};

//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

vector<Conserved> Newtonian2D::u_dot(const vector<Conserved> &u_state)
{

    int xcoordinate, ycoordinate;
    string default_coordinates = "cartesian";

    vector<Conserved> L;
    L.reserve(active_zones);

    Conserved  ux_l, ux_r, uy_l, uy_r, f_l, f_r; 
    Conserved  f1, f2, g1, g2, g_l, g_r;

    Primitive  xprims_l, xprims_r, yprims_l, yprims_r;
    Primitive  xleft_most, xleft_mid, xright_mid, xright_most;
    Primitive  yleft_most, yleft_mid, yright_mid, yright_most;
    Primitive  center;

    /**
    for (int jj = idx_shift; jj < y_bound; jj++){
        for (int ii = idx_shift; ii < x_bound; ii++){
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
            
            f_l = calc_flux(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_flux(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_flux(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_flux(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

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

            f_l = calc_flux(gamma, xprims_l.rho, xprims_l.v1, xprims_l.v2, xprims_l.p);
            f_r = calc_flux(gamma, xprims_r.rho, xprims_r.v1, xprims_r.v2, xprims_r.p);

            g_l = calc_flux(gamma, yprims_l.rho, yprims_l.v1, yprims_l.v2, yprims_l.p, false);
            g_r = calc_flux(gamma, yprims_r.rho, yprims_r.v1, yprims_r.v2, yprims_r.p, false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1.rho - f2.rho)/dx - (g1.rho - g2.rho)/dy;
            L[1][ycoordinate][xcoordinate] = - (f1.m1 - f2.m1)/dx - (g1.m1 - g2.m1)/dy;
            L[2][ycoordinate][xcoordinate] = - (f1.m2 - f2.m2)/dx - (g1.m2 - g2.m2)/dy;
            L[3][ycoordinate][xcoordinate] = - (f1.e_dens - f2.e_dens)/dx - (g1.e_dens - g2.e_dens)/dy;

        }
    }
    */

    if (coord_system == "cartesian"){
        real dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
        real dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;
        for (int jj = idx_shift; jj < y_bound; jj++){
            for (int ii = idx_shift; ii < x_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    // X Coordinate
                    xleft_most   = roll(prims, jj * NX + ii - 2);
                    xleft_mid    = roll(prims, jj * NX + ii - 1);
                    center       = prims[jj * NX + ii];
                    xright_mid   = roll(prims, NX*jj + ii + 1);
                    xright_most  = roll(prims, NX*jj + ii + 2);

                    yleft_most   = roll(prims, ii +  NX*(jj - 2) );
                    yleft_mid    = roll(prims, ii +  NX*(jj - 1) );
                    yright_mid   = roll(prims, ii +  NX*(jj + 1) );
                    yright_most  = roll(prims, ii +  NX*(jj + 2) );

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most  = prims[jj * NX + ii - 2];
                    xleft_mid   = prims[jj * NX + ii - 1];
                    center      = prims[jj * NX + ii];
                    xright_mid  = prims[jj * NX + ii + 1];
                    xright_most = prims[jj * NX + ii + 2];

                    // Coordinate Y
                    yleft_most   = prims[(jj - 2)*NX + ii];
                    yleft_mid    = prims[(jj - 1)*NX + ii];
                    yright_mid   = prims[(jj + 1)*NX + ii];
                    yright_most  = prims[(jj + 2)*NX + ii];

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
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l);
                f_r = calc_flux(xprims_r);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc){
                    f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
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
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l);
                f_r = calc_flux(xprims_r);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc){
                    f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                L.push_back(Conserved{
                    // L(rho)
                    - (f1.rho - f2.rho)/dx - (g1.rho - g2.rho)/dy 
                        + sourceRho[ycoordinate * xphysical_grid + xcoordinate],

                    - (f1.m1 - f2.m1)/dx - (g1.m1 - g2.m1)/dy 
                        + sourceM1[ycoordinate * xphysical_grid + xcoordinate],

                    - (f1.m2 - f2.m2)/dx - (g1.m2 - g2.m2)/dy
                        + sourceM2[ycoordinate * xphysical_grid + xcoordinate],

                    - (f1.e_dens - f2.e_dens)/dx - (g1.e_dens - g2.e_dens)/dy 
                        + sourceE[ycoordinate * xphysical_grid + xcoordinate]

                });
            }

        }
        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        real r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        real tcoordinate, rcoordinate;
        real upper_tsurface, lower_tsurface, right_rsurface, left_rsurface, dtheta;

        real pi = 2*acos(0.0);

        for (int jj = idx_shift; jj < y_bound; jj++){
            tcoordinate = jj - idx_shift;
            upper_tsurface = coord_lattice.x2_face_areas[tcoordinate + 1];
            lower_tsurface = coord_lattice.x2_face_areas[tcoordinate];
            for (int ii = idx_shift; ii < x_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most   = roll(prims, jj * NX + ii - 2);
                    xleft_mid    = roll(prims, jj * NX + ii - 1);
                    center       = prims[jj * NX + ii];
                    xright_mid   = roll(prims, NX*jj + ii + 1);
                    xright_most  = roll(prims, NX*jj + ii + 2);

                    yleft_most   = roll(prims, ii +  NX*(jj - 2) );
                    yleft_mid    = roll(prims, ii +  NX*(jj - 1) );
                    yright_mid   = roll(prims, ii +  NX*(jj + 1) );
                    yright_most  = roll(prims, ii +  NX*(jj + 2) );
 

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;

                    // Coordinate X
                    xleft_most      = prims[jj * NX + ii - 2];
                    xleft_mid       = prims[jj * NX + ii - 1];
                    center          = prims[jj * NX + ii];
                    xright_mid      = prims[jj * NX + ii + 1];
                    xright_most     = prims[jj * NX + ii + 2];

                    // Coordinate Y
                    yleft_most   = prims[(jj - 2) * NX + ii];
                    yleft_mid    = prims[(jj - 1) * NX + ii];
                    yright_mid   = prims[(jj + 1) * NX + ii];
                    yright_most  = prims[(jj + 2) * NX + ii];

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
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l);
                f_r = calc_flux(xprims_r);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc){
                    f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
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
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l);
                f_r = calc_flux(xprims_r);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc){
                    f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                
                rhoc = center.rho;
                pc   = center.p;
                uc   = center.v1;
                vc   = center.v2;

                // Compute the surface areas
                right_rsurface = coord_lattice.x1_face_areas[rcoordinate + 1];
                left_rsurface  = coord_lattice.x1_face_areas[rcoordinate];
                volAvg         = coord_lattice.x1mean[rcoordinate];
                deltaV1        = coord_lattice.dV1[rcoordinate];
                deltaV2        = volAvg * coord_lattice.dV2[tcoordinate];

                L.push_back(Conserved{
                    // L(rho)
                    - (f1.rho*right_rsurface - f2.rho*left_rsurface)/deltaV1
                        - (g1.rho*upper_tsurface - g2.rho*lower_tsurface)/deltaV2
                        + sourceRho[tcoordinate * yphysical_grid + rcoordinate],

                    // L(rho * v1)
                    - (f1.m1*right_rsurface - f2.m1*left_rsurface)/deltaV1
                        - (g1.m1*upper_tsurface - g2.m1*lower_tsurface)/deltaV2 
                            + rhoc*vc*vc/volAvg + 2*pc/volAvg
                                + sourceM1[tcoordinate * yphysical_grid + rcoordinate],

                    // L(rho * v2)
                    - (f1.m2*right_rsurface - f2.m2*left_rsurface)/deltaV1
                            - (g1.m2*upper_tsurface - g2.m2*lower_tsurface)/deltaV2
                            -(rhoc*uc*vc/volAvg - pc*coord_lattice.cot[tcoordinate]/(volAvg))
                            + sourceM2[tcoordinate *yphysical_grid + rcoordinate],

                    // L(E)
                    - (f1.e_dens*right_rsurface - f2.e_dens*left_rsurface)/deltaV1
                        - (g1.e_dens*upper_tsurface - g2.e_dens*lower_tsurface)/deltaV2
                            + sourceE[tcoordinate *yphysical_grid + rcoordinate]
                
                });
            }

        }

        return L;
        
    }
    

};



//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
vector<vector<real> > Newtonian2D::simulate2D(
    const vector<vector<real> > &sources, 
    real tend = 0.1,
    bool periodic = false, 
    real dt = 1.e-4, 
    bool linspace=true,
    bool hllc = false,
    real theta)
{

    // Define the swap vector for the integrated state
    float t = 0;
    int i_real, j_real;

    this->nzones      = NX*NY;
    this->sources     = sources;
    this->first_order = first_order;
    this->periodic    = periodic;
    this->hllc        = hllc;
    this->linspace    = linspace;
    this->theta       = theta;

    if (periodic){
        this->xphysical_grid = NX;
        this->yphysical_grid = NY;
        this->x_bound        = NX;
        this->y_bound        = NY;
        this->idx_shift      = 0;
    } else {
        this->xphysical_grid = NX - 4;
        this->yphysical_grid = NY - 4;
        this->idx_shift = 2;
        this->x_bound = NX - 2;
        this->y_bound = NY - 2;

    }
    this->active_zones = xphysical_grid * yphysical_grid;
    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    if (hllc){
        solver = Solver::HLLC;
    } else {
        solver = Solver::HLLE;
    }


    vector<Conserved> u, u1, u2, udot, u_p;
    u1.reserve(nzones);
    u2.reserve(nzones);
    u.reserve(nzones);
    udot.reserve(active_zones);

    for (size_t i = 0; i < nzones; i++)
    {
        u.push_back(Conserved{
                init_state[0][i], 
                init_state[1][i], 
                init_state[2][i], 
                init_state[3][i]} );
    }

    sourceRho = sources[0];
    sourceM1  = sources[1];
    sourceM2  = sources[2];
    sourceE   = sources[3];

    u_p = u;
    u1  = u; 
    u2  = u;
    
    config_ghosts2D(u, NX, NY, false);
    
    while (t <= tend){
        /* Compute the loop execution time */
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        prims = cons2prim(u);
        udot = u_dot(u);

        /* Perform Higher Order RK3 */
        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            j_real = jj + idx_shift;
            for (int ii = 0; ii < xphysical_grid; ii++){
                i_real = ii + idx_shift;
                u1[j_real * NX + i_real] = 
                    u[j_real * NX + i_real] + udot[jj * xphysical_grid + ii] * dt;
            }
        }

        if (!periodic){
            config_ghosts2D(u1, NX, NY, false);
        }

        prims = cons2prim(u1);
        udot = u_dot(u1);
        

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            j_real = jj + idx_shift;
            for (int ii = 0; ii < xphysical_grid; ii++){
                i_real = ii + idx_shift;

                u2[j_real * NX + i_real] = 
                    u[j_real * NX + i_real] * 0.5 + 
                         u1[j_real * NX + i_real] * 0.5 + 
                             udot[jj * xphysical_grid + ii] * dt * 0.5;

            }
            
        }
        
        if (!periodic){
            config_ghosts2D(u2, NX, NY, false);
        }
        
        // Swap the arrays
        u.swap(u2);
        
        t += dt;
        dt = adapt_dt(prims);
        
        /* Compute the loop execution time */
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<real> time_span = duration_cast<duration<real>>(t2 - t1);

        cout << fixed << setprecision(3) << std::scientific;
        cout << "\r" << "dt: " << setw(5) << dt 
             << "\t" << "t: " << setw(5) << t 
             << "\t" <<  "Zones per sec: " << nzones/time_span.count()
             << flush;

    }

    cout << "\n " << endl;
    std::vector<std::vector<real> > solution(4, vector<real>(nzones));
    for (size_t ii = 0; ii < nzones; ii++)
    {
        solution[0][ii] = u[ii].rho;
        solution[1][ii] = u[ii].m1;
        solution[2][ii] = u[ii].m2;
        solution[3][ii] = u[ii].e_dens;
    }
    
    // write_data(u, tend, "sod");
    return solution;

 };