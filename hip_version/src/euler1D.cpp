/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "euler1D.hpp" 
#include "helpers.hpp"
#include <cmath>
#include <algorithm>

using std::vector; using std::cout;
using namespace simbi;


// Default Constructor 
Newtonian1D::Newtonian1D () {}

// Overloaded Constructor
Newtonian1D::Newtonian1D(
    vector< vector<real> > init_state, 
    real gamma, 
    real CFL, 
    vector<real> r,
    std::string coord_system = "cartesian") :

    init_state(init_state),
    gamma(gamma),
    r(r),
    coord_system(coord_system),
    CFL(CFL) {}

// Destructor 
Newtonian1D::~Newtonian1D() {}


// Typedefs because I'm lazy
typedef hydro1d::Conserved Conserved;
typedef hydro1d::Primitive Primitive;
typedef hydro1d::Eigenvals Eigenvals;
//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------
vector<Primitive> Newtonian1D::cons2prim(const vector<Conserved> &u_state){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    vector <Primitive> prims;
    prims.reserve(nzones);
    real rho, pre, v;
    
    for (auto &u: u_state){
        rho = u.rho;
        v   = u.m/rho;
        pre = (gamma - 1.0)*(u.e_dens - 0.5 * rho * v * v);
        prims.push_back(Primitive{rho, v, pre});
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------


Eigenvals Newtonian1D::calc_eigenvals(const Primitive &left_prim, const Primitive &right_prim)
{
    Eigenvals lambdas;
    // Separate the left and right state components
    real rho_l = left_prim.rho;
    real v_l   = left_prim.v;
    real p_l   = left_prim.p;

    real rho_r    = right_prim.rho;
    real v_r      = right_prim.v;
    real p_r      = right_prim.p;

    real cs_r = std::sqrt(gamma * p_r/rho_r);
    real cs_l = std::sqrt(gamma * p_l/rho_l);

    switch (sim_solver)
    {
    case SOLVER::HLLE:
        lambdas.aR = std::max({v_l + cs_l, v_r + cs_r, (real)0.0}); 
        lambdas.aL = std::min({v_l - cs_l, v_r - cs_r, (real)0.0});
        return lambdas;
    
    case SOLVER::HLLC:
        real cbar   = 0.5*(cs_l + cs_r);
        real rhoBar = 0.5*(rho_l + rho_r);
        real pStar  = 0.5*(p_l + p_r) + 0.5*(v_l - v_r)*cbar*rhoBar;

        // Steps to Compute HLLC as described in Toro et al. 2019
        real z      = (gamma - 1.)/(2.*gamma);
        real num    = cs_l + cs_r - ( gamma-1.)/2 *(v_r - v_l);
        real denom  = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
        real p_term = num/denom;
        real qL, qR;

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

        real aL = v_l - qL*cs_l;
        real aR = v_r + qR*cs_r;

        real aStar = ( (p_r - p_l + rho_l*v_l*(aL - v_l) - rho_r*v_r*(aR - v_r))/
                        (rho_l*(aL - v_l) - rho_r*(aR - v_r) ) );

        lambdas.aL = aL;
        lambdas.aR = aR;
        lambdas.aStar = aStar;
        lambdas.pStar = pStar;

        return lambdas;
    }


};

// Adapt the CFL conditonal timestep
real Newtonian1D::adapt_dt(vector<Primitive> &prims){

    real r_left, r_right, dr, cs, cfl_dt;
    real v, pre, rho;
    int shift_i;

    real min_dt = 0.0;

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < active_zones; ii++){
        shift_i = ii + idx_shift;

        r_right = xvertices[ii + 1];
        r_left  = xvertices[ii];

        dr = r_right - r_left;

        rho = prims[shift_i].rho;
        v   = prims[shift_i].v;
        pre = prims[shift_i].p;

        cs = std::sqrt(gamma * pre/rho);
        cfl_dt = dr/(std::max({std::abs(v + cs), std::abs(v - cs)}));

        if (ii > 0){
            min_dt = std::min(min_dt, cfl_dt);
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
Conserved Newtonian1D::prims2cons(const Primitive &prim)
{
    real energy = prim.p/(gamma - 1.0) + 0.5 * prim.rho * prim.v * prim.v;

    return Conserved{prim.rho, prim.rho * prim.v, energy};
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
Conserved Newtonian1D::calc_flux(const Primitive &prim)
{
    real energy = prim.p/(gamma - 1.0) + 0.5 * prim.rho * prim.v * prim.v;

    return Conserved{
        prim.rho * prim.v,
        prim.rho * prim.v * prim.v + prim.p,
        (energy + prim.p)*prim.v

    };
};

Conserved Newtonian1D::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims)
{
    Eigenvals lambda;
    lambda = calc_eigenvals(left_prims, right_prims);
    real am = lambda.aL;
    real ap = lambda.aR;

    // Compute the HLL Flux component-wise
    return (left_flux * ap - right_flux * am + (right_state - left_state) * am * ap)  / (ap - am) ;

};

Conserved Newtonian1D::calc_hllc_flux(
                                const Conserved &left_state,
                                const Conserved &right_state,
                                const Conserved &left_flux,
                                const Conserved &right_flux,
                                const Primitive &left_prims,
                                const Primitive &right_prims
                                )
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    real aL = lambda.aL; 
    real aR = lambda.aR; 
    real ap = std::max((real)0.0, aR);
    real am = std::min((real)0.0, aL);
    if (0.0 <= aL){
        return left_flux;
    } 
    else if (0.0 >= aR){
        return right_flux;
    }

    real aStar = lambda.aStar;
    real pStar = lambda.pStar;

    auto hll_flux = (left_flux * ap + right_flux * am - (right_state - left_state) * am * ap)  / (am + ap) ;

    auto hll_state = (right_state * aR - left_state * aL - right_flux + left_flux)/(aR - aL);
    
    if (- aL <= (aStar - aL)){
        real pressure = left_prims.p;
        real v        = left_prims.v;
        real rho      = left_state.rho;
        real m        = left_state.m;
        real energy   = left_state.e_dens;
        real cofac    = 1./(aL - aStar);

        real rhoStar = cofac * (aL - v)*rho;
        real mstar   = cofac * (m*(aL - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aL - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the intermediate left flux
        return left_flux + (star_state - left_state) * aL;
    } else {
        real pressure = right_prims.p;
        real v        = right_prims.v;
        real rho      = right_state.rho;
        real m        = right_state.m;
        real energy   = right_state.e_dens;
        real cofac    = 1./(aR - aStar);

        real rhoStar = cofac * (aR - v)*rho;
        real mstar   = cofac * (m*(aR - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aR - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the intermediate right flux
        return right_flux + (star_state - right_state) * aR;
    }
    
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

vector<Conserved> Newtonian1D::u_dot(vector<Conserved> &u_state)
{
    int i_start, i_bound, coordinate;
    int n_vars = u_state.size();
    std::string default_coordinates = "cartesian";
    
    
    Conserved u_l, u_r, f_l, f_r, f1, f2; 
    Primitive prims_l, prims_r;
    
    if (first_order){
        vector<Conserved> L;

        real dx = (r[active_zones - 1] - r[0])/active_zones;
        if (periodic){
            i_start = 0;
            i_bound = nzones;
        } else{
            int true_npts = nzones - 1;
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
                    u_l = u_state[ii];
                    u_r = roll(u_state, ii + 1);

                } else {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l = u_state[ii];
                    u_r = u_state[ii + 1];
                }

                prims_l = prims[ii];
                prims_r = prims[ii + 1];
                
                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                // Calc HLL Flux at i+1/2 interface
                if (hllc){
                    f1 = calc_hllc_flux(u_l, u_r, f_l, f_r,prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                }
                

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    // u_l[0] = roll(u_state[0], ii - 1);
                    // u_l[1] = roll(u_state[1], ii - 1);
                    // u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r = u_state[ii];

                } else {
                    u_l = u_state[ii - 1];
                    u_r = u_state[ii];

                }

                prims_l = prims[ii - 1];
                prims_r = prims[ii];

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                // Calc HLL Flux at i-1/2 interface
                if (hllc){
                    f2 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                }

                L.push_back( - (f1 - f2)/dx );

        }
            
        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            real sL, sR, rmean, pc, dV;
            real log_rLeft, log_rRight;

            real delta_logr = (log(r[active_zones - 1]) - log(r[0]))/active_zones;

            real dr; 

            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l = u_state[ii];
                    u_r = roll(u_state, ii + 1);

                } else {
                    // Shift the index for C++ [0] indexing
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l = u_state[ii];
                    u_r = u_state[ii + 1];
                }

                prims_l = prims[ii];
                prims_r = prims[ii + 1];

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);

                // Get the central pressure
                pc = prims_l.p;

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    // u_l[0] = roll(u_state[0], ii - 1);
                    // u_l[1] = roll(u_state[1], ii - 1);
                    // u_l[2] = roll(u_state[2], ii - 1);     
                    u_r = u_state[ii];

                } else {
                    u_l = u_state[ii - 1];
                    u_r = u_state[ii];

                }

                prims_l = prims[ii - 1];
                prims_r = prims[ii];

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);

                sR  = coord_lattice.face_areas[coordinate + 1];
                sL  = coord_lattice.face_areas[coordinate];
                dr  = coord_lattice.dx1[coordinate];

                L.push_back(Conserved{
                        // L(rho)
                        - (sR*f1.rho - sL*f2.rho)/dV,

                        // L(rho * v)
                        - (sR*f1.m - sL * f2.m )/ dV + 2*pc/rmean,

                        // L(E)
                        - (sR*f1.e_dens - sL*f2.e_dens )/ dV

                }) ;
            }
            
            
        }

        
        return L;
    } else {
        real dx = (r[active_zones - 1] - r[0])/active_zones;
        
        // Calculate the primitives for the entire state
        Primitive left_most, left_mid, center;
        Primitive right_mid, right_most;
        vector<Conserved> L;
        L.reserve(nzones);

        // The periodic BC doesn't require ghost cells. Shift the index
        // to the beginning since all of he.
        i_start = idx_shift; 
        i_bound = active_zones;

        if (coord_system == default_coordinates){
            //==============================================
            //                  CARTESIAN
            //==============================================
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    coordinate = ii;
                    left_most  = roll(prims, ii - 2);
                    left_mid   = roll(prims, ii - 1);
                    center     = prims[ii];
                    right_mid  = roll(prims, ii + 1);
                    right_most = roll(prims, ii + 2);
                    
                } else {
                    coordinate = ii - 2;
                    left_most  = prims[ii - 2];
                    left_mid   = prims[ii - 1];
                    center     = prims[ii];
                    right_mid  = prims[ii + 1];
                    right_most = prims[ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho = center.rho + 0.5*minmod(plm_theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    plm_theta*(right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5*minmod(plm_theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    plm_theta*(right_mid.v - center.v));

                prims_l.p = center.p + 0.5*minmod(plm_theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    plm_theta*(right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho - 0.5*minmod(plm_theta*(right_mid.rho - center.rho),
                                                    0.5*(right_most.rho - center.rho),
                                                    plm_theta*(right_most.rho - right_mid.rho));

                prims_r.v = right_mid.v - 0.5*minmod(plm_theta*(right_mid.v - center.v),
                                                    0.5*(right_most.v - center.v),
                                                    plm_theta*(right_most.v - right_mid.v));

                prims_r.p = right_mid.p - 0.5*minmod(plm_theta*(right_mid.p - center.p),
                                                    0.5*(right_most.p - center.p),
                                                    plm_theta*(right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                if (hllc){
                    f1 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f1 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                }

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho = left_mid.rho + 0.5*minmod(plm_theta*(left_mid.rho - left_most.rho),
                                                        0.5*(center.rho -left_most.rho),
                                                        plm_theta*(center.rho - left_mid.rho));

                prims_l.v = left_mid.v + 0.5*minmod(plm_theta*(left_mid.v - left_most.v),
                                                        0.5*(center.v -left_most.v),
                                                        plm_theta*(center.v - left_mid.v));
                
                prims_l.p = left_mid.p + 0.5*minmod(plm_theta*(left_mid.p - left_most.p),
                                                        0.5*(center.p -left_most.p),
                                                        plm_theta*(center.p - left_mid.p));


                    
                prims_r.rho = center.rho - 0.5*minmod(plm_theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    plm_theta*(right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5*minmod(plm_theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    plm_theta*(right_mid.v - center.v));

                prims_r.p = center.p - 0.5*minmod(plm_theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    plm_theta*(right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                if (hllc){
                    f2 = calc_hllc_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                } else {
                    f2 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);
                }

                L.push_back ((f1 - f2) * (-1.0) / dx );
            }            
                                                                                                                         
            

        } else {
            //==============================================
            //                  RADIAL
            //==============================================
            real sL, sR, rmean, pc, dV;
            real log_rLeft, log_rRight;

            real delta_logr = (log(r[active_zones - 1]) - log(r[0]))/active_zones;

            real dr;
            for (int ii=i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most   = roll(prims, ii - 2);
                    left_mid    = roll(prims, ii - 1);
                    center      = prims[ii];
                    right_mid   = roll(prims, ii + 1);
                    right_most  = roll(prims, ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    coordinate  = ii - 2;
                    left_most   = prims[ii - 2];
                    left_mid    = prims[ii - 1];
                    center      = prims[ii];
                    right_mid   = prims[ii + 1];
                    right_most  = prims[ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho = center.rho + 0.5*minmod(plm_theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    plm_theta*(right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5*minmod(plm_theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    plm_theta*(right_mid.v - center.v));

                prims_l.p = center.p + 0.5*minmod(plm_theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    plm_theta*(right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho - 0.5*minmod(plm_theta*(right_mid.rho - center.rho),
                                                    0.5*(right_most.rho - center.rho),
                                                    plm_theta*(right_most.rho - right_mid.rho));

                prims_r.v = right_mid.v - 0.5*minmod(plm_theta*(right_mid.v - center.v),
                                                    0.5*(right_most.v - center.v),
                                                    plm_theta*(right_most.v - right_mid.v));

                prims_r.p = right_mid.p - 0.5*minmod(plm_theta*(right_mid.p - center.p),
                                                    0.5*(right_most.p - center.p),
                                                    plm_theta*(right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                f1 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho = left_mid.rho + 0.5*minmod(plm_theta*(left_mid.rho - left_most.rho),
                                                        0.5*(center.rho -left_most.rho),
                                                        plm_theta*(center.rho - left_mid.rho));

                prims_l.v = left_mid.v + 0.5*minmod(plm_theta*(left_mid.v - left_most.v),
                                                        0.5*(center.v -left_most.v),
                                                        plm_theta*(center.v - left_mid.v));
                
                prims_l.p = left_mid.p + 0.5*minmod(plm_theta*(left_mid.p - left_most.p),
                                                        0.5*(center.p -left_most.p),
                                                        plm_theta*(center.p - left_mid.p));


                    
                prims_r.rho = center.rho - 0.5*minmod(plm_theta*(center.rho - left_mid.rho),
                                                    0.5*(right_mid.rho - left_mid.rho),
                                                    plm_theta*(right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5*minmod(plm_theta*(center.v - left_mid.v),
                                                    0.5*(right_mid.v - left_mid.v),
                                                    plm_theta*(right_mid.v - center.v));

                prims_r.p = center.p - 0.5*minmod(plm_theta*(center.p - left_mid.p),
                                                    0.5*(right_mid.p - left_mid.p),
                                                    plm_theta*(right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                f2 = calc_hll_flux(u_l, u_r, f_l, f_r, prims_l, prims_r);

                //Get Central Pressure
                pc = center.p;

                sR  = coord_lattice.face_areas[coordinate + 1];
                sL  = coord_lattice.face_areas[coordinate];
                dr  = coord_lattice.dx1[coordinate];

                L.push_back(Conserved{
                        // L(rho)
                        - (sR*f1.rho - sL*f2.rho)/dV,

                        // L(rho * v)
                        - (sR*f1.m - sL * f2.m )/ dV + 2*pc/rmean,

                        // L(E)
                        - (sR*f1.e_dens - sL*f2.e_dens )/ dV

                }) ;
            }
        
        }

        return L; 
    }
    
};


 vector<vector<real> > Newtonian1D::simulate1D(
    float tend = 0.1, 
    float dt = 1.e-4, 
    float plm_theta=1.5,
    bool first_order = true, 
    bool periodic = false,
    bool linspace = true,
    bool hllc=false){

    // Define the swap vector for the integrated state
    this->nzones = init_state[0].size();

    vector<Conserved>  u_p, u, udot;
    float t = 0;

    this->plm_theta = plm_theta;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->nzones = init_state[0].size();

    if (periodic){
        this->idx_shift    = 0;
        this->active_zones = nzones;
    } else {
        if (first_order){
            this->idx_shift = 1;
            this->active_zones = nzones - 2;
        } else {
            this->idx_shift = 2;
            this->active_zones = nzones - 4; 
        }
    }

    if (hllc){
        this->sim_solver = simbi::SOLVER::HLLC;
    } else {
        this->sim_solver = simbi::SOLVER::HLLE;
    }
    // Copy the state array into real & profile variables
    for (size_t ii = 0; ii < nzones; ii++)
    {
        u.push_back(Conserved{init_state[0][ii], init_state[1][ii], init_state[2][ii]});
    }
    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE);
    }
    else
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }

    u_p = u;
    int i_real; 
    if (first_order){
        if (t == 0){
            u[0].m = -u[1].m;
        }

        while (t < tend){
            prims = cons2prim(u);

            // Compute the udot array in active zone region
            udot = u_dot(u);

            for (int ii = 0; ii < active_zones; ii++){
                // Get the non-ghost index 
                i_real = ii + idx_shift;
                u_p[i_real] = u[i_real] + dt*udot[ii];

            }

            // Readjust the ghost cells at i-1,i+1 if not periodic
            if (periodic == false){
                u_p[0].rho      =   u_p[1].rho;
                u_p[0].m        = - u_p[1].m;
                u_p[0].e_dens   =   u_p[1].e_dens;

                // Outflow at outer boundary
                u_p[nzones - 1].rho      =   u_p[nzones - 2].rho;
                u_p[nzones - 1].m        =   u_p[nzones - 2].m;
                u_p[nzones - 1].e_dens   =   u_p[nzones - 2].e_dens;
            }
            
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;
            dt = adapt_dt(prims);

        }   

    } else {
        
        vector<Conserved> u1, u2;
        u1.reserve(nzones);
        u2.reserve(nzones);
        udot.reserve(active_zones);

        u1 = u;
        u2 = u;

        u[0].m = - u[2].m;
        u[1].m = - u[2].m;
        while (t < tend){
            prims = cons2prim(u);
            udot = u_dot(u);
            
            for (int ii = 0; ii < active_zones; ii++){
                // Get the non-ghost index 
                i_real = ii + idx_shift;
                u1[i_real] = u[i_real] + dt*udot[ii];
            }
            
            // cout << u1[0].rho << "\n";
            // std::cin.get();
            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            if (periodic == false){
                u1[0].rho      =   u1[2].rho;
                u1[1].rho      =   u1[2].rho;
                u1[0].m        = - u1[2].m;
                u1[1].m        = - u1[2].m;
                u1[0].e_dens   =   u1[2].e_dens;
                u1[1].e_dens   =   u1[2].e_dens;

                // Outflow at outer boundary
                u1[nzones - 1].rho      =   u1[nzones - 3].rho;
                u1[nzones - 2].rho      =   u1[nzones - 3].rho;
                u1[nzones - 1].m        =   u1[nzones - 3].m;
                u1[nzones - 2].m        =   u1[nzones - 3].m;
                u1[nzones - 1].e_dens   =   u1[nzones - 3].e_dens;
                u1[nzones - 2].e_dens   =   u1[nzones - 3].e_dens;
            }
            
            prims = cons2prim(u1);
            udot = u_dot(u1);

            for (int ii = 0; ii < active_zones; ii++){
                i_real = ii + idx_shift;
                u2[i_real] = u[i_real] * 0.5 + u1[i_real] * 0.5  + udot[ii] * dt * 0.5;
            }

            
            if (periodic == false){
                u2[0].rho      =   u2[2].rho;
                u2[1].rho      =   u2[2].rho;
                u2[0].m        = - u2[2].m;
                u2[1].m        = - u2[2].m;
                u2[0].e_dens   =   u2[2].e_dens;
                u2[1].e_dens   =   u2[2].e_dens;

                // Outflow at outer boundary
                u2[nzones - 1].rho      =   u2[nzones - 3].rho;
                u2[nzones - 2].rho      =   u2[nzones - 3].rho;
                u2[nzones - 1].m        =   u2[nzones - 3].m;
                u2[nzones - 2].m        =   u2[nzones - 3].m;
                u2[nzones - 1].e_dens   =   u2[nzones - 3].e_dens;
                u2[nzones - 2].e_dens   =   u2[nzones - 3].e_dens;
            }
            
            // Swap the arrays
            u.swap(u2);
            
            t += dt;
            dt = adapt_dt(prims);

        }  

    }

    std::vector<std::vector<real> > solution(3, vector<real>(nzones));
    for (size_t ii = 0; ii < nzones; ii++)
    {
        solution[0][ii] = u[ii].rho;
        solution[1][ii] = u[ii].m;
        solution[2][ii] = u[ii].e_dens;
    }
    
    // write_data(u, tend, "sod");
    return solution;

 };
