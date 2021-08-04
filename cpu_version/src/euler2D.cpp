/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "euler2D.hpp" 
#include "helpers.hpp"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <chrono>



using namespace simbi;
using namespace std::chrono;

// Default Constructor 
Newtonian2D::Newtonian2D () {}

// Overloaded Constructor
Newtonian2D::Newtonian2D(
    std::vector<std::vector<double> > init_state, 
    int NX,
    int NY,
    double gamma, 
    std::vector<double> x1, 
    std::vector<double> x2, 
    double CFL, 
    std::string coord_system = "cartesian")
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
void Newtonian2D::cons2prim()
{
    double rho, energy;
    double v1,v2, pre;
    int gid;

    #pragma omp parallel for schedule(static)
    for (int jj = 0; jj < NY; jj++)
    {  
        for (int ii = 0; ii < NX; ii++)
        {   
            gid = jj * NX + ii;
            rho     = cons[gid].rho;
            v1      = cons[gid].m1/rho;
            v2      = cons[gid].m2/rho;

            pre = (gamma - 1.0)*(cons[gid].e_dens - 0.5 * rho * (v1 * v1 + v2 * v2));
            prims [gid] = Primitive{rho, v1, v2, pre};
        }
    }
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
            const double vx_l = left_prims.v1;
            const double vx_r = right_prims.v1;
            const double vy_l = left_prims.v2;
            const double vy_r = right_prims.v2;

            const double p_r   = right_prims.p;
            const double p_l   = left_prims.p;
            const double rho_l = left_prims.rho;
            const double rho_r = right_prims.rho;

            const double cs_r = std::sqrt(gamma * p_r/rho_r);
            const double cs_l = std::sqrt(gamma * p_l/rho_l);

            switch (ehat)
            {
                case 1:
                    {
                        // Calculate the mean velocities of sound and fluid
                        const double cbar   = 0.5*(cs_l + cs_r);
                        const double rhoBar = 0.5*(rho_l + rho_r);
                        const double z      = (gamma - 1.)/(2.0*gamma);
                        const double num    = cs_l + cs_r - (gamma - 1.) * 0.5 *(vx_r - vx_l);
                        const double denom  = cs_l/std::pow(p_l,z) + cs_r/std::pow(p_r, z);
                        const double p_term = num/denom;
                        const double pStar  = std::pow(p_term, (1./z));

                        const double qL = 
                            (pStar <= p_l) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));

                        const double qR = 
                            (pStar <= p_r) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));

                        const double aL = vx_l - qL*cs_l;
                        const double aR = vx_r + qR*cs_r;

                        const double aStar = 
                            ( (p_r - p_l + rho_l*vx_l*(aL - vx_l) - rho_r*vx_r*(aR - vx_r) )/ 
                                (rho_l*(aL - vx_l) - rho_r*(aR - vx_r) ) );

                        return Eigenvals(aL, aR, aStar, pStar);
                    }

                case 2:
                    {
                        const double cbar   = 0.5*(cs_l + cs_r);
                        const double rhoBar = 0.5*(rho_l + rho_r);
                        const double z      = (gamma - 1.)/(2.*gamma);
                        const double num    = cs_l + cs_r - (gamma - 1.) * 0.5 *(vy_r - vy_l);
                        const double denom  = cs_l/std::pow(p_l,z) + cs_r/std::pow(p_r, z);
                        const double p_term = num/denom;
                        const double pStar  = std::pow(p_term, (1./z));

                        const double qL =
                            (pStar <= p_l) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));

                        const double qR = 
                            (pStar <= p_r) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));

                        const double aL = vy_l - qL*cs_l;
                        const double aR = vy_r + qR*cs_r;

                        const double aStar = 
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
                    const double vx_l = left_prims.v1;
                    const double vx_r = right_prims.v1;
                    const double p_r   = right_prims.p;
                    const double p_l   = left_prims.p;
                    const double rho_l = left_prims.rho;
                    const double rho_r = right_prims.rho;
                    const double cs_r = std::sqrt(gamma * p_r/rho_r);
                    const double cs_l = std::sqrt(gamma * p_l/rho_l);

                    const double aL = std::min({0.0, vx_l - cs_l, vx_r - cs_r});
                    const double aR = std::max({0.0, vx_l + cs_l, vx_r + cs_r});

                    return Eigenvals(aL, aR);
                }
                
            
            case 2:
                {
                    const double vy_l  = left_prims.v2;
                    const double vy_r  = right_prims.v2;
                    const double p_r   = right_prims.p;
                    const double p_l   = left_prims.p;
                    const double rho_l = left_prims.rho;
                    const double rho_r = right_prims.rho;
                    const double cs_r = std::sqrt(gamma * p_r/rho_r);
                    const double cs_l = std::sqrt(gamma * p_l/rho_l);
                    
                    
                    const double aL = std::min({0.0, vy_l - cs_l, vy_r - cs_r});
                    const double aR = std::max({0.0, vy_l + cs_l, vy_r + cs_r});
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
    const double rho = prims.rho;
    const double vx  = prims.v1;
    const double vy  = prims.v2;
    const double pre = prims.p;
    const double et  = pre/(gamma - 1.0) + 0.5 * rho * (vx*vx + vy*vy);

    return Conserved{rho, rho*vx, rho*vy, et};
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
void Newtonian2D::adapt_dt()
{
    double dx1, cs, dx2, rho, pressure, v1, v2, rmean;
    double min_dt, cfl_dt;
    int shift_i, shift_j, aid;

    min_dt = INFINITY;
    // Compute the minimum timestep given CFL
    // #pragma omp parallel for default(shared) reduction(min: min_dt) 
    for (int jj = 0; jj < yphysical_grid; jj++){
        dx2     = coord_lattice.dx2[jj];
        shift_j = jj + idx_active;
        for (int ii = 0; ii < xphysical_grid; ii++){
            shift_i = ii + idx_active;
            dx1 = coord_lattice.dx1[ii];
            aid = shift_j * NX + shift_i;
            rho      = prims[aid].rho;
            v1       = prims[aid].v1;
            v2       = prims[aid].v2;
            pressure = prims[aid].p;

            cs       = std::sqrt(gamma * pressure / rho );

            switch (geometry[coord_system])
            {
            case simbi::Geometry::CARTESIAN:
                cfl_dt = 
                    std::min( dx1/(std::max(std::abs(v1 + cs), std::abs(v1 - cs))), dx2/(std::max(std::abs(v2 + cs), std::abs(v2 - cs))) );

                break;
            
            case simbi::Geometry::SPHERICAL:
                rmean = coord_lattice.x1mean[ii];
                cfl_dt = 
                    std::min( dx1/(std::max(std::abs(v1 + cs), std::abs(v1 - cs))), rmean*dx2/(std::max(std::abs(v2 + cs), std::abs(v2 - cs))) );

                break;
            }

            min_dt = std::min(min_dt, cfl_dt);
        }
    }
    dt =  CFL * min_dt;
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
    double am = lambda.aL;
    double ap = lambda.aR;

    // if (am == ap)
    // {
    //     std::cout << "aP: " << ap << "aM: " << "\n";
    //     std::cin.get();
    // }
    
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

    const double aL    = lambda.aL;
    const double aR    = lambda.aR;

    // Quick checks before moving on with rest of computation
    if (0.0 <= aL){
        return left_flux;
    } else if (0.0 >= aR){
        return right_flux;
    }

    const double aStar = lambda.aStar;
    const double pStar = lambda.pStar;

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

void Newtonian2D::evolve()
{
    int xcoordinate, ycoordinate;

    Conserved ux_l, ux_r, uy_l, uy_r;
    Conserved f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    Primitive xprims_l, xprims_r, yprims_l, yprims_r;

    Primitive xleft_most, xleft_mid, xright_mid, xright_most;
    Primitive yleft_most, yleft_mid, yright_mid, yright_most;
    Primitive center;

    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    const int i_start = idx_active;
    const int j_start = idx_active;
    const int i_bound = x_bound;
    const int j_bound = y_bound;
    int aid;
    double dx, dy, rmean, dV1, dV2, s1L, s1R, s2L, s2R;
    double pc, rhoc, uc, vc;

    if (first_order)
    {
        // #pragma omp parallel for
        for (int jj = j_start; jj < j_bound; jj++)
        {
            ycoordinate = jj - 1;
            s2R = coord_lattice.x2_face_areas[ycoordinate + 1];
            s2L = coord_lattice.x2_face_areas[ycoordinate];
            for (int ii = i_start; ii < i_bound; ii++)
            {
                aid = jj * NX + ii;
                xcoordinate = ii - 1;

                // i+1/2
                ux_l = cons[ii + NX * jj];
                ux_r = cons[(ii + 1) + NX * jj];

                // j+1/2
                uy_l = cons[ii + NX * jj];
                uy_r = cons[ii + NX * (jj + 1)];

                xprims_l = prims[ii + jj * NX];
                xprims_r = prims[(ii + 1) + jj * NX];

                yprims_l = prims[ii + jj * NX];
                yprims_r = prims[ii + (jj + 1) * NX];

                f_l = calc_flux(xprims_l, 1);
                f_r = calc_flux(xprims_r, 1);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                // Calc HLL Flux at i+1/2 interface
                if (hllc)
                {
                    f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                // Set up the left and right state interfaces for i-1/2

                // i-1/2
                ux_l = cons[(ii - 1) + NX * jj];
                ux_r = cons[ii + NX * jj];

                // j-1/2
                uy_l = cons[ii + NX * (jj - 1)];
                uy_r = cons[ii + NX * jj];

                xprims_l = prims[(ii - 1) + jj * NX];
                xprims_r = prims[ii + jj * NX];

                yprims_l = prims[ii + (jj - 1) * NX];
                yprims_r = prims[ii + jj * NX];

                f_l = calc_flux(xprims_l, 1);
                f_r = calc_flux(xprims_r, 1);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                // Calc HLL Flux at i+1/2 interface
                if (hllc)
                {
                    f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                int real_loc = ycoordinate * xphysical_grid + xcoordinate;
                switch (geometry[coord_system])
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice.dx1[xcoordinate];
                    dy = coord_lattice.dx2[ycoordinate];
                    cons_n[aid].rho    += dt * (- (f1.rho - f2.rho)       / dx - (g1.rho - g2.rho)       / dy + sourceRho[real_loc]);
                    cons_n[aid].m1     += dt * (- (f1.m1 - f2.m1)         / dx - (g1.m1 - g2.m1)         / dy + sourceM1[real_loc]);
                    cons_n[aid].m2     += dt * (- (f1.m2 - f2.m2)         / dx - (g1.m2 - g2.m2)         / dy + sourceM2[real_loc]);
                    cons_n[aid].e_dens += dt * (- (f1.e_dens - f2.e_dens) / dx - (g1.e_dens - g2.e_dens) / dy + sourceE[real_loc]);
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    s1R   = coord_lattice.x1_face_areas[xcoordinate + 1];
                    s1L   = coord_lattice.x1_face_areas[xcoordinate];
                    rmean = coord_lattice.x1mean[xcoordinate];
                    dV1   = coord_lattice.dV1[xcoordinate];
                    dV2   = rmean * coord_lattice.dV2[ycoordinate];

                    pc   = prims[aid].p;
                    rhoc = prims[aid].rho, 
                    uc   = prims[aid].v1;
                    vc   = prims[aid].v2;

                    cons_n[aid] += Conserved{
                        // L(D)
                        -(f1.rho * s1R - f2.rho * s1L) / dV1 
                            - (g1.rho * s2R - g2.rho * s2L) / dV2 
                                + sourceRho[real_loc] * decay_const,

                        // L(S1)
                        -(f1.m1 * s1R - f2.m1 * s1L) / dV1 
                            - (g1.m1 * s2R - g2.m1 * s2L) / dV2 
                                + rhoc * vc * vc / rmean + 2 * pc / rmean +
                                    sourceM1[real_loc] * decay_const,

                        // L(S2)
                        -(f1.m2 * s1R - f2.m2 * s1L) / dV1
                            - (g1.m2 * s2R - g2.m2 * s2L) / dV2 
                                - (rhoc * uc * vc / rmean - pc * coord_lattice.cot[ycoordinate] / rmean) 
                                    + sourceM2[real_loc] * decay_const,

                        // L(tau)
                        -(f1.e_dens * s1R - f2.e_dens * s1L) / dV1 
                            - (g1.e_dens * s2R - g2.e_dens * s2L) / dV2 
                                + sourceE[real_loc] * decay_const
                    } * dt;
                    break;
                    // printf("\nCons after: %f, cons_n after: %f\n", cons[aid].m1, cons_n[aid].m1);
                    break;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel 
        {
        int ii, jj;
        for (jj = j_start; jj < j_bound; jj++)
        {
            ycoordinate = jj - 2;
            s2L         = coord_lattice.x2_face_areas[ycoordinate];
            s2R         = coord_lattice.x2_face_areas[ycoordinate + 1];
            #pragma omp for nowait
            for (ii = i_start; ii < i_bound; ii++)
            {
                // printf("i = %d, j= %d, threadId = %d \n", ii, jj, omp_get_thread_num());
                aid = jj * NX + ii;
                if (periodic)
                {
                    xcoordinate = ii;
                    ycoordinate = jj;

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
                }
                else
                {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;

                    // Coordinate X
                    xleft_most = prims[(ii - 2) + NX * jj];
                    xleft_mid = prims[(ii - 1) + NX * jj];
                    center = prims[ii + NX * jj];
                    xright_mid = prims[(ii + 1) + NX * jj];
                    xright_most = prims[(ii + 2) + NX * jj];

                    // Coordinate Y
                    yleft_most = prims[ii + NX * (jj - 2)];
                    yleft_mid = prims[ii + NX * (jj - 1)];
                    yright_mid = prims[ii + NX * (jj + 1)];
                    yright_most = prims[ii + NX * (jj + 2)];
                }

                // Reconstructed left X Primitive vector at the i+1/2 interface
                xprims_l.rho =
                    center.rho + 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                0.5 * (xright_mid.rho - xleft_mid.rho),
                                                plm_theta * (xright_mid.rho - center.rho));

                xprims_l.v1 =
                    center.v1 + 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                plm_theta * (xright_mid.v1 - center.v1));

                xprims_l.v2 =
                    center.v2 + 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                plm_theta * (xright_mid.v2 - center.v2));

                xprims_l.p =
                    center.p + 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                            0.5 * (xright_mid.p - xleft_mid.p),
                                            plm_theta * (xright_mid.p - center.p));

                // Reconstructed right Primitive vector in x
                xprims_r.rho =
                    xright_mid.rho -
                    0.5 * minmod(plm_theta * (xright_mid.rho - center.rho),
                                    0.5 * (xright_most.rho - center.rho),
                                    plm_theta * (xright_most.rho - xright_mid.rho));

                xprims_r.v1 = xright_mid.v1 -
                                0.5 * minmod(plm_theta * (xright_mid.v1 - center.v1),
                                            0.5 * (xright_most.v1 - center.v1),
                                            plm_theta * (xright_most.v1 - xright_mid.v1));

                xprims_r.v2 = xright_mid.v2 -
                                0.5 * minmod(plm_theta * (xright_mid.v2 - center.v2),
                                            0.5 * (xright_most.v2 - center.v2),
                                            plm_theta * (xright_most.v2 - xright_mid.v2));

                xprims_r.p = xright_mid.p -
                                0.5 * minmod(plm_theta * (xright_mid.p - center.p),
                                            0.5 * (xright_most.p - center.p),
                                            plm_theta * (xright_most.p - xright_mid.p));

                // Reconstructed right Primitive vector in y-direction at j+1/2
                // interfce
                yprims_l.rho =
                    center.rho + 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                0.5 * (yright_mid.rho - yleft_mid.rho),
                                                plm_theta * (yright_mid.rho - center.rho));

                yprims_l.v1 =
                    center.v1 + 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                plm_theta * (yright_mid.v1 - center.v1));

                yprims_l.v2 =
                    center.v2 + 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                plm_theta * (yright_mid.v2 - center.v2));

                yprims_l.p =
                    center.p + 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                            0.5 * (yright_mid.p - yleft_mid.p),
                                            plm_theta * (yright_mid.p - center.p));

                yprims_r.rho =
                    yright_mid.rho -
                    0.5 * minmod(plm_theta * (yright_mid.rho - center.rho),
                                    0.5 * (yright_most.rho - center.rho),
                                    plm_theta * (yright_most.rho - yright_mid.rho));

                yprims_r.v1 = yright_mid.v1 -
                                0.5 * minmod(plm_theta * (yright_mid.v1 - center.v1),
                                            0.5 * (yright_most.v1 - center.v1),
                                            plm_theta * (yright_most.v1 - yright_mid.v1));

                yprims_r.v2 = yright_mid.v2 -
                                0.5 * minmod(plm_theta * (yright_mid.v2 - center.v2),
                                            0.5 * (yright_most.v2 - center.v2),
                                            plm_theta * (yright_most.v2 - yright_mid.v2));

                yprims_r.p = yright_mid.p -
                                0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                            0.5 * (yright_most.p - center.p),
                                            plm_theta * (yright_most.p - yright_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l, 1);
                f_r = calc_flux(xprims_r, 1);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc)
                {
                    f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                // Left side Primitive in x
                xprims_l.rho = xleft_mid.rho +
                                0.5 * minmod(plm_theta * (xleft_mid.rho - xleft_most.rho),
                                            0.5 * (center.rho - xleft_most.rho),
                                            plm_theta * (center.rho - xleft_mid.rho));

                xprims_l.v1 = xleft_mid.v1 +
                                0.5 * minmod(plm_theta * (xleft_mid.v1 - xleft_most.v1),
                                            0.5 * (center.v1 - xleft_most.v1),
                                            plm_theta * (center.v1 - xleft_mid.v1));

                xprims_l.v2 = xleft_mid.v2 +
                                0.5 * minmod(plm_theta * (xleft_mid.v2 - xleft_most.v2),
                                            0.5 * (center.v2 - xleft_most.v2),
                                            plm_theta * (center.v2 - xleft_mid.v2));

                xprims_l.p =
                    xleft_mid.p + 0.5 * minmod(plm_theta * (xleft_mid.p - xleft_most.p),
                                                0.5 * (center.p - xleft_most.p),
                                                plm_theta * (center.p - xleft_mid.p));

                // Right side Primitive in x
                xprims_r.rho =
                    center.rho - 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                0.5 * (xright_mid.rho - xleft_mid.rho),
                                                plm_theta * (xright_mid.rho - center.rho));

                xprims_r.v1 =
                    center.v1 - 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                plm_theta * (xright_mid.v1 - center.v1));

                xprims_r.v2 =
                    center.v2 - 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                plm_theta * (xright_mid.v2 - center.v2));

                xprims_r.p =
                    center.p - 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                            0.5 * (xright_mid.p - xleft_mid.p),
                                            plm_theta * (xright_mid.p - center.p));

                // Left side Primitive in y
                yprims_l.rho = yleft_mid.rho +
                                0.5 * minmod(plm_theta * (yleft_mid.rho - yleft_most.rho),
                                            0.5 * (center.rho - yleft_most.rho),
                                            plm_theta * (center.rho - yleft_mid.rho));

                yprims_l.v1 = yleft_mid.v1 +
                                0.5 * minmod(plm_theta * (yleft_mid.v1 - yleft_most.v1),
                                            0.5 * (center.v1 - yleft_most.v1),
                                            plm_theta * (center.v1 - yleft_mid.v1));

                yprims_l.v2 = yleft_mid.v2 +
                                0.5 * minmod(plm_theta * (yleft_mid.v2 - yleft_most.v2),
                                            0.5 * (center.v2 - yleft_most.v2),
                                            plm_theta * (center.v2 - yleft_mid.v2));

                yprims_l.p =
                    yleft_mid.p + 0.5 * minmod(plm_theta * (yleft_mid.p - yleft_most.p),
                                                0.5 * (center.p - yleft_most.p),
                                                plm_theta * (center.p - yleft_mid.p));

                // Right side Primitive in y
                yprims_r.rho =
                    center.rho - 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                0.5 * (yright_mid.rho - yleft_mid.rho),
                                                plm_theta * (yright_mid.rho - center.rho));

                yprims_r.v1 =
                    center.v1 - 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                plm_theta * (yright_mid.v1 - center.v1));

                yprims_r.v2 =
                    center.v2 - 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                plm_theta * (yright_mid.v2 - center.v2));

                yprims_r.p =
                    center.p - 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                            0.5 * (yright_mid.p - yleft_mid.p),
                                            plm_theta * (yright_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = prims2cons(xprims_l);
                ux_r = prims2cons(xprims_r);

                uy_l = prims2cons(yprims_l);
                uy_r = prims2cons(yprims_r);

                f_l = calc_flux(xprims_l, 1);
                f_r = calc_flux(xprims_r, 1);

                g_l = calc_flux(yprims_l, 2);
                g_r = calc_flux(yprims_r, 2);

                if (hllc)
                {
                    f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                int real_loc = ycoordinate * xphysical_grid + xcoordinate;
                switch (geometry[coord_system])
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice.dx1[xcoordinate];
                    dy = coord_lattice.dx2[ycoordinate];
                    cons_n[aid].rho    += 0.5 * dt * (- (f1.rho - f2.rho)       / dx - (g1.rho - g2.rho)       / dy + sourceRho[real_loc]);
                    cons_n[aid].m1     += 0.5 * dt * (- (f1.m1 - f2.m1)         / dx - (g1.m1 - g2.m1)         / dy + sourceM1[real_loc]);
                    cons_n[aid].m2     += 0.5 * dt * (- (f1.m2 - f2.m2)         / dx - (g1.m2 - g2.m2)         / dy + sourceM2[real_loc]);
                    cons_n[aid].e_dens += 0.5 * dt * (- (f1.e_dens - f2.e_dens) / dx - (g1.e_dens - g2.e_dens) / dy + sourceE[real_loc]);
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    s1R   = coord_lattice.x1_face_areas[xcoordinate + 1];
                    s1L   = coord_lattice.x1_face_areas[xcoordinate];
                    rmean = coord_lattice.x1mean[xcoordinate];
                    dV1   = coord_lattice.dV1[xcoordinate];
                    dV2   = rmean * coord_lattice.dV2[ycoordinate];

                    pc   = prims[aid].p;
                    rhoc = prims[aid].rho, 
                    uc   = prims[aid].v1;
                    vc   = prims[aid].v2;
                    
                    // #pragma omp atomic
                    cons_n[aid] += Conserved{
                        // L(D)
                        -(f1.rho * s1R - f2.rho * s1L) / dV1 
                            - (g1.rho * s2R - g2.rho * s2L) / dV2 
                                + sourceRho[real_loc] * decay_const,

                        // L(S1)
                        -(f1.m1 * s1R - f2.m1 * s1L) / dV1 
                            - (g1.m1 * s2R - g2.m1 * s2L) / dV2 
                                + rhoc * vc * vc / rmean + 2 * pc / rmean +
                                    sourceM1[real_loc] * decay_const,

                        // L(S2)
                        -(f1.m2 * s1R - f2.m2 * s1L) / dV1
                            - (g1.m2 * s2R - g2.m2 * s2L) / dV2 
                                - (rhoc * uc * vc / rmean - pc * coord_lattice.cot[ycoordinate] / rmean) 
                                    + sourceM2[real_loc] * decay_const,

                        // L(tau)
                        -(f1.e_dens * s1R - f2.e_dens * s1L) / dV1 
                            - (g1.e_dens * s2R - g2.e_dens * s2L) / dV2 
                                + sourceE[real_loc] * decay_const
                    } * dt * 0.5;
                    break;
                }
            }
        }
        }
    }
};



//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
std::vector<std::vector<double> > Newtonian2D::simulate2D(
    const std::vector<std::vector<double>> sources,
    double tstart, 
    double tend, 
    double init_dt, 
    double plm_theta,
    double engine_duration, 
    double chkpt_interval,
    std::string data_directory, 
    bool first_order,
    bool periodic, 
    bool linspace, 
    bool hllc)
{

    std::string tnow, tchunk, tstep, filename;
    int nzones = NX * NY;

    double round_place = 1 / chkpt_interval;
    double t = tstart;
    double t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;

    this->nzones      = NX*NY;
    this->sources     = sources;
    this->first_order = first_order;
    this->periodic    = periodic;
    this->hllc        = hllc;
    this->linspace    = linspace;
    this->plm_theta   = plm_theta;
    this->dt          = init_dt;

    if (periodic){
        this->xphysical_grid = NX;
        this->yphysical_grid = NY;
        this->x_bound        = NX;
        this->y_bound        = NY;
        this->idx_active      = 0;
    } else {
        if (first_order)
        {
            this->xphysical_grid = NX - 2;
            this->yphysical_grid = NY - 2;
            this->idx_active = 1;
            this->x_bound = NX - 1;
            this->y_bound = NY - 1;
        }
        else
        {
            this->xphysical_grid = NX - 4;
            this->yphysical_grid = NY - 4;
            this->idx_active = 2;
            this->x_bound = NX - 2;
            this->y_bound = NY - 2;
        }
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
    config_system();
    if (hllc){
        solver = Solver::HLLC;
    } else {
        solver = Solver::HLLE;
    }

    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.xmax = x1[xphysical_grid - 1];
    setup.xmin = x1[0];
    setup.ymax = x2[yphysical_grid - 1];
    setup.ymin = x2[0];
    setup.NX = NX;
    setup.NY = NY;

    cons.resize(nzones);
    cons_n.resize(nzones);
    prims.resize(nzones);

    for (size_t i = 0; i < nzones; i++)
    {
        cons[i] = Conserved{
                init_state[0][i], 
                init_state[1][i], 
                init_state[2][i], 
                init_state[3][i]};
    }
    cons_n = cons;

    sourceRho = sources[0];
    sourceM1  = sources[1];
    sourceM2  = sources[2];
    sourceE   = sources[3];
    
    high_resolution_clock::time_point t1, t2;
    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1.0 / (1.0 + exp(10.0 * (tstart - engine_duration)));

    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag, num_zeros;

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    hydro2d::PrimitiveData transfer_prims;

    if (t == 0)
    {
        config_ghosts2D(cons, NX, NY, first_order);
    }

    if (first_order)
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            t1 = high_resolution_clock::now();

            cons2prim();
            evolve();
            config_ghosts2D(cons_n, NX, NY, true);
            cons = cons_n;
            t += dt;

            /* Compute the loop execution time */
            t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                 << "dt: " << std::setw(5) << dt << "\t"
                 << "t: "  << std::setw(5) << t << "\t"
                 << "Zones per sec: " << nzones / time_span.count() << std::flush;
            
            adapt_dt();
            n++;
        }
    }
    else
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            t1 = high_resolution_clock::now();

            // First half step
            cons2prim();
            evolve();
            config_ghosts2D(cons_n, NX, NY, false);
            cons = cons_n;

            // Final half step
            cons2prim();
            evolve();
            config_ghosts2D(cons_n, NX, NY, false);
            cons = cons_n;

            t += dt;

            t2 = high_resolution_clock::now();
            auto time_span = duration_cast<duration<double>>(t2 - t1);
            
            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                 << "dt: " << std::setw(5) << dt << "\t"
                 << "t: "  << std::setw(5) << t << "\t"
                 << "Zones per sec: " << nzones / time_span.count() << std::flush;

            
            decay_const = 1.0 / (1.0 + exp(10.0 * (t - engine_duration)));

            /* Write to a File every nth of a second */
            if (t >= t_interval)
            {
                // Check if time order of magnitude exceeds 
                // the hundreds place set by the tchunk std::string
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vec2struct<hydro2d::PrimitiveData, Primitive>(prims);
                writeToProd<hydro2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow           = create_step_str(t_interval, tchunk);
                filename       = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t        = t;
                setup.dt       = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, nzones);
                t_interval += chkpt_interval;
            }
            adapt_dt();
            n++;
        }
    }
    cons2prim();
    std::cout << "\n ";
    std::vector<std::vector<double> > solution(4, std::vector<double>(nzones));
    for (size_t ii = 0; ii < nzones; ii++)
    {
        solution[0][ii] = cons[ii].rho;
        solution[1][ii] = cons[ii].m1;
        solution[2][ii] = cons[ii].m2;
        solution[3][ii] = cons[ii].e_dens;
    }

    return solution;

 };