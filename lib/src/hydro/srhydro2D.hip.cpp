/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "util/device_api.hpp"
#include "util/dual.hpp"
#include "common/helpers.hpp"
#include "helpers.hip.hpp"
#include "srhydro2D.hip.hpp"
#include "util/printb.hpp"
#include "util/parallel_for.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace std::chrono;

// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(std::vector<std::vector<real>> state2D, int nx, int ny, real gamma,
               std::vector<real> x1, std::vector<real> x2, real CFL,
               std::string coord_system = "cartesian")
:

    nx(nx),
    ny(ny),
    nzones(state2D[0].size()),
    state2D(state2D),
    gamma(gamma),
    x1(x1),
    x2(x2),
    CFL(CFL),
    coord_system(coord_system)
{
}

// Destructor
SRHD2D::~SRHD2D() {}

/* Define typedefs because I am lazy */
typedef sr2d::Primitive Primitive;
typedef sr2d::Conserved Conserved;
typedef sr2d::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD2D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const unsigned int nhat = 1)
{
    // Eigenvals lambda;

    // Separate the left and right Primitive
    const real rho_l = prims_l.rho;
    const real p_l = prims_l.p;
    const real h_l = 1. + gamma * p_l / (rho_l * (gamma - 1));

    const real rho_r = prims_r.rho;
    const real p_r = prims_r.p;
    const real h_r = 1. + gamma * p_r / (rho_r * (gamma - 1));

    const real cs_r = sqrt(gamma * p_r / (h_r * rho_r));
    const real cs_l = sqrt(gamma * p_l / (h_l * rho_l));

    switch (nhat)
    {
    case 1:
    {
        const real v1_l = prims_l.v1;
        const real v1_r = prims_r.v1;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const real vbar  = (real)0.5 * (v1_l + v1_r);
        const real cbar  = (real)0.5 * (cs_l + cs_r);
        const real bl    = (vbar - cbar)/(1. - cbar*vbar);
        const real br    = (vbar + cbar)/(1. + cbar*vbar);
        const real aL    = my_min(bl, (v1_l - cs_l)/(1. - v1_l*cs_l));
        const real aR    = my_max(br, (v1_r + cs_r)/(1. + v1_r*cs_r));

        //--------Calc the wave speeds based on Mignone and Bodo (2005)
        // const real sL = cs_l * cs_l * (1. / (gamma * gamma * (1 - cs_l * cs_l)));
        // const real sR = cs_r * cs_r * (1. / (gamma * gamma * (1 - cs_r * cs_r)));

        // // Define temporaries to save computational cycles
        // const real qfL = 1. / (1. + sL);
        // const real qfR = 1. / (1. + sR);
        // const real sqrtR = sqrt(sR * (1 - v1_r * v1_r + sR));
        // const real sqrtL = sqrt(sL * (1 - v1_l * v1_l + sL));

        // const real lamLm = (v1_l - sqrtL) * qfL;
        // const real lamRm = (v1_r - sqrtR) * qfR;
        // const real lamLp = (v1_l + sqrtL) * qfL;
        // const real lamRp = (v1_r + sqrtR) * qfR;

        // const real aL = lamLm < lamRm ? lamLm : lamRm;
        // const real aR = lamLp > lamRp ? lamLp : lamRp;

        return Eigenvals(aL, aR, cs_l, cs_r);
    }
    case 2:
        const real v2_r = prims_r.v2;
        const real v2_l = prims_l.v2;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const real vbar  = (real)0.5 * (v2_l + v2_r);
        const real cbar  = (real)0.5 * (cs_l + cs_r);
        const real bl    = (vbar - cbar)/(1. - cbar*vbar);
        const real br    = (vbar + cbar)/(1. + cbar*vbar);
        const real aL    = my_min(bl, (v2_l - cs_l)/(1. - v2_l*cs_l));
        const real aR    = my_max(br, (v2_r + cs_r)/(1. + v2_r*cs_r));

        // return Eigenvals(aL, aR);

        // Calc the wave speeds based on Mignone and Bodo (2005)
        // real sL = cs_l * cs_l * ((real)1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
        // real sR = cs_r * cs_r * ((real)1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

        // // Define some temporaries to save a few cycles
        // const real qfL = 1. / (1. + sL);
        // const real qfR = 1. / (1. + sR);
        // const real sqrtR = sqrt(sR * (1 - v2_r * v2_r + sR));
        // const real sqrtL = sqrt(sL * (1 - v2_l * v2_l + sL));

        // const real lamLm = (v2_l - sqrtL) * qfL;
        // const real lamRm = (v2_r - sqrtR) * qfR;
        // const real lamLp = (v2_l + sqrtL) * qfL;
        // const real lamRp = (v2_r + sqrtR) * qfR;
        // const real aL = lamLm < lamRm ? lamLm : lamRm;
        // const real aR = lamLp > lamRp ? lamLp : lamRp;

        return Eigenvals(aL, aR, cs_l, cs_r);
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1 - (vx * vx + vy * vy));
    const real h = 1. + gamma * pressure / (rho * (gamma - 1.));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * vx,
        rho * h * lorentz_gamma * lorentz_gamma * vy,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma};
};

Conserved SRHD2D::calc_intermed_statesSR2D(const Primitive &prims,
                                           const Conserved &state, real a,
                                           real aStar, real pStar,
                                           int nhat = 1)
{
    real Dstar, S1star, S2star, tauStar, Estar, cofactor;
    Conserved starStates;

    real pressure = prims.p;
    real v1 = prims.v1;
    real v2 = prims.v2;

    real D = state.D;
    real S1 = state.S1;
    real S2 = state.S2;
    real tau = state.tau;
    real E = tau + D;

    switch (nhat)
    {
    case 1:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v1) * D;
        S1star = cofactor * (S1 * (a - v1) - pressure + pStar);
        S2star = cofactor * (a - v1) * S2;
        Estar = cofactor * (E * (a - v1) + pStar * aStar - pressure * v1);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    case 2:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v2) * D;
        S1star = cofactor * (a - v2) * S1;
        S2star = cofactor * (S2 * (a - v2) - pressure + pStar);
        Estar = cofactor * (E * (a - v2) + pStar * aStar - pressure * v2);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    }

    return starStates;
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------

// Adapt the CFL conditonal timestep
void SRHD2D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real dx1, cs, dx2, rho, pressure, v1, v2, rmean, h;
        real cfl_dt;
        int shift_i, shift_j;
        real plus_v1, plus_v2, minus_v1, minus_v2;
        int aid; // active index id

        // Compute the minimum timestep given CFL
        for (int jj = 0; jj < yphysical_grid; jj++)
        {
            dx2 = coord_lattice.dx2[jj];
            shift_j = jj + idx_active;
            #pragma omp for schedule(static)
            for (int ii = 0; ii < xphysical_grid; ii++)
            {
                shift_i  = ii + idx_active;
                aid      = shift_i + nx * shift_j;
                dx1      = coord_lattice.dx1[ii];
                rho      = prims[aid].rho;
                v1       = prims[aid].v1;
                v2       = prims[aid].v2;
                pressure = prims[aid].p;

                h = 1. + gamma * pressure / (rho * (gamma - 1.));
                cs = sqrt(gamma * pressure / (rho * h));

                plus_v1  = (v1 + cs) / (1. + v1 * cs);
                plus_v2  = (v2 + cs) / (1. + v2 * cs);
                minus_v1 = (v1 - cs) / (1. - v1 * cs);
                minus_v2 = (v2 - cs) / (1. - v2 * cs);

                if (coord_system == "cartesian")
                {

                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }
                else
                {
                    rmean = coord_lattice.x1mean[ii];
                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                rmean * dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }

                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
                
            } // end ii 
        } // end jj
    } // end parallel region

    dt = CFL * min_dt;
};

void SRHD2D::adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p)
{
    #if GPU_CODE
    {
        dtWarpReduce<SRHD2D, Primitive, 128><<<p.gridSize, dim3(BLOCK_SIZE2D, BLOCK_SIZE2D)>>>(dev, geometry);
        simbi::gpu::api::deviceSynch();
        simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
    #endif
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2flux(const Primitive &prims, unsigned int nhat = 1)
{

    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1. - (vx * vx + vy * vy));

    const real h = 1. + gamma * pressure / (rho * (gamma - 1));
    const real D = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const real tau =
                    rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return (nhat == 1) ? Conserved(D * vx, S1 * vx + pressure, S2 * vx,
                                   (tau + pressure) * vx)
                       : Conserved(D * vy, S1 * vy, S2 * vy + pressure,
                                   (tau + pressure) * vy);
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const unsigned int nhat)
{
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    // Calculate plus/minus alphas
    const real aLminus = aL < (real)0.0 ? aL : (real)0.0;
    const real aRplus  = aR > (real)0.0 ? aR : (real)0.0;

    // Compute the HLL Flux component-wise
    return (left_flux * aRplus - right_flux * aLminus 
                + (right_state - left_state) * aRplus * aLminus) /
                    (aRplus - aLminus);
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const unsigned int nhat = 1)
{

    Conserved starStateR, starStateL;
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if ((real)0.0 <= aL)
    {
        return left_flux;
    }
    else if ((real)0.0 >= aR)
    {
        return right_flux;
    }

    const real aLminus = aL < (real)0.0 ? aL : (real)0.0;
    const real aRplus  = aR > (real)0.0 ? aR : (real)0.0;

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aR - left_state * aL - right_flux + left_flux) / (aR - aL);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux 
        = (left_flux * aRplus - right_flux * aLminus + (right_state - left_state) * aRplus * aLminus) 
            / (aRplus - aLminus);

    //------ Mignone & Bodo subtract off the rest mass density
    const real e  = hll_state.tau + hll_state.D;
    const real s  = hll_state.momentum(nhat);
    const real fe = hll_flux.tau + hll_flux.D;
    const real fs = hll_flux.momentum(nhat);

    //------Calculate the contact wave velocity and pressure
    const real a = fe;
    const real b = -(e + fs);
    const real c = s;
    const real quad = -(real)0.5 * (b + sgn(b) * sqrt(b * b - 4.0 * a * c));
    const real aStar = c * ((real)1.0 / quad);
    const real pStar = -aStar * fe + fs;

    // Apply the low-Mach HLLC fix found in: 
    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
    const real cL       = lambda.csL;
    const real cR       = lambda.csR;
    const real ma_lim   = 0.2;

    //--------------Compute the L Star State----------
    switch (nhat)
    {
    case 1:
        {
            real pressure = left_prims.p;
            real D        = left_state.D;
            real S1       = left_state.S1;
            real S2       = left_state.S2;
            real tau      = left_state.tau;
            real E        = tau + D;
            real cofactor = 1. / (aL - aStar);

            real v1 = left_prims.v1;
            // Left Star State in x-direction of coordinate lattice
            real Dstar    = cofactor * (aL - v1) * D;
            real S1star   = cofactor * (S1 * (aL - v1) - pressure + pStar);
            real S2star   = cofactor * (aL - v1) * S2;
            real Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
            real tauStar  = Estar - Dstar;
            starStateL    = Conserved(Dstar, S1star, S2star, tauStar);

            pressure = right_prims.p;
            D        = right_state.D;
            S1       = right_state.S1;
            S2       = right_state.S2;
            tau      = right_state.tau;
            E        = tau + D;
            cofactor = 1. / (aR - aStar);

            v1      = right_prims.v1;
            Dstar   = cofactor * (aR - v1) * D;
            S1star  = cofactor * (S1 * (aR - v1) - pressure + pStar);
            S2star  = cofactor * (aR - v1) * S2;
            Estar   = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
            tauStar = Estar - Dstar;
            starStateR = Conserved(Dstar, S1star, S2star, tauStar);

            const real ma_local = my_max(std::abs(left_prims.v1 / cL), std::abs(right_prims.v1 / cR));
            const real phi      = sin(my_min((real)1.0, ma_local / ma_lim) * PI * (real)0.5);
            const real aL_lm    = (phi != 0) ? phi * aL : aL;
            const real aR_lm    = (phi != 0) ? phi * aR : aR;

            return (left_flux + right_flux) * (real)0.5 + ( (starStateL - left_state) * aL_lm
                + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm ) * (real)0.5;
        }
        break;
    
    case 2:
        {
            real pressure = left_prims.p;
            real D        = left_state.D;
            real S1       = left_state.S1;
            real S2       = left_state.S2;
            real tau      = left_state.tau;
            real E        = tau + D;
            real cofactor = 1. / (aL - aStar);

            real v2 = left_prims.v2;
            // Start States in y-direction in the coordinate lattice
            real Dstar   = cofactor * (aL - v2) * D;
            real S1star  = cofactor * (aL - v2) * S1;
            real S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
            real Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
            real tauStar = Estar - Dstar;

            starStateL = Conserved(Dstar, S1star, S2star, tauStar);
            v2 = right_prims.v2;
            // Start States in y-direction in the coordinate lattice
            pressure = right_prims.p;
            D        = right_state.D;
            S1       = right_state.S1;
            S2       = right_state.S2;
            tau      = right_state.tau;
            E        = tau + D;
            cofactor = 1. / (aR - aStar);

            Dstar    = cofactor * (aR - v2) * D;
            S1star   = cofactor * (aR - v2) * S1;
            S2star   = cofactor * (S2 * (aR - v2) - pressure + pStar);
            Estar    = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
            tauStar  = Estar - Dstar;

            starStateR = Conserved(Dstar, S1star, S2star, tauStar);

            const real ma_local = my_max(std::abs(left_prims.v2 / cL), std::abs(right_prims.v2 / cR));
            const real phi      = sin(my_min((real)1.0, ma_local / ma_lim) * PI * (real)0.5);
            const real aL_lm    = (phi != 0) ? phi * aL : aL;
            const real aR_lm    = (phi != 0) ? phi * aR : aR;

            return (left_flux + right_flux) * (real)0.5 + ( (starStateL - left_state) * aL_lm
                + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm ) * (real)0.5;
        }
        break;
    }

    // return Conserved(0.0, 0.0, 0.0, 0.0);
    // if (-aL <= (aStar - aL))
    // {
    //     const real pressure = left_prims.p;
    //     const real D = left_state.D;
    //     const real S1 = left_state.S1;
    //     const real S2 = left_state.S2;
    //     const real tau = left_state.tau;
    //     const real E = tau + D;
    //     const real cofactor = 1. / (aL - aStar);
    //     //--------------Compute the L Star State----------
    //     switch (nhat)
    //     {
    //     case 1:
    //     {
    //         const real v1 = left_prims.v1;
    //         // Left Star State in x-direction of coordinate lattice
    //         const real Dstar    = cofactor * (aL - v1) * D;
    //         const real S1star   = cofactor * (S1 * (aL - v1) - pressure + pStar);
    //         const real S2star   = cofactor * (aL - v1) * S2;
    //         const real Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
    //         const real tauStar  = Estar - Dstar;

    //         const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

    //         //---------Compute the L Star Flux
    //         return left_flux + (interstate_left - left_state) * aL;
    //     }

    //     case 2:
    //         const real v2 = left_prims.v2;
    //         // Start States in y-direction in the coordinate lattice
    //         const real Dstar   = cofactor * (aL - v2) * D;
    //         const real S1star  = cofactor * (aL - v2) * S1;
    //         const real S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
    //         const real Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

    //         //---------Compute the L Star Flux
    //         return left_flux + (interstate_left - left_state) * aL;
    //     }
    // }
    // else
    // {
    //     const real pressure = right_prims.p;
    //     const real D = right_state.D;
    //     const real S1 = right_state.S1;
    //     const real S2 = right_state.S2;
    //     const real tau = right_state.tau;
    //     const real E = tau + D;
    //     const real cofactor = 1. / (aR - aStar);

    //     /* Compute the L/R Star State */
    //     switch (nhat)
    //     {
    //     case 1:
    //     {
    //         const real v1 = right_prims.v1;
    //         const real Dstar = cofactor * (aR - v1) * D;
    //         const real S1star = cofactor * (S1 * (aR - v1) - pressure + pStar);
    //         const real S2star = cofactor * (aR - v1) * S2;
    //         const real Estar = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

    //         // Compute the intermediate right flux
    //         return right_flux + (interstate_right - right_state) * aR;
    //     }

    //     case 2:
    //         const real v2 = right_prims.v2;
    //         // Start States in y-direction in the coordinate lattice
    //         const real cofactor = 1. / (aR - aStar);
    //         const real Dstar = cofactor * (aR - v2) * D;
    //         const real S1star = cofactor * (aR - v2) * S1;
    //         const real S2star = cofactor * (S2 * (aR - v2) - pressure + pStar);
    //         const real Estar = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

    //         // Compute the intermediate right flux
    //         return right_flux + (interstate_right - right_state) * aR;
    //     }
    // }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================

void SRHD2D::cons2prim(
    ExecutionPolicy<> p, 
    SRHD2D *dev, 
    simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, 0, nzones, [=] GPU_LAMBDA (int gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        #if GPU_CODE
        extern __shared__ Conserved  conserved_buff[];
        #else
        auto* const conserved_buff = &cons[0];
        #endif 

        auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * threadIdx.y + threadIdx.x : gid;
        // load shared memory
        if constexpr(BuildPlatform == Platform::GPU)
            conserved_buff[tid] = self->gpu_cons[gid];
            
        simbi::gpu::api::synchronize();
        int iter  = 0;
        real D    = conserved_buff[tid].D;
        real S1   = conserved_buff[tid].S1;
        real S2   = conserved_buff[tid].S2;
        real tau  = conserved_buff[tid].tau;
        real S    = sqrt(S1 * S1 + S2 * S2);

        #if GPU_CODE
        real peq = self->gpu_pressure_guess[gid];
        #else 
        real peq = self->pressure_guess[gid];
        #endif

        real tol = D * tol_scale;
        do
        {
            pre = peq;
            et  = tau + D + pre;
            v2 = S * S / (et * et);
            W   = (real)1.0 / sqrt((real)1.0 - v2);
            rho = D / W;

            eps = (tau + ((real)1.0 - W) * D + (1. - W * W) * pre) / (D * W);

            h = 1 + eps + pre / rho;
            c2 = self->gamma * pre / (h * rho);

            g = c2 * v2 - (real)1.0;
            f = (self->gamma - (real)1.0) * rho * eps - pre;

            peq = pre - f / g;
            iter++;
            if (iter >= MAX_ITER)
            {
                printf("\nCons2Prim cannot converge\n");
                self->dt = INFINITY;
                return;
            }

        } while (std::abs(peq - pre) >= tol);

        real inv_et = 1. / (tau + D + peq);
        real vx     = S1 * inv_et;
        real vy     = S2 * inv_et;

        #if GPU_CODE
            self->gpu_pressure_guess[gid] = peq;
            self->gpu_prims[gid]          = Primitive{D * sqrt(1 - (vx * vx + vy * vy)), vx, vy, peq};
        #else
            self->pressure_guess[gid] = peq;
            self->prims[gid]          = Primitive{D * sqrt(1 - (vx * vx + vy * vy)), vx, vy, peq};
        #endif
        

    });
}

void SRHD2D::advance(
    SRHD2D *dev, 
    const ExecutionPolicy<> p,
    const int bx,
    const int by,
    const int radius, 
    const simbi::Geometry geometry, 
    const simbi::MemSide user)
{
    auto *self = (BuildPlatform == Platform::GPU) ? dev : this;
    const int xpg                   = this->xphysical_grid;
    const int ypg                   = this->yphysical_grid;
    const bool is_first_order       = this->first_order;
    const bool is_periodic          = this->periodic;
    const bool hllc                 = this->hllc;
    const real dt                   = this->dt;
    const real decay_const          = this->decay_const;
    const real plm_theta            = this->plm_theta;
    const real gamma                = this->gamma;
    const int nx                    = this->nx;
    const int ny                    = this->ny;
    const int extent                = (BuildPlatform == Platform::GPU) ? p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const int xextent               = p.blockSize.x;
    const int yextent               = p.blockSize.y;

    const CLattice2D *coord_lattice = &(self->coord_lattice);
    const int bs                    = (BuildPlatform == Platform::GPU) ? bx : nx;
    const int nbs                   = (BuildPlatform == Platform::GPU) ? bx * by : nzones;

    simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = &prims[0];
        #endif 

        const int ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const int jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        if     constexpr(BuildPlatform == Platform::GPU) if ((ii >= xpg) || (jj >= ypg)) return;

        const int ia  = ii + radius;
        const int ja  = jj + radius;
        const int tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const int ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const int txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const int tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;

        // printf("(%d, %d) -- rho = %f\n", ia, ja, prim_buff[bs * ja + ia].rho);

        Conserved ux_l, ux_r, uy_l, uy_r;
        Conserved f_l, f_r, g_l, g_r, f1, f2, g1, g2;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r;

        int aid = ja * nx + ia;
        if  constexpr(BuildPlatform == Platform::GPU)
        {
            int txl = xextent;
            int tyl = yextent;

            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * bx + txa] = self->gpu_prims[aid];
            if (ty < radius)
            {
                if (ja + yextent > ny - 1) tyl = ny - radius - ja + threadIdx.y;
                prim_buff[(tya - radius) * bx + txa] = self->gpu_prims[(ja - radius) * nx + ia];
                prim_buff[(tya + tyl   ) * bx + txa] = self->gpu_prims[(ja + tyl   ) * nx + ia]; 
            
            }
            if (tx < radius)
            {   
                if (ia + xextent > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[tya * bx + txa - radius] =  self->gpu_prims[ja * nx + ia - radius];
                prim_buff[tya * bx + txa +    txl] =  self->gpu_prims[ja * nx + ia + txl]; 
            }
            simbi::gpu::api::synchronize();
        }

        if (is_first_order)
        {
            if (is_periodic)
            {
                xprims_l = prim_buff[txa + tya * bx];
                xprims_r = roll(prim_buff, (txa + 1) + tya * bx, nbs);

                yprims_l = prim_buff[txa + tya * bx];
                yprims_r = roll(prim_buff, txa + (tya + 1) * bx, nbs);
            }
            else
            {
                xprims_l = prim_buff[tya * bx + (txa + 0)];
                xprims_r = prim_buff[tya * bx + (txa + 1)];
                //j+1/2
                yprims_l = prim_buff[(tya + 0) * bx + txa];
                yprims_r = prim_buff[(tya + 1) * bx + txa];
            }
            
            // i+1/2
            ux_l = self->prims2cons(xprims_l); 
            ux_r = self->prims2cons(xprims_r); 
            // j+1/2
            uy_l = self->prims2cons(yprims_l);  
            uy_r = self->prims2cons(yprims_r); 

            f_l = self->prims2flux(xprims_l, 1);
            f_r = self->prims2flux(xprims_r, 1);

            g_l = self->prims2flux(yprims_l, 2);
            g_r = self->prims2flux(yprims_r, 2);

            // Calc HLL Flux at i+1/2 interface
            if (hllc)
            {
                // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                //     f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // } else {
                //     f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // }
                
                // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                //     g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // } else {
                //     g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // }
                f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            // Set up the left and right state interfaces for i-1/2
            if (is_periodic)
            {
                xprims_l = roll(prim_buff,  txa - 1 + tya * bx, nbs);
                xprims_r = prim_buff[txa + tya * bx];

                yprims_l = roll(prim_buff, txa + (tya - 1) * bx, nbs);
                yprims_r = prim_buff[txa + tya * bx];
            }
            else
            {
                xprims_l = prim_buff[tya * bx + (txa - 1)];
                xprims_r = prim_buff[tya * bx + (txa + 0)];
                //j+1/2
                yprims_l = prim_buff[(tya - 1) * bx + txa]; 
                yprims_r = prim_buff[(tya + 0) * bx + txa]; 
            }

            // i+1/2
            ux_l = self->prims2cons(xprims_l); 
            ux_r = self->prims2cons(xprims_r); 
            // j+1/2
            uy_l = self->prims2cons(yprims_l);  
            uy_r = self->prims2cons(yprims_r); 

            f_l = self->prims2flux(xprims_l, 1);
            f_r = self->prims2flux(xprims_r, 1);

            g_l = self->prims2flux(yprims_l, 2);
            g_r = self->prims2flux(yprims_r, 2);

            // Calc HLL Flux at i-1/2 interface
            if (hllc)
            {
                // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                //     f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // } else {
                //     f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // }
                
                // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                //     g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // } else {
                //     g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // }
                f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

            } else {
                f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            //Advance depending on geometry
            int real_loc = jj * xpg + ii;
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            real dx = coord_lattice->gpu_dx1[ii];
                            real dy = coord_lattice->gpu_dx2[jj];
                            self->gpu_cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + self->gpu_sourceD [real_loc] );
                            self->gpu_cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + self->gpu_sourceS1[real_loc] );
                            self->gpu_cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + self->gpu_sourceS2[real_loc] );
                            self->gpu_cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + self->gpu_sourceTau [real_loc]);
                        #else
                            real dx = coord_lattice->dx1[ii];
                            real dy = coord_lattice->dx2[jj];
                            cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + sourceD   [real_loc] );
                            cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + sourceS1  [real_loc] );
                            cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + sourceS2  [real_loc] );
                            cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + sourceTau [real_loc] );
                        #endif
                    

                    break;
                    }
                
                case simbi::Geometry::SPHERICAL:
                    {
                    #if GPU_CODE
                    real s1R        = coord_lattice->gpu_x1_face_areas[ii + 1];
                    real s1L        = coord_lattice->gpu_x1_face_areas[ii + 0];
                    real s2R        = coord_lattice->gpu_x2_face_areas[jj + 1];
                    real s2L        = coord_lattice->gpu_x2_face_areas[jj + 0];
                    real rmean      = coord_lattice->gpu_x1mean[ii]           ;
                    real dV1        = coord_lattice->gpu_dV1[ii]              ;
                    real dV2        = rmean * coord_lattice->gpu_dV2[jj]      ;
                    #else 
                    real s1R   = coord_lattice->x1_face_areas[ii + 1];
                    real s1L   = coord_lattice->x1_face_areas[ii + 0];
                    real s2R   = coord_lattice->x2_face_areas[jj + 1];
                    real s2L   = coord_lattice->x2_face_areas[jj + 0];
                    real rmean = coord_lattice->x1mean[ii];
                    real dV1   = coord_lattice->dV1[ii];
                    real dV2   = rmean * coord_lattice->dV2[jj];
                    #endif
                    // Grab central primitives
                    real rhoc = prim_buff[tya * bx + txa].rho;
                    real pc   = prim_buff[tya * bx + txa].p;
                    real uc   = prim_buff[tya * bx + txa].v1;
                    real vc   = prim_buff[tya * bx + txa].v2;

                    real hc   = (real)1.0 + gamma * pc/(rhoc * (gamma - (real)1.0));
                    real gam2 = (real)1.0/((real)1.0 - (uc * uc + vc * vc));

                    #if GPU_CODE 
                        self->gpu_cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + self->gpu_sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + (real)(real)2.0 * pc / rmean +
                                            self->gpu_sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[jj] / rmean) 
                                            + self->gpu_sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + self->gpu_sourceTau[real_loc] * decay_const
                            } * dt;
                    #else
                        cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + self->sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + (real)2.0 * pc / rmean +
                                            self->sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice->cot[jj] / rmean) 
                                            + self->sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + self->sourceTau[real_loc] * decay_const
                            } * dt;
                    #endif
                    
                    break;
                    }
            } // end switch
                
        }
        else
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            if (!is_periodic)
            {
                // Coordinate X
                xleft_most  = prim_buff[(txa - 2) + bs * tya];
                xleft_mid   = prim_buff[(txa - 1) + bs * tya];
                center      = prim_buff[ txa      + bs * tya];
                xright_mid  = prim_buff[(txa + 1) + bs * tya];
                xright_most = prim_buff[(txa + 2) + bs * tya];

                // Coordinate Y
                yleft_most  = prim_buff[txa + bs * (tya - 2)];
                yleft_mid   = prim_buff[txa + bs * (tya - 1)];
                yright_mid  = prim_buff[txa + bs * (tya + 1)];
                yright_most = prim_buff[txa + bs * (tya + 2)];
            }
            else
            {
                // X Coordinate
                xleft_most   = roll(prim_buff, tya * bx + txa - 2, nbs);
                xleft_mid    = roll(prim_buff, tya * bx + txa - 1, nbs);
                center       = prim_buff[tya * bx + txa];
                xright_mid   = roll(prim_buff, tya * bx + txa + 1, nbs);
                xright_most  = roll(prim_buff, tya * bx + txa + 2, nbs);

                yleft_most   = roll(prim_buff, txa +  bs * (tya - 2), nbs);
                yleft_mid    = roll(prim_buff, txa +  bs * (tya - 1), nbs);
                yright_mid   = roll(prim_buff, txa +  bs * (tya + 1), nbs);
                yright_most  = roll(prim_buff, txa +  bs * (tya + 2), nbs);
            }
                // Reconstructed left X Primitive vector at the i+1/2 interface
                xprims_l.rho =
                    center.rho + (real)0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                (real)0.5 * (xright_mid.rho - xleft_mid.rho),
                                                plm_theta * (xright_mid.rho - center.rho));

                xprims_l.v1 =
                    center.v1 + (real)0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                (real)0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                plm_theta * (xright_mid.v1 - center.v1));

                xprims_l.v2 =
                    center.v2 + (real)0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                (real)0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                plm_theta * (xright_mid.v2 - center.v2));

                xprims_l.p =
                    center.p + (real)0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                            (real)0.5 * (xright_mid.p - xleft_mid.p),
                                            plm_theta * (xright_mid.p - center.p));

                // Reconstructed right Primitive vector in x
                xprims_r.rho =
                    xright_mid.rho -
                    (real)0.5 * minmod(plm_theta * (xright_mid.rho - center.rho),
                                    (real)0.5 * (xright_most.rho - center.rho),
                                    plm_theta * (xright_most.rho - xright_mid.rho));

                xprims_r.v1 = xright_mid.v1 -
                                (real)0.5 * minmod(plm_theta * (xright_mid.v1 - center.v1),
                                            (real)0.5 * (xright_most.v1 - center.v1),
                                            plm_theta * (xright_most.v1 - xright_mid.v1));

                xprims_r.v2 = xright_mid.v2 -
                                (real)0.5 * minmod(plm_theta * (xright_mid.v2 - center.v2),
                                            (real)0.5 * (xright_most.v2 - center.v2),
                                            plm_theta * (xright_most.v2 - xright_mid.v2));

                xprims_r.p = xright_mid.p -
                                (real)0.5 * minmod(plm_theta * (xright_mid.p - center.p),
                                            (real)0.5 * (xright_most.p - center.p),
                                            plm_theta * (xright_most.p - xright_mid.p));

                // Reconstructed right Primitive vector in y-direction at j+1/2
                // interfce
                yprims_l.rho =
                    center.rho + (real)0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                (real)0.5 * (yright_mid.rho - yleft_mid.rho),
                                                plm_theta * (yright_mid.rho - center.rho));

                yprims_l.v1 =
                    center.v1 + (real)0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                (real)0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                plm_theta * (yright_mid.v1 - center.v1));

                yprims_l.v2 =
                    center.v2 + (real)0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                (real)0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                plm_theta * (yright_mid.v2 - center.v2));

                yprims_l.p =
                    center.p + (real)0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                            (real)0.5 * (yright_mid.p - yleft_mid.p),
                                            plm_theta * (yright_mid.p - center.p));

                yprims_r.rho =
                    yright_mid.rho -
                    (real)0.5 * minmod(plm_theta * (yright_mid.rho - center.rho),
                                    (real)0.5 * (yright_most.rho - center.rho),
                                    plm_theta * (yright_most.rho - yright_mid.rho));

                yprims_r.v1 = yright_mid.v1 -
                                (real)0.5 * minmod(plm_theta * (yright_mid.v1 - center.v1),
                                            (real)0.5 * (yright_most.v1 - center.v1),
                                            plm_theta * (yright_most.v1 - yright_mid.v1));

                yprims_r.v2 = yright_mid.v2 -
                                (real)0.5 * minmod(plm_theta * (yright_mid.v2 - center.v2),
                                            (real)0.5 * (yright_most.v2 - center.v2),
                                            plm_theta * (yright_most.v2 - yright_mid.v2));

                yprims_r.p = yright_mid.p -
                                (real)0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                            (real)0.5 * (yright_most.p - center.p),
                                            plm_theta * (yright_most.p - yright_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = self->prims2cons(xprims_l);
                ux_r = self->prims2cons(xprims_r);
                uy_l = self->prims2cons(yprims_l);
                uy_r = self->prims2cons(yprims_r);

                f_l = self->prims2flux(xprims_l, 1);
                f_r = self->prims2flux(xprims_r, 1);
                g_l = self->prims2flux(yprims_l, 2);
                g_r = self->prims2flux(yprims_r, 2);

                if (hllc)
                {
                    // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    //     f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // } else {
                    //     f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // }
                    
                    // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    //     g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // } else {
                    //     g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // }
                    f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                else
                {
                    f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side Primitive in x
                xprims_l.rho = xleft_mid.rho +
                                (real)0.5 * minmod(plm_theta * (xleft_mid.rho - xleft_most.rho),
                                            (real)0.5 * (center.rho - xleft_most.rho),
                                            plm_theta * (center.rho - xleft_mid.rho));

                xprims_l.v1 = xleft_mid.v1 +
                                (real)0.5 * minmod(plm_theta * (xleft_mid.v1 - xleft_most.v1),
                                            (real)0.5 * (center.v1 - xleft_most.v1),
                                            plm_theta * (center.v1 - xleft_mid.v1));

                xprims_l.v2 = xleft_mid.v2 +
                                (real)0.5 * minmod(plm_theta * (xleft_mid.v2 - xleft_most.v2),
                                            (real)0.5 * (center.v2 - xleft_most.v2),
                                            plm_theta * (center.v2 - xleft_mid.v2));

                xprims_l.p =
                    xleft_mid.p + (real)0.5 * minmod(plm_theta * (xleft_mid.p - xleft_most.p),
                                                (real)0.5 * (center.p - xleft_most.p),
                                                plm_theta * (center.p - xleft_mid.p));

                // Right side Primitive in x
                xprims_r.rho =
                    center.rho - (real)0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                (real)0.5 * (xright_mid.rho - xleft_mid.rho),
                                                plm_theta * (xright_mid.rho - center.rho));

                xprims_r.v1 =
                    center.v1 - (real)0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                (real)0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                plm_theta * (xright_mid.v1 - center.v1));

                xprims_r.v2 =
                    center.v2 - (real)0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                (real)0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                plm_theta * (xright_mid.v2 - center.v2));

                xprims_r.p =
                    center.p - (real)0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                            (real)0.5 * (xright_mid.p - xleft_mid.p),
                                            plm_theta * (xright_mid.p - center.p));

                // Left side Primitive in y
                yprims_l.rho = yleft_mid.rho +
                                (real)0.5 * minmod(plm_theta * (yleft_mid.rho - yleft_most.rho),
                                            (real)0.5 * (center.rho - yleft_most.rho),
                                            plm_theta * (center.rho - yleft_mid.rho));

                yprims_l.v1 = yleft_mid.v1 +
                                (real)0.5 * minmod(plm_theta * (yleft_mid.v1 - yleft_most.v1),
                                            (real)0.5 * (center.v1 - yleft_most.v1),
                                            plm_theta * (center.v1 - yleft_mid.v1));

                yprims_l.v2 = yleft_mid.v2 +
                                (real)0.5 * minmod(plm_theta * (yleft_mid.v2 - yleft_most.v2),
                                            (real)0.5 * (center.v2 - yleft_most.v2),
                                            plm_theta * (center.v2 - yleft_mid.v2));

                yprims_l.p =
                    yleft_mid.p + (real)0.5 * minmod(plm_theta * (yleft_mid.p - yleft_most.p),
                                                (real)0.5 * (center.p - yleft_most.p),
                                                plm_theta * (center.p - yleft_mid.p));

                // Right side Primitive in y
                yprims_r.rho =
                    center.rho - (real)0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                (real)0.5 * (yright_mid.rho - yleft_mid.rho),
                                                plm_theta * (yright_mid.rho - center.rho));

                yprims_r.v1 =
                    center.v1 - (real)0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                (real)0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                plm_theta * (yright_mid.v1 - center.v1));

                yprims_r.v2 =
                    center.v2 - (real)0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                (real)0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                plm_theta * (yright_mid.v2 - center.v2));

                yprims_r.p =
                    center.p - (real)0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                            (real)0.5 * (yright_mid.p - yleft_mid.p),
                                            plm_theta * (yright_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = self->prims2cons(xprims_l);
                ux_r = self->prims2cons(xprims_r);
                uy_l = self->prims2cons(yprims_l);
                uy_r = self->prims2cons(yprims_r);

                f_l = self->prims2flux(xprims_l, 1);
                f_r = self->prims2flux(xprims_r, 1);
                g_l = self->prims2flux(yprims_l, 2);
                g_r = self->prims2flux(yprims_r, 2);

                
                if (hllc)
                {
                    // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    //     f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // } else {
                    //     f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // }
                    
                    // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    //     g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // } else {
                    //     g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // }
                    f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);   
                }
                else
                {
                    f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
            //Advance depending on geometry
            int real_loc = jj * xpg + ii;
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            real dx = coord_lattice->gpu_dx1[ii];
                            real dy = coord_lattice->gpu_dx2[jj];
                            self->gpu_cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + self->gpu_sourceD [real_loc] );
                            self->gpu_cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + self->gpu_sourceS1[real_loc] );
                            self->gpu_cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx - (g1.S2  - g2.S2) / dy + self->gpu_sourceS2[real_loc] );
                            self->gpu_cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + self->gpu_sourceTau [real_loc]);
                        #else
                            real dx = self->coord_lattice.dx1[ii];
                            real dy = self->coord_lattice.dx2[jj];
                            self->cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + self->sourceD   [real_loc] );
                            self->cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + self->sourceS1  [real_loc] );
                            self->cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + self->sourceS2  [real_loc] );
                            self->cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + self->sourceTau [real_loc] );
                        #endif
                    

                    break;
                    }
                
                case simbi::Geometry::SPHERICAL:
                    {
                    #if GPU_CODE
                    real s1R        = coord_lattice->gpu_x1_face_areas[ii + 1] ;
                    real s1L        = coord_lattice->gpu_x1_face_areas[ii + 0] ;
                    real s2R        = coord_lattice->gpu_x2_face_areas[jj + 1] ;
                    real s2L        = coord_lattice->gpu_x2_face_areas[jj + 0] ;
                    real rmean      = coord_lattice->gpu_x1mean[ii]            ;
                    real dV1        = coord_lattice->gpu_dV1[ii]               ;
                    real dV2        = rmean * coord_lattice->gpu_dV2[jj]       ;
                    #else 
                    real s1R   = self->coord_lattice.x1_face_areas[ii + 1];
                    real s1L   = self->coord_lattice.x1_face_areas[ii + 0];
                    real s2R   = self->coord_lattice.x2_face_areas[jj + 1];
                    real s2L   = self->coord_lattice.x2_face_areas[jj + 0];
                    real rmean = self->coord_lattice.x1mean[ii];
                    real dV1   = self->coord_lattice.dV1[ii];
                    real dV2   = rmean * self->coord_lattice.dV2[jj];
                    #endif
                    // Grab central primitives
                    real rhoc = prim_buff[tya * bx + txa].rho;
                    real pc   = prim_buff[tya * bx + txa].p;
                    real uc   = prim_buff[tya * bx + txa].v1;
                    real vc   = prim_buff[tya * bx + txa].v2;

                    real hc   = (real)1.0 + gamma * pc/(rhoc * (gamma - (real)1.0));
                    real gam2 = (real)1.0/((real)1.0 - (uc * uc + vc * vc));

                    #if GPU_CODE 
                        self->gpu_cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + self->gpu_sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + (real)2.0 * pc / rmean +
                                            self->gpu_sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[jj] / rmean) 
                                            + self->gpu_sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + self->gpu_sourceTau[real_loc] * decay_const
                            } * dt * (real)0.5;
                    #else
                        self->cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + self->sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + (real)2.0 * pc / rmean +
                                            self->sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * self->coord_lattice.cot[jj] / rmean) 
                                            + self->sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + self->sourceTau[real_loc] * decay_const
                            } * dt * (real)0.5;
                    #endif
                    
                    break;
                    }
            } // end switch
        }

    });
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
    std::vector<std::vector<real>> &sources,
    real tstart,
    real tend,
    real init_dt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    std::string data_directory,
    bool first_order,
    bool periodic,
    bool linspace,
    bool hllc)
{
    std::string tnow, tchunk, tstep, filename;
    int total_zones = nx * ny;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + (real)0.5) / round_place
               : floor(tstart * round_place + (real)0.5) / round_place + chkpt_interval;

    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->plm_theta = plm_theta;
    this->dt    = init_dt;

    if (first_order)
    {
        this->xphysical_grid = nx - 2;
        this->yphysical_grid = ny - 2;
        this->idx_active = 1;
        this->i_start = 1;
        this->j_start = 1;
        this->i_bound = nx - 1;
        this->j_bound = ny - 1;
    }
    else
    {
        this->xphysical_grid = nx - 4;
        this->yphysical_grid = ny - 4;
        this->idx_active = 2;
        this->i_start = 2;
        this->j_start = 2;
        this->i_bound = nx - 2;
        this->j_bound = ny - 2;
    }

    this->active_zones = xphysical_grid * yphysical_grid;

    //--------Config the System Enums
    config_system();
    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    if (coord_lattice.x2vertices[yphysical_grid] == PI){
        bipolar = true;
    }
    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.xmax = x1[xphysical_grid - 1];
    setup.xmin = x1[0];
    setup.ymax = x2[yphysical_grid - 1];
    setup.ymin = x2[0];
    setup.nx = nx;
    setup.ny = ny;
    setup.linspace = linspace;

    cons.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);
    // Define the source terms
    sourceD    = sources[0];
    sourceS1   = sources[1];
    sourceS2   = sources[2];
    sourceTau  = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        auto D            = state2D[0][i];
        auto S1           = state2D[1][i];
        auto S2           = state2D[2][i];
        auto E            = state2D[3][i];
        auto S            = sqrt(S1 * S1 + S2 * S2);
        cons[i]           = Conserved(D, S1, S2, E);
        pressure_guess[i] = std::abs(S - D - E);
    }
    // deallocate initial state vector
    std::vector<int> state2D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    // if (t == 0)
    // {
    //     config_ghosts2D(cons, nx, ny, first_order);
    // }
    // Copy the current SRHD instance over to the device
    SRHD2D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD2D));
    simbi::dual::DualSpace2D<Primitive, Conserved, SRHD2D> dualMem;
    dualMem.copyHostToDev(*this, device_self);

    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // // Setup the system
    const int nxBlocks          = (nx + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int nyBlocks          = (ny + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nxBlocks = (xphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nyBlocks = (yphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;

    dim3 agridDim  = dim3(physical_nxBlocks, physical_nyBlocks);    // active grid dimensions
    dim3 fgridDim  = dim3(nxBlocks, nyBlocks);                      // full grid dimensions
    dim3 threadDim = dim3(BLOCK_SIZE2D, BLOCK_SIZE2D);              // thread block dimensions

    const int xblockdim         = xphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : xphysical_grid;
    const int yblockdim         = yphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : yphysical_grid;
    const int radius            = (first_order) ? 1 : 2;
    const int bx                = xblockdim + 2 * radius;
    const int by                = yblockdim + 2 * radius;
    const int shBlockSpace      = bx * by;
    const unsigned shBlockBytes = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);

    if constexpr(BuildPlatform == Platform::GPU)
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
    else 
        cons2prim(fullP);

    simbi::gpu::api::deviceSynch();
    
    // Some benchmarking tools 
    int      n   = 0;
    int  nfold   = 0;
    int  ncheck  = 0;
    real zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<real> delta_t;

    // Simulate :)
    if (first_order)
    {  
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            if constexpr(BuildPlatform == Platform::GPU)
            {
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, true);
            } else {
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, true);
            }
            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                // simbi::util::writeln("Iteration: {0} \t dt: {1} \t time: {2} \t Zones/sec: {3}", n, dt, t, total_zones / delta_t.count());
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< total_zones / delta_t.count() << std::flush;
                nfold += 100;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
                writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            
            n++;


            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry[coord_system], activeP);
            } else {
                adapt_dt();
            }

            // Update decay constant
            decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (t - engine_duration)));
        }
    } else {
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            if constexpr(BuildPlatform == Platform::GPU)
            {
                // First Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, false);

                // Final Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, false);
            } else {
                // First Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, false);

                // Final Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, false);
            }
            

            t += dt; 

            if (n >= nfold){
                ncheck += 1;
                simbi::gpu::api::deviceSynch();
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones/ delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< total_zones/ delta_t.count() << std::flush;
                nfold += 100;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
                writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            n++;

            // Update decay constant
            decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (t - engine_duration)));

            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry[coord_system], activeP);
            } else {
                adapt_dt();
            }
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    if constexpr(BuildPlatform == Platform::GPU)
    {
        dualMem.copyDevToHost(device_self, *this);
        simbi::gpu::api::gpuFree(device_self);
    }

    // cons2prim2D();
    transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(4, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
