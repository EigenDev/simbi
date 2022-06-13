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
using namespace simbi::util;
using namespace std::chrono;

// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(std::vector<std::vector<real>> state2D, luint nx, luint ny, real gamma,
               std::vector<real> x1, std::vector<real> x2, real cfl,
               std::string coord_system = "cartesian")
:

    nx(nx),
    ny(ny),
    nzones(state2D[0].size()),
    state2D(state2D),
    gamma(gamma),
    x1(x1),
    x2(x2),
    cfl(cfl),
    coord_system(coord_system),
    inFailureState(false)
{
    d_all_zeros  = false;
    s1_all_zeros = false;
    s2_all_zeros = false;
    e_all_zeros  = false;
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
Eigenvals SRHD2D::calc_eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const luint nhat = 1) const
{
    // Separate the left and right Primitive
    const real rho_l = prims_l.rho;
    const real p_l   = prims_l.p;
    const real h_l   = static_cast<real>(1.0) + gamma * p_l / (rho_l * (gamma - static_cast<real>(1.0)));

    const real rho_r = prims_r.rho;
    const real p_r   = prims_r.p;
    const real h_r   = static_cast<real>(1.0) + gamma * p_r / (rho_r * (gamma - static_cast<real>(1.0)));

    const real cs_r = sqrt(gamma * p_r / (h_r * rho_r));
    const real cs_l = sqrt(gamma * p_l / (h_l * rho_l));

    const real v_l = prims_l.vcomponent(nhat);
    const real v_r = prims_r.vcomponent(nhat);

    //-----------Calculate wave speeds based on Shneider et al. 1992
    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            const real vbar  = static_cast<real>(0.5) * (v_l + v_r);
            const real cbar  = static_cast<real>(0.5) * (cs_l + cs_r);
            const real bl    = (vbar - cbar)/(static_cast<real>(1.0) - cbar*vbar);
            const real br    = (vbar + cbar)/(static_cast<real>(1.0) + cbar*vbar);
            const real aL    = my_min(bl, (v_l - cs_l)/(static_cast<real>(1.0) - v_l*cs_l));
            const real aR    = my_max(br, (v_r + cs_r)/(static_cast<real>(1.0) + v_r*cs_r));

            return Eigenvals(aL, aR, cs_l, cs_r);
        }
    
    case simbi::WaveSpeeds::MIGNONE_AND_BODO_05:
        {
            //--------Calc the wave speeds based on Mignone and Bodo (2005)
            const real sL = cs_l * cs_l * (static_cast<real>(1.0) / (gamma * gamma * (static_cast<real>(1.0) - cs_l * cs_l)));
            const real sR = cs_r * cs_r * (static_cast<real>(1.0) / (gamma * gamma * (static_cast<real>(1.0) - cs_r * cs_r)));

            // Define temporaries to save computational cycles
            const real qfL   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sL);
            const real qfR   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sR);
            const real sqrtR = sqrt(sR * (static_cast<real>(1.0)- v_r * v_r + sR));
            const real sqrtL = sqrt(sL * (static_cast<real>(1.0)- v_l * v_l + sL));

            const real lamLm = (v_l - sqrtL) * qfL;
            const real lamRm = (v_r - sqrtR) * qfR;
            const real lamLp = (v_l + sqrtL) * qfL;
            const real lamRp = (v_r + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

            return Eigenvals(aL, aR, cs_l, cs_r);
        }
    case simbi::WaveSpeeds::NAIVE:
        {
            const real aLm = (v_l - cs_l) / (1 - v_l * cs_l);
            const real aLp = (v_l + cs_l) / (1 + v_l * cs_l);
            const real aRm = (v_r - cs_r) / (1 - v_r * cs_r);
            const real aRp = (v_r + cs_r) / (1 + v_r * cs_r);

            const real aL = my_min(aLm, aRm);
            const real aR = my_max(aLp, aRp);
            return Eigenvals(aL, aR, cs_l, cs_r);
        }
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2cons(const Primitive &prims) const
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (vx * vx + vy * vy));
    const real h = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * vx,
        rho * h * lorentz_gamma * lorentz_gamma * vy,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma,
        rho * lorentz_gamma * prims.chi};
};

Conserved SRHD2D::calc_intermed_statesSR2D(const Primitive &prims,
                                           const Conserved &state, real a,
                                           real aStar, real pStar,
                                           luint nhat = 1)
{
    real Dstar, S1star, S2star, tauStar, Estar, cofactor;
    Conserved starStates;

    real pressure = prims.p;
    real v1 = prims.v1;
    real v2 = prims.v2;

    real D = state.d;
    real S1 = state.s1;
    real S2 = state.s2;
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

// Adapt the cfl conditonal timestep
void SRHD2D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real v1p, v1m, v2p, v2m;
        // Compute the minimum timestep given cfl
        for (luint jj = 0; jj < yphysical_grid; jj++)
        {
            const auto shift_j = jj + idx_active;
            #pragma omp for reduction(min:min_dt)
            for (luint ii = 0; ii < xphysical_grid; ii++)
            {
                const auto shift_i  = ii + idx_active;
                const auto aid      = shift_i + nx * shift_j;
                const auto dx1      = coord_lattice.dx1[ii];
                const auto rho      = prims[aid].rho;
                const auto v1       = prims[aid].v1;
                const auto v2       = prims[aid].v2;
                const auto pressure = prims[aid].p;
                const auto h        = 1.0 + gamma * pressure / (rho * (gamma - 1.0));
                const auto cs       = sqrt(gamma * pressure / (rho * h));

                const auto plus_v1  = (v1 + cs) / (1.0 + v1 * cs);
                const auto plus_v2  = (v2 + cs) / (1.0 + v2 * cs);
                const auto minus_v1 = (v1 - cs) / (1.0 - v1 * cs);
                const auto minus_v2 = (v2 - cs) / (1.0 - v2 * cs);

                v1p = std::abs(plus_v1);
                v1m = std::abs(minus_v1);
                v2p = std::abs(plus_v2);
                v2m = std::abs(minus_v2);
                real cfl_dt;
                if (coord_system == "cartesian") {
                    if (mesh_motion) {
                        v1p = std::abs(plus_v1  - hubble_param);
                        v1m = std::abs(minus_v1 - hubble_param);
                    }
                    cfl_dt = std::min(dx1 / (std::max(v1p, v1m)), dx2 / (std::max(v2p, v2m)));
                } else {
                    const real tl     = my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                    const real tr     = my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                    const real dtheta = tr - tl;
                    const real x1l    = get_xface(ii, geometry, 0);
                    const real x1r    = get_xface(ii, geometry, 1);
                    const real dr     = x1r - x1l;
                    const real rmean  = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                    if (mesh_motion)
                    {
                        const real vfaceL   = x1l * hubble_param;
                        const real vfaceR   = x1r * hubble_param;
                        const real vzone    = 0.5 * (vfaceL + vfaceR);
                        v1p = std::abs(plus_v1  - vzone);
                        v1m = std::abs(minus_v1 - vzone);
                    }
                    cfl_dt = std::min(dr / (std::max(v1p, v1m)),  rmean * dtheta / (std::max(v2p, v2m)));
                }
                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
            } // end ii 
        } // end jj
    } // end parallel region
    dt = cfl * min_dt;
};

void SRHD2D::adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, luint bytes)
{
    #if GPU_CODE
    {
        const luint psize         = p.blockSize.x*p.blockSize.y;
        const luint dt_buff_width = bytes * sizeof(real) / sizeof(Primitive);
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                compute_dt<SRHD2D, Primitive><<<p.gridSize,p.blockSize, bytes>>>(
                    dev, 
                    geometry, 
                    psize, 
                    dx1, 
                    dx2
                );
                dtWarpReduce<SRHD2D, Primitive, 8><<<p.gridSize,p.blockSize,dt_buff_width>>>(dev);
                break;
            
            case simbi::Geometry::SPHERICAL:
                compute_dt<SRHD2D, Primitive><<<p.gridSize,p.blockSize, bytes>>> (dev, geometry, psize, dlogx1, dx2, x1min, x1max, x2min, x2max);
                dtWarpReduce<SRHD2D, Primitive, 8><<<p.gridSize,p.blockSize,dt_buff_width>>>(dev);
                break;
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
        }
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
Conserved SRHD2D::prims2flux(const Primitive &prims, luint nhat = 1) const
{
    const real vn              = prims.vcomponent(nhat);
    const auto kron            = kronecker(nhat, 1);
    const real rho             = prims.rho;
    const real vx              = prims.v1;
    const real vy              = prims.v2;
    const real pressure        = prims.p;
    const real lorentz_gamma   = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (vx * vx + vy * vy));

    const real h   = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));
    const real D   = rho * lorentz_gamma;
    const real S1  = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const real S2  = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const real Sj  = (nhat == 1) ? S1 : S2;

    return Conserved{D * vn, S1 * vn + kron * pressure, S2 * vn + !kron * pressure, Sj - D * vn, D * vn * prims.chi};
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const luint nhat,
    const real vface) const
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL = lambda.aL;
    const real aR = lambda.aR;

    // Calculate plus/minus alphas
    const real aLm = aL < static_cast<real>(0.0) ? aL : static_cast<real>(0.0);
    const real aRp = aR > static_cast<real>(0.0) ? aR : static_cast<real>(0.0);

    Conserved net_flux;
    // Compute the HLL Flux component-wise
    if (vface < aLm) {
        net_flux = left_flux - left_state * vface;
    }
    else if (vface > aRp) {
        net_flux = right_flux - right_state * vface;
    }
    else {    
        Conserved f_hll       = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aRp * aLm) / (aRp - aLm);
        const Conserved u_hll = (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);
        net_flux = f_hll - u_hll * vface;
    }
    // Upwind the scalar concentration flux
    if (net_flux.d < static_cast<real>(0.0))
        net_flux.chi = right_prims.chi * net_flux.d;
    else
        net_flux.chi = left_prims.chi  * net_flux.d;

    // Compute the HLL Flux component-wise
    return net_flux;
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint nhat,
    const real vface) const
{

    // Conserved starStateR, starStateL;
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;
    const real cL = lambda.csL;
    const real cR = lambda.csR;

    const real aLm = aL < static_cast<real>(0.0) ? aL : static_cast<real>(0.0);
    const real aRp = aR > static_cast<real>(0.0) ? aR : static_cast<real>(0.0);

    //---- Check Wave Speeds before wasting computations
    if (vface <= aLm) {
        return left_flux - left_state * vface;
    } else if (vface >= aRp) {
        return right_flux - right_state * vface;
    }

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux = 
        (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aRp * aLm) / (aRp - aLm);

    //------ Mignone & Bodo subtract off the rest mass density
    const real e  = hll_state.tau + hll_state.d;
    const real s  = hll_state.momentum(nhat);
    const real fe = hll_flux.tau + hll_flux.d;
    const real fs = hll_flux.momentum(nhat);

    //------Calculate the contact wave velocity and pressure
    const real a     = fe;
    const real b     = -(e + fs);
    const real c     = s;
    const real quad  = -static_cast<real>(0.5) * (b + sgn(b) * std::sqrt(b * b - static_cast<real>(4.0) * a * c));
    const real aStar = c * (static_cast<real>(1.0) / quad);
    const real pStar = -aStar * fe + fs;

    // Apply the low-Mach HLLC fix found in Fleichman et al 2020: 
    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
    // constexpr real ma_lim   = static_cast<real>(0.10);

    //--------------Compute the L Star State----------
    real pressure = left_prims.p;
    real D        = left_state.d;
    real S1       = left_state.s1;
    real S2       = left_state.s2;
    real tau      = left_state.tau;
    real E        = tau + D;
    real cofactor = static_cast<real>(1.0) / (aL - aStar);

    const real vL           =  left_prims.vcomponent(nhat);
    const real vR           = right_prims.vcomponent(nhat);
    const auto kdelta       = kronecker(nhat, 1);
    // Left Star State in x-direction of coordinate lattice
    real Dstar              = cofactor * (aL - vL) * D;
    real S1star             = cofactor * (S1 * (aL - vL) +  kdelta * (-pressure + pStar) );
    real S2star             = cofactor * (S2 * (aL - vL) + !kdelta * (-pressure + pStar) );
    real Estar              = cofactor * (E  * (aL - vL) + pStar * aStar - pressure * vL);
    real tauStar            = Estar - Dstar;
    const auto starStateL   = Conserved{Dstar, S1star, S2star, tauStar};

    pressure = right_prims.p;
    D        = right_state.d;
    S1       = right_state.s1;
    S2       = right_state.s2;
    tau      = right_state.tau;
    E        = tau + D;
    cofactor = static_cast<real>(1.0) / (aR - aStar);

    Dstar                 = cofactor * (aR - vR) * D;
    S1star                = cofactor * (S1 * (aR - vR) +  kdelta * (-pressure + pStar) );
    S2star                = cofactor * (S2 * (aR - vR) + !kdelta * (-pressure + pStar) );
    Estar                 = cofactor * (E  * (aR - vR) + pStar * aStar - pressure * vR);
    tauStar               = Estar - Dstar;
    const auto starStateR = Conserved{Dstar, S1star, S2star, tauStar};

    // const real voL      =  left_prims.vcomponent(!nhat);
    // const real voR      = right_prims.vcomponent(!nhat);
    // const real ma_local = my_max(std::abs(vL / cL), std::abs(vR / cR));
    // const real phi      = std::sin(my_min(static_cast<real>(1.0), ma_local / ma_lim) * M_PI * static_cast<real>(0.5));
    // const real aL_lm    = (phi != 0) ? phi * aL : aL;
    // const real aR_lm    = (phi != 0) ? phi * aR : aR;

    const Conserved face_starState = (aStar <= 0) ? starStateR : starStateL;
    Conserved net_flux = (left_flux + right_flux) * static_cast<real>(0.5) + ( (starStateL - left_state) * aL
                        + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR) * static_cast<real>(0.5) - face_starState * vface;

    // upwind the concentration flux 
    if (net_flux.d < static_cast<real>(0.0))
        net_flux.chi = right_prims.chi * net_flux.d;
    else
        net_flux.chi = left_prims.chi  * net_flux.d;

    return net_flux;

    // if (-aL <= (aStar - aL))
    // {
    //     const real pressure = left_prims.p;
    //     const real D        = left_state.d;
    //     const real S1       = left_state.s1;
    //     const real S2       = left_state.s2;
    //     const real tau      = left_state.tau;
    //     const real chi      = left_state.chi;
    //     const real E        = tau + D;
    //     const real cofactor = static_cast<real>(1.0) / (aL - aStar);

    //     const real vL           =  left_prims.vcomponent(nhat);
    //     auto       kdelta       = kronecker(nhat, 1);
    //     // Left Star State in x-direction of coordinate lattice
    //     const real Dstar         = cofactor * (aL - vL) * D;
    //     const real chistar       = cofactor * (aL - vL) * chi;
    //     const real S1star        = cofactor * (S1 * (aL - vL) +  kdelta * (-pressure + pStar) );
    //     const real S2star        = cofactor * (S2 * (aL - vL) + !kdelta * (-pressure + pStar) );
    //     const real Estar         = cofactor * (E  * (aL - vL) + pStar * aStar - pressure * vL);
    //     const real tauStar       = Estar - Dstar;
    //     auto starStateL    = Conserved{Dstar, S1star, S2star, tauStar, chistar};

    //     auto hllc_flux = left_flux + (starStateL - left_state) * aL;

    //     // upwind the concentration flux 
    //     if (hllc_flux.d < static_cast<real>(0.0))
    //         hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //     else
    //         hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //     return hllc_flux;
    // }
    // else
    // {
    //     const real pressure = right_prims.p;
    //     const real D        = right_state.d;
    //     const real S1       = right_state.s1;
    //     const real S2       = right_state.s2;
    //     const real tau      = right_state.tau;
    //     const real chi      = right_state.chi;
    //     const real E        = tau + D;
    //     const real cofactor = static_cast<real>(1.0) / (aR - aStar);

    //     const real vR         = right_prims.vcomponent(nhat);
    //     auto       kdelta     = kronecker(nhat, 1);

    //     const real Dstar      = cofactor * (aR - vR) * D;
    //     const real chistar    = cofactor * (aR - vR) * chi;
    //     const real S1star     = cofactor * (S1 * (aR - vR) +  kdelta * (-pressure + pStar) );
    //     const real S2star     = cofactor * (S2 * (aR - vR) + !kdelta * (-pressure + pStar) );
    //     const real Estar      = cofactor * (E  * (aR - vR) + pStar * aStar - pressure * vR);
    //     const real tauStar    = Estar - Dstar;
    //     auto starStateR = Conserved{Dstar, S1star, S2star, tauStar, chistar};

    //     auto hllc_flux = right_flux + (starStateR - right_state) * aR;

    //     // upwind the concentration flux 
    //     if (hllc_flux.d < static_cast<real>(0.0))
    //         hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //     else
    //         hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //     return hllc_flux;
    // }

    // if (-aL <= (aStar - aL))
    // {
    //     const real pressure = left_prims.p;
    //     const real D        = left_state.d;
    //     const real S1       = left_state.s1;
    //     const real S2       = left_state.s2;
    //     const real tau      = left_state.tau;
    //     const real E        = tau + D;
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
    //         auto hllc_flux =  left_flux + (interstate_left - left_state) * aL;

    //         // upwind the concentration flux 
    //         if (hllc_flux.d < static_cast<real>(0.0))
    //             hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //         else
    //             hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //         return hllc_flux;
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
    //         auto hllc_flux = left_flux + (interstate_left - left_state) * aL;
    //         // upwind the concentration flux 
    //         if (hllc_flux.d < static_cast<real>(0.0))
    //             hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //         else
    //             hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //         return hllc_flux;
    //     }
    // }
    // else
    // {
    //     const real pressure = right_prims.p;
    //     const real D = right_state.d;
    //     const real S1 = right_state.s1;
    //     const real S2 = right_state.s2;
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
    //         auto hllc_flux = right_flux + (interstate_right - right_state) * aR;

    //         // upwind the concentration flux 
    //         if (hllc_flux.d < static_cast<real>(0.0))
    //             hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //         else
    //             hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //         return hllc_flux;
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
    //         auto hllc_flux = right_flux + (interstate_right - right_state) * aR;
    //         // upwind the concentration flux 
    //         if (hllc_flux.d < static_cast<real>(0.0))
    //             hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //         else
    //             hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //         return hllc_flux;
    //     }
    // }
};

//===================================================================================================================
//                                            KERNEL CALCULATIONS
//===================================================================================================================
void SRHD2D::cons2prim(
    ExecutionPolicy<> p, 
    SRHD2D *dev, 
    simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    auto gamma = this->gamma;

    const real step        = (first_order) ? 1.0 : 0.5;
    const real radius      = (first_order) ? 1 : 2;

    #if GPU_CODE
    const bool mesh_motion = this->mesh_motion;
    const auto active_zones = this->active_zones;
    const auto dt           = this->dt;
    const auto hubble_param = this->hubble_param;
    const auto geometry     = this->geometry;
    #endif
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;
        const auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * threadIdx.y + threadIdx.x : gid;
        #if GPU_CODE
        if (tid == 0) found_failure = self->inFailureState;
        simbi::gpu::api::synchronize();
        #else 
        found_failure = self->inFailureState;
        #endif
        
        real invdV = 1.0;
        while (!found_failure && workLeftToDo)
        {
            if (tid == 0 && self->inFailureState) 
                found_failure = true;
            
            if (mesh_motion && (geometry == simbi::Geometry::SPHERICAL))
            {
                const auto ii = gid % self->nx;
                const auto jj = gid / self->nx;
                lint ireal;
                lint jreal; 

                if (ii > self->xphysical_grid + 1) {
                    ireal = self->xphysical_grid - 1;
                } else {
                    ireal = (ii - radius > 0 ) * (ii - radius);
                }

                if (jj > self->yphysical_grid + 1) {
                    jreal = self->yphysical_grid - 1;
                } else {
                    jreal = (jj - radius > 0) * (jj - radius);
                }
                
                const real dV = self->get_cell_volume(ireal, jreal, geometry, step);
                invdV = 1.0 / dV;
            }
            #if GPU_CODE
            extern __shared__ Conserved  conserved_buff[];
            // load shared memory
            conserved_buff[tid] = self->gpu_cons[gid];
            simbi::gpu::api::synchronize();
            #else
            auto* const conserved_buff = &cons[0];
            #endif 

            luint iter  = 0;
            const real D    = conserved_buff[tid].d   * invdV;
            const real S1   = conserved_buff[tid].s1  * invdV;
            const real S2   = conserved_buff[tid].s2  * invdV;
            const real tau  = conserved_buff[tid].tau * invdV;
            const real Dchi = conserved_buff[tid].chi * invdV; 
            const real S    = std::sqrt(S1 * S1 + S2 * S2);

            // const auto ii = gid % nx;
            // const auto jj = gid / nx;
            // if (true)
            // {
            //     if (D != 1)
            //         writeln("ii: {}, jj: {}, M: {}, D: {}, dV: {}", ii, jj, conserved_buff[tid].d, D, 1.0 / invdV);
            //     // pause_program();
            // }
            #if GPU_CODE
            real peq = self->gpu_pressure_guess[gid];
            #else 
            real peq = pressure_guess[gid];
            #endif

            const real tol = D * tol_scale;
            do
            {
                pre = peq;
                et  = tau + D + pre;
                v2  = S * S / (et * et);
                W   = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - v2);
                rho = D / W;
                eps = (tau + (static_cast<real>(1.0) - W) * D + (static_cast<real>(1.0) - W * W) * pre) / (D * W);

                h  = static_cast<real>(1.0) + eps + pre / rho;
                c2 = gamma * pre / (h * rho);

                g = c2 * v2 - static_cast<real>(1.0);
                f = (gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                iter++;
                if (iter >= MAX_ITER)
                {
                    const auto ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % self->xphysical_grid;
                    const auto jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / self->xphysical_grid;
                    printf("\nCons2Prim cannot converge\n");
                    printf("Density: %f, Pressure: %f, Vsq: %f, xindex: %lu, yindex: %lu\n", rho, peq, v2, ii, jj);
                    self->dt             = INFINITY;
                    found_failure        = true;
                    self->inFailureState = true;
                    simbi::gpu::api::synchronize();
                    break;
                }

            } while (std::abs(peq - pre) >= tol);

            real inv_et = static_cast<real>(1.0) / (tau + D + peq);
            real vx     = S1 * inv_et;
            real vy     = S2 * inv_et;
            #if GPU_CODE
                self->gpu_pressure_guess[gid] = peq;
                self->gpu_prims[gid]          = Primitive{D / W, vx, vy, peq, Dchi / D};
            #else
                self->pressure_guess[gid] = peq;
                self->prims[gid]          = Primitive{D / W, vx, vy, peq, Dchi / D};
            #endif
            workLeftToDo = false;
        }
    });
}

void SRHD2D::advance(
    SRHD2D *dev, 
    const ExecutionPolicy<> p,
    const luint bx,
    const luint by,
    const luint radius, 
    const simbi::Geometry geometry, 
    const simbi::MemSide user)
{
    auto *self = (BuildPlatform == Platform::GPU) ? dev : this;
    const luint xpg                   = this->xphysical_grid;
    const luint ypg                   = this->yphysical_grid;
    const luint extent                = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const luint xextent               = p.blockSize.x;
    const luint yextent               = p.blockSize.y;
    const auto step                   = (first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);

    #if GPU_CODE
    const bool first_order         = this->first_order;
    const bool periodic            = this->periodic;
    const bool hllc                = this->hllc;
    const real dt                  = this->dt;
    const real decay_const         = this->decay_const;
    const real plm_theta           = this->plm_theta;
    const real gamma               = this->gamma;
    const luint nx                 = this->nx;
    const luint ny                 = this->ny;
    const real dx2                 = this->dx2;
    const real dlogx1              = this->dlogx1;
    const real dx1                 = this->dx1;
    const real imax                = this->xphysical_grid - 1;
    const real jmax                = this->yphysical_grid - 1;
    const bool d_all_zeros         = this->d_all_zeros;
    const bool s1_all_zeros        = this->s1_all_zeros;
    const bool s2_all_zeros        = this->s2_all_zeros;
    const bool e_all_zeros         = this->e_all_zeros;
    const real x1min               = this->x1min;
    const real x1max               = this->x1max;
    const real x2min               = this->x2min;
    const real x2max               = this->x2max;
    const bool quirk_smoothing     = this->quirk_smoothing;
    const real pow_dlogr           = pow(10, dlogx1);
    const real hubble_param        = this->hubble_param;
    // const real x1min_init          = this->x1min_init;
    // const real 
    #endif

    // const CLattice2D *coord_lattice = &(self->coord_lattice);
    const luint nbs = (BuildPlatform == Platform::GPU) ? bx * by : nzones;

    // Choice of column major striding by user
    const luint sx = (col_maj) ? 1  : bx;
    const luint sy = (col_maj) ? by :  1;

    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx) {
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = &prims[0];
        #endif 

        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        #if GPU_CODE 
        if ((ii >= xpg) || (jj >= ypg)) return;
        #endif

        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;

        Conserved ux_l, ux_r, uy_l, uy_r;
        Conserved f_l, f_r, g_l, g_r, frf, flf, grf, glf;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r;

        const luint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = self->gpu_prims[aid];
            if (ty < radius)
            {
                if (ja + yextent > ny - 1) tyl = ny - radius - ja + ty;
                prim_buff[(tya - radius) * sx + txa] = self->gpu_prims[(ja - radius) * nx + ia];
                prim_buff[(tya + tyl   ) * sx + txa] = self->gpu_prims[(ja + tyl   ) * nx + ia]; 
            }
            if (tx < radius)
            {   
                if (ia + xextent > nx - 1) txl = nx - radius - ia + tx;
                prim_buff[tya * sx + txa - radius] =  self->gpu_prims[ja * nx + ia - radius];
                prim_buff[tya * sx + txa +    txl] =  self->gpu_prims[ja * nx + ia + txl]; 
                if ((radius == 2) && (txl < xextent))
                {
                    prim_buff[tya * sx + txa + txl + 1] =  self->gpu_prims[ja * nx + ia + txl + 1]; 
                }
            }
            
            simbi::gpu::api::synchronize();
        #endif


        const real x1l    = self->get_xface(ii, geometry, 0);
        const real x1r    = self->get_xface(ii, geometry, 1);
        const real vfaceR = x1r * hubble_param;
        const real vfaceL = x1l * hubble_param;
        if (first_order)
        {
            xprims_l = prim_buff[( (txa + 0) * sy + (tya + 0) * sx)];
            xprims_r = prim_buff[( (txa + 1) * sy + (tya + 0) * sx)];
            //j+1/2
            yprims_l = prim_buff[( (txa + 0) * sy + (tya + 0) * sx)];
            yprims_r = prim_buff[( (txa + 0) * sy + (tya + 1) * sx)];
            
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
                frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            } else {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            }

            // Set up the left and right state interfaces for i-1/2
            xprims_l = prim_buff[( (txa - 1) * sy + (tya + 0) * sx )];
            xprims_r = prim_buff[( (txa - 0) * sy + (tya + 0) * sx )];
            //j+1/2
            yprims_l = prim_buff[( (txa - 0) * sy + (tya - 1) * sx )]; 
            yprims_r = prim_buff[( (txa + 0) * sy + (tya - 0) * sx )]; 

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
                flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            } else {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            }   
        }
        else 
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            // Coordinate X
            xleft_most  = prim_buff[((txa - 2) * sy + tya * sx) % nbs];
            xleft_mid   = prim_buff[((txa - 1) * sy + tya * sx) % nbs];
            center      = prim_buff[((txa + 0) * sy + tya * sx) % nbs];
            xright_mid  = prim_buff[((txa + 1) * sy + tya * sx) % nbs];
            xright_most = prim_buff[((txa + 2) * sy + tya * sx) % nbs];

            // Coordinate Y
            yleft_most  = prim_buff[(txa * sy + (tya - 2) * sx) % nbs];
            yleft_mid   = prim_buff[(txa * sy + (tya - 1) * sx) % nbs];
            yright_mid  = prim_buff[(txa * sy + (tya + 1) * sx) % nbs];
            yright_most = prim_buff[(txa * sy + (tya + 2) * sx) % nbs];

            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprims_l = center     + minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center) * plm_theta) * static_cast<real>(0.5); 
            xprims_r = xright_mid - minmod((xright_mid - center) * plm_theta, (xright_most - center) * static_cast<real>(0.5), (xright_most - xright_mid)*plm_theta) * static_cast<real>(0.5);
            yprims_l = center     + minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center) * plm_theta) * static_cast<real>(0.5);  
            yprims_r = yright_mid - minmod((yright_mid - center) * plm_theta, (yright_most - center) * static_cast<real>(0.5), (yright_most - yright_mid)*plm_theta) * static_cast<real>(0.5);


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
                if(quirk_smoothing)
                {
                    if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                        frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                    } else {
                        frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                    }

                    if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                        grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                    } else {
                        grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                    }
                } else {
                    frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                    grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                }
            }
            else
            {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceR);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            xprims_l = xleft_mid + minmod((xleft_mid - xleft_most) * plm_theta, (center - xleft_most) * static_cast<real>(0.5), (center - xleft_mid)*plm_theta) * static_cast<real>(0.5);
            xprims_r = center    - minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center)*plm_theta)*static_cast<real>(0.5);
            yprims_l = yleft_mid + minmod((yleft_mid - yleft_most) * plm_theta, (center - yleft_most) * static_cast<real>(0.5), (center - yleft_mid)*plm_theta) * static_cast<real>(0.5);
            yprims_r = center    - minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center)*plm_theta)*static_cast<real>(0.5);

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
                if (quirk_smoothing)
                {
                    if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                        flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                    } else {
                        flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                    }
                    
                    if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                        glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                    } else {
                        glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                    } 
                } else {
                    flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                    glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
                }
            }
            else
            {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1, vfaceL);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2, 0.0);
            }
        }

        //Advance depending on geometry
        const luint real_loc = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    #if GPU_CODE
                        const real d_source  = (d_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceD[real_loc];
                        const real s1_source = (s1_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS1[real_loc];
                        const real s2_source = (s2_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS2[real_loc];
                        const real e_source  = (e_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceTau[real_loc];
                        const Conserved source_terms = Conserved{d_source, s1_source, s2_source, e_source} * decay_const;
                        self->gpu_cons[aid]   -= ( (frf - flf) / dx1 + (grf - glf)/ dx2 - source_terms) * step;
                    #else
                        const real d_source  = (d_all_zeros)   ? static_cast<real>(0.0) : sourceD[real_loc];
                        const real s1_source = (s1_all_zeros)  ? static_cast<real>(0.0) : sourceS1[real_loc];
                        const real s2_source = (s2_all_zeros)  ? static_cast<real>(0.0) : sourceS2[real_loc];
                        const real e_source  = (e_all_zeros)   ? static_cast<real>(0.0) : sourceTau[real_loc];
                        const real dx1 = self->coord_lattice.dx1[ii];
                        const real dx2  = self->coord_lattice.dx2[jj];
                        const Conserved source_terms = Conserved{d_source, s1_source, s2_source, e_source} * decay_const;
                        cons[aid] -= ( (frf - flf) / dx1 + (grf - glf)/dx2 - source_terms) * step;
                    #endif
                

                break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                // #if GPU_CODE
                const real rl           = x1l + vfaceL * step * dt; 
                const real rr           = x1r + vfaceR * step * dt;
                const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real tl           = my_max(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                const real tr           = my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real s1R          = rr * rr; 
                const real s1L          = rl * rl; 
                const real s2R          = std::sin(tr);
                const real s2L          = std::sin(tl);
                const real thmean       = static_cast<real>(0.5) * (tl + tr);
                const real sint         = std::sin(thmean);
                const real dV1          = rmean * rmean * (rr - rl);             
                const real dV2          = rmean * sint * (tr - tl); 
                const real cot          = std::cos(thmean) / sint;
                const real dcos         = std::cos(tl) - std::cos(tr);
                const real dVtot        = (2.0 * M_PI * (1.0 / 3.0) * (rr * rr * rr - rl * rl * rl) * dcos);
                const real factor       = (self->mesh_motion) ? dVtot : 1;  
                // writeln("vfaceL: {}, vfaceR: {}, factor: {}, frf: {}, flf: {}", vfaceL, vfaceR, factor, frf.d, flf.d);
                // pause_program();
                #if GPU_CODE
                const real d_source  = (d_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceD[real_loc];
                const real s1_source = (s1_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS1[real_loc];
                const real s2_source = (s2_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS2[real_loc];
                const real e_source  = (e_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceTau[real_loc];
                #else
                const real d_source  = (d_all_zeros)   ? static_cast<real>(0.0) : sourceD[real_loc];
                const real s1_source = (s1_all_zeros)  ? static_cast<real>(0.0) : sourceS1[real_loc];
                const real s2_source = (s2_all_zeros)  ? static_cast<real>(0.0) : sourceS2[real_loc];
                const real e_source  = (e_all_zeros)   ? static_cast<real>(0.0) : sourceTau[real_loc];
                #endif

                // Grab central primitives
                const real rhoc = prim_buff[txa * sy + tya * sx].rho;
                const real uc   = prim_buff[txa * sy + tya * sx].v1;
                const real vc   = prim_buff[txa * sy + tya * sx].v2;
                const real pc   = prim_buff[txa * sy + tya * sx].p;

                const real hc   = static_cast<real>(1.0) + gamma * pc/(rhoc * (gamma - static_cast<real>(1.0)));
                const real gam2 = static_cast<real>(1.0)/(static_cast<real>(1.0) - (uc * uc + vc * vc));

                const Conserved geom_source  = {static_cast<real>(0.0), (rhoc * hc * gam2 * vc * vc) / rmean + pc * (s1R - s1L) / dV1, - (rhoc * hc * gam2 * uc * vc) / rmean + pc * (s2R - s2L)/dV2 , static_cast<real>(0.0)};
                const Conserved source_terms = Conserved{d_source, s1_source, s2_source, e_source} * decay_const;
                #if GPU_CODE 
                    self->gpu_cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 - geom_source - source_terms) * dt * step * factor;
                #else
                    cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 - geom_source - source_terms) * dt * step * factor;
                #endif
                
                break;
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
        } // end switch
    });
    // update x1 enpoints
    const real x1l    = self->get_xface(0, geometry, 0);
    const real x1r    = self->get_xface(xphysical_grid, geometry, 1);
    const real vfaceR = x1r * hubble_param;
    const real vfaceL = x1l * hubble_param;
    self->x1min      += step * dt * vfaceL;
    self->x1max      += step * dt * vfaceR;
    if constexpr(BuildPlatform == Platform::GPU) {
        this->x1max = self->x1max;
        this->x1min = self->x1min;
    }
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
    std::vector<std::vector<real>> & sources,
    real tstart,
    real tend,
    real init_dt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    std::string data_directory,
    std::string boundary_condition,
    bool first_order,
    bool linspace,
    bool hllc,
    bool quirk_smoothing,
    std::function<double(double)> a,
    std::function<double(double)> adot,
    std::function<double(double, double)> d_outer,
    std::function<double(double, double)> s1_outer,
    std::function<double(double, double)> s2_outer,
    std::function<double(double, double)> e_outer)
{
    std::string tnow, tchunk, tstep, filename;
    luint total_zones = nx * ny;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

    this->first_order     = first_order;
    this->periodic        = boundary_condition == "periodic";
    this->hllc            = hllc;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->dt              = init_dt;
    this->xphysical_grid  = (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid  = (first_order) ? ny - 2 : ny - 4;
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = xphysical_grid * yphysical_grid;
    this->quirk_smoothing = quirk_smoothing;
    this->bc              = boundary_cond_map.at(boundary_condition);
    this->geometry        = geometry_map.at(coord_system);

    if ((coord_system == "spherical") && (linspace))
    {
        this->geometry = simbi::Geometry::SPHERICAL;
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->geometry = simbi::Geometry::SPHERICAL;
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE);
    } else {
        this->geometry = simbi::Geometry::CARTESIAN;
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (coord_lattice.x2vertices[yphysical_grid] == M_PI){
        this->bipolar = true;
    }

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
        auto Dchi         = state2D[4][i];
        auto S            = std::sqrt(S1 * S1 + S2 * S2);
        cons[i]           = Conserved(D, S1, S2, E, Dchi);
        const auto ii = i % nx;
        const auto jj = i / nx;
        if (ii == 0)
        pressure_guess[i] = std::abs(S - D - E);
    }

    this->dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->x1min   = x1[0];
    this->x1max   = x1[xphysical_grid - 1];
    this->x2min   = x2[0];
    this->x2max   = x2[yphysical_grid - 1];

    this->d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    this->s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    this->s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    this->e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});

    // deallocate initial state vector
    std::vector<int> state2D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1 / (1 + std::exp(static_cast<real>(10.0) * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;
    
    // Copy the current SRHD instance over to the device
    // if compiling for CPU, these functions do nothing
    SRHD2D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(SRHD2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD2D));
    simbi::dual::DualSpace2D<Primitive, Conserved, SRHD2D> dualMem;
    dualMem.copyHostToDev(*this, device_self);
    
    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.x1max          = x1[xphysical_grid - 1];
    setup.x1min          = x1[0];
    setup.x2max          = x2[yphysical_grid - 1];
    setup.x2min          = x2[0];
    setup.nx             = nx;
    setup.ny             = ny;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;

    // // Setup the system
    const luint xblockdim       = xphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : xphysical_grid;
    const luint yblockdim       = yphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : yphysical_grid;
    const luint radius          = (periodic) ? 0 : (first_order) ? 1 : 2;
    const luint bx              = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint by              = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint shBlockSpace    = bx * by;
    const luint shBlockBytes    = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);
    
    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU)
        {
            config_ghosts2D(fullP, device_self, nx, ny, first_order, bc);
        } else {
            config_ghosts2D(fullP, this, nx, ny, first_order, bc);
        }
    }

    const auto dtShBytes = xblockdim * yblockdim * sizeof(Primitive);
    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, geometry, activeP, dtShBytes);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    Conserved *outer_zones = nullptr;
    Conserved *dev_outer_zones = nullptr;
    // Fill outer zones if user-defined conservative functions provided
    const real step = (first_order) ? 1.0 : 0.5;
    if (d_outer)
    {
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::gpuMalloc(&dev_outer_zones, ny * sizeof(Conserved));
        }

        outer_zones = new Conserved[ny];
        lint jreal = 0;
        for (int jj = 0; jj < ny; jj++) {
           if (jj > yphysical_grid + 1) {
                jreal = yphysical_grid - 1;
            } else {
                jreal = (jj - radius) > 0 ? jj - radius : 0;
            }
            const real dV = get_cell_volume(xphysical_grid - 1, jreal, geometry, step);
            outer_zones[jj] = Conserved{d_outer(x1max, x2[jreal]), s1_outer(x1max, x2[jreal]), s2_outer(x1max, x2[jreal]), e_outer(x1max, x2[jreal])} * dV;
        }
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::copyHostToDevice(dev_outer_zones, outer_zones, ny * sizeof(Conserved));
        }
    }
    
    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
        transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
        writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
        tnow = create_step_str(t_interval, tchunk);
        filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
        setup.t = t;
        setup.dt = dt;
        write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
        t_interval += chkpt_interval;
    }
    // Some benchmarking tools 
    luint      n   = 0;
    luint  nfold   = 0;
    luint  ncheck  = 0;
    real    zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<real> delta_t;
    
    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;
    const auto ozones  = (BuildPlatform == Platform::GPU) ? dev_outer_zones : outer_zones;
    // Simulate :)
    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            advance(self, activeP, bx, by, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts2D(fullP, self, nx, ny, true, bc, ozones, bipolar);
            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\r Iteration: {>8} \t dt: {>8} \t Time: {>8} \t Zones/sec: {>8}", n, dt, t, total_zones/delta_t.count());
                nfold += 100;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU){
                    simbi::gpu::api::copyDevToHost(&x1min, &(device_self->x1min),  sizeof(real));
                    simbi::gpu::api::copyDevToHost(&x1max, &(device_self->x1max),  sizeof(real));
                    dualMem.copyDevToHost(device_self, *this);
                } 
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
                writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t     = t;
                setup.dt    = dt;
                setup.x1max = x1max;
                setup.x1min = x1min;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            
            n++;
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            } else {
                adapt_dt();
            }
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            hubble_param = adot(t) / a(t);
            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));

            if (d_outer)
            {
                simbi::gpu::api::deviceSynch();
                lint jreal = 0;
                for (int jj = 0; jj < ny; jj++) {
                if (jj > yphysical_grid + 1) {
                        jreal = yphysical_grid - 1;
                    } else {
                        jreal = (jj - radius) > 0 ? jj - radius : 0;
                    }
                    const real dV = get_cell_volume(xphysical_grid - 1, jreal, geometry, step);
                    outer_zones[jj] = Conserved{d_outer(x1max, x2[jreal]), s1_outer(x1max, x2[jreal]), s2_outer(x1max, x2[jreal]), e_outer(x1max, x2[jreal])} * dV;
                }
                 if constexpr(BuildPlatform == Platform::GPU) {
                    simbi::gpu::api::copyHostToDevice(ozones, outer_zones, ny * sizeof(Conserved));
                }
            }
        }
    } else {
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            advance(self, activeP, bx, by, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts2D(fullP, self, nx, ny, false, bc, ozones, bipolar);

            // Final Half Step
            advance(self, activeP, bx, by, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts2D(fullP, self, nx, ny, false, bc, ozones, bipolar);

            t += dt; 

            if (n >= nfold){
                ncheck += 1;
                simbi::gpu::api::deviceSynch();
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones/ delta_t.count();
                writefl("Iteration: {>8} \t dt: {>8} \t Time: {>8} \t Zones/sec: {>8} \t\r", n, dt, t, total_zones/delta_t.count());
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
                setup.t     = t;
                setup.dt    = dt;
                setup.x1max = x1max;
                setup.x1min = x1min;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            n++;
            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));

            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            } else {
                adapt_dt();
            }
            hubble_param = adot(t) / a(t);
            if (d_outer)
            {
                lint jreal = 0;
                for (int jj = 0; jj < ny; jj++) {
                if (jj > yphysical_grid + 1) {
                        jreal = yphysical_grid - 1;
                    } else {
                        jreal = (jj - radius) > 0 ? jj - radius : 0;
                    }
                    const real dV = get_cell_volume(xphysical_grid - 1, jreal, geometry, step);
                    outer_zones[jj] = Conserved{d_outer(x1max, x2[jreal]), s1_outer(x1max, x2[jreal]), s2_outer(x1max, x2[jreal]), e_outer(x1max, x2[jreal])} * dV;
                }
                 if constexpr(BuildPlatform == Platform::GPU) {
                    simbi::gpu::api::copyHostToDevice(ozones, outer_zones, ny * sizeof(Conserved));
                }
            }
        }
    }
    if (ncheck > 0) {
        writeln("Average zone_updates/sec for {} iterations was: {} zones/sec", n, zu_avg / ncheck);
    }

    if constexpr(BuildPlatform == Platform::GPU)
    {
        dualMem.copyDevToHost(device_self, *this);
        simbi::gpu::api::gpuFree(device_self);
    }

    if (outer_zones) {

        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::deviceSynch();
            simbi::gpu::api::gpuFree(ozones);
        } else {
            delete[] ozones;
        }
    }

    transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(5, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;
    solution[4] = transfer_prims.chi;

    return solution;
};
