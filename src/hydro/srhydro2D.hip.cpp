/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */
#include <chrono>
#include <cmath>
#include "util/device_api.hpp"
#include "common/helpers.hip.hpp"
#include "srhydro2D.hip.hpp"
#include "util/printb.hpp"
#include "util/parallel_for.hpp"
#include "util/logger.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;
constexpr auto write2file = helpers::write_to_file<sr2d::PrimitiveSOA, 2, SRHD2D>;

/* Define typedefs because I am lazy */
using Primitive           = sr2d::Primitive;
using Conserved           = sr2d::Conserved;
using Eigenvals           = sr2d::Eigenvals;
// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(
    std::vector<std::vector<real>> state, 
    luint nx, 
    luint ny, 
    real gamma,
    std::vector<real> x1, 
    std::vector<real> x2, 
    real cfl,
    std::string coord_system = "cartesian")
:
    HydroBase(
        state,
        nx,
        ny,
        gamma,
        x1,
        x2,
        cfl,
        coord_system
    )
{
    d_all_zeros  = false;
    s1_all_zeros = false;
    s2_all_zeros = false;
    e_all_zeros  = false;
}

// Destructor
SRHD2D::~SRHD2D() {}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD2D::calc_eigenvals(const Primitive &primsL,
                                 const Primitive &primsR,
                                 const luint nhat = 1) const
{
    // Separate the left and right Primitive
    const real rhoL = primsL.rho;
    const real pL   = primsL.p;
    const real hL   = static_cast<real>(1.0) + gamma * pL / (rhoL * (gamma - static_cast<real>(1.0)));

    const real rhoR  = primsR.rho;
    const real pR    = primsR.p;
    const real hR    = static_cast<real>(1.0) + gamma * pR  / (rhoR  * (gamma - static_cast<real>(1.0)));

    const real csR  = std::sqrt(gamma * pR  / (hR  * rhoR));
    const real csL = std::sqrt(gamma * pL / (hL * rhoL));

    const real vL = primsL.vcomponent(nhat);
    const real vR  = primsR.vcomponent(nhat);

    //-----------Calculate wave speeds based on Shneider et al. 1992
    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            const real vbar  = static_cast<real>(0.5) * (vL + vR);
            const real cbar  = static_cast<real>(0.5) * (csL + csR);
            const real bl    = (vbar - cbar)/(static_cast<real>(1.0) - cbar*vbar);
            const real br    = (vbar + cbar)/(static_cast<real>(1.0) + cbar*vbar);
            const real aL    = helpers::my_min(bl, (vL - csL)/(static_cast<real>(1.0) - vL*csL));
            const real aR    = helpers::my_max(br, (vR  + csR)/(static_cast<real>(1.0) + vR*csR));

            return Eigenvals(aL, aR, csL, csR);
        }
    
    case simbi::WaveSpeeds::MIGNONE_AND_BODO_05:
        {
            // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
            const real sL = csL*csL/(gamma*gamma*(static_cast<real>(1.0) - csL*csL));
            const real sR = csR*csR/(gamma*gamma*(static_cast<real>(1.0) - csR*csR));
            // Define temporaries to save computational cycles
            const real qfL   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sL);
            const real qfR   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sR);
            const real sqrtR = std::sqrt(sR * (static_cast<real>(1.0) - vR  * vR  + sR));
            const real sqrtL = std::sqrt(sL * (static_cast<real>(1.0) - vL * vL + sL));

            const real lamLm = (vL - sqrtL) * qfL;
            const real lamRm = (vR  - sqrtR) * qfR;
            const real lamLp = (vL + sqrtL) * qfL;
            const real lamRp = (vR  + sqrtR) * qfR;

            real aL = lamLm < lamRm ? lamLm : lamRm;
            real aR = lamLp > lamRp ? lamLp : lamRp;

            // Smoothen for rarefaction fan
            aL = helpers::my_min(aL, (vL - csL) / (1 - vL * csL));
            aR = helpers::my_max(aR, (vR  + csR) / (1 + vR  * csR));

            return Eigenvals(aL, aR, csL, csR);
        }
    case simbi::WaveSpeeds::NAIVE:
        {
            const real aLm = (vL - csL) / (1 - vL * csL);
            const real aLp = (vL + csL) / (1 + vL * csL);
            const real aRm = (vR  - csR) / (1 - vR  * csR);
            const real aRp = (vR  + csR) / (1 + vR  * csR);

            const real aL = helpers::my_min(aLm, aRm);
            const real aR = helpers::my_max(aLp, aRp);
            return Eigenvals(aL, aR, csL, csR);
        }
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2cons(const Primitive &prims) const
{
    const real rho           = prims.rho;
    const real v1            = prims.v1;
    const real v2            = prims.v2;
    const real pressure      = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (v1 * v1 + v2 * v2));
    const real h             = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * v1,
        rho * h * lorentz_gamma * lorentz_gamma * v2,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma,
        rho * lorentz_gamma * prims.chi
    };
};

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
                const auto rho      = prims[aid].rho;
                const auto v1       = prims[aid].v1;
                const auto v2       = prims[aid].v2;
                const auto pressure = prims[aid].p;
                const auto h        = 1.0 + gamma * pressure / (rho * (gamma - 1.0));
                const auto cs       = std::sqrt(gamma * pressure / (rho * h));

                //================ Plus / Minus Wave speed components -================
                const auto plus_v1  = (v1 + cs) / (1.0 + v1 * cs);
                const auto plus_v2  = (v2 + cs) / (1.0 + v2 * cs);
                const auto minus_v1 = (v1 - cs) / (1.0 - v1 * cs);
                const auto minus_v2 = (v2 - cs) / (1.0 - v2 * cs);

                v1p = std::abs(plus_v1);
                v1m = std::abs(minus_v1);
                v2p = std::abs(plus_v2);
                v2m = std::abs(minus_v2);
                real cfl_dt;
                switch (geometry)
                {
                    case simbi::Geometry::CARTESIAN:
                        {
                            if (mesh_motion) {
                                v1p = std::abs(plus_v1  - hubble_param);
                                v1m = std::abs(minus_v1 - hubble_param);
                            }
                            cfl_dt = helpers::my_min(dx1 / (helpers::my_max(v1p, v1m)), dx2 / (helpers::my_max(v2p, v2m)));
                        }
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                        {
                            const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                            const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                            const real dtheta = tr - tl;
                            const real x1l    = get_x1face(ii, geometry, 0);
                            const real x1r    = get_x1face(ii, geometry, 1);
                            const real dr     = x1r - x1l;
                            const real rmean  = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                            if (mesh_motion)
                            {
                                const real vfaceL   = x1l * hubble_param;
                                const real vfaceR   = x1r * hubble_param;
                                v1p = std::abs(plus_v1  - vfaceR);
                                v1m = std::abs(minus_v1 - vfaceL);
                            }
                            cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                        }
                        break;

                    case simbi::Geometry::CYLINDRICAL:
                    break;
                } // end switch
                min_dt = helpers::my_min(min_dt, cfl_dt);
            } // end ii 
        } // end jj
    } // end parallel region
    dt = cfl * min_dt;
};

void SRHD2D::adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, luint bytes)
{
    #if GPU_CODE
    {
        const luint psize = p.blockSize.x*p.blockSize.y;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                compute_dt<Primitive><<<p.gridSize,p.blockSize, bytes>>>(
                    dev, 
                    prims.data(),
                    dt_min.data(),
                    geometry, 
                    psize, 
                    dx1, 
                    dx2
                );
                deviceReduceKernel<2><<<p.gridSize,p.blockSize>>>(dev, dt_min.data(), active_zones);
                deviceReduceKernel<2><<<1,1024>>>(dev, dt_min.data(), p.gridSize.x * p.gridSize.y);
                break;
            
            case simbi::Geometry::SPHERICAL:
                compute_dt<Primitive><<<p.gridSize,p.blockSize, bytes>>> (
                    dev, 
                    prims.data(), 
                    dt_min.data(),
                    geometry, 
                    psize, 
                    dlogx1, 
                    dx2, 
                    x1min, 
                    x1max, 
                    x2min, 
                    x2max);
                deviceReduceKernel<2><<<p.gridSize,p.blockSize>>>(dev, dt_min.data(), active_zones);
                deviceReduceKernel<2><<<1,1024>>>(dev, dt_min.data(), p.gridSize.x * p.gridSize.y);
                // dtWarpReduce<SRHD2D, Primitive, 64><<<p.gridSize,p.blockSize,dt_buff_width>>>(dev);
                break;
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
        }
        simbi::gpu::api::deviceSynch();
        this->dt = dev->dt;
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
    const real v1              = prims.v1;
    const real v2              = prims.v2;
    const real pressure        = prims.p;
    const real lorentz_gamma   = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (v1 * v1 + v2 * v2));

    const real h   = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));
    const real D   = rho * lorentz_gamma;
    const real S1  = rho * lorentz_gamma * lorentz_gamma * h * v1;
    const real S2  = rho * lorentz_gamma * lorentz_gamma * h * v2;
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
    // const real cL = lambda.csL;
    // const real cR = lambda.csR;

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
    const real quad  = -static_cast<real>(0.5) * (b + helpers::sgn(b) * std::sqrt(b * b - static_cast<real>(4.0) * a * c));
    const real aStar = c * (static_cast<real>(1.0) / quad);
    const real pStar = -aStar * fe + fs;

    // Apply the low-Mach HLLC fix found in Fleichman et al 2020: 
    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
    constexpr real ma_lim   = static_cast<real>(0.10);

    // --------------Compute the L Star State----------
    real pressure = left_prims.p;
    real D        = left_state.d;
    real S1       = left_state.s1;
    real S2       = left_state.s2;
    real tau      = left_state.tau;
    real E        = tau + D;
    real cofactor = static_cast<real>(1.0) / (aL - aStar);

    const real vL           = left_prims.vcomponent(nhat);
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
    // const real ma_local = helpers::my_max(std::abs(vL / cL), std::abs(vR / cR));
    // const real phi      = std::sin(helpers::my_min(static_cast<real>(1.0), ma_local / ma_lim) * M_PI * static_cast<real>(0.5));
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

    // if (vface <= aStar)
    // {
    //     const real pressure = left_prims.p;
    //     const real D        = left_state.d;
    //     const real S1       = left_state.s1;
    //     const real S2       = left_state.s2;
    //     const real tau      = left_state.tau;
    //     const real chi      = left_state.chi;
    //     const real E        = tau + D;
    //     const real cofactor = static_cast<real>(1.0) / (aL - aStar);

    //     const real vL     =  left_prims.vcomponent(nhat);
    //     const auto kdelta = kronecker(nhat, 1);
    //     // Left Star State in x-direction of coordinate lattice
    //     const real Dstar         = cofactor * (aL - vL) * D;
    //     const real chistar       = cofactor * (aL - vL) * chi;
    //     const real S1star        = cofactor * (S1 * (aL - vL) +  kdelta * (-pressure + pStar) );
    //     const real S2star        = cofactor * (S2 * (aL - vL) + !kdelta * (-pressure + pStar) );
    //     const real Estar         = cofactor * (E  * (aL - vL) + pStar * aStar - pressure * vL);
    //     const real tauStar       = Estar - Dstar;
    //     auto starStateL          = Conserved{Dstar, S1star, S2star, tauStar, chistar};

    //     auto hllc_flux = left_flux + (starStateL - left_state) * aL - starStateL * vface;

    //     // upwind the concentration flux 
    //     if (hllc_flux.d < static_cast<real>(0.0))
    //         hllc_flux.chi = right_prims.chi * hllc_flux.d;
    //     else
    //         hllc_flux.chi = left_prims.chi  * hllc_flux.d;

    //     return hllc_flux;
    // } else {
    //     const real pressure = right_prims.p;
    //     const real D        = right_state.d;
    //     const real S1       = right_state.s1;
    //     const real S2       = right_state.s2;
    //     const real tau      = right_state.tau;
    //     const real chi      = right_state.chi;
    //     const real E        = tau + D;
    //     const real cofactor = static_cast<real>(1.0) / (aR - aStar);

    //     const real vR         = right_prims.vcomponent(nhat);
    //     const auto kdelta     = kronecker(nhat, 1);
    //     const real Dstar      = cofactor * (aR - vR) * D;
    //     const real chistar    = cofactor * (aR - vR) * chi;
    //     const real S1star     = cofactor * (S1 * (aR - vR) +  kdelta * (-pressure + pStar) );
    //     const real S2star     = cofactor * (S2 * (aR - vR) + !kdelta * (-pressure + pStar) );
    //     const real Estar      = cofactor * (E  * (aR - vR) + pStar * aStar - pressure * vR);
    //     const real tauStar    = Estar - Dstar;
    //     auto starStateR       = Conserved{Dstar, S1star, S2star, tauStar, chistar};

    //     auto hllc_flux = right_flux + (starStateR - right_state) * aR - starStateR * vface;

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
    const luint radius = (first_order) ? 1 : 2;
    const auto n = this->n;
    auto* const cons_data  = cons.data();
    auto* const prim_data  = prims.data();
    auto* const press_data = pressure_guess.data();
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;        
        const auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * threadIdx.y + threadIdx.x : gid;
        if (tid == 0)
            found_failure = self->inFailureState;
        simbi::gpu::api::synchronize();
        
        real invdV = 1.0;
        while (!found_failure && workLeftToDo)
        {
            if (self->mesh_motion && (self->geometry == simbi::Geometry::SPHERICAL))
            {
                const luint ii   = gid % self->nx;
                const luint jj   = gid / self->nx;
                const auto ireal = helpers::get_real_idx(ii, radius, self->xphysical_grid);
                const auto jreal = helpers::get_real_idx(jj, radius, self->yphysical_grid); 
                const real dV    = self->get_cell_volume(ireal, jreal, self->geometry);
                invdV = 1.0 / dV;
            }

            const real D    = cons_data[gid].d   * invdV;
            const real S1   = cons_data[gid].s1  * invdV;
            const real S2   = cons_data[gid].s2  * invdV;
            const real tau  = cons_data[gid].tau * invdV;
            const real Dchi = cons_data[gid].chi * invdV; 
            const real S    = std::sqrt(S1 * S1 + S2 * S2);
            
            real peq = press_data[gid];
            luint iter  = 0;
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
                c2 = self->gamma * pre / (h * rho);

                g = c2 * v2 - static_cast<real>(1.0);
                f = (self->gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                iter++;
                if (iter >= MAX_ITER || std::isnan(peq))
                {
                    const auto ii     = gid % self->nx;
                    const auto jj     = gid / self->nx;
                    const lint ireal  = helpers::get_real_idx(ii, radius, self->xphysical_grid);
                    const lint jreal  = helpers::get_real_idx(jj, radius, self->yphysical_grid); 
                    const real x1l    = self->get_x1face(ireal, self->geometry, 0);
                    const real x1r    = self->get_x1face(ireal, self->geometry, 1);
                    const real x2l    = self->get_x2face(jreal, 0);
                    const real x2r    = self->get_x2face(jreal, 1);
                    const real x1mean = helpers::calc_any_mean(x1l, x1r, self->x1cell_spacing);
                    const real x2mean = helpers::calc_any_mean(x2l, x2r, self->x2cell_spacing);
                    printf("\nCons2Prim cannot converge:\n");
                    printf("Density: %f, Pressure: %f, Vsq: %f, et: %f, xcoord: %.2e, ycoord: %.2e, iter: %lu\n", rho, peq, v2, et,  x1mean, x2mean, iter);
                    self->dt             = INFINITY;
                    found_failure        = true;
                    self->inFailureState = true;
                    simbi::gpu::api::synchronize();
                    break;
                }
            } while (std::abs(peq - pre) >= tol);

            const real inv_et = static_cast<real>(1.0) / (tau + D + peq);
            const real v1     = S1 * inv_et;
            const real v2     = S2 * inv_et;

            press_data[gid] = peq;
            prim_data[gid]  = Primitive{rho, v1, v2, peq, Dchi / D};
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
    const auto xpg      = this->xphysical_grid;
    const auto ypg      = this->yphysical_grid;
    const real step     = (first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);

    #if GPU_CODE
    const auto xextent             = p.blockSize.x;
    const auto yextent             = p.blockSize.y;
    const bool hllc                = this->hllc;
    const real decay_constant      = this->decay_constant;
    const real plm_theta           = this->plm_theta;
    const real gamma               = this->gamma;
    const luint nx                 = this->nx;
    const luint ny                 = this->ny;
    const real dx2                 = this->dx2;
    const real dlogx1              = this->dlogx1;
    const real dx1                 = this->dx1;
    const bool d_all_zeros         = this->d_all_zeros;
    const bool s1_all_zeros        = this->s1_all_zeros;
    const bool s2_all_zeros        = this->s2_all_zeros;
    const bool e_all_zeros         = this->e_all_zeros;
    const real x2min               = this->x2min;
    const real x2max               = this->x2max;
    const bool quirk_smoothing     = this->quirk_smoothing;
    const real pow_dlogr           = std::pow(10, dlogx1);
    const real hubble_param        = this->hubble_param;
    const auto n = this->n;
    #endif

    const luint nbs = (BuildPlatform == Platform::GPU) ? bx * by : nzones;
    const luint extent                = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    // Choice of column major striding by user
    const luint sx = (col_maj) ? 1  : bx;
    const luint sy = (col_maj) ? by :  1;

    auto* const prim_data   = prims.data();
    auto* const cons_data   = cons.data();
    auto* const dens_source = sourceD.data();
    auto* const mom1_source = sourceS1.data();
    auto* const mom2_source = sourceS2.data();
    auto* const erg_source  = sourceTau.data();
    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx) {
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        // auto *const prim_buff = prim_data;
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const auto ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const auto jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        #if GPU_CODE 
        if ((ii >= xpg) || (jj >= ypg)) return;
        #endif

        const auto ia  = ii + radius;
        const auto ja  = jj + radius;
        const auto tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const auto ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const auto txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const auto tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;

        Conserved uxL, uxR, uyL, uyR;
        Conserved fL, fR, gL, gR, frf, flf, grf, glf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR;

        const lint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        // Load Shared memory luinto buffer for active zones plus ghosts
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = prim_data[aid];
            if (ty < radius)
            {
                if (blockIdx.y == p.gridSize.y - 1 && (ja + yextent > ny - radius + ty)) {
                    tyl = ny - radius - ja + ty;
                }
                prim_buff[(tya - radius) * sx + txa] = prim_data[(ja - radius) * nx + ia];
                prim_buff[(tya + tyl   ) * sx + txa] = prim_data[(ja + tyl   ) * nx + ia]; 
            }
            if (tx < radius)
            {   
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                prim_buff[tya * sx + txa - radius] =  prim_data[ja * nx + ia - radius];
                prim_buff[tya * sx + txa +    txl] =  prim_data[ja * nx + ia + txl]; 
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = self->get_x1face(ii, geometry, 0);
        const real x1r    = self->get_x1face(ii, geometry, 1);
        const real vfaceR = x1r * hubble_param;
        const real vfaceL = x1l * hubble_param;
        if (self->first_order)
        {
            xprimsL = prim_buff[( (txa + 0) * sy + (tya + 0) * sx)];
            xprimsR = prim_buff[( (txa + 1) * sy + (tya + 0) * sx)];
            //j+1/2
            yprimsL = prim_buff[( (txa + 0) * sy + (tya + 0) * sx)];
            yprimsR = prim_buff[( (txa + 0) * sy + (tya + 1) * sx)];
            
            // i+1/2
            uxL = self->prims2cons(xprimsL); 
            uxR = self->prims2cons(xprimsR); 
            // j+1/2
            uyL = self->prims2cons(yprimsL);  
            uyR = self->prims2cons(yprimsR); 

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);

            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            // Calc HLL Flux at i+1/2 interface
            if (hllc) {
                frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            } else {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[( (txa - 1) * sy + (tya + 0) * sx )];
            xprimsR = prim_buff[( (txa - 0) * sy + (tya + 0) * sx )];
            //j+1/2
            yprimsL = prim_buff[( (txa - 0) * sy + (tya - 1) * sx )]; 
            yprimsR = prim_buff[( (txa + 0) * sy + (tya - 0) * sx )]; 

            // i+1/2
            uxL = self->prims2cons(xprimsL); 
            uxR = self->prims2cons(xprimsR); 
            // j+1/2
            uyL = self->prims2cons(yprimsL);  
            uyR = self->prims2cons(yprimsR); 

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);

            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            // Calc HLL Flux at i-1/2 interface
            if (self->hllc) {
                flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            } else {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }   
        } else {
            // Coordinate X
            const Primitive xleft_most  = prim_buff[((txa - 2) * sy + tya * sx) % nbs];
            const Primitive xleft_mid   = prim_buff[((txa - 1) * sy + tya * sx) % nbs];
            const Primitive center      = prim_buff[((txa + 0) * sy + tya * sx) % nbs];
            const Primitive xright_mid  = prim_buff[((txa + 1) * sy + tya * sx) % nbs];
            const Primitive xright_most = prim_buff[((txa + 2) * sy + tya * sx) % nbs];

            // Coordinate Y
            const Primitive yleft_most  = prim_buff[(txa * sy + (tya - 2) * sx) % nbs];
            const Primitive yleft_mid   = prim_buff[(txa * sy + (tya - 1) * sx) % nbs];
            const Primitive yright_mid  = prim_buff[(txa * sy + (tya + 1) * sx) % nbs];
            const Primitive yright_most = prim_buff[(txa * sy + (tya + 2) * sx) % nbs];

            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprimsL  = center     + helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center) * plm_theta) * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::minmod((xright_mid - center) * plm_theta, (xright_most - center) * static_cast<real>(0.5), (xright_most - xright_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsL  = center     + helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center) * plm_theta) * static_cast<real>(0.5);  
            yprimsR  = yright_mid - helpers::minmod((yright_mid - center) * plm_theta, (yright_most - center) * static_cast<real>(0.5), (yright_most - yright_mid)*plm_theta) * static_cast<real>(0.5);


            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL  = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);
            uyL  = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);

            fL  = self->prims2flux(xprimsL, 1);
            fR  = self->prims2flux(xprimsR, 1);
            gL  = self->prims2flux(yprimsL, 2);
            gR  = self->prims2flux(yprimsR, 2);

            if (self->hllc)
            {
                if(quirk_smoothing)
                {
                    if (quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                        frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    } else {
                        frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    }

                    if (quirk_strong_shock(yprimsL.p, yprimsR.p)){
                        grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } else {
                        grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    }
                } else {
                    frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                }
            }
            else
            {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            xprimsL  = xleft_mid + helpers::minmod((xleft_mid - xleft_most) * plm_theta, (center - xleft_most) * static_cast<real>(0.5), (center - xleft_mid)*plm_theta) * static_cast<real>(0.5);
            xprimsR  = center    - helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center)*plm_theta)*static_cast<real>(0.5);
            yprimsL  = yleft_mid + helpers::minmod((yleft_mid - yleft_most) * plm_theta, (center - yleft_most) * static_cast<real>(0.5), (center - yleft_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsR  = center    - helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL  = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);
            uyL  = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);

            fL  = self->prims2flux(xprimsL, 1);
            fR  = self->prims2flux(xprimsR, 1);
            gL  = self->prims2flux(yprimsL, 2);
            gR  = self->prims2flux(yprimsR, 2);

            if (self->hllc) {
                if (quirk_smoothing)
                {
                    if (quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                        flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    } else {
                        flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    }
                    
                    if (quirk_strong_shock(yprimsL.p, yprimsR.p)){
                        glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } else {
                        glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } 
                } else {
                    flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                }
            } else {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }
        }

        //Advance depending on geometry
        const luint real_loc = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
        const real d_source  = dens_source[real_loc];
        const real s1_source = mom1_source[real_loc];
        const real s2_source = mom2_source[real_loc];
        const real e_source  = erg_source[real_loc];
        const Conserved source_terms = Conserved{d_source, s1_source, s2_source, e_source} * decay_constant;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
            {
                cons_data[aid] -= ( (frf - flf) / dx1 + (grf - glf)/dx2 - source_terms) * step * self->dt;
                break;
            }
            
            case simbi::Geometry::SPHERICAL:
                {
                const real rl           = x1l + vfaceL * step * self->dt; 
                const real rr           = x1r + vfaceR * step * self->dt;
                const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real tl           = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                const real tr           = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                // const real thmean       = static_cast<real>(0.5) * (tl + tr);
                // const real sint         = std::sin(thmean);
                // const real dV1          = rmean * rmean * (rr - rl);             
                // const real dV2          = rmean * sint * (tr - tl); 
                // const real cot          = std::cos(thmean) / std::sin(thmean);
                const real dcos         = std::cos(tl) - std::cos(tr);
                const real dVtot        = 2.0 * M_PI * (1.0 / 3.0) * (rr * rr * rr - rl * rl * rl) * dcos;
                const real invdV        = 1.0 / dVtot;
                const real s1R          = 2.0 * M_PI * rr * rr * dcos; 
                const real s1L          = 2.0 * M_PI * rl * rl * dcos; 
                const real s2R          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tr); // std::sin(tr);
                const real s2L          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tl); // std::sin(tl);
                const real factor       = (self->mesh_motion) ? dVtot : 1;  


                // Grab central primitives
                const real rhoc = prim_buff[txa * sy + tya * sx].rho;
                const real uc   = prim_buff[txa * sy + tya * sx].v1;
                const real vc   = prim_buff[txa * sy + tya * sx].v2;
                const real pc   = prim_buff[txa * sy + tya * sx].p;

                const real hc   = static_cast<real>(1.0) + gamma * pc/(rhoc * (gamma - static_cast<real>(1.0)));
                const real gam2 = static_cast<real>(1.0)/(static_cast<real>(1.0) - (uc * uc + vc * vc));

                const Conserved geom_source  = {static_cast<real>(0.0), (rhoc * hc * gam2 * vc * vc) / rmean + pc * (s1R - s1L) / dVtot, - (rhoc * hc * gam2 * uc * vc) / rmean + pc * (s2R - s2L) * invdV , static_cast<real>(0.0)};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * self->dt * step * factor;
                break;
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
        } // end switch
    });
    // update x1 enpoints
    const real x1l    = self->get_x1face(0, geometry, 0);
    const real x1r    = self->get_x1face(xphysical_grid, geometry, 1);
    const real vfaceR = x1r * hubble_param;
    const real vfaceL = x1l * hubble_param;
    self->x1min      += step * self->dt * vfaceL;
    self->x1max      += step * self->dt * vfaceR;
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
    real dlogt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    int chkpt_idx,
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
    anyDisplayProps();
    real round_place = 1 / chkpt_interval;
    this->t = tstart;
    this->t_interval =
        t == 0 ? 0
               : dlogt !=0 ? tstart
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

    // Define the source terms
    this->sourceD         = sources[0];
    this->sourceS1        = sources[1];
    this->sourceS2        = sources[2];
    this->sourceTau       = sources[3];

    // Define sim state params
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->init_chkpt_idx  = chkpt_idx;
    this->total_zones     = nx * ny;
    this->first_order     = first_order;
    this->periodic        = boundary_condition == "periodic";
    this->hllc            = hllc;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->dlogt           = dlogt;
    this->xphysical_grid  = (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid  = (first_order) ? ny - 2 : ny - 4;
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = xphysical_grid * yphysical_grid;
    this->quirk_smoothing = quirk_smoothing;
    this->bc              = helpers::boundary_cond_map.at(boundary_condition);
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->x2cell_spacing  = simbi::Cellspacing::LINSPACE;
    this->dx2             = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1          = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1             = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->x1min           = x1[0];
    this->x1max           = x1[xphysical_grid - 1];
    this->x2min           = x2[0];
    this->x2max           = x2[yphysical_grid - 1];
    this->checkpoint_zones= yphysical_grid;
    this->d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    this->s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    this->s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    this->e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (x2max == 0.5 * M_PI){
        this->reflecting_theta = true;
    }

    cons.resize(nzones);
    prims.resize(nzones);
    dt_min.resize(active_zones);
    pressure_guess.resize(nzones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < nzones; i++)
    {
        auto D            = state[0][i];
        auto S1           = state[1][i];
        auto S2           = state[2][i];
        auto E            = state[3][i];
        auto Dchi         = state[4][i];
        auto S            = std::sqrt(S1 * S1 + S2 * S2);
        cons[i]           = Conserved(D, S1, S2, E, Dchi);
        pressure_guess[i] = std::abs(S - D - E);
    }

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_constant = 1 / (1 + std::exp(static_cast<real>(10.0) * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    sr2d::PrimitiveSOA transfer_prims;
    
    // Copy the current SRHD instance over to the device
    // if compiling for CPU, these functions do nothing
    SRHD2D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(SRHD2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD2D));
    cons.copyToGpu();
    prims.copyToGpu();
    pressure_guess.copyToGpu();
    dt_min.copyToGpu();
    sourceD.copyToGpu();
    sourceS1.copyToGpu();
    sourceS2.copyToGpu();
    sourceTau.copyToGpu();

    // Write some info about the setup for writeup later
    setup.x1max          = x1[xphysical_grid - 1];
    setup.x1min          = x1[0];
    setup.x2max          = x2[yphysical_grid - 1];
    setup.x2min          = x2[0];
    setup.nx             = nx;
    setup.ny             = ny;
    setup.xactive_zones  = xphysical_grid;
    setup.yactive_zones  = yphysical_grid;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;
    setup.regime         = "relativistic";
    setup.using_fourvelocity = false;
    setup.x1                 = x1;
    setup.x2                 = x2;

    // // Setup the system
    const luint xblockdim    = xphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : xphysical_grid;
    const luint yblockdim    = yphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : yphysical_grid;
    this->radius             = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->pseudo_radius      = (first_order) ? 1 : 2;
    const luint bx           = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint by           = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint shBlockSpace = bx * by;
    const luint shBlockBytes = shBlockSpace * sizeof(Primitive);
    const auto fullP         = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim}, shBlockBytes);
    const auto activeP       = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);
    
    if (t == 0)
    {
        config_ghosts2D(fullP, cons.data(), nx, ny, first_order, bc);
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
    if (d_outer)
    {
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::gpuMalloc(&dev_outer_zones, ny * sizeof(Conserved));
        }

        outer_zones = new Conserved[ny];
        // #pragma omp parallel for 
        for (luint jj = 0; jj < ny; jj++) {
            const auto jreal = helpers::get_real_idx(jj, radius, yphysical_grid);
            const real dV    = get_cell_volume(xphysical_grid - 1, jreal, geometry);
            outer_zones[jj]  = Conserved{d_outer(x1max, x2[jreal]), s1_outer(x1max, x2[jreal]), s2_outer(x1max, x2[jreal]), e_outer(x1max, x2[jreal])} * dV;
        }
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::copyHostToDevice(dev_outer_zones, outer_zones, ny * sizeof(Conserved));
        }
    }
     
    // Save initial condition
    if (t == 0) {
        write2file(*this, setup, data_directory, t, t_interval, chkpt_interval, yphysical_grid);
        t_interval += chkpt_interval;
    }
    
    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;
    const auto ozones  = (BuildPlatform == Platform::GPU) ? dev_outer_zones : outer_zones;

    // Simulate :)
    while (t < tend & !inFailureState)
    {
        simbi::detail::with_logger(*this, [&](){
            advance(self, activeP, bx, by, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts2D(fullP, cons.data(), nx, ny, first_order, bc, ozones, reflecting_theta);
        });

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(device_self, geometry, activeP, dtShBytes);
        } else {
            adapt_dt();
        }
        t += dt;
        if constexpr(BuildPlatform == Platform::GPU) {
            if (device_self->inFailureState) {
                this->inFailureState = device_self->inFailureState;
            }
        }
    }

    if (detail::logger::ncheck > 0) {
        writeln("Average zone update/sec for:{:>5} iterations was {:>5.2e} zones/sec", detail::logger::n, detail::logger::zu_avg/ detail::logger::ncheck);
    }

    if (outer_zones) {
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::deviceSynch();
            simbi::gpu::api::gpuFree(dev_outer_zones);
            delete[] outer_zones;
        } else {
            delete[] outer_zones;
        }
    }

    std::vector<std::vector<real>> final_prims(5, std::vector<real>(nzones, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v1;
        final_prims[2][ii] = prims[ii].v2;
        final_prims[3][ii] = prims[ii].p;
        final_prims[4][ii] = prims[ii].chi;
    }

    return final_prims;
};
