/*
 * C++ Source to perform 3D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */
#include <chrono>
#include <cmath>
#include "util/device_api.hpp"
#include "util/parallel_for.hpp"
#include "util/printb.hpp"
#include "common/helpers.hip.hpp"
#include "srhydro3D.hip.hpp"
#include "util/logger.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;
constexpr auto write2file = helpers::write_to_file<sr3d::PrimitiveSOA, 3, SRHD3D>;

/* Define typedefs because I am lazy */
using Primitive           = sr3d::Primitive;
using Conserved           = sr3d::Conserved;
using Eigenvals           = sr3d::Eigenvals;
// Default Constructor
SRHD3D::SRHD3D() {}

// Overloaded Constructor
SRHD3D::SRHD3D(
    std::vector<std::vector<real>> &state, 
    luint nx, luint ny, luint nz, real gamma,
    std::vector<real> &x1, 
    std::vector<real> &x2,
    std::vector<real> &x3, 
    real cfl,
    std::string coord_system = "cartesian")
:
    HydroBase(
        state,
        nx,
        ny,
        nz,
        gamma,
        x1,
        x2,
        x3,
        cfl,
        coord_system
    )
{

}

// Destructor
SRHD3D::~SRHD3D() {}
//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------
/**
 * Return a 3D matrix containing the primitive
 * variables density , pressure, and three-velocity
 * 
 * @param  none 
 * @return none
 */
void SRHD3D::cons2prim(const ExecutionPolicy<> &p)
{
    auto* const prim_data  = prims.data();
    auto* const cons_data  = cons.data();
    auto* const press_data = pressure_guess.data(); 
    auto* const troubled_data = troubled_cells.data();
    simbi::parallel_for(p, (luint)0, nzones, [CAPTURE_THIS]   GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;

        auto tid = get_threadId();
        if (tid == 0) 
            found_failure = inFailureState;
        simbi::gpu::api::synchronize();

        while (!found_failure && workLeftToDo)
        {
            const real D    = cons_data[gid].d;
            const real S1   = cons_data[gid].s1;
            const real S2   = cons_data[gid].s2;
            const real S3   = cons_data[gid].s3;
            const real tau  = cons_data[gid].tau;
            const real Dchi = cons_data[gid].chi; 
            const real S    = std::sqrt(S1 * S1 + S2 * S2 + S3 * S3);

            real peq = press_data[gid];
            luint iter  = 0;
            const real tol = D * tol_scale;
            do
            {
                pre = peq;
                et  = tau + D + pre;
                v2  = S * S / (et * et);
                W   = 1 / std::sqrt(1 - v2);
                rho = D / W;
                eps = (tau + (1 - W) * D + (1 - W * W) * pre) / (D * W);

                h  = 1 + eps + pre / rho;
                c2 = gamma * pre / (h * rho);

                g = c2 * v2 - 1;
                f = (gamma - 1) * rho * eps - pre;

                peq = pre - f / g;
                iter++;
                if (iter >= MAX_ITER || std::isnan(peq))
                {
                    troubled_data[gid] = iter;
                    found_failure  = true;
                    inFailureState = true;
                    dt             = INFINITY;
                    break;
                }

            } while (std::abs(peq - pre) >= tol);

            const real inv_et = 1 / (tau + D + peq); 
            const real v1 = S1 * inv_et;
            const real v2 = S2 * inv_et;
            const real v3 = S3 * inv_et;
            press_data[gid] = peq;
            #if FOUR_VELOCITY
                prim_data[gid] = Primitive{D/ W, v1 * W, v2 * W, v3 * W, peq, Dchi / D};
            #else
                prim_data[gid] = Primitive{D/ W, v1, v2, v3, peq, Dchi / D};
            #endif
            workLeftToDo = false;

            if (peq < 0) {
                troubled_data[gid] = iter;
                inFailureState = true;
                found_failure  = true;
                dt = INFINITY;
            }
            simbi::gpu::api::synchronize();
        }
    });
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD3D::calc_eigenvals(
    const Primitive &primsL,
    const Primitive &primsR,
    const luint nhat)
{
    // Separate the left and right Primitive
    const real rhoL = primsL.rho;
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p;
    const real hL   = 1 + gamma * pL / (rhoL * (gamma - 1));

    const real rhoR  = primsR.rho;
    const real vR    = primsR.vcomponent(nhat);
    const real pR    = primsR.p;
    const real hR    = 1 + gamma * pR  / (rhoR * (gamma - 1));

    const real csR = std::sqrt(gamma * pR / (hR * rhoR));
    const real csL = std::sqrt(gamma * pL / (hL * rhoL));

    //-----------Calculate wave speeds based on Shneider et al. 1992
    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            const real vbar  = static_cast<real>(0.5) * (vL + vR);
            const real cbar  = static_cast<real>(0.5) * (csL + csR);
            const real bl    = (vbar - cbar)/(1 - cbar*vbar);
            const real br    = (vbar + cbar)/(1 + cbar*vbar);
            const real aL    = helpers::my_min(bl, (vL - csL)/(1 - vL*csL));
            const real aR    = helpers::my_max(br, (vR + csR)/(1 + vR*csR));

            return Eigenvals(aL, aR, csL, csR);
        }
    
    case simbi::WaveSpeeds::MIGNONE_AND_BODO_05:
        {
            // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
            const real gammaL = 1 / std::sqrt(1 - (vL * vL));
            const real gammaR = 1 / std::sqrt(1 - (vR * vR));
            const real sL = csL*csL/(gammaL*gammaL*(1 - csL*csL));
            const real sR = csR*csR/(gammaR*gammaR*(1 - csR*csR));
            // Define temporaries to save computational cycles
            const real qfL   = 1 / (1 + sL);
            const real qfR   = 1 / (1 + sR);
            const real sqrtR = std::sqrt(sR * (1 - vR * vR + sR));
            const real sqrtL = std::sqrt(sL * (1 - vL * vL + sL));

            const real lamLm = (vL - sqrtL) * qfL;
            const real lamRm = (vR - sqrtR) * qfR;
            const real lamLp = (vL + sqrtL) * qfL;
            const real lamRp = (vR + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

            return Eigenvals(aL, aR, csL, csR);
        }
    case simbi::WaveSpeeds::HUBER_AND_KISSMANN_2021:
        {
            const real gammaL = 1 / std::sqrt(1 - (vL * vL));
            const real gammaR = 1 / std::sqrt(1 - (vR * vR));
            const real uL = gammaL * vL;
            const real uR = gammaR * vR;
            const real sL = csL*csL/(1 - csL * csL);
            const real sR = csR*csR/(1 - csR * csR);
            const real sqrtR = std::sqrt(sR * (gammaR * gammaR - uR * uR + sR));
            const real sqrtL = std::sqrt(sL * (gammaL * gammaL - uL * uL + sL));
            const real qfL   = 1 / (gammaL * gammaL + sL);
            const real qfR   = 1 / (gammaR * gammaR + sR);

            const real lamLm = (gammaL * uL - sqrtL) * qfL;
            const real lamRm = (gammaR * uR - sqrtR) * qfR;
            const real lamLp = (gammaL * uL + sqrtL) * qfL;
            const real lamRp = (gammaR * uR + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

            return Eigenvals(aL, aR, csL, csR);
        }
    default:
        {
            const real aLm = (vL - csL) / (1 - vL * csL);
            const real aLp = (vL + csL) / (1 + vL * csL);
            const real aRm = (vR - csR) / (1 - vR * csR);
            const real aRp = (vR + csR) / (1 + vR * csR);

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
Conserved SRHD3D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real v1  = prims.get_v1();
    const real v2  = prims.get_v2();
    const real v3  = prims.get_v3();
    const real pressure = prims.p;
    const real lorentz_gamma = 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));
    const real h = 1 + gamma * pressure / (rho * (gamma - 1));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * v1,
        rho * h * lorentz_gamma * lorentz_gamma * v2,
        rho * h * lorentz_gamma * lorentz_gamma * v3,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma};
};
//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditonal timestep
template<TIMESTEP_TYPE dt_type>
void SRHD3D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real cfl_dt, v1p, v1m, v2p, v2m, v3p, v3m;
        // Compute the minimum timestep given cfl
        for (luint kk = 0; kk < zphysical_grid; kk++)
        {
            const auto x3l     = get_x3face(kk, 0);
            const auto x3r     = get_x3face(kk, 1);
            const auto dx3     = x3r - x3l; 
            const auto shift_k = kk + idx_active;
            for (luint jj = 0; jj < yphysical_grid; jj++)
            {
                const auto x2l     = get_x2face(jj, 0);
                const auto x2r     = get_x2face(jj, 1);
                const auto dx2     = x2r - x2l; 
                const auto shift_j = jj + idx_active;
                #pragma omp for nowait schedule(static) reduction(min:min_dt)
                for (luint ii = 0; ii < xphysical_grid; ii++)
                {
                    const auto shift_i  = ii + idx_active;
                    const auto aid      = shift_k * nx * ny + shift_j * nx + shift_i;
                    const auto rho      = prims[aid].rho;
                    const auto v1       = prims[aid].get_v1();
                    const auto v2       = prims[aid].get_v2();
                    const auto v3       = prims[aid].get_v3();
                    const auto pressure = prims[aid].p;
                    const auto h        = 1 + gamma * pressure / (rho * (gamma - 1.));
                    const auto cs       = std::sqrt(gamma * pressure / (rho * h));

                    // Left/Right wave speeds
                    if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        v1p = std::abs(v1 + cs) / (1 + v1 * cs);
                        v2p = std::abs(v2 + cs) / (1 + v2 * cs);
                        v3p = std::abs(v3 + cs) / (1 + v3 * cs);
                        v1m = std::abs(v1 - cs) / (1 - v1 * cs);
                        v2m = std::abs(v2 - cs) / (1 - v2 * cs);
                        v3m = std::abs(v3 - cs) / (1 - v3 * cs);
                    } else {
                        v1p = 1;
                        v1m = 1;
                        v2p = 1;
                        v2m = 1;
                        v3p = 1;
                        v3m = 1;
                    }

                    const auto x1l     = get_x1face(ii, geometry, 0);
                    const auto x1r     = get_x1face(ii, geometry, 1);
                    const auto dx1     = x1r - x1l; 
                    switch (geometry)
                    {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = std::min(
                                    {dx1 / (std::max(v1p, v1m)),
                                     dx2 / (std::max(v2p, v2m)),
                                     dx3 / (std::max(v3p, v3m))});
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                        {
                            const auto rmean = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                            const real th    = static_cast<real>(0.5) * (x2r + x2l);
                            const auto rproj = rmean * std::sin(th);
                            cfl_dt = std::min(
                                        {       dx1 / (std::max(v1p, v1m)),
                                        rmean * dx2 / (std::max(v2p, v2m)),
                                        rproj * dx3 / (std::max(v3p, v3m))});
                            break;
                        }
                    default:
                        {
                            const auto rmean = static_cast<real>(2.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                            cfl_dt = std::min(
                                        {       dx1 / (std::max(v1p, v1m)),
                                        rmean * dx2 / (std::max(v2p, v2m)),
                                                dx3 / (std::max(v3p, v3m))});
                            break;
                        }
                    }
                        
                    min_dt = helpers::my_min(min_dt, cfl_dt);
                    
                } // end ii 
            } // end jj
        } // end kk
    } // end parallel region
    dt = cfl * min_dt;
};

template<TIMESTEP_TYPE dt_type>
void SRHD3D::adapt_dt(const ExecutionPolicy<> &p)
{
    
    #if GPU_CODE
    {
        helpers::compute_dt<Primitive, dt_type><<<p.gridSize,p.blockSize>>>(this, prims.data(), dt_min.data(), geometry);
        helpers::deviceReduceWarpAtomicKernel<3><<<p.gridSize, p.blockSize>>>(this, dt_min.data(), active_zones);
        gpu::api::deviceSynch();
    }
    #endif
}
//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
GPU_CALLABLE_MEMBER
Conserved SRHD3D::prims2flux(const Primitive &prims, const luint nhat = 1)
{
    const real rho      = prims.rho;
    const real v1       = prims.get_v1();
    const real v2       = prims.get_v2();
    const real v3       = prims.get_v3();
    const real chi      = prims.chi;
    const real vn       = prims.vcomponent(nhat);
    const real pressure = prims.p;
    const real lorentz_gamma = 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));

    const real h  = 1 + gamma * pressure / (rho * (gamma - 1));
    const real D  = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * v1;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * v2;
    const real S3 = rho * lorentz_gamma * lorentz_gamma * h * v3;
    const real Sj = (nhat == 1) ? S1 : (nhat == 2) ? S2 : S3;
    // const real tau = rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return Conserved{
        D  * vn, 
        S1 * vn + helpers::kronecker(nhat, 1) * pressure, 
        S2 * vn + helpers::kronecker(nhat, 2) * pressure, 
        S3 * vn + helpers::kronecker(nhat, 3) * pressure,  
        Sj - D * vn, 
        D * vn * chi
    };
};

GPU_CALLABLE_MEMBER
Conserved SRHD3D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const   luint nhat)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL = lambda.aL;
    const real aR = lambda.aR;

    // Calculate plus/minus alphas
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;
    const real vface = 0.0;
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
    if (net_flux.d < 0)
        net_flux.chi = right_prims.chi * net_flux.d;
    else
        net_flux.chi = left_prims.chi  * net_flux.d;

    // Compute the HLL Flux component-wise
    return net_flux;
};

GPU_CALLABLE_MEMBER
Conserved SRHD3D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const   luint nhat = 1)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL = lambda.aL;
    const real aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if (0 <= aL) {
        return left_flux;
    }
    else if (0 >= aR) {
        return right_flux;
    }

    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux 
        = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aRp * aLm) 
            / (aRp - aLm);

    const real uhlld   = hll_state.d;
    const real uhlls1  = hll_state.s1;
    const real uhlls2  = hll_state.s2;
    const real uhlls3  = hll_state.s3;
    const real uhlltau = hll_state.tau;
    const real fhlld   = hll_flux.d;
    const real fhlls1  = hll_flux.s1;
    const real fhlls2  = hll_flux.s2;
    const real fhlls3  = hll_flux.s3;
    const real fhlltau = hll_flux.tau;
    const real e  = uhlltau + uhlld;
    const real s  = (nhat == 1) ? uhlls1 : (nhat == 2) ? uhlls2 : uhlls3;
    const real fe = fhlltau + fhlld;
    const real fs = (nhat == 1) ? fhlls1 : (nhat == 2) ? fhlls2 : fhlls3;

    //------Calculate the contact wave velocity and pressure
    const real a = fe;
    const real b = -(e + fs);
    const real c = s;
    const real quad  = -static_cast<real>(0.5) * (b + helpers::sgn(b) * std::sqrt(b * b - 4.0 * a * c));
    const real aStar = c * (1 / quad);
    const real pStar = -aStar * fe + fs;

    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D   = left_state.d;
        const real S1  = left_state.s1;
        const real S2  = left_state.s2;
        const real S3  = left_state.s3;
        const real tau = left_state.tau;
        const real E   = tau + D;
        const real cofactor = 1 / (aL - aStar);
        //--------------Compute the L Star State----------
        switch (nhat)
        {
            case 1:
            {
                const real v1 = left_prims.get_v1();
                // Left Star State in x-direction of coordinate lattice
                const real Dstar    = cofactor * (aL - v1) * D;
                const real S1star   = cofactor * (S1 * (aL - v1) - pressure + pStar);
                const real S2star   = cofactor * (aL - v1) * S2;
                const real S3star   = cofactor * (aL - v1) * S3;
                const real Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
                const real tauStar  = Estar - Dstar;

                const auto interstate_left = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return left_flux + (interstate_left - left_state) * aL;
            }

            case 2:
            {
                const real v2 = left_prims.get_v2();
                // Start States in y-direction in the coordinate lattice
                const real Dstar   = cofactor * (aL - v2) * D;
                const real S1star  = cofactor * (aL - v2) * S1;
                const real S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
                const real S3star  = cofactor * (aL - v2) * S3;
                const real Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
                const real tauStar = Estar - Dstar;

                const auto interstate_left = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return left_flux + (interstate_left - left_state) * aL;
            }

            default: // nhat == 3
            {
                const real v3 = left_prims.get_v3();
                // Start States in y-direction in the coordinate lattice
                const real Dstar   = cofactor * (aL - v3) * D;
                const real S1star  = cofactor * (aL - v3) * S1;
                const real S2star  = cofactor * (aL - v3) * S2;
                const real S3star  = cofactor * (S3 * (aL - v3) - pressure + pStar);
                const real Estar   = cofactor * (E * (aL - v3) + pStar * aStar - pressure * v3);
                const real tauStar = Estar - Dstar;

                const auto interstate_left = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return left_flux + (interstate_left - left_state) * aL;
            }
            
        } // end switch
    } else {
        const real pressure = right_prims.p;
        const real D   = right_state.d;
        const real S1  = right_state.s1;
        const real S2  = right_state.s2;
        const real S3  = right_state.s3;
        const real tau = right_state.tau;
        const real E   = tau + D;
        const real cofactor = 1 / (aR - aStar);

        /* Compute the L/R Star State */
        switch (nhat)
        {
            case 1:
            {
                const real v1 = right_prims.get_v1();
                // Left Star State in x-direction of coordinate lattice
                const real Dstar    = cofactor * (aR - v1) * D;
                const real S1star   = cofactor * (S1 * (aR - v1) - pressure + pStar);
                const real S2star   = cofactor * (aR - v1) * S2;
                const real S3star   = cofactor * (aR - v1) * S3;
                const real Estar    = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
                const real tauStar  = Estar - Dstar;

                const auto interstate_right = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return right_flux + (interstate_right - right_state) * aR;
            }

            case 2:
            {
                const real v2 = right_prims.get_v2();
                // Start States in y-direction in the coordinate lattice
                const real Dstar   = cofactor * (aR - v2) * D;
                const real S1star  = cofactor * (aR - v2) * S1;
                const real S2star  = cofactor * (S2 * (aR - v2) - pressure + pStar);
                const real S3star  = cofactor * (aR - v2) * S3;
                const real Estar   = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
                const real tauStar = Estar - Dstar;

                const auto interstate_right = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return right_flux + (interstate_right - right_state) * aR;
            }

            default: //nhat == 3
            {
                const real v3 = right_prims.get_v3();
                // Start States in y-direction in the coordinate lattice
                const real Dstar   = cofactor * (aR - v3) * D;
                const real S1star  = cofactor * (aR - v3) * S1;
                const real S2star  = cofactor * (aR - v3) * S2;
                const real S3star  = cofactor * (S3 * (aR - v3) - pressure + pStar);
                const real Estar   = cofactor * (E * (aR - v3) + pStar * aStar - pressure * v3);
                const real tauStar = Estar - Dstar;

                const auto interstate_right = Conserved(Dstar, S1star, S2star, S3star, tauStar);

                //---------Compute the L Star Flux
                return right_flux + (interstate_right - right_state) * aR;
            }
        } // end switch
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
void SRHD3D::advance(
    const ExecutionPolicy<> &p,
    const luint sx,
    const luint sy)
{
    const luint xpg = this->xphysical_grid;
    const luint ypg = this->yphysical_grid;
    const luint zpg = this->zphysical_grid;

    #if GPU_CODE
    const luint xextent = p.blockSize.x;
    const luint yextent = p.blockSize.y;
    const luint zextent = p.blockSize.z;
    #endif 

    const luint extent = p.get_full_extent();
    auto* const prim_data   = prims.data();
    auto* const cons_data   = cons.data();
    auto* const dens_source = sourceD.data();
    auto* const mom1_source = sourceS1.data();
    auto* const mom2_source = sourceS2.data();
    auto* const mom3_source = sourceS3.data();
    auto* const erg_source  = sourceTau.data();
    auto* const object_data = object_pos.data();

    simbi::parallel_for(p, (luint)0, extent, [CAPTURE_THIS] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint kk  = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::detail::get_height(idx, xpg, ypg);
        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::detail::get_row(idx, xpg, ypg, kk);
        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::detail::get_column(idx, xpg, ypg, kk);
        #if GPU_CODE
        if ((ii >= xpg) || (jj >= ypg) || (kk >= zpg)) return;
        #endif 

        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint ka  = kk + radius;
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x : 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y : 0;
        const luint tz  = (BuildPlatform == Platform::GPU) ? threadIdx.z : 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;
        const luint tza = (BuildPlatform == Platform::GPU) ? tz + radius : ka;

        Conserved uxL, uxR, uyL, uyR, uzL, uzR;
        Conserved fL, fR, gL, gR, hL, hR, frf, flf, grf, glf, hrf, hlf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR, zprimsL, zprimsR;

        const luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            luint tzl = zextent;
            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tza * sx * sy + tya * sx + txa] = prim_data[aid];
            if (tz == 0)    
            {
                if ((blockIdx.z == p.gridSize.z - 1) && (ka + zextent > nz - radius + tz)) {
                    tzl = nz - radius - ka + tz;
                }
                for (int q = 1; q < radius + 1; q++) {
                    const auto re = tzl + q - 1;
                    prim_buff[(tza - q) * sx * sy + tya * sx + txa]  = prim_data[(ka - q) * nx * ny + ja * nx + ia];
                    prim_buff[(tza + re) * sx * sy + tya * sx + txa] = prim_data[(ka + re) * nx * ny + ja * nx + ia];
                } 
            }
            if (ty == 0)    
            {
                if ((blockIdx.y == p.gridSize.y - 1) && (ja + yextent > ny - radius + ty)) {
                    tyl = ny - radius - ja + ty;
                }
                for (int q = 1; q < radius + 1; q++) {
                    const auto re = tyl + q - 1;
                    prim_buff[tza * sx * sy + (tya - q) * sx + txa]  = prim_data[ka * nx * ny + (ja - q) * nx + ia];
                    prim_buff[tza * sx * sy + (tya + re) * sx + txa] = prim_data[ka * nx * ny + (ja + re) * nx + ia];
                } 
            }
            if (tx == 0)
            {   
                if ((blockIdx.x == p.gridSize.x - 1) && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                for (int q = 1; q < radius + 1; q++) {
                    const auto re = txl + q - 1;
                    prim_buff[tza * sx * sy + tya * sx + txa - q]  =  prim_data[ka * nx * ny + ja * nx + ia - q];
                    prim_buff[tza * sx * sy + tya * sx + txa + re] =  prim_data[ka * nx * ny + ja * nx + ia + re]; 
                }
            }
            simbi::gpu::api::synchronize();
        #endif

        const bool object_to_my_left  = object_data[kk * xpg * ypg + jj * xpg +  helpers::my_max(static_cast<lint>(ii - 1), static_cast<lint>(0))];
        const bool object_to_my_right = object_data[kk * xpg * ypg + jj * xpg +  helpers::my_min(ii + 1,  xpg - 1)];
        const bool object_in_front    = object_data[kk * xpg * ypg + helpers::my_min(jj + 1, ypg - 1) * xpg +  ii];
        const bool object_behind      = object_data[kk * xpg * ypg + helpers::my_max(static_cast<lint>(jj - 1), static_cast<lint>(0)) * xpg + ii];
        const bool object_above_me    = object_data[helpers::my_min(kk + 1, zpg - 1)  * xpg * ypg + jj * xpg +  ii];
        const bool object_below_me    = object_data[helpers::my_max(static_cast<lint>(kk - 1), static_cast<lint>(0)) * xpg * ypg + jj * xpg +  ii];

        if (first_order) [[unlikely]] {
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            //j+1/2
            yprimsL = prim_buff[tza * sx * sy + (tya + 0) * sx + txa];
            yprimsR = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
            //j+1/2
            zprimsL = prim_buff[(tza + 0) * sx * sy + tya * sx + txa];
            zprimsR = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];

            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                xprimsR.v2  =  xprimsL.v2;
                xprimsR.v3  =  xprimsL.v3;
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                yprimsR.v2  = -yprimsL.v2;
                yprimsR.v3  =  yprimsL.v3;
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                zprimsR.v2  =  zprimsL.v2;
                zprimsR.v3  = -zprimsL.v3;
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);

            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);

            uzL = prims2cons(zprimsL);
            uzR = prims2cons(zprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);

            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            hL = prims2flux(zprimsL, 3);
            hR = prims2flux(zprimsR, 3);

            // Calc HLL Flux at i+1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            //j+1/2
            yprimsL = prim_buff[tza * sx * sy + (tya - 1) * sx + txa]; 
            yprimsR = prim_buff[tza * sx * sy + (tya + 0) * sx + txa]; 
            //k+1/2
            zprimsL = prim_buff[(tza - 1) * sx * sy + tya * sx + txa]; 
            zprimsR = prim_buff[(tza - 0) * sx * sy + tya * sx + txa]; 

            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                xprimsL.v2  =  xprimsR.v2;
                xprimsL.v3  =  xprimsR.v3;
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                yprimsL.v2  = -yprimsR.v2;
                yprimsL.v3  =  yprimsR.v3;
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                zprimsL.v2  =  zprimsR.v2;
                zprimsL.v3  = -zprimsR.v3;
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);

            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);

            uzL = prims2cons(zprimsL);
            uzR = prims2cons(zprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);

            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            hL = prims2flux(zprimsL, 3);
            hR = prims2flux(zprimsR, 3);

            // Calc HLL Flux at i-1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            }
        } else{
            // Coordinate X
            const Primitive xleft_most  = prim_buff[tza * sx * sy + tya * sx + (txa - 2)];
            const Primitive xleft_mid   = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            const Primitive center      = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            const Primitive xright_mid  = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            const Primitive xright_most = prim_buff[tza * sx * sy + tya * sx + (txa + 2)];

            // Coordinate Y
            const Primitive yleft_most  = prim_buff[tza * sx * sy + (tya - 2) * sx + txa];
            const Primitive yleft_mid   = prim_buff[tza * sx * sy + (tya - 1) * sx + txa];
            const Primitive yright_mid  = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
            const Primitive yright_most = prim_buff[tza * sx * sy + (tya + 2) * sx + txa];

            // Coordinate z
            const Primitive zleft_most  = prim_buff[(tza - 2) * sx * sy + tya * sx + txa];
            const Primitive zleft_mid   = prim_buff[(tza - 1) * sx * sy + tya * sx + txa];
            const Primitive zright_mid  = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
            const Primitive zright_most = prim_buff[(tza + 2) * sx * sy + tya * sx + txa];
            
            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprimsL  = center     + helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta)   * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::plm_gradient(xright_mid, center, xright_most, plm_theta) * static_cast<real>(0.5);
            yprimsL  = center     + helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta)   * static_cast<real>(0.5);  
            yprimsR  = yright_mid - helpers::plm_gradient(yright_mid, center, yright_most, plm_theta) * static_cast<real>(0.5);
            zprimsL  = center     + helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta)   * static_cast<real>(0.5);  
            zprimsR  = zright_mid - helpers::plm_gradient(zright_mid, center, zright_most, plm_theta) * static_cast<real>(0.5);


            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                xprimsR.v2  =  xprimsL.v2;
                xprimsR.v3  =  xprimsL.v3;
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                yprimsR.v2  = -yprimsL.v2;
                yprimsR.v3  =  yprimsL.v3;
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                zprimsR.v2  =  zprimsL.v2;
                zprimsR.v3  = -zprimsL.v3;
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);

            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);

            uzL = prims2cons(zprimsL);
            uzR = prims2cons(zprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);
            hL = prims2flux(zprimsL, 3);
            hR = prims2flux(zprimsR, 3);

            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            xprimsL  = xleft_mid  + helpers::plm_gradient(xleft_mid, xleft_most, center, plm_theta) * static_cast<real>(0.5); 
            xprimsR  = center     - helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta) * static_cast<real>(0.5);
            yprimsL  = yleft_mid  + helpers::plm_gradient(yleft_mid, yleft_most, center, plm_theta) * static_cast<real>(0.5); 
            yprimsR  = center     - helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta) * static_cast<real>(0.5);
            zprimsL  = zleft_mid  + helpers::plm_gradient(zleft_mid, zleft_most, center, plm_theta) * static_cast<real>(0.5);
            zprimsR  = center     - helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta) * static_cast<real>(0.5);

            
            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                xprimsL.v2  =  xprimsR.v2;
                xprimsL.v3  =  xprimsR.v3;
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                yprimsL.v2  = -yprimsR.v2;
                yprimsL.v3  =  yprimsR.v3;
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                zprimsL.v2  =  zprimsR.v2;
                zprimsL.v3  = -zprimsR.v3;
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            // Calculate the left and right states using the reconstructed PLM Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);
            uzL = prims2cons(zprimsL);
            uzR = prims2cons(zprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);
            hL = prims2flux(zprimsL, 3);
            hR = prims2flux(zprimsR, 3);

            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
                break;
            }

        }// end else 

        //Advance depending on geometry
        const luint real_loc = kk * xpg * ypg + jj * xpg + ii;
        const real d_source  = den_source_all_zeros     ? 0.0 : dens_source[real_loc];
        const real s1_source = mom1_source_all_zeros    ? 0.0 : mom1_source[real_loc];
        const real s2_source = mom2_source_all_zeros    ? 0.0 : mom2_source[real_loc];
        const real s3_source = mom3_source_all_zeros    ? 0.0 : mom3_source[real_loc];
        const real e_source  = energy_source_all_zeros  ? 0.0 : erg_source[real_loc];
        const Conserved source_terms = Conserved{d_source, s1_source, s2_source, s3_source, e_source} * time_constant;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    cons_data[aid] -= ( (frf  - flf ) * invdx1 + (grf - glf) * invdx2 + (hrf - hlf) * invdx3 - source_terms) * dt * step;
                    break;
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl     = (ii > 0 ) ? x1min * std::pow(10, (ii -static_cast<real>(0.5)) * dlogx1) :  x1min;
                    const real rr     = (ii < xpg - 1) ? rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                    const real tl     = (jj > 0 ) ? x2min + (jj - static_cast<real>(0.5)) * dx2 :  x2min;
                    const real tr     = (jj < ypg - 1) ? tl + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                    const real ql     = (kk > 0 ) ? x3min + (kk - static_cast<real>(0.5)) * dx3 :  x3min;
                    const real qr     = (kk < zpg - 1) ? ql + dx3 * (kk == 0 ? 0.5 : 1.0) :  x3max; 
                    const real rmean  = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                    const real s1R    = rr * rr; 
                    const real s1L    = rl * rl; 
                    const real s2R    = std::sin(tr);
                    const real s2L    = std::sin(tl);
                    const real thmean = static_cast<real>(0.5) * (tl + tr);
                    const real sint   = std::sin(thmean);
                    const real dV1    = rmean * rmean * (rr - rl);             
                    const real dV2    = rmean * sint  * (tr - tl); 
                    const real dV3    = rmean * sint  * (qr - ql); 
                    const real cot    = std::cos(thmean) / sint;

                    // Grab central primitives
                    const real rhoc = prim_buff[txa + tya * sx + tza * sx * sy].rho;
                    const real uc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v1();
                    const real vc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v2();
                    const real wc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v3();
                    const real pc   = prim_buff[txa + tya * sx + tza * sx * sy].p;

                    const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                    const real gam2 = 1/(1 - (uc * uc + vc * vc + wc * wc));

                    const Conserved geom_source  = {0, 
                        (rhoc * hc * gam2 * (vc * vc + wc * wc)) / rmean + pc * (s1R - s1L) / dV1,
                        rhoc * hc * gam2 * (wc * wc * cot - uc * vc) / rmean + pc * (s2R - s2L)/dV2 , 
                        - rhoc * hc * gam2 * wc * (uc + vc * cot) / rmean, 
                        0
                    };
                    cons_data[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 + (hrf - hlf) / dV3 - geom_source - source_terms) * dt * step;
                    break;
                }
            default:
                {
                    const real rl           = (ii > 0 ) ? x1min * std::pow(10, (ii -static_cast<real>(0.5)) * dlogx1) :  x1min;
                    const real rr           = (ii < xpg - 1) ? rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                    const real tl           = (jj > 0 ) ? x2min + (jj - static_cast<real>(0.5)) * dx2 :  x2min;
                    const real tr           = (jj < ypg - 1) ? tl + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                    const real zl           = (kk > 0 ) ? x3min + (kk - static_cast<real>(0.5)) * dx3 :  x3min;
                    const real zr           = (kk < zpg - 1) ? zl + dx3 * (kk == 0 ? 0.5 : 1.0) :  x3max; 
                    const real rmean        = static_cast<real>(2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                    const real s1R          = rr * (zr - zl) * (tr - tl); 
                    const real s1L          = rl * (zr - zl) * (tr - tl); 
                    const real s2R          = (rr - rl) * (zr - rl);
                    const real s2L          = (rr - rl) * (zr - rl);
                    // const real s3L          = rmean * (rr - rl) * (tr - tl);
                    // const real s3R          = s3L;
                    // const real thmean       = static_cast<real>(0.5) * (tl + tr);
                    const real dV           = rmean  * (rr - rl) * (zr - zl) * (tr - tl);
                    const real invdV        = 1/ dV;

                    // Grab central primitives
                    const real rhoc = prim_buff[txa + tya * sx + tza * sx * sy].rho;
                    const real uc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v1();
                    const real vc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v2();
                    const real wc   = prim_buff[txa + tya * sx + tza * sx * sy].get_v3();
                    const real pc   = prim_buff[txa + tya * sx + tza * sx * sy].p;

                    const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                    const real gam2 = 1/(1 - (uc * uc + vc * vc + wc * wc));

                    const Conserved geom_source  = {0, (rhoc * hc * gam2 * (vc * vc + wc * wc)) / rmean + pc * (s1R - s1L) * invdV, - (rhoc * hc * gam2 * uc * vc) / rmean , 0, 0};
                    cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV + (hrf - hlf) * invdV - geom_source - source_terms) * dt * step;
                    break;
                }
        } // end switch
    });
}
//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD3D::simulate3D(
    const std::vector<std::vector<real>> &sources,
    const std::vector<bool> &object_cells,
    real tstart, 
    real tend, 
    real dlogt, 
    real plm_theta,
    real engine_duration, 
    real chkpt_interval,
    int  chkpt_idx,
    std::string data_directory, 
    std::vector<std::string> boundary_conditions,
    bool first_order,
    bool linspace, 
    const std::string solver,
    bool constant_sources,
    std::vector<std::vector<real>> boundary_sources)
{   
    helpers::anyDisplayProps();

    // Define the source terms
    this->sourceD        = sources[0];
    this->sourceS1       = sources[1];
    this->sourceS2       = sources[2];
    this->sourceS3       = sources[3];
    this->sourceTau      = sources[4];
    
    // Define simulation params
    this->t               = tstart;
    this->object_pos      = object_cells;
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->engine_duration = engine_duration;
    this->total_zones     = nx * ny * nz;
    this->first_order     = first_order;
    this->sim_solver      = helpers::solver_map.at(solver);
    this->dlogt           = dlogt;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->xphysical_grid  = (first_order) ? nx - 2: nx - 4;
    this->yphysical_grid  = (first_order) ? ny - 2: ny - 4;
    this->zphysical_grid  = (first_order) ? nz - 2: nz - 4;
    this->idx_active      = (first_order) ? 1     : 2;
    this->active_zones    = xphysical_grid * yphysical_grid * zphysical_grid;
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->x2cell_spacing  = simbi::Cellspacing::LINSPACE;
    this->x3cell_spacing  = simbi::Cellspacing::LINSPACE;
    this->dx3             = (x3[zphysical_grid - 1] - x3[0]) / (zphysical_grid - 1);
    this->dx2             = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1          = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1             = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->invdx1          = 1 / dx1;
    this->invdx2          = 1 / dx2;
    this->invdx3          = 1 / dx3;
    this->x1min           = x1[0];
    this->x1max           = x1[xphysical_grid - 1];
    this->x2min           = x2[0];
    this->x2max           = x2[yphysical_grid - 1];
    this->x3min           = x3[0];
    this->x3max           = x3[zphysical_grid - 1];
    this->checkpoint_zones= zphysical_grid;
    this->den_source_all_zeros     = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    this->mom1_source_all_zeros    = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    this->mom2_source_all_zeros    = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    this->mom3_source_all_zeros    = std::all_of(sourceS3.begin(),  sourceS3.end(),  [](real i) {return i == 0;});
    this->energy_source_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    define_tinterval(t, dlogt, chkpt_interval, chkpt_idx);
    define_chkpt_idx(chkpt_idx);

    // Stuff for moving mesh 
    // TODO: make happen at some point
    this->hubble_param = 0.0; //adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (x2max == 0.5 * M_PI){
        this->half_sphere = true;
    }

    inflow_zones.resize(6);
    for (int i = 0; i < 6; i++) {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = Conserved{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2], boundary_sources[i][3], boundary_sources[i][4]};
    }

    // Write some info about the setup for writeup later
    setup.x1max              = x1[xphysical_grid - 1];
    setup.x1min              = x1[0];
    setup.x2max              = x2[yphysical_grid - 1];
    setup.x2min              = x2[0];
    setup.x3max              = x3[zphysical_grid - 1];
    setup.x3min              = x3[0];
    setup.nx                 = nx;
    setup.ny                 = ny;
    setup.nz                 = nz;
    setup.xactive_zones      = xphysical_grid;
    setup.yactive_zones      = yphysical_grid;
    setup.zactive_zones      = zphysical_grid;
    setup.linspace           = linspace;
    setup.ad_gamma           = gamma;
    setup.first_order        = first_order;
    setup.coord_system       = coord_system;
    setup.using_fourvelocity = (VelocityType == Velocity::FourVelocity);
    setup.regime             = "relativistic";
    setup.x1                 = x1;
    setup.x2                 = x2;
    setup.x3                 = x3;
    setup.mesh_motion        = mesh_motion;
    setup.boundary_conditions  = boundary_conditions;
    setup.dimensions           = 3;

    cons.resize(nzones);
    prims.resize(nzones);
    troubled_cells.resize(nzones, 0);
    dt_min.resize(active_zones);
    pressure_guess.resize(nzones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state[0].size(); i++)
    {
        auto D            = state[0][i];
        auto S1           = state[1][i];
        auto S2           = state[2][i];
        auto S3           = state[3][i];
        auto E            = state[4][i];
        auto S            = std::sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        cons[i]           = Conserved{D, S1, S2, S3, E};
        pressure_guess[i] = std::abs(S - D - E);
    }

    cons.copyToGpu();
    prims.copyToGpu();
    pressure_guess.copyToGpu();
    dt_min.copyToGpu();
    sourceD.copyToGpu();
    sourceS1.copyToGpu();
    sourceS2.copyToGpu();
    sourceS3.copyToGpu();
    sourceTau.copyToGpu();
    object_pos.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();
    troubled_cells.copyToGpu();

    // Setup the system
    const luint xblockdim    = xphysical_grid > gpu_block_dimx ? gpu_block_dimx : xphysical_grid;
    const luint yblockdim    = yphysical_grid > gpu_block_dimy ? gpu_block_dimy : yphysical_grid;
    const luint zblockdim    = zphysical_grid > gpu_block_dimz ? gpu_block_dimz : zphysical_grid;
    this->radius             = (first_order) ? 1 : 2;
    this->step               = (first_order) ? 1 : static_cast<real>(0.5);
    const luint xstride      = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint ystride      = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint shBlockSpace = (xblockdim + 2 * radius) * (yblockdim + 2 * radius) * (zblockdim + 2 * radius);
    const luint shBlockBytes = shBlockSpace * sizeof(Primitive);
    const auto fullP         = simbi::ExecutionPolicy({nx, ny, nz}, {xblockdim, yblockdim, zblockdim});
    const auto activeP       = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid, zphysical_grid}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU){
        writeln("Requested shared memory: {} bytes", shBlockBytes);
    }
    
    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP);
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(activeP);
    } else {
        cons2prim(fullP);
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }
    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || chkpt_idx == 0) {
        write2file(*this, setup, data_directory, t, 0, chkpt_interval, zphysical_grid);
        helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
    }
    
    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&](){
        if (inFailureState){
            return;
        }
        advance(activeP, xstride, ystride);
        cons2prim(fullP);
        helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
    });

    if (inFailureState){
        emit_troubled_cells();
    }
    
    std::vector<std::vector<real>> final_prims(5, std::vector<real>(nzones, 0));
    for (luint ii = 0; ii < nzones; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v1;
        final_prims[2][ii] = prims[ii].v2;
        final_prims[3][ii] = prims[ii].v3;
        final_prims[4][ii] = prims[ii].p;
    }

    return final_prims;
};
