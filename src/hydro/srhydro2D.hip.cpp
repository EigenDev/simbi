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
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p;
    const real hL   = 1 + gamma * pL / (rhoL * (gamma - 1));

    const real rhoR  = primsR.rho;
    const real vR    = primsR.vcomponent(nhat);
    const real pR    = primsR.p;
    const real hR    = 1 + gamma * pR / (rhoR * (gamma - 1));

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
            const real sL = csL*csL/(gamma*gamma*(1 - csL*csL));
            const real sR = csR*csR/(gamma*gamma*(1 - csR*csR));
            // Define temporaries to save computational cycles
            const real qfL   = 1 / (1 + sL);
            const real qfR   = 1 / (1 + sR);
            const real sqrtR = std::sqrt(sR * (1 - vR * vR + sR));
            const real sqrtL = std::sqrt(sL * (1 - vL * vL + sL));

            const real lamLm = (vL - sqrtL) * qfL;
            const real lamRm = (vR - sqrtR) * qfR;
            const real lamLp = (vL + sqrtL) * qfL;
            const real lamRp = (vR + sqrtR) * qfR;

            real aL = lamLm < lamRm ? lamLm : lamRm;
            real aR = lamLp > lamRp ? lamLp : lamRp;

            // Smoothen for rarefaction fan
            aL = helpers::my_min(aL, (vL - csL) / (1 - vL * csL));
            aR = helpers::my_max(aR, (vR + csR) / (1 + vR * csR));

            return Eigenvals(aL, aR, csL, csR);
        }
    default: // NAIVE estimates
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
Conserved SRHD2D::prims2cons(const Primitive &prims) const
{
    const real rho           = prims.rho;
    const real v1            = prims.get_v1();
    const real v2            = prims.get_v2();
    const real pressure      = prims.p;
    const real lorentz_gamma = 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));
    const real h             = 1 + gamma * pressure / (rho * (gamma - 1));

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
template<TIMESTEP_TYPE dt_type>
void SRHD2D::adapt_dt()
{
    if (use_omp) {
        real min_dt = INFINITY;
        #pragma omp parallel 
        {
            real v1p, v1m, v2p, v2m, cfl_dt;
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
                    const auto v1       = prims[aid].get_v1();
                    const auto v2       = prims[aid].get_v2();
                    const auto pressure = prims[aid].p;
                    const auto h        = 1.0 + gamma * pressure / (rho * (gamma - 1.0));
                    const auto cs       = std::sqrt(gamma * pressure / (rho * h));

                    //================ Plus / Minus Wave speed components -================

                    if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        v1p = std::abs(v1 + cs) / (1.0 + v1 * cs);
                        v2p = std::abs(v2 + cs) / (1.0 + v2 * cs);
                        v1m = std::abs(v1 - cs) / (1.0 - v1 * cs);
                        v2m = std::abs(v2 - cs) / (1.0 - v2 * cs);
                    } else {
                        v1p = 1;
                        v1m = 1;
                        v2p = 1;
                        v2m = 1;
                    }
                    switch (geometry)
                    {
                        case simbi::Geometry::CARTESIAN:
                            {
                                if (mesh_motion) {
                                    v1p = std::abs(v1p - hubble_param);
                                    v1m = std::abs(v1m - hubble_param);
                                }
                                cfl_dt = helpers::my_min(dx1 / (helpers::my_max(v1p, v1m)), dx2 / (helpers::my_max(v2p, v2m)));
                                break;
                            }
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
                                    v1p = std::abs(v1p - vfaceR);
                                    v1m = std::abs(v1m - vfaceL);
                                }
                                cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                                break;
                            }

                        case simbi::Geometry::PLANAR_CYLINDRICAL:
                            {
                                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                                const real dtheta = tr - tl;
                                const real x1l    = get_x1face(ii, geometry, 0);
                                const real x1r    = get_x1face(ii, geometry, 1);
                                const real dr     = x1r - x1l;
                                const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                                if (mesh_motion)
                                {
                                    const real vfaceL   = x1l * hubble_param;
                                    const real vfaceR   = x1r * hubble_param;
                                    v1p = std::abs(v1p - vfaceR);
                                    v1m = std::abs(v1m - vfaceL);
                                }
                                cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                                break;
                            }
                        default:
                            {
                                const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                                const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                                const real dz     = zr - zl;
                                const real x1l    = get_x1face(ii, geometry, 0);
                                const real x1r    = get_x1face(ii, geometry, 1);
                                const real dr     = x1r - x1l;
                                if (mesh_motion)
                                {
                                    const real vfaceL   = hubble_param;
                                    const real vfaceR   = hubble_param;
                                    v1p = std::abs(v1p - vfaceR);
                                    v1m = std::abs(v1m - vfaceL);
                                }
                                cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  dz / (helpers::my_max(v2p, v2m)));
                                break;
                            }
                    } // end switch
                    min_dt = helpers::my_min(min_dt, cfl_dt);
                } // end ii 
            } // end jj
        } // end parallel region
        dt = cfl * min_dt;
    } else {
        std::atomic<real> min_dt = INFINITY;
        thread_pool.parallel_for(static_cast<luint>(0), active_zones, [&] (luint gid) {
            real cfl_dt, v1p, v1m, v2p ,v2m;

            const auto ii       = gid % xphysical_grid;
            const auto jj       = gid / xphysical_grid;
            const auto aid      = (jj + radius) * nx + (ii + radius);
            const auto rho      = prims[aid].rho;
            const auto v1       = prims[aid].get_v1();
            const auto v2       = prims[aid].get_v2();
            const auto pressure = prims[aid].p;
            const auto h        = 1.0 + gamma * pressure / (rho * (gamma - 1.0));
            const auto cs       = std::sqrt(gamma * pressure / (rho * h) );

            //================ Plus / Minus Wave speed components -================
            if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                v1p = std::abs(v1 + cs) / (1.0 + v1 * cs);
                v2p = std::abs(v2 + cs) / (1.0 + v2 * cs);
                v1m = std::abs(v1 - cs) / (1.0 - v1 * cs);
                v2m = std::abs(v2 - cs) / (1.0 - v2 * cs);
            } else {
                v1p = 1;
                v1m = 1;
                v2p = 1;
                v2m = 1;
            }
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        if (mesh_motion) {
                            v1p = std::abs(v1p - hubble_param);
                            v1m = std::abs(v1m - hubble_param);
                        }
                        cfl_dt = helpers::my_min(dx1 / (helpers::my_max(v1p, v1m)), dx2 / (helpers::my_max(v2p, v2m)));
                        break;
                    }
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
                            v1p = std::abs(v1p - vfaceR);
                            v1m = std::abs(v1m - vfaceL);
                        }
                        cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                        break;
                    }

                case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dtheta = tr - tl;
                        const real x1l    = get_x1face(ii, geometry, 0);
                        const real x1r    = get_x1face(ii, geometry, 1);
                        const real dr     = x1r - x1l;
                        const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                        if (mesh_motion)
                        {
                            const real vfaceL   = x1l * hubble_param;
                            const real vfaceR   = x1r * hubble_param;
                            v1p = std::abs(v1p - vfaceR);
                            v1m = std::abs(v1m - vfaceL);
                        }
                        cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                case simbi::Geometry::AXIS_CYLINDRICAL:
                    {
                        const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dz     = zr - zl;
                        const real x1l    = get_x1face(ii, geometry, 0);
                        const real x1r    = get_x1face(ii, geometry, 1);
                        const real dr     = x1r - x1l;
                        if (mesh_motion)
                        {
                            const real vfaceL   = hubble_param;
                            const real vfaceR   = hubble_param;
                            v1p = std::abs(v1p - vfaceR);
                            v1m = std::abs(v1m - vfaceL);
                        }
                        cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  dz / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                default:
                    break;
            } // end switch
            pooling::update_minimum(min_dt, cfl_dt);
        });
        dt = cfl * min_dt;
    }
};

template<TIMESTEP_TYPE dt_type>
void SRHD2D::adapt_dt(const ExecutionPolicy<> &p)
{
    
    #if GPU_CODE
    {
        compute_dt<Primitive, dt_type><<<p.gridSize,p.blockSize>>>(
            this, 
            prims.data(),
            dt_min.data(),
            geometry
        );
        deviceReduceWarpAtomicKernel<2><<<p.gridSize, p.blockSize>>>(this, dt_min.data(), active_zones);
    }
    #endif
    gpu::api::deviceSynch();
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2flux(const Primitive &prims, const luint nhat) const
{
    const real rho             = prims.rho;
    const real v1              = prims.get_v1();
    const real v2              = prims.get_v2();
    const real pressure        = prims.p;
    const real chi             = prims.chi;
    const real vn              = (nhat == 1) ? v1 : v2;
    const auto kron            = kronecker(nhat, 1);
    const real lorentz_gamma   = 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));

    const real h   = 1 + gamma * pressure / (rho * (gamma - 1));
    const real D   = rho * lorentz_gamma;
    const real S1  = rho * lorentz_gamma * lorentz_gamma * h * v1;
    const real S2  = rho * lorentz_gamma * lorentz_gamma * h * v2;
    const real Sj  = (nhat == 1) ? S1 : S2;

    return Conserved{D * vn, S1 * vn + kron * pressure, S2 * vn + !kron * pressure, Sj - D * vn, D * vn * chi};
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
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

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

    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

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

    
    const real uhlld   = hll_state.d;
    const real uhlls1  = hll_state.s1;
    const real uhlls2  = hll_state.s2;
    const real uhlltau = hll_state.tau;
    const real fhlld   = hll_flux.d;
    const real fhlls1  = hll_flux.s1;
    const real fhlls2  = hll_flux.s2;
    const real fhlltau = hll_flux.tau;
    const real e  = uhlltau + uhlld;
    const real s  = (nhat == 1) ? uhlls1 : uhlls2;
    const real fe = fhlltau + fhlld;
    const real fs = (nhat == 1) ? fhlls1 : fhlls2;

    //------Calculate the contact wave velocity and pressure
    const real a     = fe;
    const real b     = -(e + fs);
    const real c     = s;
    const real quad  = -static_cast<real>(0.5) * (b + helpers::sgn(b) * std::sqrt(b * b - static_cast<real>(4.0) * a * c));
    const real aStar = c * (1 / quad);
    const real pStar = -aStar * fe + fs;

    switch (comp_hllc_type)
    {
    case HLLCTYPE::FLEISCHMANN:
        {
            constexpr real ma_lim = static_cast<real>(0.1);

            // --------------Compute the L Star State----------
            real pressure = left_prims.p;
            real D        = left_state.d;
            real S1       = left_state.s1;
            real S2       = left_state.s2;
            real tau      = left_state.tau;
            real E        = tau + D;
            real cofactor = 1 / (aL - aStar);

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
            cofactor = 1 / (aR - aStar);

            Dstar                 = cofactor * (aR - vR) * D;
            S1star                = cofactor * (S1 * (aR - vR) +  kdelta * (-pressure + pStar) );
            S2star                = cofactor * (S2 * (aR - vR) + !kdelta * (-pressure + pStar) );
            Estar                 = cofactor * (E  * (aR - vR) + pStar * aStar - pressure * vR);
            tauStar               = Estar - Dstar;
            const auto starStateR = Conserved{Dstar, S1star, S2star, tauStar};
            const real ma_left    = vL / cL; // * std::sqrt((1 - cL * cL) / (1 - vL * vL));
            const real ma_right   = vR / cR; // * std::sqrt((1 - cR * cR) / (1 - vR * vR));
            const real ma_local   = helpers::my_max(std::abs(ma_left), std::abs(ma_right));
            const real phi        = std::sin(helpers::my_min(static_cast<real>(1.0), ma_local / ma_lim) * M_PI * static_cast<real>(0.5));
            const real aL_lm      = (phi - (phi == 0) * (phi - 1)) * aL;
            const real aR_lm      = (phi - (phi == 0) * (phi - 1)) * aR;

            const Conserved face_starState = (aStar <= 0) ? starStateR : starStateL;
            Conserved net_flux = (left_flux + right_flux) * static_cast<real>(0.5) + ( (starStateL - left_state) * aL_lm
                                + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm) * static_cast<real>(0.5) - face_starState * vface;

            // upwind the concentration flux 
            if (net_flux.d < 0)
                net_flux.chi = right_prims.chi * net_flux.d;
            else
                net_flux.chi = left_prims.chi  * net_flux.d;
            return net_flux;
        }
    
    default:
        {
            if (vface <= aStar)
            {
                const real pressure = left_prims.p;
                const real D        = left_state.d;
                const real S1       = left_state.s1;
                const real S2       = left_state.s2;
                const real tau      = left_state.tau;
                const real chi      = left_state.chi;
                const real E        = tau + D;
                const real cofactor = 1 / (aL - aStar);

                const real vL     =  left_prims.vcomponent(nhat);
                const auto kdelta = kronecker(nhat, 1);
                // Left Star State in x-direction of coordinate lattice
                const real Dstar         = cofactor * (aL - vL) * D;
                const real chistar       = cofactor * (aL - vL) * chi;
                const real S1star        = cofactor * (S1 * (aL - vL) +  kdelta * (-pressure + pStar) );
                const real S2star        = cofactor * (S2 * (aL - vL) + !kdelta * (-pressure + pStar) );
                const real Estar         = cofactor * (E  * (aL - vL) + pStar * aStar - pressure * vL);
                const real tauStar       = Estar - Dstar;
                auto starStateL          = Conserved{Dstar, S1star, S2star, tauStar, chistar};

                auto hllc_flux = left_flux + (starStateL - left_state) * aL - starStateL * vface;

                // upwind the concentration flux 
                if (hllc_flux.d < 0)
                    hllc_flux.chi = right_prims.chi * hllc_flux.d;
                else
                    hllc_flux.chi = left_prims.chi  * hllc_flux.d;

                return hllc_flux;
            } else {
                const real pressure = right_prims.p;
                const real D        = right_state.d;
                const real S1       = right_state.s1;
                const real S2       = right_state.s2;
                const real tau      = right_state.tau;
                const real chi      = right_state.chi;
                const real E        = tau + D;
                const real cofactor = 1 / (aR - aStar);

                const real vR         = right_prims.vcomponent(nhat);
                const auto kdelta     = kronecker(nhat, 1);
                const real Dstar      = cofactor * (aR - vR) * D;
                const real chistar    = cofactor * (aR - vR) * chi;
                const real S1star     = cofactor * (S1 * (aR - vR) +  kdelta * (-pressure + pStar) );
                const real S2star     = cofactor * (S2 * (aR - vR) + !kdelta * (-pressure + pStar) );
                const real Estar      = cofactor * (E  * (aR - vR) + pStar * aStar - pressure * vR);
                const real tauStar    = Estar - Dstar;
                auto starStateR       = Conserved{Dstar, S1star, S2star, tauStar, chistar};

                auto hllc_flux = right_flux + (starStateR - right_state) * aR - starStateR * vface;

                // upwind the concentration flux 
                if (hllc_flux.d < 0)
                    hllc_flux.chi = right_prims.chi * hllc_flux.d;
                else
                    hllc_flux.chi = left_prims.chi  * hllc_flux.d;

                return hllc_flux;
            }
        }
    } // end switch 
};

//===================================================================================================================
//                                            KERNEL CALCULATIONS
//===================================================================================================================
void SRHD2D::cons2prim(const ExecutionPolicy<> &p)
{
    auto* const cons_data  = cons.data();
    auto* const prim_data  = prims.data();
    auto* const press_data = pressure_guess.data();
    auto* const troubled_data = troubled_cells.data();
    simbi::parallel_for(p, (luint)0, nzones, [CAPTURE_THIS]   GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;        
        const auto tid = get_threadId();
        if (tid == 0)
            found_failure = inFailureState;
        simbi::gpu::api::synchronize();
        
        real invdV = 1.0;
        while (!found_failure && workLeftToDo)
        {
            if (mesh_motion &&  (geometry != simbi::Geometry::CARTESIAN))
            {
                const luint ii   = gid % nx;
                const luint jj   = gid / nx;
                const auto ireal = helpers::get_real_idx(ii, radius, xphysical_grid);
                const auto jreal = helpers::get_real_idx(jj, radius, yphysical_grid); 
                const real dV    = get_cell_volume(ireal, jreal, geometry);
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
                    dt             = INFINITY;
                    found_failure  = true;
                    inFailureState = true;
                    break;
                }
            } while (std::abs(peq - pre) >= tol);
            

            const real inv_et = 1 / (tau + D + peq);
            const real v1     = S1 * inv_et;
            const real v2     = S2 * inv_et;
            press_data[gid] = peq;
            #if FOUR_VELOCITY
                prim_data[gid] = Primitive{D/ W, v1 * W, v2 * W, peq, Dchi / D};
            #else
                prim_data[gid] = Primitive{D/ W, v1, v2, peq, Dchi / D};
            #endif
            workLeftToDo = false;

            if (peq < 0) {
                troubled_data[gid] = iter;
                dt             = INFINITY;
                inFailureState = true;
                found_failure  = true;
            }
            simbi::gpu::api::synchronize();
        }
    });
}

void SRHD2D::advance(
    const ExecutionPolicy<> &p,
    const luint bx,
    const luint by)
{
    const auto xpg      = this->xphysical_grid;
    const auto ypg      = this->yphysical_grid;
    const auto pseudo_radius = (first_order) ? 1 : 2;
    #if GPU_CODE
    const auto xextent             = p.get_xextent();
    const auto yextent             = p.get_yextent();
    const luint max_ii             = (col_maj) ? ypg : xpg;
    const luint max_jj             = (col_maj) ? xpg : ypg;
    #endif

    const luint extent= p.get_full_extent();
    // Choice of column major striding by user
    const luint sx = (col_maj) ? 1  : bx;
    const luint sy = (col_maj) ? by :  1;

    auto* const prim_data   = prims.data();
    auto* const cons_data   = cons.data();
    auto* const dens_source = sourceD.data();
    auto* const mom1_source = sourceS1.data();
    auto* const mom2_source = sourceS2.data();
    auto* const erg_source  = sourceTau.data();
    auto* const object_data = object_pos.data();
    auto* const grav_source = sourceG.data();
    simbi::parallel_for(p, (luint)0, extent, [CAPTURE_THIS]   GPU_LAMBDA (const luint idx) {
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        // auto *const prim_buff = prim_data;
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint ii  = (BuildPlatform == Platform::GPU) ? get_ii_in2D() : idx % xpg;
        const luint jj  = (BuildPlatform == Platform::GPU) ? get_jj_in2D() : idx / xpg;
        #if GPU_CODE 
        if ((ii >= max_ii) || (jj >= max_jj)) return;
        #endif

        const bool object_to_my_left  = object_data[jj * xpg +  helpers::my_max(static_cast<lint>(ii - 1), static_cast<lint>(0))];
        const bool object_to_my_right = object_data[jj * xpg +  helpers::my_min(ii + 1,  xpg - 1)];
        const bool object_above_me    = object_data[helpers::my_min(jj + 1, ypg - 1)  * xpg +  ii];
        const bool object_below_me    = object_data[helpers::my_max(static_cast<lint>(jj - 1), static_cast<lint>(0)) * xpg +  ii];

        const lint ia  = ii + radius;
        const lint ja  = jj + radius;
        const lint tx  = (BuildPlatform == Platform::GPU) ? get_tx(): 0;
        const lint ty  = (BuildPlatform == Platform::GPU) ? get_ty(): 0;
        const lint txa = (BuildPlatform == Platform::GPU) ? tx + pseudo_radius : ia;
        const lint tya = (BuildPlatform == Platform::GPU) ? ty + pseudo_radius : ja;

        Conserved uxL, uxR, uyL, uyR;
        Conserved fL, fR, gL, gR, frf, flf, grf, glf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR;

        const lint aid = get_2d_idx(ia, ja, nx, ny); 
        // Load Shared memory into buffer for active zones plus ghosts
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = prim_data[aid];
            if (ty < pseudo_radius)
            {
                if (blockIdx.y == p.gridSize.y - 1 && (ja + yextent > ny - radius + ty)) {
                    tyl = ny - radius - ja + ty;
                }
                prim_buff[(tya - pseudo_radius) * sx + txa] = prim_data[helpers::mod(ja - pseudo_radius, ny) * nx + ia];
                prim_buff[(tya + tyl) * sx + txa]           = prim_data[(ja + tyl) % ny * nx + ia]; 
            }
            if (tx < pseudo_radius)
            {   
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                prim_buff[tya * sx + txa - pseudo_radius] =  prim_data[ja * nx + helpers::mod(ia - pseudo_radius, nx)];
                prim_buff[tya * sx + txa +    txl]        =  prim_data[ja * nx +    (ia + txl) % nx]; 
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = get_x1face(ii, geometry, 0);
        const real x1r    = get_x1face(ii, geometry, 1);
        const real vfaceR = (geometry == simbi::Geometry::SPHERICAL) ? x1r * hubble_param : hubble_param;
        const real vfaceL = (geometry == simbi::Geometry::SPHERICAL) ? x1l * hubble_param : hubble_param;
        if (first_order) [[unlikely]]
        {
            //i+1/2
            xprimsL = prim_buff[(txa + 0)      * sy + (tya + 0) * sx];
            xprimsR = prim_buff[(txa + 1) % bx * sy + (tya + 0) * sx];
            //j+1/2
            yprimsL = prim_buff[(txa + 0) * sy + (tya + 0)      * sx];
            yprimsR = prim_buff[(txa + 0) * sy + (tya + 1) % by * sx];
            
            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                xprimsR.v1  =  xprimsL.v2;
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_above_me){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v2  =  yprimsL.v1;
                yprimsR.v2  = -yprimsL.v2;
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            // i+1/2
            uxL = prims2cons(xprimsL); 
            uxR = prims2cons(xprimsR); 
            // j+1/2
            uyL = prims2cons(yprimsL);  
            uyR = prims2cons(yprimsR); 

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            // Calc HLL Flux at i+1/2 interface
            if (hllc) {
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            } else {
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[(txa - 1) * sy + (tya + 0) * sx];
            xprimsR = prim_buff[(txa - 0) * sy + (tya + 0) * sx];
            //j-1/2
            yprimsL = prim_buff[(txa - 0) * sy + (tya - 1) * sx]; 
            yprimsR = prim_buff[(txa + 0) * sy + (tya - 0) * sx]; 

            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                xprimsL.v2  =  xprimsR.v2;
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_below_me){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                yprimsL.v2  = -yprimsR.v2;
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }
            // i-1/2
            uxL = prims2cons(xprimsL); 
            uxR = prims2cons(xprimsR); 
            // j-1/2
            uyL = prims2cons(yprimsL);  
            uyR = prims2cons(yprimsR); 

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            // Calc HLL Flux at i-1/2 interface
            if (hllc) {
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            } else {
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }   
        } else { 
            // Coordinate X
            const Primitive xleft_most  = prim_buff[(helpers::mod(txa - 2, bx)    * sy + tya * sx)];
            const Primitive xleft_mid   = prim_buff[(helpers::mod(txa - 1, bx)    * sy + tya * sx)];
            const Primitive center      = prim_buff[(            (txa + 0)        * sy + tya * sx)];
            const Primitive xright_mid  = prim_buff[(            (txa + 1) % bx   * sy + tya * sx)];
            const Primitive xright_most = prim_buff[(            (txa + 2) % bx   * sy + tya * sx)];

            // Coordinate Y
            const Primitive yleft_most  = prim_buff[(txa * sy + helpers::mod(tya - 2, by)  * sx)];
            const Primitive yleft_mid   = prim_buff[(txa * sy + helpers::mod(tya - 1, by)  * sx)];
            const Primitive yright_mid  = prim_buff[(txa * sy +             (tya + 1) % by * sx)];
            const Primitive yright_most = prim_buff[(txa * sy +             (tya + 2) % by * sx)];

            // Reconstructed left X Primitive vector at the [i,j +1/2] interface
            xprimsL  = center     + helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta)   * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::plm_gradient(xright_mid, center, xright_most, plm_theta) * static_cast<real>(0.5);
            yprimsL  = center     + helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta)   * static_cast<real>(0.5);  
            yprimsR  = yright_mid - helpers::plm_gradient(yright_mid, center, yright_most, plm_theta) * static_cast<real>(0.5);


            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                xprimsR.v2  =  xprimsL.v2;
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_above_me){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                yprimsR.v2  = -yprimsL.v2;
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            // Calculate the left and right states using the reconstructed PLM
            // Primitive (i,j + 1/2)
            uxL  = prims2cons(xprimsL);
            uxR  = prims2cons(xprimsR);
            uyL  = prims2cons(yprimsL);
            uyR  = prims2cons(yprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            if (hllc) {
                if(quirk_smoothing)
                {
                    if (quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                        frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    } else {
                        frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    }

                    if (quirk_strong_shock(yprimsL.p, yprimsR.p)){
                        grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } else {
                        grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    }
                } else {
                    frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                }
            } else {
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }

            // Do the same thing, but for the left side interface [i,j - 1/2]
            xprimsL  = xleft_mid  + helpers::plm_gradient(xleft_mid, xleft_most, center, plm_theta) * static_cast<real>(0.5); 
            xprimsR  = center     - helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta) * static_cast<real>(0.5);
            yprimsL  = yleft_mid  + helpers::plm_gradient(yleft_mid, yleft_most, center, plm_theta) * static_cast<real>(0.5);  
            yprimsR  = center     - helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta) * static_cast<real>(0.5);
            
            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                xprimsL.v2  =  xprimsR.v2;
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_below_me){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                yprimsL.v2  = -yprimsR.v2;
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }


            // Calculate the left and right states using the reconstructed PLM
            // Primitive (i,j -1/2)
            uxL  = prims2cons(xprimsL);
            uxR  = prims2cons(xprimsR);
            uyL  = prims2cons(yprimsL);
            uyR  = prims2cons(yprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            if (hllc) {
                if (quirk_smoothing)
                {
                    if (quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                        flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    } else {
                        flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    }
                    
                    if (quirk_strong_shock(yprimsL.p, yprimsR.p)){
                        glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } else {
                        glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                    } 
                } else {
                    flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                }
            } else {
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
            }
        }

        //Advance depending on geometry
        const luint real_loc = get_2d_idx(ii, jj, xpg, ypg);
        const real d_source  = den_source_all_zeros    ? 0.0 : dens_source[real_loc];
        const real s1_source = mom1_source_all_zeros   ? 0.0 : mom1_source[real_loc];
        const real s2_source = mom2_source_all_zeros   ? 0.0 : mom2_source[real_loc];
        const real e_source  = energy_source_all_zeros ? 0.0 : erg_source[real_loc];
        // Gravity
        const auto g_source   = grav_source_all_zeros  ? 0.0 :  grav_source[ii];
        const auto gs1_source = g_source * cons_data[aid].d;
        const auto gs2_source = 0; 
        const auto ge_source  = gs1_source * prim_buff[txa].v1;
        const auto gravity    = Conserved{0, gs1_source, gs2_source, ge_source};
        const Conserved source_terms = Conserved{d_source, s1_source, s2_source, e_source} * time_constant;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
            {
                cons_data[aid] -= ( (frf - flf) * invdx1 + (grf - glf) * invdx2 - source_terms) * step * dt;
                break;
            }
            
            case simbi::Geometry::SPHERICAL:
                {
                const real rl           = x1l + vfaceL * step * dt; 
                const real rr           = x1r + vfaceR * step * dt;
                const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real tl           = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                const real tr           = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dcos         = std::cos(tl) - std::cos(tr);
                const real dVtot        = 2.0 * M_PI * (1.0 / 3.0) * (rr * rr * rr - rl * rl * rl) * dcos;
                const real invdV        = 1.0 / dVtot;
                const real s1R          = 2.0 * M_PI * rr * rr * dcos; 
                const real s1L          = 2.0 * M_PI * rl * rl * dcos; 
                const real s2R          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tr);
                const real s2L          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tl);
                const real factor       = (mesh_motion) ? dVtot : 1;  


                // Grab central primitives
                const real rhoc = prim_buff[txa * sy + tya * sx].rho;
                const real uc   = prim_buff[txa * sy + tya * sx].get_v1();
                const real vc   = prim_buff[txa * sy + tya * sx].get_v2();
                const real pc   = prim_buff[txa * sy + tya * sx].p;

                const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                const real gam2 = 1/(1 - (uc * uc + vc * vc));

                const Conserved geom_source  = {0, (rhoc * hc * gam2 * vc * vc) / rmean + pc * (s1R - s1L) * invdV, - (rhoc * hc * gam2 * uc * vc) / rmean + pc * (s2R - s2L) * invdV , 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms - gravity) * dt * step * factor;
                break;
                }
             case simbi::Geometry::PLANAR_CYLINDRICAL:
                {
                const real rl           = x1l + vfaceL * step * dt; 
                const real rr           = x1r + vfaceR * step * dt;
                const real rmean        = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                // const real tl           = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                // const real tr           = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dVtot        = rmean * (rr - rl) * dx2;
                const real invdV        = 1.0 / dVtot;
                const real s1R          = rr * dx2; 
                const real s1L          = rl * dx2; 
                const real s2R          = (rr - rl); 
                const real s2L          = (rr - rl); 

                // Grab central primitives
                const real rhoc = prim_buff[tya * bx + txa].rho;
                const real uc   = prim_buff[tya * bx + txa].get_v1();
                const real vc   = prim_buff[tya * bx + txa].get_v2();
                const real pc   = prim_buff[tya * bx + txa].p;
                
                const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                const real gam2 = 1/(1 - (uc * uc + vc * vc));

                const Conserved geom_source  = {0, (rhoc * hc * gam2 * vc * vc) / rmean + pc * (s1R - s1L) * invdV, - (rhoc * hc * gam2 * uc * vc) / rmean, 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * dt * step;
                break;
                }
            case simbi::Geometry::AXIS_CYLINDRICAL:
                {
                const real rl           = x1l + vfaceL * step * dt; 
                const real rr           = x1r + vfaceR * step * dt;
                const real rmean        = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                const real dVtot        = rmean * (rr - rl) * dx2;
                const real invdV        = 1.0 / dVtot;
                const real s1R          = rr * dx2; 
                const real s1L          = rl * dx2; 
                const real s2R          = rmean * (rr - rl); 
                const real s2L          = rmean * (rr - rl); 
                const real factor       = (mesh_motion) ? dVtot : 1;  

                // Grab central primitives
                const real pc   = prim_buff[tya * bx + txa].p;
                
                const Conserved geom_source  = {0, pc * (s1R - s1L) * invdV, 0, 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * dt * step * factor;
                break;
                }
            default:
                break;
        } // end switch
    });
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
    std::vector<std::vector<real>> &sources,
    std::vector<bool> &object_cells,
    std::vector<real> &gsource,
    real tstart,
    real tend,
    real dlogt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    int chkpt_idx,
    std::string data_directory,
    std::vector<std::string> boundary_conditions,
    bool first_order,
    bool linspace,
    bool hllc,
    bool quirk_smoothing,
    bool constant_sources,
    std::vector<std::vector<real>> boundary_sources,
    std::function<double(double)> a,
    std::function<double(double)> adot,
    std::function<double(double, double)> d_outer,
    std::function<double(double, double)> s1_outer,
    std::function<double(double, double)> s2_outer,
    std::function<double(double, double)> e_outer)
{
    anyDisplayProps();
    define_periodic(boundary_conditions);
    this->t = tstart;
    // Define the source terms
    this->object_pos      = object_cells;
    this->sourceD         = sources[0];
    this->sourceS1        = sources[1];
    this->sourceS2        = sources[2];
    this->sourceTau       = sources[3];
    this->sourceG         = gsource;
    // Define sim state params
    this->engine_duration = engine_duration;
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->total_zones     = nx * ny;
    this->first_order     = first_order;
    this->hllc            = hllc;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->dlogt           = dlogt;
    this->xphysical_grid  = (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid  = (first_order) ? ny - 2 : ny - 4;
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = xphysical_grid * yphysical_grid;
    this->quirk_smoothing = quirk_smoothing;
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->x2cell_spacing  = simbi::Cellspacing::LINSPACE;
    this->dx2             = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1          = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1             = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->invdx1          = 1 / dx1;
    this->invdx2          = 1 / dx2;
    this->x1min           = x1[0];
    this->x1max           = x1[xphysical_grid - 1];
    this->x2min           = x2[0];
    this->x2max           = x2[yphysical_grid - 1];
    this->checkpoint_zones= yphysical_grid;
    this->den_source_all_zeros    = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    this->mom1_source_all_zeros   = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    this->mom2_source_all_zeros   = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    this->energy_source_all_zeros = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    this->grav_source_all_zeros   = std::all_of(sourceG.begin(), sourceG.end(), [](real i) {return i == 0;});
    define_tinterval(t, dlogt, chkpt_interval, chkpt_idx);
    define_chkpt_idx(chkpt_idx);
    // Params moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->all_outer_bounds = (d_outer && s1_outer && s2_outer && e_outer);
    if (all_outer_bounds){
        dens_outer = d_outer;
        mom1_outer = s1_outer;
        mom2_outer = s2_outer;
        nrg_outer  = e_outer;
    }
    if (x2max == 0.5 * M_PI){
        this->half_sphere = true;
    }
    inflow_zones.resize(4);
    for (int i = 0; i < 4; i++) {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = Conserved{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2], boundary_sources[i][3]};
    }

    cons.resize(nzones);
    prims.resize(nzones);
    troubled_cells.resize(nzones, 0);
    dt_min.resize(active_zones);
    pressure_guess.resize(nzones);
    if (mesh_motion && all_outer_bounds) {
        outer_zones.resize(ny);
        for (luint jj = 0; jj < ny; jj++) {
            const auto jreal = helpers::get_real_idx(jj, radius, yphysical_grid);
            const real dV    = get_cell_volume(xphysical_grid - 1, jreal, geometry);
            outer_zones[jj]  = conserved_t{
                dens_outer(x1max, x2[jreal]), 
                mom1_outer(x1max, x2[jreal]), 
                mom2_outer(x1max, x2[jreal]), 
                nrg_outer(x1max, x2[jreal])} * dV;
        }
        outer_zones.copyToGpu();
    }

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
    
    cons.copyToGpu();
    prims.copyToGpu();
    pressure_guess.copyToGpu();
    dt_min.copyToGpu();
    sourceD.copyToGpu();
    sourceS1.copyToGpu();
    sourceS2.copyToGpu();
    sourceTau.copyToGpu();
    object_pos.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();
    troubled_cells.copyToGpu();
    sourceG.copyToGpu();

    // Write some info about the setup for writeup later
    setup.x1max              = x1[xphysical_grid - 1];
    setup.x1min              = x1[0];
    setup.x2max              = x2[yphysical_grid - 1];
    setup.x2min              = x2[0];
    setup.nx                 = nx;
    setup.ny                 = ny;
    setup.xactive_zones      = xphysical_grid;
    setup.yactive_zones      = yphysical_grid;
    setup.linspace           = linspace;
    setup.ad_gamma           = gamma;
    setup.first_order        = first_order;
    setup.coord_system       = coord_system;
    setup.boundary_conditions  = boundary_conditions;
    setup.regime             = "relativistic";
    setup.using_fourvelocity = (VelocityType == Velocity::FourVelocity);
    setup.x1                 = x1;
    setup.x2                 = x2;
    setup.mesh_motion        = mesh_motion;
    setup.dimensions         = 2;

    // // Setup the system
    const luint xblockdim    = xphysical_grid > gpu_block_dimx ? gpu_block_dimx : xphysical_grid;
    const luint yblockdim    = yphysical_grid > gpu_block_dimy ? gpu_block_dimy : yphysical_grid;
    this->radius             = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->pseudo_radius      = (first_order) ? 1 : 2;
    this->step               = (first_order) ? 1 : static_cast<real>(0.5);
    const luint shBlockSpace = (xblockdim + 2 * radius) * (yblockdim + 2 * radius);
    const luint shBlockBytes = shBlockSpace * sizeof(Primitive);
    const auto fullP         = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim});
    const auto activeP       = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU){
        std::cout << "  Requested shared memory:   " << shBlockBytes << std::endl;

    }
    if (t == 0) {
        config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
    }
    
    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP);
        adapt_dt(activeP);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
    
    // Save initial condition
    if (t == 0 || chkpt_idx == 0) {
        write2file(*this, setup, data_directory, t, t_interval, chkpt_interval, yphysical_grid);
        if (dlogt != 0) {
            t_interval *= std::pow(10, dlogt);
        } else {
            t_interval += chkpt_interval;
        }
    }
    
    // Simulate :)
    const luint xstride = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint ystride = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    simbi::detail::logger::with_logger(*this, tend, [&](){
        if (inFailureState){
            return;
        }
        
        advance(activeP, xstride, ystride);
        cons2prim(fullP);
        config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
        
        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
        if (mesh_motion){
            // update x1 endpoints  
            const real vmin = (geometry == simbi::Geometry::SPHERICAL) ? x1min * hubble_param : hubble_param;
            const real vmax = (geometry == simbi::Geometry::SPHERICAL) ? x1max * hubble_param : hubble_param;
            x1max += step * dt * vmax;
            x1min += step * dt * vmin;
            hubble_param = adot(t) / a(t);
        }
    });

    if (inFailureState){
        emit_troubled_cells();
    }

    std::vector<std::vector<real>> final_prims(5, std::vector<real>(nzones, 0));
    for (luint ii = 0; ii < nzones; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v1;
        final_prims[2][ii] = prims[ii].v2;
        final_prims[3][ii] = prims[ii].p;
        final_prims[4][ii] = prims[ii].chi;
    }

    return final_prims;
};
