/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/
#include <cmath>
#include <chrono>
#include "euler2D.hpp" 
#include "util/parallel_for.hpp"
#include "util/printb.hpp"
#include "common/helpers.hip.hpp"
#include "util/logger.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;
constexpr auto write2file = helpers::write_to_file<hydro2d::PrimitiveSOA, 2, Newtonian2D>;


// Typedefs because I'm lazy
using Conserved = hydro2d::Conserved;
using Primitive = hydro2d::Primitive;
using Eigenvals = hydro2d::Eigenvals;

// Default Constructor 
Newtonian2D::Newtonian2D () {}

// Overloaded Constructor
Newtonian2D::Newtonian2D(
    std::vector<std::vector<real>> &state, 
    luint nx,
    luint ny,
    real gamma, 
    std::vector<real> &x1, 
    std::vector<real> &x2, 
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
        coord_system)
{

}

// Destructor 
Newtonian2D::~Newtonian2D() {}

// Typedefs because I'm lazy
typedef hydro2d::Conserved Conserved;
typedef hydro2d::Primitive Primitive;
typedef hydro2d::Eigenvals Eigenvals;
//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------
void Newtonian2D::cons2prim(const ExecutionPolicy<> &p)
{
    auto* const cons_data = cons.data();
    auto* const prim_data = prims.data();
    simbi::parallel_for(p, (luint)0, nzones, [CAPTURE_THIS]   GPU_LAMBDA (const luint gid) {
        const real rho     = cons_data[gid].rho;
        const real v1      = cons_data[gid].m1 / rho;
        const real v2      = cons_data[gid].m2 / rho;
        const real rho_chi = cons_data[gid].chi;
        const real pre     = (gamma - 1)*(cons_data[gid].e_dens - static_cast<real>(0.5) * rho * (v1 * v1 + v2 * v2));
        prim_data[gid] = Primitive{rho, v1, v2, pre, rho_chi / rho};
    });
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals Newtonian2D::calc_eigenvals(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
{   
    const real rhoL = left_prims.rho;
    const real vL   = left_prims.vcomponent(ehat);
    const real pL   = left_prims.p;
    
    const real rhoR = right_prims.rho;
    const real vR   = right_prims.vcomponent(ehat);
    const real pR   = right_prims.p;
    switch (sim_solver)
    {
    case Solver::HLLC:
        {
            const real csR = std::sqrt(gamma * pR/rhoR);
            const real csL = std::sqrt(gamma * pL/rhoL);

            // Calculate the mean velocities of sound and fluid
            // const real cbar   = 0.5*(csL + csR);
            // const real rhoBar = 0.5*(rhoL + rhoR);
            const real num       = csL + csR- (gamma - 1.) * 0.5 *(vR- vR);
            const real denom     = csL * std::pow(pL, - hllc_z) + csR * std::pow(pR, - hllc_z);
            const real p_term    = num/denom;
            const real pStar     = std::pow(p_term, (1./hllc_z));

            const real qL = 
                (pStar <= pL) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pL - 1.));

            const real qR = 
                (pStar <= pR) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pR- 1.));

            const real aL = vR - qL*csL;
            const real aR = vR + qR*csR;

            const real aStar = 
                ( (pR- pL + rhoL*vL*(aL - vL) - rhoR*vR*(aR - vR) )/ 
                    (rhoL*(aL - vL) - rhoR*(aR - vR) ) );

            return Eigenvals(aL, aR, csL, csR, aStar, pStar);
        }

    default:
        {
            const real csR  = std::sqrt(gamma * pR/rhoR);
            const real csL  = std::sqrt(gamma * pL/rhoL);

            const real aL = helpers::my_min(vL - csL, vR - csR);
            const real aR = helpers::my_max(vL + csL, vR + csR);

            return Eigenvals(aL, aR);
        }

    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state array for computation. 
// It is being doing poluintwise in this case as opposed to over
// the entire array since we are in c++
GPU_CALLABLE_MEMBER
Conserved Newtonian2D::prims2cons(const Primitive &prims)
 {
    const real rho = prims.rho;
    const real v1  = prims.v1;
    const real v2  = prims.v2;
    const real pre = prims.p;
    const real et  = pre/(gamma - 1.0) + 0.5 * rho * (v1*v1 + v2*v2);

    return Conserved{rho, rho*v1, rho*v2, et};
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the cfl conditonal timestep
void Newtonian2D::adapt_dt()
{
    if (use_omp) {
        real min_dt = INFINITY;
        #pragma omp parallel
        {
            real cfl_dt;
            for (luint jj = 0; jj < yphysical_grid; jj++)
            {
                const auto shift_j = jj + idx_active;
                #pragma omp for schedule(static) reduction(min:min_dt)
                for (luint ii = 0; ii < xphysical_grid; ii++)
                {
                    const auto shift_i  = ii + idx_active;
                    const auto aid      = shift_j * nx + shift_i;
                    const auto rho      = prims[aid].rho;
                    const auto v1       = prims[aid].v1;
                    const auto v2       = prims[aid].v2;
                    const auto pressure = prims[aid].p;
                    const auto cs       = std::sqrt(gamma * pressure / rho );

                    //================ Plus / Minus Wave speed components -================
                    const auto plus_v1  = (v1 + cs);
                    const auto plus_v2  = (v2 + cs);
                    const auto minus_v1 = (v1 - cs);
                    const auto minus_v2 = (v2 - cs);

                    auto v1p = std::abs(plus_v1);
                    auto v1m = std::abs(minus_v1);
                    auto v2p = std::abs(plus_v2);
                    auto v2m = std::abs(minus_v2);
                    switch (geometry)
                    {
                        case simbi::Geometry::CARTESIAN:
                            {
                                if (mesh_motion) {
                                    v1p = std::abs(plus_v1  - hubble_param);
                                    v1m = std::abs(minus_v1 - hubble_param);
                                }
                                cfl_dt = helpers::my_min(dx1 / (helpers::my_max(v1p, v1m)), dx2 / (helpers::my_max(v2p, v2m)));
                                break;
                            }
                        case simbi::Geometry::SPHERICAL:
                            {
                                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                                const real dtheta = tr - tl;
                                const real x1l    = get_x1face(ii, 0);
                                const real x1r    = get_x1face(ii, 1);
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
                                break;
                            }

                        case simbi::Geometry::PLANAR_CYLINDRICAL:
                            {
                                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                                const real dtheta = tr - tl;
                                const real x1l    = get_x1face(ii, 0);
                                const real x1r    = get_x1face(ii, 1);
                                const real dr     = x1r - x1l;
                                const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                                if (mesh_motion)
                                {
                                    const real vfaceL   = x1l * hubble_param;
                                    const real vfaceR   = x1r * hubble_param;
                                    v1p = std::abs(plus_v1  - vfaceR);
                                    v1m = std::abs(minus_v1 - vfaceL);
                                }
                                cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                                break;
                            }
                        default:
                            {
                                const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                                const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                                const real dz     = zr - zl;
                                const real x1l    = get_x1face(ii, 0);
                                const real x1r    = get_x1face(ii, 1);
                                const real dr     = x1r - x1l;
                                // const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                                if (mesh_motion)
                                {
                                    const real vfaceL   = x1l * hubble_param;
                                    const real vfaceR   = x1r * hubble_param;
                                    v1p = std::abs(plus_v1  - vfaceR);
                                    v1m = std::abs(minus_v1 - vfaceL);
                                }
                                cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  dz / (helpers::my_max(v2p, v2m)));
                                break;
                            }
                    } // end switch
                    min_dt = std::min(min_dt, cfl_dt);
                } // end ii
            } // end jj
        }// end parallel region
        dt = cfl * min_dt;
    } else {
        std::atomic<real> min_dt = INFINITY;
        thread_pool.parallel_for(static_cast<luint>(0), active_zones, [&] (luint gid) {
            real cfl_dt;

            const auto ii       = gid % xphysical_grid;
            const auto jj       = gid / xphysical_grid;
            const auto aid      = (jj + radius) * nx + (ii + radius);
            const auto rho      = prims[aid].rho;
            const auto v1       = prims[aid].v1;
            const auto v2       = prims[aid].v2;
            const auto pressure = prims[aid].p;
            const auto cs       = std::sqrt(gamma * pressure / rho );

            //================ Plus / Minus Wave speed components -================
            const auto plus_v1  = (v1 + cs);
            const auto plus_v2  = (v2 + cs);
            const auto minus_v1 = (v1 - cs);
            const auto minus_v2 = (v2 - cs);

            auto v1p = std::abs(plus_v1);
            auto v1m = std::abs(minus_v1);
            auto v2p = std::abs(plus_v2);
            auto v2m = std::abs(minus_v2);

            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        if (mesh_motion) {
                            v1p = std::abs(plus_v1  - hubble_param);
                            v1m = std::abs(minus_v1 - hubble_param);
                        }
                        cfl_dt = helpers::my_min(dx1 / (helpers::my_max(v1p, v1m)), dx2 / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                case simbi::Geometry::SPHERICAL:
                    {
                        const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dtheta = tr - tl;
                        const real x1l    = get_x1face(ii, 0);
                        const real x1r    = get_x1face(ii, 1);
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
                        break;
                    }

                case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dtheta = tr - tl;
                        const real x1l    = get_x1face(ii, 0);
                        const real x1r    = get_x1face(ii, 1);
                        const real dr     = x1r - x1l;
                        const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                        if (mesh_motion)
                        {
                            const real vfaceL   = x1l * hubble_param;
                            const real vfaceR   = x1r * hubble_param;
                            v1p = std::abs(plus_v1  - vfaceR);
                            v1m = std::abs(minus_v1 - vfaceL);
                        }
                        cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  rmean * dtheta / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                default:
                    {
                        const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dz     = zr - zl;
                        const real x1l    = get_x1face(ii, 0);
                        const real x1r    = get_x1face(ii, 1);
                        const real dr     = x1r - x1l;
                        // const real rmean  = (2.0 / 3.0)* (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                        if (mesh_motion)
                        {
                            const real vfaceL   = x1l * hubble_param;
                            const real vfaceR   = x1r * hubble_param;
                            v1p = std::abs(plus_v1  - vfaceR);
                            v1m = std::abs(minus_v1 - vfaceL);
                        }
                        cfl_dt = helpers::my_min(dr / (helpers::my_max(v1p, v1m)),  dz / (helpers::my_max(v2p, v2m)));
                        break;
                    }
            } // end switch
            pooling::update_minimum(min_dt, cfl_dt);
        });
        dt = cfl * min_dt;
    }
};

void Newtonian2D::adapt_dt(const ExecutionPolicy<> &p)
{
    #if GPU_CODE
    {
        helpers::compute_dt<Primitive><<<p.gridSize,p.blockSize>>>(this, prims.data(),dt_min.data(), geometry);
        helpers::deviceReduceWarpAtomicKernel<2><<<p.gridSize, p.blockSize>>>(this, dt_min.data(), active_zones);
    }
    gpu::api::deviceSynch();
    #endif
}
//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 2D Flux array (4,1). Either return F or G depending on directional flag
GPU_CALLABLE_MEMBER
Conserved Newtonian2D::prims2flux(const Primitive &prims, const luint ehat)
{
    const auto vn  = prims.vcomponent(ehat);
    const auto rho = prims.rho;
    const auto v1  = prims.v1;
    const auto v2  = prims.v2;
    const auto pre = prims.p;
    const auto et  = pre / (gamma - 1.0) + 0.5 * rho * (v1*v1 + v2*v2);
    
    const auto dens  = rho*vn;
    const auto momx  = rho*v1*vn + pre * helpers::kronecker(1, ehat);
    const auto momy  = rho*v2*vn + pre * helpers::kronecker(2, ehat);
    const auto edens = (et + pre)*vn;

    return Conserved{dens, momx, momy, edens};

};

GPU_CALLABLE_MEMBER
Conserved Newtonian2D::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
                                        
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);
    real am = helpers::my_min(static_cast<real>(0.0), lambda.aL);
    real ap = helpers::my_max(static_cast<real>(0.0), lambda.aR);
    
    // Compute the HLL Flux 
    return  ( left_flux * ap - right_flux * am 
                + (right_state - left_state ) * am * ap )  /
                    (ap - am);
};

GPU_CALLABLE_MEMBER
Conserved Newtonian2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);

    const real aL    = lambda.aL;
    const real aR    = lambda.aR;
    const real cL    = lambda.csL;
    const real cR    = lambda.csR;
    const real aStar = lambda.aStar;
    const real pStar = lambda.pStar;

    // Quick checks before moving on with rest of computation
    if (0.0 <= aL){
        return left_flux;
    } else if (0.0 >= aR){
        return right_flux;
    }

    // Apply the low-Mach HLLC fix found in Fleichman et al 2020: 
    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
    constexpr real ma_lim   = static_cast<real>(0.10);

    // --------------Compute the L Star State----------
    real pressure = left_prims.p;
    real rho      = left_state.rho;
    real m1       = left_state.m1;
    real m2       = left_state.m2;
    real edens    = left_state.e_dens;
    real cofactor = 1 / (aL - aStar);

    const real vL           = left_prims.vcomponent(ehat);
    const real vR           = right_prims.vcomponent(ehat);
    const auto kdelta       = helpers::kronecker(ehat, 1);
    // Left Star State in x-direction of coordinate lattice
    real rhostar            = cofactor * (aL - vL) * rho;
    real m1star             = cofactor * (m1 * (aL - vL) +  kdelta * (-pressure + pStar) );
    real m2star             = cofactor * (m2 * (aL - vL) + !kdelta * (-pressure + pStar) );
    real estar              = cofactor * (edens  * (aL - vL) + pStar * aStar - pressure * vL);
    const auto starStateL   = Conserved{rhostar, m1star, m2star, estar};

    pressure = right_prims.p;
    rho      = right_state.rho;
    m1       = right_state.m1;
    m2       = right_state.m2;
    edens    = right_state.e_dens;
    cofactor = 1 / (aR - aStar);

    rhostar               = cofactor * (aR - vR) * rho;
    m1star                = cofactor * (m1 * (aR - vR) +  kdelta * (-pressure + pStar) );
    m2star                = cofactor * (m2 * (aR - vR) + !kdelta * (-pressure + pStar) );
    estar                 = cofactor * (edens  * (aR - vR) + pStar * aStar - pressure * vR);
    const auto starStateR = Conserved{rhostar, m1star, m2star, estar};

    const real ma_local = helpers::my_max(std::abs(vL / cL), std::abs(vR / cR));
    const real phi      = std::sin(helpers::my_min(static_cast<real>(1.0), ma_local / ma_lim) * M_PI * static_cast<real>(0.5));
    const real aL_lm    = phi * aL;
    const real aR_lm    = phi * aR;

    // const Conserved face_starState = (aStar <= 0) ? starStateR : starStateL;
    Conserved net_flux = (left_flux + right_flux) * static_cast<real>(0.5) + ( (starStateL - left_state) * aL_lm
                        + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm) * static_cast<real>(0.5);

    // upwind the concentration flux 
    if (net_flux.rho < 0)
        net_flux.chi = right_prims.chi * net_flux.rho;
    else
        net_flux.chi = left_prims.chi  * net_flux.rho;

    return net_flux;

    // if (-aL <= (aStar - aL) ){
    //     const auto pre      = left_prims.p;
    //     const auto v1       = left_prims.v1;
    //     const auto v2       = left_prims.v2;
    //     const auto rho      = left_prims.rho;
    //     const auto m1       = left_state.m1;
    //     const auto m2       = left_state.m2;
    //     const auto energy   = left_state.e_dens;
    //     const auto cofac    = 1./(aL - aStar);

    //     switch (ehat)
    //     {
    //     case 1:
    //         {
    //             const auto rhoStar = cofac * (aL - v1) * rho;
    //             const auto m1star  = cofac * (m1*(aL - v1) - pre + pStar);
    //             const auto m2star  = cofac * (aL - v1) * m2;
    //             const auto eStar   = cofac * (energy*(aL - v1) + pStar*aStar - pre*v1);

    //             const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

    //             return left_flux + (starstate - left_state)*aL;
    //         }
        
    //     case 2:
    //             const auto rhoStar = cofac * (aL - v2) * rho;
    //             const auto m1star  = cofac * (aL - v2) * m1; 
    //             const auto m2star  = cofac * (m2 * (aL - v2) - pre + pStar);
    //             const auto eStar   = cofac * (energy*(aL - v2) + pStar*aStar - pre*v2);

    //             const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

    //             return left_flux + (starstate - left_state)*aL;
    //     }

    // } else {
    //     const auto pre      = right_prims.p;
    //     const auto v1       = right_prims.v1;
    //     const auto v2       = right_prims.v2;
    //     const auto rho      = right_prims.rho;
    //     const auto m1       = right_state.m1;
    //     const auto m2       = right_state.m2;
    //     const auto energy   = right_state.e_dens;
    //     const auto cofac    = 1./(aR - aStar);

    //     switch (ehat)
    //     {
    //     case 1:
    //         {
    //             const auto rhoStar = cofac * (aR - v1) * rho;
    //             const auto m1star  = cofac * (m1*(aR - v1) - pre + pStar);
    //             const auto m2star  = cofac * (aR - v1) * m2;
    //             const auto eStar   = cofac * (energy*(aR - v1) + pStar*aStar - pre*v1);

    //             const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

    //             return right_flux + (starstate - right_state)*aR;
    //         }
        
    //     case 2:
    //             const auto rhoStar = cofac * (aR - v2) * rho;
    //             const auto m1star  = cofac * (aR - v2) * m1; 
    //             const auto m2star  = cofac * (m2 * (aR - v2) - pre + pStar);
    //             const auto eStar   = cofac * (energy*(aR - v2) + pStar*aStar - pre*v2);

    //             const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

    //             return right_flux + (starstate - right_state)*aR;
    //     }


    // }
    
};

//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

void Newtonian2D::advance(
    const ExecutionPolicy<> &p,
    const luint bx,
    const luint by)
{
    const luint xpg                   = this->xphysical_grid;
    const luint ypg                   = this->yphysical_grid;
    const luint extent                = p.get_full_extent();
    #if GPU_CODE
    const luint xextent            = p.blockSize.x;
    const luint yextent            = p.blockSize.y;
    #endif

    // Compile-time choice of coloumn major indexing
    const lint sx            = (col_maj) ? 1  : bx;
    const lint sy            = (col_maj) ? by :  1;

    auto* const prim_data    = prims.data();
    auto* const cons_data    = cons.data();
    auto* const dens_source  = sourceRho.data();
    auto* const mom1_source  = sourceM1.data();
    auto* const mom2_source  = sourceM2.data();
    auto* const erg_source   = sourceE.data();
    simbi::parallel_for(p, (luint)0, extent, [CAPTURE_THIS]   GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        #if GPU_CODE 
        if ((ii >= xpg) || (jj >= ypg)) return;
        #endif

        const lint ia  = ii + radius;
        const lint ja  = jj + radius;
        const lint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const lint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const lint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const lint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;

        Conserved uxL, uxR, uyL, uyR;
        Conserved fL, fR, gL, gR, frf, flf, grf, glf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR;

        const lint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        // Load Shared memory into buffer for active zones plus ghosts
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
                prim_buff[(tya + tyl) * sx + txa]    = prim_data[(ja + tyl)    * nx + ia]; 
            }
            if (tx < radius)
            {   
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                prim_buff[tya * sx + txa - radius] =  prim_data[ja * nx + helpers::mod(ia - radius, nx)];
                prim_buff[tya * sx + txa +    txl]        =  prim_data[ja * nx +    (ia + txl) % nx]; 
            }
            simbi::gpu::api::synchronize();
        #endif

        if (first_order) [[unlikely]]
        {
            //i+1/2
            xprimsL = prim_buff[(txa + 0) * sy + (tya + 0) * sx];
            xprimsR = prim_buff[(txa + 1) * sy + (tya + 0) * sx];
            //j+1/2
            yprimsL = prim_buff[(txa + 0) * sy + (tya + 0) * sx];
            yprimsR = prim_buff[(txa + 0) * sy + (tya + 1) * sx];
            
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
            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            }

            //i-1/2
            xprimsL = prim_buff[(txa - 1) * sy + (tya + 0) * sx];
            xprimsR = prim_buff[(txa - 0) * sy + (tya + 0) * sx];
            //j-1/2
            yprimsL = prim_buff[(txa - 0) * sy + (tya - 1) * sx]; 
            yprimsR = prim_buff[(txa + 0) * sy + (tya - 0) * sx]; 

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

            // Calc HLL Flux at i-1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            } 
        }
        else 
        {
            // Coordinate X
            const Primitive xleft_most  = prim_buff[(txa - 2) * sy + tya * sx];
            const Primitive xleft_mid   = prim_buff[(txa - 1) * sy + tya * sx];
            const Primitive center      = prim_buff[(txa + 0) * sy + tya * sx];
            const Primitive xright_mid  = prim_buff[(txa + 1) * sy + tya * sx];
            const Primitive xright_most = prim_buff[(txa + 2) * sy + tya * sx];

            // Coordinate Y
            const Primitive yleft_most  = prim_buff[txa * sy + (tya - 2) * sx];
            const Primitive yleft_mid   = prim_buff[txa * sy + (tya - 1) * sx];
            const Primitive yright_mid  = prim_buff[txa * sy + (tya + 1) * sx];
            const Primitive yright_most = prim_buff[txa * sy + (tya + 2) * sx];

            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprimsL  = center     + helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta)   * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::plm_gradient(xright_mid, center, xright_most, plm_theta) * static_cast<real>(0.5);
            yprimsL  = center     + helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta)   * static_cast<real>(0.5);  
            yprimsR  = yright_mid - helpers::plm_gradient(yright_mid, center, yright_most, plm_theta) * static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            }

            // Do the same thing, but for the left side interface [i,j - 1/2]
            xprimsL  = xleft_mid  + helpers::plm_gradient(xleft_mid, xleft_most, center, plm_theta) * static_cast<real>(0.5); 
            xprimsR  = center     - helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta) * static_cast<real>(0.5);
            yprimsL  = yleft_mid  + helpers::plm_gradient(yleft_mid, yleft_most, center, plm_theta) * static_cast<real>(0.5);  
            yprimsR  = center     - helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta) * static_cast<real>(0.5);
            
            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            uyL = prims2cons(yprimsL);
            uyR = prims2cons(yprimsR);

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            gL = prims2flux(yprimsL, 2);
            gR = prims2flux(yprimsR, 2);

            
            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                break;
            } 
        }

        //Advance depending on geometry
        luint real_loc  = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
        const real rho_source = den_source_all_zeros      ? 0.0 : dens_source[real_loc];
        const real m1_source  = mom1_source_all_zeros     ? 0.0 : mom1_source[real_loc];
        const real m2_source  = mom2_source_all_zeros     ? 0.0 : mom2_source[real_loc];
        const real e_source   = energy_source_all_zeros   ? 0.0 : erg_source[real_loc];
        const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    cons_data[aid] -= ( (frf - flf) * invdx1 + (grf - glf) * invdx2 - source_terms) * step * dt;
                    break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                const real rl           = get_x1face(ii, 0); 
                const real rr           = get_x1face(ii, 1);
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

                // Grab central primitives
                const real rhoc = prim_buff[tya * bx + txa].rho;
                const real uc   = prim_buff[tya * bx + txa].v1;
                const real vc   = prim_buff[tya * bx + txa].v2;
                const real pc   = prim_buff[tya * bx + txa].p;
                
                const Conserved geom_source  = {0, (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV, - (rhoc * uc * vc) / rmean + pc * (s2R - s2L) * invdV , 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * dt * step;
                break;
                }
            case simbi::Geometry::PLANAR_CYLINDRICAL:
                {
                const real rl           = get_x1face(ii, 0); 
                const real rr           = get_x1face(ii, 1);
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
                const real uc   = prim_buff[tya * bx + txa].v1;
                const real vc   = prim_buff[tya * bx + txa].v2;
                const real pc   = prim_buff[tya * bx + txa].p;

                const Conserved geom_source  = {0, (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV, - (rhoc * uc * vc) / rmean, 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * dt * step;
                break;
                }
            case simbi::Geometry::AXIS_CYLINDRICAL:
                {
                const real rl           = get_x1face(ii, 0); 
                const real rr           = get_x1face(ii, 1);
                const real rmean        = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                const real dVtot        = rmean * (rr - rl) * dx2;
                const real invdV        = 1.0 / dVtot;
                const real s1R          = rr * dx2; 
                const real s1L          = rl * dx2; 
                const real s2R          = rmean * (rr - rl); 
                const real s2L          = rmean * (rr - rl); 

                // Grab central primitives
                const real pc   = prim_buff[tya * bx + txa].p;
                
                const Conserved geom_source  = {0, pc * (s1R - s1L) * invdV, 0, 0};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV - geom_source - source_terms) * dt * step;
                break;
                }
            default:
                break;
        } // end switch
    });
}



//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
std::vector<std::vector<real> > Newtonian2D::simulate2D(
    const std::vector<std::vector<real>> sources,
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
    this->t = tstart;
    // Define the simulation members
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->total_zones     = nx * ny;
    this->sourceRho       = sources[0];
    this->sourceM1        = sources[1];
    this->sourceM2        = sources[2];
    this->sourceE         = sources[3];
    this->first_order     = first_order;
    this->sim_solver      = helpers::solver_map.at(solver);
    this->engine_duration = engine_duration;
    this->dlogt           = dlogt;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->xphysical_grid  = (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid  = (first_order) ? ny - 2 : ny - 4;
    this->idx_active      = (first_order) ? 1 : 2;
    this->active_zones    = xphysical_grid * yphysical_grid;
    this->quirk_smoothing = quirk_smoothing;
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->checkpoint_zones= yphysical_grid;
    this->dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->x1min   = x1[0];
    this->x1max   = x1[xphysical_grid - 1];
    this->x2min   = x2[0];
    this->x2max   = x2[yphysical_grid - 1];

    this->den_source_all_zeros    = std::all_of(sourceRho.begin(), sourceRho.end(),   [](real i) {return i == 0;});
    this->mom1_source_all_zeros   = std::all_of(sourceM1.begin(),  sourceM1.end(),  [](real i) {return i == 0;});
    this->mom2_source_all_zeros   = std::all_of(sourceM2.begin(),  sourceM2.end(),  [](real i) {return i == 0;});
    this->energy_source_all_zeros = std::all_of(sourceE.begin(),   sourceE.end(), [](real i) {return i == 0;});
    define_tinterval(t, dlogt, chkpt_interval, chkpt_idx);
    define_chkpt_idx(chkpt_idx);
    // Stuff for moving mesh
    this->hubble_param = 0.0; ///adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (x2max == 0.5 * M_PI){
        this->half_sphere = true;
    }

    inflow_zones.resize(4);
    for (int i = 0; i < 4; i++) {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = Conserved{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2], boundary_sources[i][3]};
    }
    
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
    setup.boundary_conditions = boundary_conditions;
    setup.regime         = "classical";
    setup.x1             = x1;
    setup.x2             = x2;
    setup.mesh_motion    = mesh_motion;
    setup.dimensions = 2;

    cons.resize(nzones);
    prims.resize(nzones);
    dt_min.resize(active_zones);
    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state[0].size(); i++)
    {
        auto rho      = state[0][i];
        auto m1       = state[1][i];
        auto m2       = state[2][i];
        auto e        = state[3][i];
        auto rho_chi  = state[4][i];
        cons[i]    = Conserved(rho, m1, m2, e, rho_chi);
    }

    cons.copyToGpu();
    prims.copyToGpu();
    dt_min.copyToGpu();
    sourceRho.copyToGpu();
    sourceM1.copyToGpu();
    sourceM2.copyToGpu();
    sourceE.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();

    // TODO: Implement moving mesh at some point
    if (false) {
        outer_zones.resize(ny);
    }


   
    dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    invdx1  = 1 / dx1;
    invdx2  = 1 / dx2;
    x1min   = x1[0];
    x1max   = x1[xphysical_grid - 1];
    x2min   = x2[0];
    x2max   = x2[yphysical_grid - 1];


    // Setup the system
    const luint xblockdim       = xphysical_grid > gpu_block_dimx ? gpu_block_dimx : xphysical_grid;
    const luint yblockdim       = yphysical_grid > gpu_block_dimy ? gpu_block_dimy : yphysical_grid;
    this->radius                = (first_order) ? 1 : 2;
    this->step                  = (first_order) ? 1 : static_cast<real>(0.5);
    const luint bx              = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint by              = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint shBlockSpace    = bx * by;
    const luint shBlockBytes    = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim});
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU){
        writeln("Requested shared memory: {} bytes", shBlockBytes);
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
        write2file(*this, setup, data_directory, t, 0, chkpt_interval, yphysical_grid);
        helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
    }
    
    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&](){
        advance(activeP, bx, by);
        cons2prim(fullP);
        helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
    });
    
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