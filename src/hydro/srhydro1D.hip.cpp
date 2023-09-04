/*
 * C++ Library to perform extensive hydro calculations
 * to be later wrapped and plotted in Python
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */
#include <chrono>
#include <cmath>
#include "srhydro1D.hip.hpp"
#include "common/helpers.hip.hpp"
#include "util/device_api.hpp"
#include "util/printb.hpp"
#include "util/parallel_for.hpp"
#include "util/logger.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;
constexpr auto write2file = helpers::write_to_file<sr1d::PrimitiveSOA, 1, SRHD>;
//================================================
//              DATA STRUCTURES
//================================================
using Conserved  =  sr1d::Conserved;
using Primitive  =  sr1d::Primitive;
using Eigenvals  =  sr1d::Eigenvals;

// Overloaded Constructor
SRHD::SRHD(
    std::vector<std::vector<real>> &state, 
    real gamma, 
    real cfl,
    std::vector<real> &x1, 
    std::string coord_system = "cartesian") 
:
    HydroBase(
        state,
        gamma,
        cfl,
        x1,
        coord_system
    )
{

}

GPU_CALLABLE_MEMBER
real SRHD::calc_vface(const lint ii, const real hubble_const, const simbi::Geometry geometry, const int side) const
{
    switch(geometry)
    {
        case simbi::Geometry::CARTESIAN:
        {
            // Rigid motion
            return  hubble_const;
        }
        default:
        {
            const real xl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1), x1min); 
            if (side == 0) {
                return xl;
            } else {
                const real xr = helpers::my_min(xl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)),  x1max);
                return xr;
            }
        }
    }
}
//=====================================================================
//                          KERNEL CALLS
//=====================================================================
void SRHD::advance(const ExecutionPolicy<> &p)
{
    #if GPU_CODE
    const auto xextent          = p.get_xextent();
    #endif 
    auto* const cons_data       = cons.data();
    auto* const prim_data       = prims.data();
    auto* const dens_source     = sourceD.data();
    auto* const mom_source      = sourceS.data();
    auto* const erg_source      = source0.data();
    auto* const grav_source     = sourceG.data();
    simbi::parallel_for(p, (luint)0, active_zones, [CAPTURE_THIS]   GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = prim_data;
        #endif 

        Conserved uL, uR;
        Conserved fL, fR, frf, flf;
        Primitive primsL, primsR;

        lint ia  = ii + radius;
        lint txa = (BuildPlatform == Platform::GPU) ?  threadIdx.x + radius : ia;
        #if GPU_CODE
            luint txl = xextent;
            // Check if the active index exceeds the active zones
            // if it does, then this thread buffer will taken on the
            // ghost index at the very end and return
            prim_buff[txa] = prim_data[ia];
            if (threadIdx.x < radius)
            {
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + threadIdx.x)) {
                    txl = nx - radius - ia + threadIdx.x;
                }
                prim_buff[txa - radius] = prim_data[ia - radius];
                prim_buff[txa + txl]        = prim_data[ia + txl];
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = get_xface(ii, geometry, 0);
        const real x1r    = get_xface(ii, geometry, 1);
        const real vfaceL = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
        const real vfaceR = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
        if (first_order) [[unlikely]]
        {
            // Set up the left and right state interfaces for i+1/2
            primsL = prim_buff[txa + 0];
            primsR = prim_buff[txa + 1];
            uL     = prims2cons(primsL);
            uR     = prims2cons(primsR);
            fL     = prims2flux(primsL);
            fR     = prims2flux(primsR);

            // Calc HLL Flux at i+1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
                break;
            
            default:
                frf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
                break;
            }

            // Set up the left and right state interfaces for i-1/2
            primsL = prim_buff[txa - 1];
            primsR = prim_buff[txa - 0];

            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            // Calc HLL Flux at i-1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
                break;
            
            default:
                flf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
                break;
            } 
        } else {
            const Primitive left_most  = prim_buff[txa - 2];
            const Primitive left_mid   = prim_buff[txa - 1];
            const Primitive center     = prim_buff[txa + 0];
            const Primitive right_mid  = prim_buff[txa + 1];
            const Primitive right_most = prim_buff[txa + 2];

            // Compute the reconstructed primitives at the i+1/2 interface
            // Reconstructed left primitives vector
            primsL  = center     + helpers::plm_gradient(center, left_mid, right_mid, plm_theta)   * static_cast<real>(0.5); 
            primsR  = right_mid  - helpers::plm_gradient(right_mid, center, right_most, plm_theta) * static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM primitives
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
                break;
            
            default:
                frf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
                break;
            }
            
            // Do the same thing, but for the right side interface [i - 1/2]
            primsL  = left_mid + helpers::plm_gradient(left_mid, left_most, center, plm_theta) * static_cast<real>(0.5); 
            primsR  = center   - helpers::plm_gradient(center, left_mid, right_mid, plm_theta) * static_cast<real>(0.5);
            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
                break;
            
            default:
                flf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
                break;
            } 
        }

        const auto d_source = den_source_all_zeros    ? 0.0 :  dens_source[ii];
        const auto s_source = mom1_source_all_zeros   ? 0.0 :  mom_source[ii];
        const auto e_source = energy_source_all_zeros ? 0.0 :  erg_source[ii];
        const auto gs_source = grav_source_all_zeros  ? 0.0 :  cons_data[ia].d * grav_source[ii];
        const auto ge_source = gs_source * prim_buff[txa].v;
        const auto sources = Conserved{d_source, s_source, e_source} * time_constant;
        const auto gravity = Conserved{0, gs_source, ge_source};
        switch(geometry)
        {
            case simbi::Geometry::CARTESIAN:
            {
                cons_data[ia] -= ((frf - flf) * invdx1) * dt * step;
                break;
            }
            default:
            {
                const real rlf    = x1l + vfaceL * step * dt; 
                const real rrf    = x1r + vfaceR * step * dt;
                const real rmean  = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                const real sR     = rrf * rrf; 
                const real sL     = rlf * rlf; 
                const real dV     = rmean * rmean * (rrf - rlf);    
                const real factor = (mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;
                const real invdV  = 1 / dV;

                const auto geom_sources = Conserved{0.0, pc * (sR - sL) * invdV, 0.0};
                cons_data[ia] -= ( (frf * sR - flf * sL) * invdV - geom_sources - sources - gravity) * step * dt * factor;
                break;
            }
        } // end switch
    });	
}

void SRHD::cons2prim(const ExecutionPolicy<> &p)
{
    auto* const cons_data  = cons.data();
    auto* const prims_data = prims.data();
    auto* const press_data = pressure_guess.data();
    auto* const troubled_data = troubled_cells.data();
    simbi::parallel_for(p, (luint)0, nx, [CAPTURE_THIS]   GPU_LAMBDA (luint ii){
        real g, f, peq;
        // pre, pstar;
        volatile __shared__ bool found_failure;
        luint tx = get_threadId();

        if (tx == 0) 
            found_failure = inFailureState;
        simbi::gpu::api::synchronize();
        
        real invdV = 1.0;
        bool workLeftToDo = true;
        while (!found_failure && workLeftToDo)
        {   
            if (mesh_motion && (geometry == simbi::Geometry::SPHERICAL))
            {
                const luint idx  = helpers::get_real_idx(ii, radius, active_zones);
                const real xl    = get_xface(idx, geometry, 0);
                const real xr    = get_xface(idx, geometry, 1);
                const real xmean = static_cast<real>(0.75) * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl);
                invdV            = 1 / (xmean * xmean * (xr - xl));
            }
            peq            = press_data[ii];
            // pstar          = peq;
            const real D   = cons_data[ii].d   * invdV;
            const real S   = cons_data[ii].s   * invdV;
            const real tau = cons_data[ii].tau * invdV;
            int iter       = 0;

            // Perform modified Newton Raphson based on
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930
            // so far, the convergence rate is the same, but perhaps I need a slight tweak

            // compute f(x_0)
            // f = helpers::newton_f(gamma, tau, D, S, peq);
            const real tol = D * tol_scale;
            do
            {
                // compute x_[k+1]
                f     = helpers::newton_f(gamma, tau, D, S, peq);
                g     = helpers::newton_g(gamma, tau, D, S, peq);
                peq  -= f / g;

                // compute x*_k
                // f     = helpers::newton_f(gamma, tau, D, S, peq);
                // pstar = peq - f / g;

                if (iter >= MAX_ITER || std::isnan(peq))
                {
                    troubled_data[ii] = iter;
                    dt                = INFINITY;
                    inFailureState    = true;
                    found_failure     = true;
                    break;
                }
                iter++;

            } while (std::abs(f / g) >= tol);
            
            real v = S / (tau + D + peq);
            real W = 1 / std::sqrt(1 - v * v);
            // real mach_ceiling = 100.0;
            // real u            = v /std::sqrt(1 - v * v);
            // real e            = peq / rho * 3.0;
            // real emin         = u * u / (1.0 + u * u) / std::pow(mach_ceiling, 2.0);

            // if (e < emin) {
            //     // printf("peq: %f, npew: %f\n", rho * emin * (gamma - 1.0));
            //     peq = rho * emin * (gamma - 1.0);
            // }
            press_data[ii] = peq;
            #if FOUR_VELOCITY
                prims_data[ii] = Primitive{D/ W, v * W, peq};
            #else
                prims_data[ii] = Primitive{D/ W, v, peq};
            #endif

            if (peq < 0) {
                troubled_data[ii] = iter;
                dt                = INFINITY;
                inFailureState    = true;
                found_failure     = true;
            }
            simbi::gpu::api::synchronize();
            workLeftToDo = false;
        }
    });
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD::calc_eigenvals(
    const Primitive &primsL,
    const Primitive &primsR) const
{
    // Compute L/R Sound Speeds
    const real rhoL = primsL.rho;
    const real vL   = primsL.get_v();
    const real pL   = primsL.p;
    const real hL   = 1 + gamma * pL / (rhoL * (gamma - 1));
    const real csL  = std::sqrt(gamma * pL / (rhoL * hL));

    const real rhoR  = primsR.rho;
    const real vR    = primsR.get_v();
    const real pR    = primsR.p;
    const real hR    = 1 + gamma * pR  / (rhoR * (gamma - 1));
    const real csR   = std::sqrt(gamma * pR  / (rhoR * hR));

    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
            const real vbar = static_cast<real>(0.5) * (vL + vR);
            const real cbar = static_cast<real>(0.5) * (csR + csL);

            const real bR   = (vbar + cbar) / (1 + vbar * cbar);
            const real bL   = (vbar - cbar) / (1 - vbar * cbar);

            const real aL = helpers::my_min(bL, (vL - csL) / (1 - vL * csL));
            const real aR = helpers::my_max(bR, (vR + csR) / (1 + vR * csR));

            return Eigenvals(aL, aR);
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
            
            return Eigenvals(aL, aR);
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

            return Eigenvals(aL, aR);
        }
    default: // NAIVE estimates
    {
        const real aL = helpers::my_min((vR - csR) / (1 - vR * csR), (vL - csL) / (1 - vL * csL));
        const real aR = helpers::my_max((vL + csL) / (1 + vL * csL), (vR + csR) / (1 + vR * csR));

        return Eigenvals(aL, aR);
    }
    }  
};

// Adapt the cfl conditonal timestep
template<TIMESTEP_TYPE dt_type>
void SRHD::adapt_dt()
{   
    if (use_omp) {
        real min_dt = INFINITY;
        // Compute the minimum timestep given cfl
        #pragma omp parallel for schedule(static) reduction(min:min_dt)
        for (luint ii = 0; ii < active_zones; ii++){
            const auto shift_i  = ii + idx_active;
            const real rho      = prims[shift_i].rho;
            const real v        = prims[shift_i].get_v();
            const real pre      = prims[shift_i].p;
            const real h        = 1 + pre * gamma / (rho * (gamma - 1));
            const real cs       = std::sqrt(gamma * pre / (rho * h));

            const real vPlus = ([&]{
                if constexpr(dt_type == simbi::TIMESTEP_TYPE::ADAPTIVE) {
                    return  (v + cs) / (1 + v * cs);
                } else {
                    return  1;
                }
            })();
            const real vMinus = ([&]{
                if constexpr(dt_type == simbi::TIMESTEP_TYPE::ADAPTIVE) {
                    return (v - cs) / (1 - v * cs);
                } else {
                    return 1;
                }
            })();

            const real x1l      = get_xface(ii, geometry, 0);
            const real x1r      = get_xface(ii, geometry, 1);
            const real dx1      = x1r - x1l;
            const real vfaceL   = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
            const real vfaceR   = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
            const real cfl_dt   = dx1 / (helpers::my_max(std::abs(vPlus - vfaceR), std::abs(vMinus - vfaceL)));
            min_dt              = std::min(min_dt, cfl_dt);
        }
        dt = cfl * min_dt;
    } else {
        std::atomic<real> min_dt = INFINITY;
        thread_pool.parallel_for(static_cast<luint>(0), active_zones, [&](int ii) {
            const auto shift_i  = ii + idx_active;
            const real rho      = prims[shift_i].rho;
            const real v        = prims[shift_i].get_v();
            const real pre      = prims[shift_i].p;
            const real h        = 1 + pre * gamma / (rho * (gamma - 1));
            const real cs       = std::sqrt(gamma * pre / (rho * h));

            const real vPlus = ([&]{
                if constexpr(dt_type == simbi::TIMESTEP_TYPE::ADAPTIVE) {
                    return (v + cs) / (1 + v * cs);
                } else {
                    return 1;
                }
            })();

            const real vMinus = ([&]{
                if constexpr(dt_type == simbi::TIMESTEP_TYPE::ADAPTIVE) {
                    return (v - cs) / (1 - v * cs);
                } else {
                    return 1;
                }
            })();

            const real x1l      = get_xface(ii, geometry, 0);
            const real x1r      = get_xface(ii, geometry, 1);
            const real dx1      = x1r - x1l;
            const real vfaceL   = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
            const real vfaceR   = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
            const real cfl_dt   = dx1 / (helpers::my_max(std::abs(vPlus - vfaceR), std::abs(vMinus - vfaceL)));
            pooling::update_minimum(min_dt, cfl_dt);
        });
        dt = cfl * min_dt;
    }
};

template<TIMESTEP_TYPE dt_type>
void SRHD::adapt_dt(const luint blockSize)
{   
    #if GPU_CODE
        compute_dt<Primitive><<<dim3(blockSize), dim3(gpu_block_dimx)>>>(this, prims.data(), dt_min.data());
        deviceReduceWarpAtomicKernel<1><<<blockSize, gpu_block_dimx>>>(this, dt_min.data(), active_zones);
        gpu::api::deviceSynch();
    #endif
};

//----------------------------------------------------------------------------------------------------
//              STATE ARRAY CALCULATIONS
//----------------------------------------------------------------------------------------------------

// Get the (3,1) state array for computation. Used for Higher Order
// Reconstruction
GPU_CALLABLE_MEMBER
Conserved SRHD::prims2cons(const Primitive &prim) const
{
    const real rho = prim.rho;
    const real v   = prim.get_v();
    const real pre = prim.p;  
    const real h   = 1 + gamma * pre / (rho * (gamma - 1));
    const real W   = 1 / std::sqrt(1 - v * v);
    return Conserved{rho * W, rho * h * W * W * v, rho * h * W * W - pre - rho * W};
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------
// Get the 1D Flux array (3,1)
GPU_CALLABLE_MEMBER
Conserved SRHD::prims2flux(const Primitive &prim) const
{
    const real rho = prim.rho;
    const real v   = prim.get_v();
    const real pre = prim.p;
    const real W   = 1 / std::sqrt(1 - v * v);
    const real h   = 1 + gamma * pre / (rho * (gamma - 1));
    const real D   = rho * W;
    const real S   = rho * h * W * W * v;
    return Conserved{D*v, S*v + pre, S - D*v};
};

GPU_CALLABLE_MEMBER Conserved SRHD::calc_hll_flux(
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux,  
    const Conserved &right_flux,
    const real      vface) const
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    // Grab the necessary wave speeds
    const real aL  = lambda.aL;
    const real aR  = lambda.aR;
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

    // Compute the HLL Flux component-wise
    if (vface < aLm) {
        return left_flux - left_state * vface;
    } else if (vface > aRp) {
        return right_flux - right_state * vface;
    } else {    
        const Conserved f_hll = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aLm * aRp) / (aRp - aLm);
        const Conserved u_hll = (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);
        return f_hll - u_hll * vface;
    }
};

GPU_CALLABLE_MEMBER Conserved SRHD::calc_hllc_flux(
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux,  
    const Conserved &right_flux,
    const real       vface) const
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);
    const real aL  = lambda.aL;
    const real aR  = lambda.aR;
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

    if (vface <= aLm) {
        return left_flux - left_state * vface;
    } else if (vface >= aRp) {
        return right_flux - right_state * vface;
    }
    const Conserved hll_flux  = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aLm * aRp) / (aRp - aLm);
    const Conserved hll_state = (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);

    const real uhlld   = hll_state.d;
    const real uhlls   = hll_state.s;
    const real uhlltau = hll_state.tau;
    const real fhlld   = hll_flux.d;
    const real fhlls   = hll_flux.s;
    const real fhlltau = hll_flux.tau;
    const real e    = uhlltau + uhlld;
    const real s    = uhlls;
    const real fs   = fhlls;
    const real fe   = fhlltau + fhlld;
    
    const real a     = fe;
    const real b     = - (e + fs);
    const real c     = s;
    // const real disc  = std::sqrt(b*b - 4.0*a*c);
    // const real quad  = -static_cast<real>(0.5)*(b + helpers::sgn(b)*disc);
    // const real aStar = c/quad;
    const real scrh  = 1.0 + std::sqrt(1.0 - 4.0*a*c/(b*b));
    const real aStar = - 2.0*c/(b*scrh);
    const real pStar = -fe * aStar + fs;

    
    if (vface <= aStar) {
        const real v        = left_prims.get_v();
        const real pressure = left_prims.p;
        const real D        = left_state.d;
        const real S        = left_state.s;
        const real tau      = left_state.tau;
        const real E        = tau + D;
        const real cofactor = 1 / (aLm - aStar);

        //--------------Compute the L Star State----------
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aLm - v) * D;
        const real Sstar    = cofactor * (S * (aLm - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aLm - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto star_stateL = Conserved{Dstar, Sstar, tauStar};

        //---------Compute the L Star Flux
        // Compute the HLL Flux component-wise
        Conserved hllc_flux = left_flux + (star_stateL - left_state) * aLm;
        return    hllc_flux - star_stateL * vface;
    } else {
        const real v         = right_prims.get_v();
        const real pressure  = right_prims.p;
        const real D         = right_state.d;
        const real S         = right_state.s;
        const real tau       = right_state.tau;
        const real E         = tau + D;
        const real cofactor  = 1 / (aRp - aStar);

        //--------------Compute the R Star State----------
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aRp - v) * D;
        const real Sstar    = cofactor * (S * (aRp - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aRp - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto star_stateR = Conserved{Dstar, Sstar, tauStar};

        //---------Compute the R Star Flux  
        Conserved hllc_flux = right_flux + (star_stateR - right_state) * aRp;
        return    hllc_flux -  star_stateR * vface;
    }
};
//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------
std::vector<std::vector<real>>
SRHD::simulate1D(
    std::vector<std::vector<real>> &sources,
    std::vector<real> &gsource,
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
    std::vector<std::vector<real>> boundary_sources,
    std::function<double(double)> a,
    std::function<double(double)> adot,
    std::function<double(double)> d_outer,
    std::function<double(double)> s_outer,
    std::function<double(double)> e_outer)
{
    anyDisplayProps();
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->first_order     = first_order;
    this->plm_theta       = plm_theta;
    this->linspace        = linspace;
    this->sourceD         = sources[0];
    this->sourceS         = sources[1];
    this->source0         = sources[2];
    this->sourceG         = gsource;
    this->sim_solver      = helpers::solver_map.at(solver);
    this->engine_duration = engine_duration;
    this->t               = tstart;
    this->tend            = tend;
    this->dlogt           = dlogt;
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->idx_active      = (first_order) ? 1 : 2;
    this->active_zones    = (first_order) ? nx - 2 : nx - 4;
    this->dlogx1          = std::log10(x1[active_zones - 1]/ x1[0]) / (active_zones - 1);
    this->dx1             = (x1[active_zones - 1] - x1[0]) / (active_zones - 1);
    this->invdx1          = 1.0 / this->dx1;
    this->x1min           = x1[0];
    this->x1max           = x1[active_zones - 1];
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->total_zones     = nx;
    this->checkpoint_zones= active_zones;
    this->den_source_all_zeros    = std::all_of(sourceD.begin(), sourceD.end(), [](real i) {return i==0;});
    this->mom1_source_all_zeros   = std::all_of(sourceS.begin(), sourceS.end(), [](real i) {return i==0;});
    this->energy_source_all_zeros = std::all_of(source0.begin(), source0.end(), [](real i) {return i==0;});
    this->grav_source_all_zeros = std::all_of(sourceG.begin(), sourceG.end(), [](real i){return i==0;});
    define_tinterval(tstart, dlogt, chkpt_interval, chkpt_idx);
    define_chkpt_idx(chkpt_idx);
    inflow_zones.resize(2);
    for (size_t i = 0; i < 2; i++)
    {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = Conserved{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2]};
    }
    
    this->hubble_param       = adot(t) / a(t);
    this->mesh_motion        = (hubble_param != 0);
    this->all_outer_bounds   = (d_outer && s_outer && e_outer);
    if (all_outer_bounds){
        dens_outer = d_outer;
        mom_outer  = s_outer;
        nrg_outer  = e_outer;
    }
    
    setup.x1max              = x1[active_zones - 1];
    setup.x1min              = x1[0];
    setup.xactive_zones      = active_zones;
    setup.nx                 = nx;
    setup.linspace           = linspace;
    setup.ad_gamma           = gamma;
    setup.first_order        = first_order;
    setup.coord_system       = coord_system;
    setup.using_fourvelocity = (VelocityType == Velocity::FourVelocity);
    setup.x1                 = x1;
    setup.regime             = "relativistic";
    setup.mesh_motion        = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions = 1;
    cons.resize(nx);
    prims.resize(nx);
    pressure_guess.resize(nx);
    troubled_cells.resize(nx, 0);
    dt_min.resize(active_zones);
    if (mesh_motion && all_outer_bounds) {
        outer_zones.resize(2);
        const real dV  = get_cell_volume(active_zones - 1, geometry);
        outer_zones[0] = conserved_t{
            dens_outer(x1max), 
            mom_outer(x1max), 
            nrg_outer(x1max)} * dV;
        outer_zones.copyToGpu();
    }

    // Copy the state array into real & profile variables
    for (luint ii = 0; ii < nx; ii++) {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
        // initial pressure guess is | |S| - D - tau|
        pressure_guess[ii] = std::abs((state[1][ii]) - state[0][ii] - state[2][ii]);
    }
    cons.copyToGpu();
    prims.copyToGpu();
    pressure_guess.copyToGpu();
    dt_min.copyToGpu();
    sourceD.copyToGpu();
    sourceS.copyToGpu();
    source0.copyToGpu();
    sourceG.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();
    troubled_cells.copyToGpu();

    const auto xblockdim      = nx > gpu_block_dimx ? gpu_block_dimx : nx;
    this->radius              = (first_order) ? 1 : 2;
    this->step                = (first_order) ? 1 : static_cast<real>(0.5);
    const luint shBlockSize   = gpu_block_dimx + 2 * radius;
    const luint shBlockBytes  = shBlockSize * sizeof(Primitive);
    const auto fullP          = simbi::ExecutionPolicy(nx, xblockdim);
    const auto activeP        = simbi::ExecutionPolicy(active_zones, xblockdim, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU){
        writeln("Requested shared memory: {} bytes", shBlockBytes);
    }

    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP);
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(activeP.gridSize.x);
    } else {
        cons2prim(fullP);
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }
    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);

    // Save initial condition
    if (t == 0 || chkpt_idx == 0) {
        write2file(*this, setup, data_directory, t, 0, chkpt_interval, active_zones);
        config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
    }

    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&](){
        if (inFailureState){
            return;
        }
        advance(activeP);
        cons2prim(fullP);
        config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP.gridSize.x);
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
    // Check if in failure state, and emit troubled cells
    if (inFailureState){
        emit_troubled_cells();
    }

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};