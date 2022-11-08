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

// Default Constructor
SRHD::SRHD(){}

// Overloaded Constructor
SRHD::SRHD(
    std::vector<std::vector<real>> state, 
    real gamma, 
    real cfl,
    std::vector<real> x1, 
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
// Destructor
SRHD::~SRHD()
{}

void SRHD::set_mirror_ptrs()
{
    cons_ptr   = &cons[0];
    prims_ptr  = &prims[0];
    pguess_ptr = &pressure_guess[0];
}

GPU_CALLABLE_MEMBER
real SRHD::calc_vface(const lint ii, const real hubble_const, const simbi::Geometry geometry, const int side) const
{
    switch(geometry)
    {
    case simbi::Geometry::SPHERICAL:
        {
            const real xl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1), x1min); 
            if (side == 0) {
                return xl;
            } else {
                const real xr = helpers::my_min(xl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)),  x1max);
                return xr;
            }
        }
    case simbi::Geometry::CARTESIAN:
        {
            // Rigid motion
            return  hubble_const;
        }
    case simbi::Geometry::CYLINDRICAL:
        // TODO: Implement Cylindrical coordinates at some point
        break;
    }
}
//=====================================================================
//                          KERNEL CALLS
//=====================================================================
void SRHD::advance(
    SRHD *dev,  
    const luint sh_block_size,
    const luint radius, 
    const simbi::Geometry geometry,
    const simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;

    // const bool mesh_motion      = (hubble_param != 0);
    const unsigned shBlockBytes = sh_block_size * sizeof(Primitive);
    auto p                      = simbi::ExecutionPolicy(nx);
    p.blockSize                 = BLOCK_SIZE;
    p.sharedMemBytes            = shBlockBytes;
    // const real xpg            = this->active_zones;
    const lint bx               = (BuildPlatform == Platform::GPU) ? sh_block_size : self->nx;
    const lint  pseudo_radius   = (first_order) ? 1 : 2;
    const real step             = (first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
    const real inv_dx           = 1.0 / this->dx1;
    auto* const cons_data       = cons.data();
    auto* const prim_data       = prims.data();
    auto* const dens_source     = sourceD.data();
    auto* const mom_source      = sourceS.data();
    auto* const erg_source      = source0.data();
    simbi::parallel_for(p, (luint)0, self->active_zones, [=] GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = prim_data;
        #endif 

        Conserved uL, uR;
        Conserved fL, fR, frf, flf;
        Primitive primsL, primsR;

        lint ia  = ii + radius;
        lint txa = (BuildPlatform == Platform::GPU) ?  threadIdx.x + pseudo_radius : ia;
        #if GPU_CODE
            luint txl = BLOCK_SIZE;
            // Check if the active index exceeds the active zones
            // if it does, then this thread buffer will taken on the
            // ghost index at the very end and return
            prim_buff[txa] = prim_data[ia];
            if (threadIdx.x < pseudo_radius)
            {
                if (blockIdx.x == p.gridSize.x - 1 && (ia + BLOCK_SIZE > self->nx - radius + threadIdx.x)) {
                    txl = self->nx - radius - ia + threadIdx.x;
                }
                prim_buff[txa - pseudo_radius] = prim_data[helpers::mod(ia - pseudo_radius, self->nx)];
                prim_buff[txa + txl   ]        = prim_data[(ia + txl ) % self->nx];
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = self->get_xface(ii, geometry, 0);
        const real x1r    = self->get_xface(ii, geometry, 1);
        const real vfaceL = (geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1l * self->hubble_param;
        const real vfaceR = (geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1r * self->hubble_param;
        if (self->first_order)
        {
            // Set up the left and right state interfaces for i+1/2
            primsL = prim_buff[(txa + 0) % bx];
            primsR = prim_buff[(txa + 1) % bx];
            uL     = self->prims2cons(primsL);
            uR     = self->prims2cons(primsR);
            fL     = self->prims2flux(primsL);
            fR     = self->prims2flux(primsR);

            // Calc HLL Flux at i+1/2 interface
            if (self->hllc){
                frf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
            } else {
                frf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
            }

            // Set up the left and right state interfaces for i-1/2
            primsL = prim_buff[helpers::mod(txa - 1, bx)];
            primsR  = prim_buff[(txa - 0) % bx];

            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            // Calc HLL Flux at i-1/2 interface
            if (self->hllc) {
                flf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
            } else {
                flf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
            }   
        } else {
            const Primitive left_most  = prim_buff[helpers::mod(txa - 2, bx)];
            const Primitive left_mid   = prim_buff[helpers::mod(txa - 1, bx)];
            const Primitive center     = prim_buff[(txa + 0) % bx];
            const Primitive right_mid  = prim_buff[(txa + 1) % bx];
            const Primitive right_most = prim_buff[(txa + 2) % bx];

            // Compute the reconstructed primitives at the i+1/2 interface
            // Reconstructed left primitives vector
            primsL = center    + helpers::minmod((center - left_mid)*self->plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*self->plm_theta)*static_cast<real>(0.5); 
            primsR = right_mid - helpers::minmod((right_mid - center)*self->plm_theta, (right_most - center)*static_cast<real>(0.5), (right_most - right_mid)*self->plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM primitives
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            if (self->hllc) {
                frf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
            } else {
                frf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceR);
            }
            
            // Do the same thing, but for the right side interface [i - 1/2]
            primsL = left_mid + helpers::minmod((left_mid - left_most)*self->plm_theta, (center - left_most)*static_cast<real>(0.5), (center - left_mid)*self->plm_theta)*static_cast<real>(0.5);
            primsR = center   - helpers::minmod((center - left_mid)*self->plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*self->plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            if (self->hllc) {
                flf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
            } else {
                flf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR, vfaceL);
            }
        }

        const auto sources = Conserved{dens_source[ii], mom_source[ii], erg_source[ii]} * self->decay_constant;
        switch(geometry)
        {
            case simbi::Geometry::CARTESIAN:
                cons_data[ia] -= ((frf - flf) * inv_dx) * self->dt * step;
                break;
            case simbi::Geometry::SPHERICAL:
            {
                const real rlf    = x1l + vfaceL * step * self->dt; 
                const real rrf    = x1r + vfaceR * step * self->dt;
                const real rmean  = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                const real sR     = rrf * rrf; 
                const real sL     = rlf * rlf; 
                const real dV     = rmean * rmean * (rrf - rlf);    
                const real factor = (self->mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;

                const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                cons_data[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * self->dt * factor;
                break;
            }
                
            case simbi::Geometry::CYLINDRICAL:
            {
                // const real rlf    = x1l + vfaceL * step * self->dt; 
                // const real rrf    = x1r + vfaceR * step * self->dt;
                // const real rmean  = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                // const real sR     = rrf; 
                // const real sL     = rlf; 
                // const real dV     = rmean * (rrf - rlf);           
                // const real pc     = prim_buff[txa].p;
                
                // #if GPU_CODE
                //     const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                //     const auto sources = Conserved{self->gpu_sourceD[ii], self->gpu_sourceS[ii],self->gpu_source0[ii]} * self->decay_constant;
                //     self->gpu_cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * self->dt;
                // #else 
                //     const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                //     const auto sources = Conserved{sourceD[ii], sourceS[ii],source0[ii]} * self->decay_constant;
                //     cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * self->dt;
                // #endif 
                break;
            }
        } // end switch
        if (ii == self->active_zones - 1) {
            self->x1max += step * self->dt * vfaceR;
        } else if (ii == 0) {
            self->x1min += step * self->dt * vfaceL;
        }
    });	
    if constexpr(BuildPlatform == Platform::GPU) {
        this->x1min = dev->x1min;
        this->x1max = dev->x1max;
    }
}

void SRHD::cons2prim(ExecutionPolicy<> p, SRHD *dev, simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    #if GPU_CODE
    const real radius = (first_order) ? 1 : 2;
    #endif
    auto* const cons_data  = cons.data();
    auto* const prims_data = prims.data();
    auto* const press_data = pressure_guess.data();
    simbi::parallel_for(p, (luint)0, nx, [=] GPU_LAMBDA (luint ii){
        real eps, pre, v2, et, c2, h, g, f, W, rho, peq;
        volatile __shared__ bool found_failure;
        luint tx = (BuildPlatform == Platform::GPU) ? threadIdx.x : ii;

        if (tx == 0) 
            found_failure = self->inFailureState;
        simbi::gpu::api::synchronize();
        
        real invdV = 1.0;
        bool workLeftToDo = true;
        while (!found_failure && workLeftToDo)
        {   
            if (self->mesh_motion && (self->geometry == simbi::Geometry::SPHERICAL))
            {
                const luint idx  = helpers::get_real_idx(ii, radius, self->active_zones);
                const real xl    = self->get_xface(idx, self->geometry, 0);
                const real xr    = self->get_xface(idx, self->geometry, 1);
                const real xmean = static_cast<real>(0.75) * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl);
                invdV            = static_cast<real>(1.0) / (xmean * xmean * (xr - xl));
            }
            peq = press_data[ii];
            const real D   = cons_data[ii].d   * invdV;
            const real S   = cons_data[ii].s   * invdV;
            const real tau = cons_data[ii].tau * invdV;
            int iter       = 0;
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
                c2 = self->gamma *pre / (h * rho); 

                g = c2 * v2 - static_cast<real>(1.0);
                f = (self->gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                if (iter >= MAX_ITER || std::isnan(peq))
                {
                    const luint idx       = helpers::get_real_idx(ii, radius, self->active_zones);
                    const real xl         = self->get_xface(idx, self->geometry, 0);
                    const real xr         = self->get_xface(idx, self->geometry, 1);
                    const real xmean      = helpers::calc_any_mean(xl, xr, self->x1cell_spacing);
                    printf("\nCons2Prim cannot converge\n");
                    printf("Density: %.3e, Pressure: %.3e, vsq: %.3e, coord: %.2e\n", rho, peq, v2, xmean);
                    self->dt             = INFINITY;
                    self->inFailureState = true;
                    found_failure        = true;
                    simbi::gpu::api::synchronize();
                    break;
                }
                iter++;

            } while (std::abs(peq - pre) >= tol);

            real v = S / (tau + D + peq);
            // real mach_ceiling = 100.0;
            // real u            = v /std::sqrt(1 - v * v);
            // real e            = peq / rho * 3.0;
            // real emin         = u * u / (1.0 + u * u) / std::pow(mach_ceiling, 2.0);

            // if (e < emin) {
            //     // printf("peq: %f, npew: %f\n", rho * emin * (self->gamma - 1.0));
            //     peq = rho * emin * (self->gamma - 1.0);
            // }
            press_data[ii] = peq;
            prims_data[ii] = Primitive{rho, v, peq};
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
    const real pL   = primsL.p;
    const real vL   = primsL.v;
    const real hL   = static_cast<real>(1.0) + gamma * pL / (rhoL * (gamma - 1));
    const real csL  = std::sqrt(gamma * pL / (rhoL * hL));

    const real rhoR  = primsR.rho;
    const real pR    = primsR.p;
    const real vR    = primsR.v;
    const real hR    = static_cast<real>(1.0) + gamma * pR  / (rhoR  * (gamma - 1));
    const real csR   = std::sqrt(gamma * pR  / (rhoR  * hR));

    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
            const real vbar = static_cast<real>(0.5) * (vL + vR);
            const real cbar = static_cast<real>(0.5) * (csR  + csL);

            const real bR   = (vbar + cbar) / (1 + vbar * cbar);
            const real bL   = (vbar - cbar) / (1 - vbar * cbar);

            const real aL = helpers::my_min(bL, (vL - csL) / (1 - vL * csL));
            const real aR = helpers::my_max(bR, (vR  + csR) / (1 + vR  * csR));

            return Eigenvals(aL, aR);
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

            return Eigenvals(aL, aR);
        }
    case simbi::WaveSpeeds::NAIVE:
    {
        const real aL = helpers::my_min((vR  - csR) / (1 - vR  * csR), (vL - csL) / (1 - vL * csL));
        const real aR = helpers::my_max((vL + csL) / (1 + vL * csL), (vR  + csR) / (1 + vR  * csR));

        return Eigenvals(aL, aR);
    }
    }  
};

// Adapt the cfl conditonal timestep
void SRHD::adapt_dt()
{   
    real min_dt = INFINITY;
    #pragma omp parallel 
    {   
        // Compute the minimum timestep given cfl
        #pragma omp for schedule(static) reduction(min:min_dt)
        for (luint ii = 0; ii < active_zones; ii++)
        {
            const real rho    = prims[ii + idx_active].rho;
            const real p      = prims[ii + idx_active].p;
            const real v      = prims[ii + idx_active].v;
            const real h      = static_cast<real>(1.0) + gamma * p / (rho * (gamma - 1));
            const real cs     = std::sqrt(gamma * p / (rho * h));
            const real vPLus  = (v + cs) / (1 + v * cs);
            const real vMinus = (v - cs) / (1 - v * cs);

            const real x1l    = get_xface(ii, geometry, 0);
            const real x1r    = get_xface(ii, geometry, 1);
            const real dx1    = x1r - x1l;
            const real vfaceL = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
            const real vfaceR = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
            const real cfl_dt = dx1 / (helpers::my_max(std::abs(vPLus - vfaceR), std::abs(vMinus - vfaceL)));
            min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
        }
    }   
    dt = cfl * min_dt;
};

void SRHD::adapt_dt(SRHD *dev, luint blockSize)
{   
    #if GPU_CODE
        compute_dt<Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev, prims.data(), dt_min.data());
        deviceReduceKernel<1><<<blockSize, BLOCK_SIZE>>>(dev, dt_min.data(), active_zones);
        deviceReduceKernel<1><<<1,1024>>>(dev, dt_min.data(), blockSize);
        simbi::gpu::api::deviceSynch();
        this->dt = dev->dt;
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
    const real v   = prim.v;
    const real pre = prim.p;  
    const real h   = static_cast<real>(1.0) + gamma * pre / (rho * (gamma - 1));
    const real W   = static_cast<real>(1.0) / std::sqrt(1 - v * v);

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
    const real pre = prim.p;
    const real v   = prim.v;
    const real W   = static_cast<real>(1.0) / std::sqrt(1 - v * v);
    const real h   = static_cast<real>(1.0) + gamma * pre / (rho * (gamma - 1));
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
    const real aR  = lambda.aR;
    const real aL  = lambda.aL;
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

    const real e  = hll_state.tau + hll_state.d;
    const real s  = hll_state.s;
    const real fs = hll_flux.s;
    const real fe = hll_flux.tau + hll_flux.d;
    
    const real a     = fe;
    const real b     = - (e + fs);
    const real c     = s;
    const real disc  = std::sqrt(b*b - 4.0*a*c);
    const real quad  = -static_cast<real>(0.5)*(b + helpers::sgn(b)*disc);
    const real aStar = c/quad;
    const real pStar = -fe * aStar + fs;

    
    if (vface <= aStar) {
        const real pressure = left_prims.p;
        const real D        = left_state.d;
        const real S        = left_state.s;
        const real tau      = left_state.tau;
        const real E        = tau + D;
        const real cofactor = static_cast<real>(1.0) / (aLm - aStar);

        //--------------Compute the L Star State----------
        const real v = left_prims.v;
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
        const real pressure  = right_prims.p;
        const real D         = right_state.d;
        const real S         = right_state.s;
        const real tau       = right_state.tau;
        const real E         = tau + D;
        const real cofactor  = static_cast<real>(1.0) / (aRp - aStar);

        //--------------Compute the R Star State----------
        const real v = right_prims.v;
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
    real tstart,
    real tend,
    real dlogt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    int  chkpt_idx,
    std::string data_directory,
    std::string boundary_condition,
    bool first_order,
    bool linspace,
    bool hllc,
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
    this->init_chkpt_idx  = chkpt_idx;
    this->periodic        = boundary_condition == "periodic";
    this->first_order     = first_order;
    this->plm_theta       = plm_theta;
    this->linspace        = linspace;
    this->sourceD         = sources[0];
    this->sourceS         = sources[1];
    this->source0         = sources[2];
    this->hllc            = hllc;
    this->engine_duration = engine_duration;
    this->t               = tstart;
    this->tend            = tend;
    this->dlogt           = dlogt;
    this->bc              = helpers::boundary_cond_map.at(boundary_condition);
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->idx_active      = (periodic) ? 0  : (first_order) ? 1 : 2;
    this->active_zones    = (periodic) ? nx : (first_order) ? nx - 2 : nx - 4;
    this->dlogx1          = std::log10(x1[active_zones - 1]/ x1[0]) / (active_zones - 1);
    this->dx1             = (x1[active_zones - 1] - x1[0]) / (active_zones - 1);
    this->x1min           = x1[0];
    this->x1max           = x1[active_zones - 1];
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->total_zones     = nx;
    this->checkpoint_zones= active_zones;
    luint n = 0;
    
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / this->chkpt_interval;
    this->t_interval =
        t == 0 ? 0
               : dlogt !=0 ? tstart
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + this->chkpt_interval;

    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    setup.x1max          = x1[active_zones - 1];
    setup.x1min          = x1[0];
    setup.xactive_zones  = active_zones;
    setup.nx             = nx;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;
    setup.using_fourvelocity = false;
    setup.x1                 = x1;
    setup.regime = "relativistic";

    cons.resize(nx);
    prims.resize(nx);
    pressure_guess.resize(nx);
    dt_min.resize(active_zones);
    // Copy the state array into real & profile variables
    for (luint ii = 0; ii < nx; ii++) {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
        // initial pressure guess is | |S| - D - tau|
        pressure_guess[ii] = std::abs((state[1][ii]) - state[0][ii] - state[2][ii]);
    }

    // Copy the current SRHD instance over to the device
    SRHD *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(SRHD));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD));
    cons.copyToGpu();
    prims.copyToGpu();
    pressure_guess.copyToGpu();
    dt_min.copyToGpu();
    sourceD.copyToGpu();
    sourceS.copyToGpu();
    source0.copyToGpu();

    const auto fullP          = simbi::ExecutionPolicy(nx);
    const auto activeP        = simbi::ExecutionPolicy(active_zones);
    this->radius              = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->pseudo_radius       = (first_order) ? 1 : 2;
    const luint shBlockSize   = BLOCK_SIZE + 2 * pseudo_radius;
    
    Conserved *outer_zones     = nullptr;
    Conserved *dev_outer_zones = nullptr;
    if (d_outer)
    {
        outer_zones = new Conserved[2];
        const real dV  = get_cell_volume(active_zones - 1, geometry);
        outer_zones[0] = Conserved{d_outer(x1max), s_outer(x1max), e_outer(x1max)} * dV;
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::gpuMalloc(&dev_outer_zones, 2 * sizeof(Conserved));
            simbi::gpu::api::copyHostToDevice(dev_outer_zones, outer_zones, 2 * sizeof(Conserved));
        }
    }
    // Save initial condition
    if (t == 0) {
        write2file(*this, setup, data_directory, t, t_interval, chkpt_interval, active_zones);
        t_interval += chkpt_interval;
    }
    // Determine the memory side and state position
    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;
    const auto ozones  = (BuildPlatform == Platform::GPU) ? dev_outer_zones : outer_zones;

    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, activeP.gridSize.x);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    // Simulate :)
    while (t < tend & !inFailureState)
    {
        // Using a sigmoid decay function to represent when the source terms turn off.
        decay_constant = helpers::sigmoid(t, engine_duration);
        
        simbi::detail::with_logger(*this, [&](){
            advance(self, shBlockSize, radius, geometry, memside);
            cons2prim(fullP, device_self, memside);
            if (!periodic) {
                config_ghosts1D(fullP, cons.data(), nx, first_order, bc, ozones);
            }
        });

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(device_self, activeP.gridSize.x);
        } else {
            adapt_dt();
        }

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
            simbi::gpu::api::gpuFree(dev_outer_zones);
            delete[] outer_zones;
        } else {
            delete[] outer_zones;
        }
    }

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};