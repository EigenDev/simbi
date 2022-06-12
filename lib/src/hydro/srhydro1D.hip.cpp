/*
 * C++ Library to perform extensive hydro calculations
 * to be later wrapped and plotted in Python
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "srhydro1D.hip.hpp"
#include "helpers.hip.hpp"
#include "util/device_api.hpp"
#include "util/dual.hpp"
#include "util/printb.hpp"
#include "common/helpers.hpp"
#include "util/parallel_for.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <math.h>
#include <memory>

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;

//================================================
//              DATA STRUCTURES
//================================================
typedef sr1d::Conserved Conserved;
typedef sr1d::Primitive Primitive;
typedef sr1d::Eigenvals Eigenvals;

// Default Constructor
SRHD::SRHD(){}

// Overloaded Constructor
SRHD::SRHD(std::vector<std::vector<real>> u_state, real gamma, real cfl,
           std::vector<real> x1, std::string coord_system = "cartesian") :

    inFailureState(false),
    state(u_state),
    gamma(gamma),
    x1(x1),
    coord_system(coord_system),
    cfl(cfl)
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
real SRHD::calc_vface(const lint ii, const real hubble_const, const simbi::Geometry geometry, const int side)
{
    switch (geometry)
    {
    case simbi::Geometry::SPHERICAL:
        {
            const real xl = (ii > 0 ) ? x1min * pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min; 
            const real xr = (ii < active_zones - 1) ? xl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;

            return xl * kronecker(0, side) * hubble_const + xr * kronecker(1, side) * hubble_const;
        }
    case simbi::Geometry::CARTESIAN:
        {
            // Rigid motion
            return  hubble_const;
        }
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

    const bool mesh_motion      = (hubble_param != 0);
    const unsigned shBlockBytes = sh_block_size * sizeof(Primitive);
    auto p                      = simbi::ExecutionPolicy(nx);
    p.blockSize                 = BLOCK_SIZE;
    p.sharedMemBytes            = shBlockBytes;

    real x1max                      = this->x1max;
    real x1min                      = this->x1min;
    const real dx1                  = this->dx1;
    const real hubble_param         = this->hubble_param;
    const real dlogx1               = this->dlogx1;
    const real xpg                  = this->active_zones;
    const real dt                   = this->dt;
    const real plm_theta            = this->plm_theta;
    const auto nx                   = this->nx;
    const lint bx                   = (BuildPlatform == Platform::GPU) ? sh_block_size : nx;
    const real decay_constant       = this->decay_constant;
    const CLattice1D *coord_lattice = &(self->coord_lattice);
    const lint  pseudo_radius       = (first_order) ? 1 : 2;

    const auto step                 = (self->first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
    simbi::parallel_for(p, (luint)0, active_zones, [=] GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = &prims[0];
        #endif 

        Conserved u_l, u_r;
        Conserved f_l, f_r, frf, flf;
        Primitive prims_l, prims_r;

        lint ia = ii + radius;
        lint txa = (BuildPlatform == Platform::GPU) ?  threadIdx.x + pseudo_radius : ia;
        #if GPU_CODE
            luint txl = BLOCK_SIZE;
            // Check if the active index exceeds the active zones
            // if it does, then this thread buffer will taken on the
            // ghost index at the very end and return
            prim_buff[txa] = self->gpu_prims[ia];
            if (threadIdx.x < pseudo_radius)
            {
                if (ia + BLOCK_SIZE > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[txa - pseudo_radius] = self->gpu_prims[mod(ia - pseudo_radius, nx)];
                prim_buff[txa + txl   ]        = self->gpu_prims[(ia + txl ) % nx];
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = self->get_xface(ii, geometry, 0);
        const real x1r    = self->get_xface(ii, geometry, 1);
        const real vfaceR = x1r * hubble_param;
        const real vfaceL = x1l * hubble_param;
        if (self->first_order)
        {

            // Set up the left and right state interfaces for i+1/2
            prims_l = prim_buff[(txa + 0) % bx];
            prims_r = prim_buff[(txa + 1) % bx];
            u_l     = self->prims2cons(prims_l);
            u_r     = self->prims2cons(prims_r);
            f_l     = self->prims2flux(prims_l);
            f_r     = self->prims2flux(prims_r);

            // Calc HLL Flux at i+1/2 interface
            if (self->hllc)
            {
                frf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceR);
            }
            else
            {
                frf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceR);
            }

            // Set up the left and right state interfaces for i-1/2
            prims_l = prim_buff[mod(txa - 1, bx)];
            prims_r = prim_buff[(txa - 0) % bx];

            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            // Calc HLL Flux at i-1/2 interface
            if (self->hllc)
            {
                flf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceL);
            }
            else
            {
                flf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceL);
            }   
        } else {
            const Primitive left_most  = prim_buff[mod(txa - 2, bx)];
            const Primitive left_mid   = prim_buff[mod(txa - 1, bx)];
            const Primitive center     = prim_buff[(txa + 0) % bx];
            const Primitive right_mid  = prim_buff[(txa + 1) % bx];
            const Primitive right_most = prim_buff[(txa + 2) % bx];

            // Compute the reconstructed primitives at the i+1/2 interface
            // Reconstructed left primitives vector
            prims_l = center    + minmod((center - left_mid)*plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*plm_theta)*static_cast<real>(0.5); 
            prims_r = right_mid - minmod((right_mid - center)*plm_theta, (right_most - center)*static_cast<real>(0.5), (right_most - right_mid)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM primitives
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            if (self->hllc) {
                frf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceR);
            } else {
                frf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceR);
            }
            
            // Do the same thing, but for the right side interface [i - 1/2]
            prims_l = left_mid + minmod((left_mid - left_most)*plm_theta, (center - left_most)*static_cast<real>(0.5), (center - left_mid)*plm_theta)*static_cast<real>(0.5);
            prims_r = center   - minmod((center - left_mid)*plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            if (self->hllc) 
            {
                flf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceL);
            } else {
                flf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r, vfaceL);
            }
        }

        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                #if GPU_CODE
                    self->gpu_cons[ia] -= ((frf - flf) / dx1) * dt * step;
                #else 
                    cons[ia] -= ((frf - flf)  / dx1) * dt * step;
                #endif 
                break;
            case simbi::Geometry::SPHERICAL:
            {
                const real rlf    = x1l + vfaceL * step * dt; 
                const real rrf    = x1r + vfaceR * step * dt;
                const real rmean  = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                const real sR     = rrf * rrf; 
                const real sL     = rlf * rlf; 
                const real dV     = rmean * rmean * (rrf - rlf);    
                const real factor = (mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;

                #if GPU_CODE
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{self->gpu_sourceD[ii], self->gpu_sourceS[ii],self->gpu_source0[ii]} * decay_constant;
                    self->gpu_cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                #else 
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{sourceD[ii], sourceS[ii],source0[ii]} * decay_constant;
                    cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                #endif 
                
                break;
            }
                
            case simbi::Geometry::CYLINDRICAL:
            {
                const real rl           = (ii > 0 ) ? x1min * pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min;
                const real rlf          = rl + vfaceL * step * dt; 
                const real rr           = (ii < xpg - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                const real rrf          = rr + vfaceR * step * dt;
                const real rmean        = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                const real sR           = rrf; 
                const real sL           = rlf; 
                const real dV           = rmean * (rrf - rlf);             
                const real pc           = prim_buff[txa].p;
                
                #if GPU_CODE
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{self->gpu_sourceD[ii], self->gpu_sourceS[ii],self->gpu_source0[ii]} * decay_constant;
                    self->gpu_cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt;
                #else 
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{sourceD[ii], sourceS[ii],source0[ii]} * decay_constant;
                    cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt;
                #endif 
                
                break;
            }
        } // end switch
    });	

    // shift the grid max and mins
    const real x1l    = self->get_xface(0, geometry, 0);
    const real x1r    = self->get_xface(active_zones, geometry, 1);
    const real vfaceR = x1r * hubble_param;
    const real vfaceL = x1l * hubble_param;

    self->x1min += step * dt * vfaceL;
    self->x1max += step * dt * vfaceR;
    #if GPU_CODE
    this->x1max = self->x1max;
    this->x1min = self->x1min;
    #endif
}

void SRHD::cons2prim(ExecutionPolicy<> p, SRHD *dev, simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    const bool mesh_motion = (hubble_param != 0);
    const real step        = (first_order) ? 1.0 : 0.5;
    const real radius      = (first_order) ? 1 : 2;
    #if GPU_CODE
    const auto active_zones = this->active_zones;
    const auto dt           = this->dt;
    const auto hubble_param = this->hubble_param;
    const auto geometry     = this->geometry;
    const auto gamma        = this->gamma;
    #endif
    simbi::parallel_for(p, (luint)0, nx, [=] GPU_LAMBDA (luint ii){
        #if GPU_CODE
        __shared__ Conserved  conserved_buff[BLOCK_SIZE];
        #else 
        auto* const conserved_buff = &cons[0];
        #endif 
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
            if (mesh_motion && (geometry == simbi::Geometry::SPHERICAL))
            {
                lint idx = (ii - radius > 0) * (ii - radius);
                if (ii > active_zones + 1) {
                    idx = active_zones - 1;
                }
                const real xl    = self->get_xface(idx, geometry, 0);
                const real xr    = self->get_xface(idx, geometry, 1);
                const real xlf   = xl * (1.0 + step * dt * hubble_param);
                const real xrf   = xr * (1.0 + step * dt * hubble_param);
                const real xmean = static_cast<real>(0.75) * (xrf * xrf * xrf * xrf - xlf * xlf * xlf * xlf) / (xrf * xrf * xrf - xlf * xlf * xlf);
                invdV            = static_cast<real>(1.0) / (xmean * xmean * (xrf - xlf));
            }
            // Compile time thread selection
            #if GPU_CODE
                conserved_buff[tx] = self->gpu_cons[ii];  
                peq = self->gpu_pressure_guess[ii];  
            #else
                peq  = self->pressure_guess[ii];
            #endif

            const real D   = conserved_buff[tx].d   * invdV;
            const real S   = conserved_buff[tx].s   * invdV;
            const real tau = conserved_buff[tx].tau * invdV;
            
            // if (ii == 0)
            // {
            //     writeln("D: {}, V: {}", D, 1.0 / invdV);
            //     pause_program();
            // }
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
                c2 = gamma *pre / (h * rho); 

                g = c2 * v2 - static_cast<real>(1.0);
                f = (gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                if (iter >= MAX_ITER)
                {
                    printf("\nCons2Prim cannot converge\n");
                    printf("Density: %.3e, Pressure: %.3e, vsq: %.3e, coord: %lu\n", rho, peq, v2, ii);
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
            // real emin         = u * u / (1.0 + u * u) / pow(mach_ceiling, 2.0);

            // if (e < emin) {
            //     // printf("peq: %f, npew: %f\n", rho * emin * (self->gamma - 1.0));
            //     peq = rho * emin * (self->gamma - 1.0);
            // }
            #if GPU_CODE
                self->gpu_pressure_guess[ii] = peq;
                self->gpu_prims[ii]          = Primitive{D / W, v, peq};
            #else
                pressure_guess[ii] = peq;
                prims[ii]  = Primitive{D / W, v, peq};
            #endif
            workLeftToDo = false;
        }
    });
    if constexpr(BuildPlatform == Platform::GPU) {
        this->inFailureState = self->inFailureState;
    }
    
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD::calc_eigenvals(const Primitive &prims_l,
                               const Primitive &prims_r)
{
    // Compute L/R Sound Speeds
    const real rho_l = prims_l.rho;
    const real p_l   = prims_l.p;
    const real v_l   = prims_l.v;
    const real h_l   = static_cast<real>(1.0) + gamma * p_l / (rho_l * (gamma - 1));
    const real cs_l  = std::sqrt(gamma * p_l / (rho_l * h_l));

    const real rho_r = prims_r.rho;
    const real p_r   = prims_r.p;
    const real v_r   = prims_r.v;
    const real h_r   = static_cast<real>(1.0) + gamma * p_r / (rho_r * (gamma - 1));
    const real cs_r  = std::sqrt(gamma * p_r / (rho_r * h_r));

    switch (comp_wave_speed)
    {
    case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
        {
            // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
            const real vbar = static_cast<real>(0.5) * (v_l + v_r);
            const real cbar = static_cast<real>(0.5) * (cs_r + cs_l);

            const real bR   = (vbar + cbar) / (1 + vbar * cbar);
            const real bL   = (vbar - cbar) / (1 - vbar * cbar);

            const real aL = my_min(bL, (v_l - cs_l) / (1 - v_l * cs_l));
            const real aR = my_max(bR, (v_r + cs_r) / (1 + v_r * cs_r));

            return Eigenvals(aL, aR);
        }
    case simbi::WaveSpeeds::MIGNONE_AND_BODO_05:
        {
            // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
            const real sL = cs_l*cs_l/(gamma*gamma*(static_cast<real>(1.0) - cs_l*cs_l));
            const real sR = cs_r*cs_r/(gamma*gamma*(static_cast<real>(1.0) - cs_r*cs_r));
            // Define temporaries to save computational cycles
            const real qfL   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sL);
            const real qfR   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sR);
            const real sqrtR = std::sqrt(sR * (static_cast<real>(1.0) - v_r * v_r + sR));
            const real sqrtL = std::sqrt(sL * (static_cast<real>(1.0) - v_l * v_l + sL));

            const real lamLm = (v_l - sqrtL) * qfL;
            const real lamRm = (v_r - sqrtR) * qfR;
            const real lamLp = (v_l + sqrtL) * qfL;
            const real lamRp = (v_r + sqrtR) * qfR;

            real aL = lamLm < lamRm ? lamLm : lamRm;
            real aR = lamLp > lamRp ? lamLp : lamRp;

            // Smoothen for rarefaction fan
            aL = my_min(aL, (v_l - cs_l) / (1 - v_l * cs_l));
            aR = my_max(aR, (v_r + cs_r) / (1 + v_r * cs_r));

            return Eigenvals(aL, aR);
        }
    case simbi::WaveSpeeds::NAIVE:
    {
        const real aL = my_min((v_r - cs_r) / (1 - v_r * cs_r), (v_l - cs_l) / (1 - v_l * cs_l));
        const real aR = my_max((v_l + cs_l) / (1 + v_l * cs_l), (v_r + cs_r) / (1 + v_r * cs_r));

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
        real dr, cs, cfl_dt, vfaceL, vfaceR, vzone, x1l, x1r;
        real h, rho, p, v, vPLus, vMinus;

        // Compute the minimum timestep given cfl
        #pragma omp for schedule(static) reduction(min:min_dt)
        for (luint ii = 0; ii < active_zones; ii++)
        {
            x1l    = get_xface(ii, geometry, 0);
            x1r    = get_xface(ii, geometry, 1);
            vfaceL = x1l * hubble_param;
            vfaceR = x1r * hubble_param;
            vzone  = 0.5 * (vfaceL + vfaceR);
            dr     = coord_lattice.dx1[ii];
            rho    = prims[ii + idx_active].rho;
            p      = prims[ii + idx_active].p;
            v      = prims[ii + idx_active].v;

            h = static_cast<real>(1.0) + gamma * p / (rho * (gamma - 1));
            cs = std::sqrt(gamma * p / (rho * h));

            vPLus  = (v + cs) / (1 + v * cs);
            vMinus = (v - cs) / (1 - v * cs);

            cfl_dt = dr / (my_max(std::abs(vPLus - vzone), std::abs(vMinus - vzone)));
            min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
        }
    }   

    dt = cfl * min_dt;
};

void SRHD::adapt_dt(SRHD *dev, luint blockSize)
{   
    #if GPU_CODE
        compute_dt<SRHD, Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        dtWarpReduce<SRHD, Primitive, 4><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
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
Conserved SRHD::prims2cons(const Primitive &prim)
{
    const real rho = prim.rho;
    const real v   = prim.v;
    const real pre = prim.p;  
    const real h   = static_cast<real>(1.0) + gamma * pre / (rho * (gamma - 1));
    const real W   = static_cast<real>(1.0) / std::sqrt(1 - v * v);

    return Conserved{rho * W, rho * h * W * W * v, rho * h * W * W - pre - rho * W};
};

GPU_CALLABLE_MEMBER
Conserved SRHD::calc_hll_state(const Conserved &left_state,
                               const Conserved &right_state,
                               const Conserved &left_flux,
                               const Conserved &right_flux,
                               const Primitive &left_prims,
                               const Primitive &right_prims)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);
    const real aL          = lambda.aL;
    const real aR          = lambda.aR;
    const real aLm         = aL < 0 ? aL : 0;
    const real aRp         = aR > 0 ? aR : 0;
    
    return (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);
}

Conserved SRHD::calc_intermed_state(
    const Primitive &prims,
    const Conserved &state, 
    const real a,
    const real aStar, 
    const real pStar)
{
    const real pressure = prims.p;
    const real v = prims.v;

    const real D = state.d;
    const real S = state.s;
    const real tau = state.tau;
    const real E = tau + D;

    const real DStar   = ((a - v)   / (a - aStar)) * D;
    const real Sstar   = (static_cast<real>(1.0) / (a - aStar)) * (S * (a - v) - pressure + pStar);
    const real Estar   = (static_cast<real>(1.0) / (a - aStar)) * (E * (a - v) + pStar * aStar - pressure * v);
    const real tauStar = Estar - DStar;

    return Conserved{DStar, Sstar, tauStar};
}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
GPU_CALLABLE_MEMBER
Conserved SRHD::prims2flux(const Primitive &prim)
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
    const real      vface)
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
    const real       vface)
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
    const real quad  = -static_cast<real>(0.5)*(b + sgn(b)*disc);
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
    real init_dt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
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
    this->periodic    = boundary_condition == "periodic";
    this->first_order = first_order;
    this->plm_theta   = plm_theta;
    this->linspace    = linspace;
    this->sourceD     = sources[0];
    this->sourceS     = sources[1];
    this->source0     = sources[2];
    this->hllc        = hllc;
    this->engine_duration = engine_duration;
    this->t           = tstart;
    this->tend        = tend;
    // Define the swap vector for the integrated state
    this->nx         = state[0].size();
    this->bc         = boundary_cond_map.at(boundary_condition);
    this->geometry   = geometry_map.at(coord_system);

    this->idx_active   = (periodic) ? 0  : (first_order) ? 1 : 2;
    this->active_zones = (periodic) ? nx : (first_order) ? nx - 2 : nx - 4;


    this->dlogx1  = std::log10(x1[active_zones - 1]/ x1[0]) / (active_zones - 1);
    this->dx1     = (x1[active_zones - 1] - x1[0]) / (active_zones - 1);
    this->x1min   = x1[0];
    this->x1max   = x1[active_zones - 1];

    n = 0;
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / chkpt_interval;
    real t_interval =
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    DataWriteMembers setup;
    setup.x1max          = x1[active_zones - 1];
    setup.x1min          = x1[0];
    setup.xactive_zones  = active_zones;
    setup.nx             = nx;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;

    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    sr1d::PrimitiveArray transfer_prims;

    cons.resize(nx);
    prims.resize(nx);
    pressure_guess.resize(nx);
    dt_arr.resize(nx);
    // Copy the state array into real & profile variables
    for (luint ii = 0; ii < nx; ii++)
    {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
        // initial pressure guess is | |S| - D - tau|
        pressure_guess[ii] = std::abs((state[1][ii]) - state[0][ii] - state[2][ii]);
    }
    // deallocate the init state vector now
    std::vector<int> state;

    if ((coord_system == "spherical") && (linspace))
    {
        this->geometry = simbi::Geometry::SPHERICAL;
        this->coord_lattice = CLattice1D(x1, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->geometry = simbi::Geometry::SPHERICAL;
        this->coord_lattice = CLattice1D(x1, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE);
    }
    else
    {
        this->geometry = simbi::Geometry::CARTESIAN;
        this->coord_lattice = CLattice1D(x1, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }

    // Copy the current SRHD instance over to the device
    SRHD *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD));
    simbi::dual::DualSpace1D<Primitive, Conserved, SRHD> dualMem;
    dualMem.copyHostToDev(*this, device_self);

    const auto fullP          = simbi::ExecutionPolicy(nx);
    const auto activeP        = simbi::ExecutionPolicy(active_zones);
    const luint radius        = (periodic) ? 0 : (first_order) ? 1 : 2;
    const luint pseudo_radius = (first_order) ? 1 : 2;
    const luint shBlockSize   = BLOCK_SIZE + 2 * pseudo_radius;
    const luint shBlockBytes  = shBlockSize * sizeof(Primitive);

    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, activeP.gridSize.x);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }
    // Some variables to handle file automatic file string formatting 
    tchunk = "000000";
    lint tchunk_order_of_mag = 2;
    lint time_order_of_mag;
    double tbefore;

    // Some benchmarking tools 
    luint   nfold   = 0;
    luint   ncheck  = 0;
    real     zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<real> delta_t;

    // Determine the memory side and state position
    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;
    // Simulate :)
    
    Conserved * outer_zones = nullptr;
    if (d_outer)
    {
        #if GPU_CODE
        simbi::gpu::api::gpuMalloc(&outer_zones, 2 * sizeof(Conserved));
        #else
        outer_zones = new Conserved[2];
        #endif
        const real dV  = get_cell_volume(active_zones - 1, geometry);
        outer_zones[0] = Conserved{d_outer(x1max), s_outer(x1max), e_outer(x1max)} * dV;
    }

    // Save initial condition
    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU) 
            dualMem.copyDevToHost(device_self, *this);

        setup.x1max = x1max;
        setup.x1min = x1min;
        time_order_of_mag = std::floor(std::log10(t));
        if (time_order_of_mag > tchunk_order_of_mag)
        {
            tchunk.insert(0, "0");
            tchunk_order_of_mag += 1;
        }
        transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
        writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
        tnow = create_step_str(t_interval, tchunk);
        filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
        setup.t  = t;
        setup.dt = t - tbefore;
        tbefore  = t;
        write_hdf5(data_directory, filename, prods, setup, 1, nx);
        t_interval += chkpt_interval;
    }

    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            
            advance(self, shBlockSize, radius, geometry, memside);
            cons2prim(fullP, device_self, memside);
            if (!periodic) {
                config_ghosts1D(fullP, self, nx, true, bc, outer_zones);
            }
            t += dt; 
            
            if (n >= nfold){
                ncheck += 1;
                simbi::gpu::api::deviceSynch();
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                writefl("Iteration: {>08} \t dt: {>08} \t Time: {>08} \t Zones/sec: {>08} \t\r", n, dt, t, nx/delta_t.count());
                nfold += 100;
            }

            /* Write to a file every nth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU) 
                    dualMem.copyDevToHost(device_self, *this);

                setup.x1max = x1max;
                setup.x1min = x1min;
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag) {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
                writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                tnow      = create_step_str(t_interval, tchunk);
                filename  = string_format("%lu.chkpt." + tnow + ".h5", active_zones);
                setup.t   = t;
                setup.dt  = t - tbefore;
                tbefore   = t;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, activeP.gridSize.x);
            } else {
                adapt_dt();
            }

            // Update the outer zones with the necessary configs if they exists
            if (d_outer)
            {
                const real dV  = get_cell_volume(active_zones - 1, geometry);
                outer_zones[0] = Conserved{d_outer(x1max), s_outer(x1max), e_outer(x1max)} * dV;
            }

            hubble_param = adot(t) / a(t);
            if (inFailureState)
                simbi::gpu::api::deviceSynch();
        }
    } else {
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();

            // First Half Step
            cons2prim(fullP, self, memside);
            advance(self, shBlockSize, radius, geometry, memside);
            if (!periodic) {
                config_ghosts1D(fullP, self, nx, false, bc, outer_zones);
            }
            // Final Half Step
            cons2prim(fullP, self, memside);
            advance(self, shBlockSize, radius, geometry, memside);
            if (!periodic) {
                config_ghosts1D(fullP, self, nx, false, bc, outer_zones);
            }

            t += dt; 
            if (n >= nfold) {
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                writefl("Iteration: {>08} \t dt: {>08} \t Time: {>08} \t Zones/sec: {>08} \t\r", n, dt, t, nx/delta_t.count());
                nfold += 100;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_interval && t != INFINITY) {
                if constexpr(BuildPlatform == Platform::GPU) 
                    dualMem.copyDevToHost(device_self, *this);

                setup.x1max = x1max;
                setup.x1min = x1min;
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
                writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t  = t;
                setup.dt = t - tbefore;
                tbefore  = t;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));

            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU) {
                adapt_dt(device_self, activeP.gridSize.x);
            } else {
                adapt_dt();
            }

            // Update the outer zones with the necessary configs if they exists
            if (d_outer) {
                const real dV  = get_cell_volume(active_zones - 1, geometry);
                outer_zones[0] = Conserved{d_outer(x1max), s_outer(x1max), e_outer(x1max)} * dV;
            }

            hubble_param = adot(t) / a(t);
            if (inFailureState)
                simbi::gpu::api::deviceSynch();
            
        }

    }
    writeln("Average zone update/sec for:{>5} iterations was {>5} zones/sec", n, zu_avg/ncheck);

    if constexpr (BuildPlatform == Platform::GPU)
    {
        dualMem.copyDevToHost(device_self, *this);
        simbi::gpu::api::gpuFree(device_self);
    } 

    if (outer_zones) {
        if constexpr(BuildPlatform == Platform::GPU) {
            simbi::gpu::api::gpuFree(outer_zones);
        } else {
            delete[] outer_zones;
        }
    }

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (luint ii = 0; ii < nx; ii++)
    {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};