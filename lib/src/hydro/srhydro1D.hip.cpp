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
#include "common/helpers.hpp"
#include "util/parallel_for.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
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
SRHD::SRHD(std::vector<std::vector<real>> u_state, real gamma, real CFL,
           std::vector<real> r, std::string coord_system = "cartesian")
{
    this->state = u_state;
    this->gamma = gamma;
    this->r = r;
    this->coord_system = coord_system;
    this->CFL = CFL;
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


//=====================================================================
//                          KERNEL CALLS
//=====================================================================
void SRHD::advance(
    SRHD *dev,  
    const int sh_block_size,
    const int radius, 
    const simbi::Geometry geometry,
    const simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    const unsigned shBlockBytes = sh_block_size * sizeof(Primitive);
    auto p = simbi::ExecutionPolicy(nx);
    p.blockSize      = BLOCK_SIZE;
    p.sharedMemBytes = shBlockBytes;
    
    const real dt                   = this->dt;
    const real plm_theta            = this->plm_theta;
    const int nx                    = this->nx;
    const real decay_constant       = this->decay_constant;
    const CLattice1D *coord_lattice = &(self->coord_lattice);

    simbi::parallel_for(p, 0, active_zones, [=] GPU_LAMBDA (int ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = &prims[0];
        #endif 
        
        Conserved u_l, u_r;
        Conserved f_l, f_r, f1, f2;
        Primitive prims_l, prims_r;
        real rmean, dV, sL, sR, pc, dx;
        
        auto ia = ii + radius;
        auto txa = (BuildPlatform == Platform::GPU) ?  threadIdx.x + radius : ia;
        if(BuildPlatform == Platform::GPU)
        {
            int txl = BLOCK_SIZE;
            // Check if the active index exceeds the active zones
            // if it does, then this thread buffer will taken on the
            // ghost index at the very end and return
            prim_buff[txa] = self->gpu_prims[ia];
            if (threadIdx.x < radius)
            {
                if (ia + BLOCK_SIZE > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[txa - radius] = self->gpu_prims[ia - radius];
                prim_buff[txa + txl   ] = self->gpu_prims[ia + txl   ];
            }
            __syncthreads();
        } 

        if (self->first_order)
        {
            if (self->periodic)
            {
                // Set up the left and right state interfaces for i+1/2
                prims_l = prim_buff[txa];
                prims_r = roll(prim_buff, txa + 1, sh_block_size);
            }
            else
            {
                // Set up the left and right state interfaces for i+1/2
                prims_l = prim_buff[txa];
                prims_r = prim_buff[txa + 1];
            }
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            // Calc HLL Flux at i+1/2 interface
            if (self->hllc)
            {
                f1 = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f1 = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            // Set up the left and right state interfaces for i-1/2
            if (self->periodic)
            {
                prims_l = roll(prim_buff, txa - 1, sh_block_size);
                prims_r = prim_buff[txa];
            }
            else
            {
                prims_l = prim_buff[txa - 1];
                prims_r = prim_buff[txa];
            }

            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);

            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            // Calc HLL Flux at i-1/2 interface
            if (self->hllc)
            {
                f2 = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f2 = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                if constexpr(BuildPlatform == Platform::GPU)
                {
                    dx = coord_lattice->gpu_dx1[ii];
                    self->gpu_cons[ia].D   += dt * ( -(f1.D - f2.D)     / dx +  self->gpu_sourceD[ii] );
                    self->gpu_cons[ia].S   += dt * ( -(f1.S - f2.S)     / dx +  self->gpu_sourceS[ii] );
                    self->gpu_cons[ia].tau += dt * ( -(f1.tau - f2.tau) / dx  + self->gpu_source0[ii] );
                } else {
                    dx = self->coord_lattice.dx1[ii];
                    cons[ia].D   += dt * ( -(f1.D - f2.D)     / dx +  sourceD[ii] );
                    cons[ia].S   += dt * ( -(f1.S - f2.S)     / dx +  sourceS[ii] );
                    cons[ia].tau += dt * ( -(f1.tau - f2.tau) / dx  + source0[ii] );
                }
                
                break;  
            
            case simbi::Geometry::SPHERICAL:
                if constexpr(BuildPlatform == Platform::GPU)
                {
                    pc    = prim_buff[txa].p;
                    sL    = coord_lattice->gpu_face_areas[ii + 0];
                    sR    = coord_lattice->gpu_face_areas[ii + 1];
                    dV    = coord_lattice->gpu_dV[ii];
                    rmean = coord_lattice->gpu_x1mean[ii];

                    self->gpu_cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        self->gpu_sourceD[ii] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + (real)2.0 * pc / rmean +
                        self->gpu_sourceS[ii] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        self->gpu_source0[ii] * decay_constant
                    } * dt;
                } else {
                    pc    = prim_buff[txa].p;
                    sL    = self->coord_lattice.face_areas[ii + 0];
                    sR    = self->coord_lattice.face_areas[ii + 1];
                    dV    = self->coord_lattice.dV[ii];
                    rmean = self->coord_lattice.x1mean[ii];

                    self->cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        self->sourceD[ii] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + (real)2.0 * pc / rmean +
                        self->sourceS[ii] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        self->source0[ii] * decay_constant
                    } * dt;
                }
                break;
            }
                
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            // if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (self->periodic)
                {
                    left_most  = roll(prim_buff, txa - 2, sh_block_size);
                    left_mid   = roll(prim_buff, txa - 1, sh_block_size);
                    center     = prim_buff[txa];
                    right_mid  = roll(prim_buff, txa + 1, sh_block_size);
                    right_most = roll(prim_buff, txa + 2, sh_block_size);
                }
                else
                {
                    left_most  = prim_buff[txa - 2];
                    left_mid   = prim_buff[txa - 1];
                    center     = prim_buff[txa];
                    right_mid  = prim_buff[txa + 1];
                    right_most = prim_buff[txa + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                
                prims_l.rho =
                    center.rho + (real)0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                            (real)0.5 * (right_mid.rho - left_mid.rho),
                                            plm_theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + (real)0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    (real)0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_l.p = center.p + (real)0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    (real)0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                            (real)0.5 * minmod(plm_theta * (right_mid.rho - center.rho),
                                        (real)0.5 * (right_most.rho - center.rho),
                                        plm_theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - (real)0.5 * minmod(plm_theta * (right_mid.v - center.v),
                                            (real)0.5 * (right_most.v - center.v),
                                            plm_theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - (real)0.5 * minmod(plm_theta * (right_mid.p - center.p),
                                            (real)0.5 * (right_most.p - center.p),
                                            plm_theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = self->prims2cons(prims_l);
                u_r = self->prims2cons(prims_r);
                f_l = self->prims2flux(prims_l);
                f_r = self->prims2flux(prims_r);

                if (self->hllc)
                {
                    f1 = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                
                // printf("plm_theta: %f", plm_theta);
                // std::cin.get();
                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho =
                    left_mid.rho + (real)0.5 * minmod(plm_theta * (left_mid.rho - left_most.rho),
                                                (real)0.5 * (center.rho - left_most.rho),
                                                plm_theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + (real)0.5 * minmod(plm_theta * (left_mid.v - left_most.v),
                                            (real)0.5 * (center.v - left_most.v),
                                            plm_theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + (real)0.5 * minmod(plm_theta * (left_mid.p - left_most.p),
                                            (real)0.5 * (center.p - left_most.p),
                                            plm_theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - (real)0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                            (real)0.5 * (right_mid.rho - left_mid.rho),
                                            plm_theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - (real)0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    (real)0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_r.p = center.p - (real)0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    (real)0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = self->prims2cons(prims_l);
                u_r = self->prims2cons(prims_r);
                f_l = self->prims2flux(prims_l);
                f_r = self->prims2flux(prims_r);

                if (self->hllc)
                {
                    f2 = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    if constexpr(BuildPlatform == Platform::GPU)
                    {
                        dx = coord_lattice->gpu_dx1[ii];
                        self->gpu_cons[ia].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx +  self->gpu_sourceD[ii] );
                        self->gpu_cons[ia].S   += (real)0.5 * dt * ( -(f1.S - f2.S)     / dx +  self->gpu_sourceS[ii] );
                        self->gpu_cons[ia].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx  + self->gpu_source0[ii] );
                    } else {
                        dx = coord_lattice->dx1[ii];
                        cons[ia].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx +  sourceD[ii] );
                        cons[ia].S   += (real)0.5 * dt * ( -(f1.S - f2.S)     / dx +  sourceS[ii] );
                        cons[ia].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx  + source0[ii] );
                    }
                    
                    break;
                case simbi::Geometry::SPHERICAL:
                    if constexpr(BuildPlatform == Platform::GPU)
                    {
                        pc    = prim_buff[txa].p;
                        sL    = coord_lattice->gpu_face_areas[ii + 0];
                        sR    = coord_lattice->gpu_face_areas[ii + 1];
                        dV    = coord_lattice->gpu_dV[ii];
                        rmean = coord_lattice->gpu_x1mean[ii];

                        self->gpu_cons[ia] += Conserved{ 
                            -(sR * f1.D - sL * f2.D) / dV +
                            self->gpu_sourceD[ii] * decay_constant,

                            -(sR * f1.S - sL * f2.S) / dV + (real)2.0 * pc / rmean +
                            self->gpu_sourceS[ii] * decay_constant,

                            -(sR * f1.tau - sL * f2.tau) / dV +
                            self->gpu_source0[ii] * decay_constant
                        } * dt * (real)0.5;
                    } else {
                        pc    = prim_buff[txa].p;
                        sL    = coord_lattice->face_areas[ii + 0];
                        sR    = coord_lattice->face_areas[ii + 1];
                        dV    = coord_lattice->dV[ii];
                        rmean = coord_lattice->x1mean[ii];

                        cons[ia] += Conserved{ 
                            -(sR * f1.D - sL * f2.D) / dV +
                            self->sourceD[ii] * decay_constant,

                            -(sR * f1.S - sL * f2.S) / dV + (real)2.0 * pc / rmean +
                            self->sourceS[ii] * decay_constant,

                            -(sR * f1.tau - sL * f2.tau) / dV +
                            self->source0[ii] * decay_constant
                        } * dt * (real)0.5;
                    }
                    
                    break;
                }
            }
        }
    });	
}

void SRHD::cons2prim(ExecutionPolicy<> p, SRHD *dev, simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, 0, nx, [=] GPU_LAMBDA (int ii){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        #if GPU_CODE
        __shared__ Conserved  conserved_buff[BLOCK_SIZE];
        #else 
        auto* const conserved_buff = &cons[0];
        #endif 
        int tx = (BuildPlatform == Platform::GPU) ? threadIdx.x : ii;

        // Compile time thread selection
        if constexpr(BuildPlatform == Platform::GPU)
        {
            conserved_buff[tx] = self->gpu_cons[ii];
        } 

        const real D       = conserved_buff[tx].D;
        const real S       = conserved_buff[tx].S;
        const real tau     = conserved_buff[tx].tau;
        real peq           = (BuildPlatform == Platform::GPU) ? self->gpu_pressure_guess[ii] : self->pressure_guess[ii];
        int iter           = 0;
        const real tol     = D * tol_scale;
        do
        {
            pre = peq;
            et = tau + D + pre;
            v2 = S * S / (et * et);
            W = (real)1.0 / real_sqrt((real)1.0 - v2);
            rho = D / W;

            eps = (tau + ((real)1.0 - W) * D + ((real)1.0 - W * W) * pre) / (D * W);

            h  = (real)1.0 + eps + pre / rho;
            c2 = self->gamma * pre / (h * rho); 

            g = c2 * v2 - (real)1.0;
            f = (self->gamma - (real)1) * rho * eps - pre;

            peq = pre - f / g;
            if (iter >= MAX_ITER)
            {
                printf("\n");
                printf("Cons2Prim cannot converge");
                printf("\n");
                self->dt = INFINITY;
                return;
            }
            iter++;

        } while (std::abs(peq - pre) >= tol);

        real v = S / (tau + D + peq);

        if constexpr(BuildPlatform == Platform::GPU)
        {
            self->gpu_pressure_guess[ii] = peq;
            self->gpu_prims[ii]          = Primitive{D * real_sqrt(1 - v * v), v, peq};
        } else {
            pressure_guess[ii] = peq;
            prims[ii]  = Primitive{D * real_sqrt(1 - v * v), v, peq};
        }   
    });
}
//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */


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
    const real h_l   = (real)1.0 + gamma * p_l / (rho_l * (gamma - 1));
    const real cs_l  = real_sqrt(gamma * p_l / (rho_l * h_l));

    const real rho_r = prims_r.rho;
    const real p_r   = prims_r.p;
    const real v_r   = prims_r.v;
    const real h_r   = (real)1.0 + gamma * p_r / (rho_r * (gamma - 1));
    const real cs_r  = real_sqrt(gamma * p_r / (rho_r * h_r));

    // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
    const real vbar = (real)0.5 * (v_l + v_r);
    const real cbar = (real)0.5 * (cs_r + cs_l);
    const real br = (vbar + cbar) / (1 + vbar * cbar);
    const real bl = (vbar - cbar) / (1 - vbar * cbar);

    const real aL = my_min(bl, (v_l - cs_l) / (1 - v_l * cs_l));
    const real aR = my_max(br, (v_r + cs_r) / (1 + v_r * cs_r));


    // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
    // const real sL = cs_l*cs_l/(gamma*gamma*(1.0 - cs_l*cs_l));
    // const real sR = cs_r*cs_r/(gamma*gamma*(1.0 - cs_r*cs_r));
    // // Define temporaries to save computational cycles
    // const real qfL = (real)1.0 / ((real)1.0 + sL);
    // const real qfR = (real)1.0 / ((real)1.0 + sR);
    // const real sqrtR = real_sqrt(sR * (1.0 - v_r * v_r + sR));
    // const real sqrtL = real_sqrt(sL * (1.0 - v_l * v_l + sL));

    // const real lamLm = (v_l - sqrtL) * qfL;
    // const real lamRm = (v_r - sqrtR) * qfR;
    // const real lamLp = (v_l + sqrtL) * qfL;
    // const real lamRp = (v_r + sqrtR) * qfR;

    // const real aL = lamLm < lamRm ? lamLm : lamRm;
    // const real aR = lamLp > lamRp ? lamLp : lamRp;

    return Eigenvals(aL, aR);
};

// Adapt the CFL conditonal timestep
// Adapt the timestep on the Host
void SRHD::adapt_dt()
{   
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real dr, cs, cfl_dt;
        real h, rho, p, v, vPLus, vMinus;

        // Compute the minimum timestep given CFL
        #pragma omp for schedule(static)
        for (int ii = 0; ii < active_zones; ii++)
        {
            dr  = coord_lattice.dx1[ii];
            rho = prims[ii + idx_shift].rho;
            p   = prims[ii + idx_shift].p;
            v   = prims[ii + idx_shift].v;

            h = (real)1.0 + gamma * p / (rho * (gamma - 1));
            cs = real_sqrt(gamma * p / (rho * h));

            vPLus  = (v + cs) / (1 + v * cs);
            vMinus = (v - cs) / (1 - v * cs);

            cfl_dt = dr / (std::max(std::abs(vPLus), std::abs(vMinus)));

            min_dt = std::min(min_dt, cfl_dt);
        }
    }   

    dt = CFL * min_dt;
};

void SRHD::adapt_dt(SRHD *dev, int blockSize)
{   
    #if GPU_CODE
    {
        dtWarpReduce<SRHD, Primitive, 128><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        simbi::gpu::api::deviceSynch();
        simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
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
    const real h = (real)1.0 + gamma * pre / (rho * (gamma - 1));
    const real W = (real)1.0 / real_sqrt(1 - v * v);

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

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    return (right_state * aR - left_state * aL - right_flux + left_flux) / (aR - aL);
}

Conserved SRHD::calc_intermed_state(const Primitive &prims,
                                    const Conserved &state, const real a,
                                    const real aStar, const real pStar)
{
    const real pressure = prims.p;
    const real v = prims.v;

    const real D = state.D;
    const real S = state.S;
    const real tau = state.tau;
    const real E = tau + D;

    const real DStar = ((a - v) / (a - aStar)) * D;
    const real Sstar = ((real)1.0 / (a - aStar)) * (S * (a - v) - pressure + pStar);
    const real Estar = ((real)1.0 / (a - aStar)) * (E * (a - v) + pStar * aStar - pressure * v);
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

    const real W = (real)1.0 / real_sqrt(1 - v * v);
    const real h = (real)1.0 + gamma * pre / (rho * (gamma - 1));
    const real D = rho * W;
    const real S = rho * h * W * W * v;

    return Conserved{D*v, S*v + pre, S - D*v};
};

GPU_CALLABLE_MEMBER
Conserved
SRHD::calc_hll_flux(
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux,  
    const Conserved &right_flux)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    // Grab the necessary wave speeds
    const real aR  = lambda.aR;
    const real aL  = lambda.aL;
    const real aLm = aL < (real)0.0 ? aL : (real)0.0;
    const real aRp = aR > (real)0.0 ? aR : (real)0.0;

    // Compute the HLL Flux component-wise
    return (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aLm * aRp) / (aRp - aLm);
};

GPU_CALLABLE_MEMBER
Conserved
SRHD::calc_hllc_flux(
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux,  
    const Conserved &right_flux)
{
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);
    const real aL = lambda.aL;
    const real aR = lambda.aR;

    if ((real)0.0 <= aL)
    {
        return left_flux;
    }
    else if ((real)0.0 >= aR)
    {
        return right_flux;
    }

    const Conserved hll_flux = calc_hll_flux(left_prims, right_prims, left_state, right_state,
                             left_flux, right_flux);

    const Conserved hll_state = calc_hll_state(left_state, right_state, left_flux, right_flux,
                               left_prims, right_prims);

    const real e = hll_state.tau + hll_state.D;
    const real s = hll_state.S;
    const real fs = hll_flux.S;
    const real fe = hll_flux.tau + hll_flux.D;
    
    const real a = fe;
    const real b = - (e + fs);
    const real c = s;
    const real disc = real_sqrt( b*b - 4.0*a*c);
    const real quad = -(real)0.5*(b + sgn(b)*disc);
    const real aStar = c/quad;
    const real pStar = -fe * aStar + fs;

    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D        = left_state.D;
        const real S        = left_state.S;
        const real tau      = left_state.tau;
        const real E        = tau + D;
        const real cofactor = (real)1.0 / (aL - aStar);
        //--------------Compute the L Star State----------
        const real v = left_prims.v;
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aL - v) * D;
        const real Sstar    = cofactor * (S * (aL - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aL - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto interstate_left = Conserved{Dstar, Sstar, tauStar};

        //---------Compute the L Star Flux
        return left_flux + (interstate_left - left_state) * aL;
    }
    else
    {
        const real pressure  = right_prims.p;
        const real D         = right_state.D;
        const real S         = right_state.S;
        const real tau       = right_state.tau;
        const real E         = tau + D;
        const real cofactor  = (real)1.0 / (aR - aStar);
        //--------------Compute the L Star State----------
        const real v = right_prims.v;
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aR - v) * D;
        const real Sstar    = cofactor * (S * (aR - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aR - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto interstate_right = Conserved{Dstar, Sstar, tauStar};

        //---------Compute the R Star Flux
        return right_flux + (interstate_right - right_state) * aR;
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
    bool first_order,
    bool periodic,
    bool linspace,
    bool hllc)
{
    this->periodic = periodic;
    this->first_order = first_order;
    this->plm_theta = plm_theta;
    this->linspace = linspace;
    this->sourceD = sources[0];
    this->sourceS = sources[1];
    this->source0 = sources[2];
    this->hllc = hllc;
    this->engine_duration = engine_duration;
    this->t    = tstart;
    this->tend = tend;
    // Define the swap vector for the integrated state
    this->nx = state[0].size();

    if (periodic)
    {
        this->idx_shift = 0;
        this->i_start   = 0;
        this->i_bound   = nx;
    }
    else
    {
        if (first_order)
        {
            this->idx_shift  = 1;
            this->active_zones = nx - 2;
            this->i_start    = 1;
            this->i_bound    = nx - 1;
        }
        else
        {
            this->idx_shift  = 2;
            this->active_zones = nx - 4;
            this->i_start    = 2;
            this->i_bound    = nx - 2;
        }
    }
    config_system();
    n = 0;
    std::vector<Conserved> u, u1, udot, udot1;
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / chkpt_interval;
    real t_interval =
        t == 0 ? floor(tstart * round_place + (real)0.5) / round_place
               : floor(tstart * round_place + (real)0.5) / round_place + chkpt_interval;
    DataWriteMembers setup;
    setup.xmax = r[active_zones - 1];
    setup.xmin = r[0];
    setup.xactive_zones = active_zones;
    setup.nx = nx;
    setup.linspace = linspace;

    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    sr1d::PrimitiveArray transfer_prims;

    cons.resize(nx);
    prims.resize(nx);
    pressure_guess.resize(nx);
    dt_arr.resize(nx);
    // Copy the state array into real & profile variables
    for (int ii = 0; ii < nx; ii++)
    {

        cons[ii] = Conserved{state[0][ii],
                          state[1][ii],
                          state[2][ii]};

        // initial pressure guess is | |S| - D - tau|
        pressure_guess[ii] = std::abs((state[1][ii]) - state[0][ii] - state[2][ii]);
    }
    // deallocate the init state vector now
    std::vector<int> state;

    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE);
    }
    else
    {
        this->coord_lattice = CLattice1D(r, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
    }

    // Copy the current SRHD instance over to the device
    SRHD *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD));
    simbi::dual::DualSpace1D<Primitive, Conserved, SRHD> dualMem;
    dualMem.copyHostToDev(*this, device_self);

    // Config the system
    const int nBlocks          = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int physical_nBlocks = (active_zones + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dim3 fgridDim   = dim3(nBlocks);
    const dim3 agridDim   = dim3(physical_nBlocks);
    const dim3 threadDim  = dim3(BLOCK_SIZE);

    const auto fullP            = simbi::ExecutionPolicy(nx);
    const auto activeP          = simbi::ExecutionPolicy(active_zones);
    const unsigned int radius   = (first_order) ? 1 : 2;
    const int shBlockSize       = BLOCK_SIZE + 2 * radius;
    const unsigned shBlockBytes = shBlockSize * sizeof(Primitive);

    if constexpr(BuildPlatform == Platform::GPU)
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
    else 
        cons2prim(fullP);

    // Check if user input a valid initial timestep. 
    // if not, adapt it
    // Adapt the timestep
    if constexpr(BuildPlatform == Platform::GPU)
    {
        adapt_dt(device_self, physical_nBlocks);
    } else {
        adapt_dt();
    }

    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;


    // Some benchmarking tools 
    int   nfold   = 0;
    int   ncheck  = 0;
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
                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts1DGPU(fullP, device_self, nx, true);
            } else {
                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts1DGPU(fullP, this, nx, true);
            }
            
            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< nx / delta_t.count() << std::flush;
                nfold += 100;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
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
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, physical_nBlocks);
            } else {
                adapt_dt();
            }
        }
    } else {
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            if constexpr(BuildPlatform == Platform::GPU)
            {
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Dev);
                config_ghosts1DGPU(fullP, device_self, nx, false);

                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Dev);
                config_ghosts1DGPU(fullP, device_self, nx, false);
            } else {
                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts1DGPU(fullP, this, nx, false);

                advance(device_self, shBlockSize, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts1DGPU(fullP, this, nx, false);
            }
            
            t += dt; 
            

            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< nx / delta_t.count() << std::flush;
                nfold += 100;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
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
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, physical_nBlocks);
            } else {
                adapt_dt();
            }
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    if constexpr (BuildPlatform == Platform::GPU)
    {
        dualMem.copyDevToHost(device_self, *this);
        simbi::gpu::api::gpuFree(device_self);
    } 
    

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (int ii = 0; ii < nx; ii++)
    {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};