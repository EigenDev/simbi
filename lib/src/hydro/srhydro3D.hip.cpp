/*
 * C++ Source to perform 3D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "util/device_api.hpp"
#include "util/dual.hpp"
#include "common/helpers.hpp"
#include "util/parallel_for.hpp"
#include "util/printb.hpp"
#include "helpers.hip.hpp"
#include "srhydro3D.hip.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;

// Default Constructor
SRHD3D::SRHD3D() {}

// Overloaded Constructor
SRHD3D::SRHD3D(
    std::vector<std::vector<real>> state3D, 
    luint nx, luint ny, luint nz, real gamma,
    std::vector<real> x1, 
    std::vector<real> x2,
    std::vector<real> x3, 
    real cfl,
    std::string coord_system = "cartesian")
:
    nx(nx),
    ny(ny),
    nz(nz),
    nzones(state3D[0].size()),
    state3D(state3D),
    gamma(gamma),
    x1(x1),
    x2(x2),
    x3(x3),
    cfl(cfl),
    coord_system(coord_system)
{

}

// Destructor
SRHD3D::~SRHD3D() {}

/* Define typedefs because I am lazy */
typedef sr3d::Primitive Primitive;
typedef sr3d::Conserved Conserved;
typedef sr3d::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------

void SRHD3D::cons2prim()
{
    /**
   * Return a 3D matrix containing the primitive
   * variables density , pressure, and
   * three-velocity
   */

    real S1, S2, S3, S, D, tau, tol;
    real W, v1, v2, v3;

    // Define Newton-Raphson Vars
    real etotal, c2, f, g, p, peq;
    real Ws, rhos, eps, h;

    luint idx;
    luint iter = 0;
    for (luint kk = 0; kk < nz; kk++)
    {
        for (luint jj = 0; jj < ny; jj++)
        {
            for (luint ii = 0; ii < nx; ii++)
            {
                idx = ii + nx * jj + nx * ny * kk;
                D   = cons[idx].D;     // Relativistic Mass Density
                S1  = cons[idx].S1;   // X1-Momentum Denity
                S2  = cons[idx].S2;   // X2-Momentum Density
                S3  = cons[idx].S3;   // X2-Momentum Density
                tau = cons[idx].tau;  // Energy Density
                S = sqrt(S1 * S1 + S2 * S2 + S3 * S3);

                peq = (n != 0.0) ? pressure_guess[idx] : std::abs(S - D - tau);

                tol = D * 1.e-12;

                //--------- Iteratively Solve for Pressure using Newton-Raphson
                // Note: The NR scheme can be modified based on:
                // https://www.sciencedirect.com/science/article/pii/S0893965913002930
                iter = 0;
                do
                {
                    p = peq;
                    etotal = tau + p + D;
                    v2 = S * S / (etotal * etotal);
                    Ws = (real)1.0 / sqrt((real)1.0 - v2);
                    rhos = D / Ws;
                    eps = (tau + D * ((real)1.0 - Ws) + ((real)1.0 - Ws * Ws) * p) / (D * Ws);
                    f = (gamma - (real)1.0) * rhos * eps - p;

                    h = (real)1.0 + eps + p / rhos;
                    c2 = gamma * p / (h * rhos);
                    g = c2 * v2 - (real)1.0;
                    peq = p - f / g;
                    iter++;

                    if (iter > MAX_ITER)
                    {
                        std::cout << "\n";
                        std::cout << "p: " << p       << "\n";
                        std::cout << "S: " << S       << "\n";
                        std::cout << "tau: " << tau   << "\n";
                        std::cout << "D: " << D       << "\n";
                        std::cout << "et: " << etotal << "\n";
                        std::cout << "Ws: " << Ws     << "\n";
                        std::cout << "v2: " << v2     << "\n";
                        std::cout << "W: " << W       << "\n";
                        std::cout << "n: " << n       << "\n";
                        std::cout << "\n Cons2Prim Cannot Converge" << "\n";
                        exit(EXIT_FAILURE);
                    }

                } while (std::abs(peq - p) >= tol);
            

                v1 = S1 / (tau + D + peq);
                v2 = S2 / (tau + D + peq);
                v3 = S3 / (tau + D + peq);
                Ws = (real)1.0 / sqrt((real)1.0 - (v1 * v1 + v2 * v2 + v3 * v3));

                // Update the pressure guess for the next time step
                pressure_guess[idx] = peq;
                prims[idx]          = Primitive{D  / Ws, v1, v2, v3, peq};
            }
        }
    }
};

void SRHD3D::cons2prim(
    ExecutionPolicy<> p, 
    SRHD3D *dev, 
    simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        #if GPU_CODE 
        extern __shared__ Conserved conserved_buff[];
        #else 
        auto *const conserved_buff = &cons[0];
        #endif 

        auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x : gid;

        // load shared memory
        #if GPU_CODE
            conserved_buff[tid] = self->gpu_cons[gid];
        #endif

        simbi::gpu::api::synchronize();
        luint iter  = 0;
        real D    = conserved_buff[tid].D;
        real S1   = conserved_buff[tid].S1;
        real S2   = conserved_buff[tid].S2;
        real S3   = conserved_buff[tid].S3;
        real tau  = conserved_buff[tid].tau;
        real S    = sqrt(S1 * S1 + S2 * S2 + S3 * S3);

        #if GPU_CODE
        real peq = self->gpu_pressure_guess[gid];
        #else 
        real peq = pressure_guess[gid];
        #endif

        real tol = D * tol_scale;
        do
        {
            pre = peq;
            et  = tau + D + pre;
            v2 = S * S / (et * et);
            W   = (real)1.0 / sqrt((real)1.0 - v2);
            rho = D / W;

            eps = (tau + ((real)1.0 - W) * D + ((real)1.0 - W * W) * pre) / (D * W);

            h = (real)1.0 + eps + pre / rho;
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
        
        real inv_et = (real)1.0 / (tau + D + peq); 
        real vx = S1 * inv_et;
        real vy = S2 * inv_et;
        real vz = S3 * inv_et;

        #if GPU_CODE
            self->gpu_pressure_guess[gid] = peq;
            self->gpu_prims[gid]          = Primitive{D * sqrt((real)1.0 - (vx * vx + vy * vy + vz * vz)), vx, vy, vz, peq};
        #else
            pressure_guess[gid] = peq;
            prims[gid]          = Primitive{D * sqrt((real)1.0 - (vx * vx + vy * vy + vz * vz)), vx, vy, vz,  peq};
        #endif
    });
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD3D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const luint nhat = 1)
{
    // Eigenvals lambda;

    // Separate the left and right Primitive
    const real rho_l = prims_l.rho;
    const real p_l   = prims_l.p;
    const real h_l   = (real)1.0 + gamma * p_l / (rho_l * (gamma - 1));

    const real rho_r = prims_r.rho;
    const real p_r   = prims_r.p;
    const real h_r   = (real)1.0 + gamma * p_r / (rho_r * (gamma - 1));

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
            const real bl    = (vbar - cbar)/((real)1.0 - cbar*vbar);
            const real br    = (vbar + cbar)/((real)1.0 + cbar*vbar);
            const real aL    = my_min(bl, (v1_l - cs_l)/((real)1.0 - v1_l*cs_l));
            const real aR    = my_max(br, (v1_r + cs_r)/((real)1.0 + v1_r*cs_r));

            return Eigenvals(aL, aR);

            //--------Calc the wave speeds based on Mignone and Bodo (2005)
            // const real sL = cs_l * cs_l * ((real)1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
            // const real sR = cs_r * cs_r * ((real)1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

            // // Define temporaries to save computational cycles
            // const real qfL = (real)1.0 / ((real)1.0 + sL);
            // const real qfR = (real)1.0 / ((real)1.0 + sR);
            // const real sqrtR = sqrt(sR * (1 - v1_r * v1_r + sR));
            // const real sqrtL = sqrt(sL * (1 - v1_l * v1_l + sL));

            // const real lamLm = (v1_l - sqrtL) * qfL;
            // const real lamRm = (v1_r - sqrtR) * qfR;
            // const real lamLp = (v1_l + sqrtL) * qfL;
            // const real lamRp = (v1_r + sqrtR) * qfR;

            // const real aL = lamLm < lamRm ? lamLm : lamRm;
            // const real aR = lamLp > lamRp ? lamLp : lamRp;

            // return Eigenvals(aL, aR);
        }
        case 2:
        {
            const real v2_r = prims_r.v2;
            const real v2_l = prims_l.v2;

            //-----------Calculate wave speeds based on Shneider et al. 1992
            const real vbar  = (real)0.5 * (v2_l + v2_r);
            const real cbar  = (real)0.5 * (cs_l + cs_r);
            const real bl    = (vbar - cbar)/((real)1.0 - cbar*vbar);
            const real br    = (vbar + cbar)/((real)1.0 + cbar*vbar);
            const real aL    = my_min(bl, (v2_l - cs_l)/((real)1.0 - v2_l*cs_l));
            const real aR    = my_max(br, (v2_r + cs_r)/((real)1.0 + v2_r*cs_r));

            return Eigenvals(aL, aR);

            // Calc the wave speeds based on Mignone and Bodo (2005)
            // real sL = cs_l * cs_l * ((real)1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
            // real sR = cs_r * cs_r * ((real)1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

            // Define some temporaries to save a few cycles
            // const real qfL = (real)1.0 / ((real)1.0 + sL);
            // const real qfR = (real)1.0 / ((real)1.0 + sR);
            // const real sqrtR = sqrt(sR * (1 - v2_r * v2_r + sR));
            // const real sqrtL = sqrt(sL * (1 - v2_l * v2_l + sL));

            // const real lamLm = (v2_l - sqrtL) * qfL;
            // const real lamRm = (v2_r - sqrtR) * qfR;
            // const real lamLp = (v2_l + sqrtL) * qfL;
            // const real lamRp = (v2_r + sqrtR) * qfR;
            // const real aL = lamLm < lamRm ? lamLm : lamRm;
            // const real aR = lamLp > lamRp ? lamLp : lamRp;

            // return Eigenvals(aL, aR);
        }
        case 3:
        {
            const real v3_r = prims_r.v3;
            const real v3_l = prims_l.v3;

            //-----------Calculate wave speeds based on Shneider et al. 1992
            const real vbar  = (real)0.5 * (v3_l + v3_r);
            const real cbar  = (real)0.5 * (cs_l + cs_r);
            const real bl    = (vbar - cbar)/((real)1.0 - cbar*vbar);
            const real br    = (vbar + cbar)/((real)1.0 + cbar*vbar);
            const real aL    = my_min(bl, (v3_l - cs_l)/((real)1.0 - v3_l*cs_l));
            const real aR    = my_max(br, (v3_r + cs_r)/((real)1.0 + v3_r*cs_r));

            return Eigenvals(aL, aR);
        }
    } // end switch
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Conserved SRHD3D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real vz = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = (real)1.0 / sqrt((real)1.0 - (vx * vx + vy * vy + vz * vz));
    const real h = (real)1.0 + gamma * pressure / (rho * (gamma - 1));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * vx,
        rho * h * lorentz_gamma * lorentz_gamma * vy,
        rho * h * lorentz_gamma * lorentz_gamma * vz,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma};
};

// Conserved SRHD3D::calc_intermed_statesSR2D(const Primitive &prims,
//                                            const Conserved &state, real a,
//                                            real aStar, real pStar,
//                                            luint nhat = 1)
// {
//     real Dstar, S1star, S2star, tauStar, Estar, cofactor;
//     Conserved starStates;

//     real pressure = prims.p;
//     real v1 = prims.v1;
//     real v2 = prims.v2;

//     real D = state.D;
//     real S1 = state.S1;
//     real S2 = state.S2;
//     real tau = state.tau;
//     real E = tau + D;

//     switch (nhat)
//     {
//     case 1:
//         cofactor = (real)1.0 / (a - aStar);
//         Dstar = cofactor * (a - v1) * D;
//         S1star = cofactor * (S1 * (a - v1) - pressure + pStar);
//         S2star = cofactor * (a - v1) * S2;
//         Estar = cofactor * (E * (a - v1) + pStar * aStar - pressure * v1);
//         tauStar = Estar - Dstar;

//         starStates = Conserved(Dstar, S1star, S2star, tauStar);

//         return starStates;
//     case 2:
//         cofactor = (real)1.0 / (a - aStar);
//         Dstar = cofactor * (a - v2) * D;
//         S1star = cofactor * (a - v2) * S1;
//         S2star = cofactor * (S2 * (a - v2) - pressure + pStar);
//         Estar = cofactor * (E * (a - v2) + pStar * aStar - pressure * v2);
//         tauStar = Estar - Dstar;

//         starStates = Conserved(Dstar, S1star, S2star, tauStar);

//         return starStates;
//     }

//     return starStates;
// }

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditonal timestep
void SRHD3D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real cs, dx1, dx2, dx3, rho, pressure, v1, v2,v3, rmean, rproj, h, sint;
        real cfl_dt;
        luint shift_i, shift_j, shift_k;
        real plus_v1, plus_v2, minus_v1, minus_v2, plus_v3, minus_v3;
        luint aid; // active index id

        // Compute the minimum timestep given cfl
        for (luint kk = 0; kk < zphysical_grid; kk++)
        {
            dx3  = coord_lattice.dx3[kk];
            shift_k = kk + idx_active;
            for (luint jj = 0; jj < yphysical_grid; jj++)
            {
                dx2 = coord_lattice.dx2[jj];
                shift_j = jj + idx_active;
                sint = coord_lattice.sin[jj];
                #pragma omp for nowait schedule(static) reduction(min:min_dt)
                for (luint ii = 0; ii < xphysical_grid; ii++)
                {
                    shift_i  = ii + idx_active;
                    aid      = shift_k * nx * ny + shift_j * nx + shift_i;
                    dx1      = coord_lattice.dx1[ii];
                    rho      = prims[aid].rho;
                    v1       = prims[aid].v1;
                    v2       = prims[aid].v2;
                    v3       = prims[aid].v3;
                    pressure = prims[aid].p;

                    h = (real)1.0 + gamma * pressure / (rho * (gamma - 1.));
                    cs = sqrt(gamma * pressure / (rho * h));

                    plus_v1  = (v1 + cs) / ((real)1.0 + v1 * cs);
                    plus_v2  = (v2 + cs) / ((real)1.0 + v2 * cs);
                    plus_v3  = (v3 + cs) / ((real)1.0 + v3 * cs);
                    minus_v1 = (v1 - cs) / ((real)1.0 - v1 * cs);
                    minus_v2 = (v2 - cs) / ((real)1.0 - v2 * cs);
                    minus_v3 = (v3 - cs) / ((real)1.0 - v3 * cs);

                    if (coord_system == "cartesian")
                    {

                        cfl_dt = std::min(
                                    {dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                     dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))),
                                     dx3 / (std::max(std::abs(plus_v3), std::abs(minus_v3)))});
                    }
                    else
                    {
                        rmean = coord_lattice.x1mean[ii];
                        rproj = rmean * sint;

                        // At either pole, we are just in the r,theta plane
                        if (rproj == 0) 
                            cfl_dt = std::min(
                                        {       dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                        rmean * dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2)))});
                        else
                            cfl_dt = std::min(
                                        {       dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                        rmean * dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))),
                                        rproj * dx3 / (std::max(std::abs(plus_v3), std::abs(minus_v3)))});
                            
                    }

                    min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
                    
                } // end ii 
            } // end jj
        } // end kk
    } // end parallel region

    dt = cfl * min_dt;
};

void SRHD3D::adapt_dt(SRHD3D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p)
{
    #if GPU_CODE
    {
        dtWarpReduce<SRHD3D, Primitive, 128><<<p.gridSize, dim3(BLOCK_SIZE3D, BLOCK_SIZE3D, BLOCK_SIZE3D)>>>(dev, geometry);
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
Conserved SRHD3D::calc_Flux(const Primitive &prims,   luint nhat = 1)
{

    const real rho      = prims.rho;
    const real vx       = prims.v1;
    const real vy       = prims.v2;
    const real vz       = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = (real)1.0 / sqrt((real)1.0 - (vx * vx + vy * vy + vz*vz));

    const real h  = (real)1.0 + gamma * pressure / (rho * (gamma - (real)1.0));
    const real D  = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const real S3 = rho * lorentz_gamma * lorentz_gamma * h * vz;
    const real tau =
                    rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return (nhat == 1) ? Conserved{D * vx, S1 * vx + pressure, S2 * vx, S3 * vx,  (tau + pressure) * vx}
          :(nhat == 2) ? Conserved{D * vy, S1 * vy, S2 * vy + pressure, S3 * vy,  (tau + pressure) * vy}
          :              Conserved{D * vz, S1 * vz, S2 * vz, S3 * vz + pressure,  (tau + pressure) * vz};
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
Conserved SRHD3D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const   luint nhat = 1)
{

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

    // return Conserved(0.0, 0.0, 0.0, 0.0);
    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D = left_state.D;
        const real S1 = left_state.S1;
        const real S2 = left_state.S2;
        const real S3 = left_state.S3;
        const real tau = left_state.tau;
        const real E = tau + D;
        const real cofactor = (real)1.0 / (aL - aStar);
        //--------------Compute the L Star State----------
        switch (nhat)
        {
            case 1:
            {
                const real v1 = left_prims.v1;
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
                const real v2 = left_prims.v2;
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

            case 3:
            {
                const real v3 = left_prims.v3;
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
    }
    else
    {
        const real pressure = right_prims.p;
        const real D  = right_state.D;
        const real S1 = right_state.S1;
        const real S2 = right_state.S2;
        const real S3 = right_state.S3;
        const real tau = right_state.tau;
        const real E = tau + D;
        const real cofactor = (real)1.0 / (aR - aStar);

        /* Compute the L/R Star State */
        switch (nhat)
        {
            case 1:
            {
                const real v1 = right_prims.v1;
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
                const real v2 = right_prims.v2;
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

            case 3:
            {
                const real v3 = right_prims.v3;
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
    SRHD3D *dev, 
    const ExecutionPolicy<> p,
    const luint sh_block_size,
    const luint radius, 
    const simbi::Geometry geometry, 
    const simbi::MemSide user)
{
    auto *self = (BuildPlatform == Platform::GPU) ? dev : this;
    const luint xpg                   = this->xphysical_grid;
    const luint ypg                   = this->yphysical_grid;
    const luint zpg                   = this->zphysical_grid;
    const bool is_first_order       = this->first_order;
    const bool is_periodic          = this->periodic;
    const bool hllc                 = this->hllc;
    const luint bx                    = (BuildPlatform == Platform::GPU) ? sh_block_size : nx;
    const luint by                    = (BuildPlatform == Platform::GPU) ? sh_block_size : ny;
    const luint nbs                   = (BuildPlatform == Platform::GPU) ? sh_block_size * sh_block_size * sh_block_size : nzones;
    const real dt                   = this->dt;
    const real decay_const          = this->decay_const;
    const real plm_theta            = this->plm_theta;
    const real gamma                = this->gamma;
    const luint nx                    = this->nx;
    const luint ny                    = this->ny;
    const luint nz                    = this->nz;
    const luint extent                = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.z * p.gridSize.z * p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const luint xextent               = p.blockSize.x;
    const luint yextent               = p.blockSize.y;
    const luint zextent               = p.blockSize.z;
    const CLattice3D *coord_lattice = &(self->coord_lattice);

    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = &prims[0];
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
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const luint tz  = (BuildPlatform == Platform::GPU) ? threadIdx.z: 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;
        const luint tza = (BuildPlatform == Platform::GPU) ? tz + radius : ka;

        // printf("(%d, %d, %d)\n", txa, tya, tza);
        Conserved ux_l, ux_r, uy_l, uy_r, uz_l, uz_r;
        Conserved f_l, f_r, g_l, g_r, h_l, h_r, f1, f2, g1, g2, h1, h2;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r, zprims_l, zprims_r;

        luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            luint tzl = zextent;

            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tza * bx * by + tya * bx + txa] = self->gpu_prims[aid];
            if (threadIdx.z < radius)    
            {
                if (ka + zextent > nz - 1) tzl = nz - radius - ka + threadIdx.z;
                prim_buff[(tza - radius) * bx * by + tya * bx + txa] = self->gpu_prims[(ka - radius) * nx * ny + ja * nx + ia];
                prim_buff[(tza + tzl   ) * bx * by + tya * bx + txa] = self->gpu_prims[(ka + tzl   ) * nx * ny + ja * nx + ia];

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if ((tzl == 1) && (radius == 2)) 
                {
                    prim_buff[(tza + 1 - radius) * bx * by + tya * bx + txa] =  self->gpu_prims[(ka + 1 - radius) * nx * ny + ja * nx + ia];
                    prim_buff[(tza + 1 + tzl   ) * bx * by + tya * bx + txa] =  self->gpu_prims[(ka + 1 + tzl   ) * nx * ny + ja * nx + ia]; 
                }  
            }
            if (threadIdx.y < radius)    
            {
                if (ja + yextent > ny - 1) tyl = ny - radius - ja + threadIdx.y;
                prim_buff[tza * bx * by + (tya - radius) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja - radius) * nx + ia];
                prim_buff[tza * bx * by + (tya + tyl   ) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja + tyl   ) * nx + ia];

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if ((tyl == 1) && (radius == 2)) 
                {
                    prim_buff[tza * bx * by + (tya + 1 - radius) * bx + txa] =  self->gpu_prims[ka * nx * ny + ((ja + 1 - radius) * nx) + ia];
                    prim_buff[tza * bx * by + (tya + 1 + tyl) * bx + txa]    =  self->gpu_prims[ka * nx * ny + ((ja + 1 + txl   ) * nx) + ia]; 
                } 
            }
            if (threadIdx.x < radius)
            {   
                if (ia + xextent > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[tza * bx * by + tya * bx + txa - radius] =  self->gpu_prims[ka * nx * ny + (ja * nx) + ia - radius];
                prim_buff[tza * bx * by + tya * bx + txa +    txl] =  self->gpu_prims[ka * nx * ny + (ja * nx) + ia + txl]; 

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if ((txl == 1) && (radius == 2)) 
                {
                    prim_buff[tza * bx * by + tya * bx + (txa + 1) - radius] =  self->gpu_prims[ka * nx * ny + (ja * nx) + (ia + 1) - radius];
                    prim_buff[tza * bx * by + tya * bx + (txa + 1) +    txl] =  self->gpu_prims[ka * nx * ny + (ja * nx) + (ia + 1) + txl]; 
                }
            }
            simbi::gpu::api::synchronize();
        #endif

        if (is_first_order)
        {
            if (is_periodic)
            {
                // Set up the left and right state interfaces for i+1/2
                // u_l   = cons_buff[txa];
                // u_r   = roll(cons_buff, txa + 1, sh_block_size);
            }
            else
            {

                xprims_l = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
                xprims_r = prim_buff[tza * bx * by + tya * bx + (txa + 1)];
                //j+1/2
                yprims_l = prim_buff[tza * bx * by + (tya + 0) * bx + txa];
                yprims_r = prim_buff[tza * bx * by + (tya + 1) * bx + txa];
                //j+1/2
                zprims_l = prim_buff[(tza + 0) * bx * by + tya * bx + txa];
                zprims_r = prim_buff[(tza + 1) * bx * by + tya * bx + txa];
            }
            ux_l = self->prims2cons(xprims_l);
            ux_r = self->prims2cons(xprims_r);

            uy_l = self->prims2cons(yprims_l);
            uy_r = self->prims2cons(yprims_r);

            uz_l = self->prims2cons(zprims_l);
            uz_r = self->prims2cons(zprims_r);

            f_l = self->calc_Flux(xprims_l, 1);
            f_r = self->calc_Flux(xprims_r, 1);

            g_l = self->calc_Flux(yprims_l, 2);
            g_r = self->calc_Flux(yprims_r, 2);

            h_l = self->calc_Flux(zprims_l, 3);
            h_r = self->calc_Flux(zprims_r, 3);

            // Calc HLL Flux at i+1/2 interface
            if (self-> hllc)
            {
                f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h1 = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);

            } else {
                f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h1 = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

            // Set up the left and right state interfaces for i-1/2
            if (is_periodic)
            {
                ;
            }
            else
            {
                xprims_l = prim_buff[tza * bx * by + tya * bx + (txa - 1)];
                xprims_r = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
                //j+1/2
                yprims_l = prim_buff[tza * bx * by + (tya - 1) * bx + txa]; 
                yprims_r = prim_buff[tza * bx * by + (tya + 0) * bx + txa]; 
                //k+1/2
                zprims_l = prim_buff[(tza - 1) * bx * by + tya * bx + txa]; 
                zprims_r = prim_buff[(tza - 0) * bx * by + tya * bx + txa]; 
            }

            ux_l = self->prims2cons(xprims_l);
            ux_r = self->prims2cons(xprims_r);

            uy_l = self->prims2cons(yprims_l);
            uy_r = self->prims2cons(yprims_r);

            uz_l = self->prims2cons(zprims_l);
            uz_r = self->prims2cons(zprims_r);

            f_l = self->calc_Flux(xprims_l, 1);
            f_r = self->calc_Flux(xprims_r, 1);

            g_l = self->calc_Flux(yprims_l, 2);
            g_r = self->calc_Flux(yprims_r, 2);

            h_l = self->calc_Flux(zprims_l, 3);
            h_r = self->calc_Flux(zprims_r, 3);

            // Calc HLL Flux at i-1/2 interface
            if (self-> hllc)
            {
                f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h2 = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);

            } else {
                f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h2 = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

            //Advance depending on geometry
            luint real_loc = kk * xpg * ypg + jj * xpg + ii;

            // printf("f1: %f, f2: %f, g1: %f, g2: %f, h1: %f, h2: %f\n",
            // f1.D,
            // f2.D,
            // g1.D,
            // g2.D,
            // h1.D,
            // h2.D);
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            real dx1 = coord_lattice->gpu_dx1[ii];
                            real dx2  = coord_lattice->gpu_dx2[jj];
                            real dz = coord_lattice->gpu_dx3[kk];
                            self->gpu_cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D )  / dx2 - (h1.D - h2.D)     / dz + self->gpu_sourceD   [real_loc] );
                            self->gpu_cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1)  / dx2 - (h1.S1 - h2.S3)   / dz + self->gpu_sourceS1  [real_loc] );
                            self->gpu_cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx1  - (g1.S2  - g2.S2) / dx2 - (h1.S2 - h2.S3)   / dz + self->gpu_sourceS2  [real_loc] );
                            self->gpu_cons[aid].S3  += dt * ( -(f1.S3 - f2.S3)   / dx1  - (g1.S3  - g2.S3) / dx2 - (h1.S3 - h2.S3)   / dz + self->gpu_sourceS3  [real_loc] );
                            self->gpu_cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau) / dx2 - (h1.tau - h2.tau) / dz + self->gpu_sourceTau [real_loc] );
                        #else
                            real dx1 = coord_lattice->dx1[ii];
                            real dx2  = coord_lattice->dx2[jj];
                            real dz = coord_lattice->dx3[kk];
                            self->cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D )  / dx2 - (h1.D - h2.D)     / dz + sourceD   [real_loc] );
                            self->cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1)  / dx2 - (h1.S1 - h2.S3)   / dz + sourceS1  [real_loc] );
                            self->cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx1  - (g1.S2  - g2.S2) / dx2 - (h1.S2 - h2.S3)   / dz + sourceS2  [real_loc] );
                            self->cons[aid].S3  += dt * ( -(f1.S3 - f2.S3)   / dx1  - (g1.S3  - g2.S3) / dx2 - (h1.S3 - h2.S3)   / dz + sourceS3  [real_loc] );
                            self->cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau) / dx2 - (h1.tau - h2.tau) / dz + sourceTau [real_loc] );
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
                        real s3R        = coord_lattice->gpu_x3_face_areas[kk + 1];
                        real s3L        = coord_lattice->gpu_x3_face_areas[kk + 0];
                        real rmean      = coord_lattice->gpu_x1mean[ii]           ;
                        real dV1        = coord_lattice->gpu_dV1[ii]              ;
                        real dV2        = rmean * coord_lattice->gpu_dV2[jj]      ;
                        real dV3        = rmean * coord_lattice->gpu_sin[jj] * coord_lattice->gpu_dx3[kk];
                        #else 
                        real s1R   =  coord_lattice->x1_face_areas[ii + 1];
                        real s1L   =  coord_lattice->x1_face_areas[ii + 0];
                        real s2R   =  coord_lattice->x2_face_areas[jj + 1];
                        real s2L   =  coord_lattice->x2_face_areas[jj + 0];
                        real s3R   =  coord_lattice->x3_face_areas[kk + 1];
                        real s3L   =  coord_lattice->x3_face_areas[kk + 0];
                        real rmean =  coord_lattice->x1mean[ii];
                        real dV1   =  coord_lattice->dV1[ii];
                        real dV2   =  rmean * coord_lattice->dV2[jj];
                        real dV3   =  rmean * coord_lattice->sin[jj] * coord_lattice->dx3[kk];
                        #endif
                        // // Grab central primitives
                        real rhoc = prim_buff[tza * bx * by + tya * bx + txa].rho;
                        real pc   = prim_buff[tza * bx * by + tya * bx + txa].p;
                        real uc   = prim_buff[tza * bx * by + tya * bx + txa].v1;
                        real vc   = prim_buff[tza * bx * by + tya * bx + txa].v2;
                        real wc   = prim_buff[tza * bx * by + tya * bx + txa].v3;

                        real hc    = (real)1.0 + gamma * pc/(rhoc * (gamma - (real)1.0));
                        real gam2  = (real)1.0/((real)1.0 - (uc * uc + vc * vc + wc * wc));

                        #if GPU_CODE
                            self->gpu_cons[aid] +=
                            Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        - (h1.D * s3R - h2.D * s3L) / dV3 
                                            + self->gpu_sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        - (h1.S1 * s3R - h2.S1 * s3L) / dV3 
                                        + rhoc * hc * gam2 * (vc * vc + wc * wc) / rmean + 2 * pc / rmean +
                                                self->gpu_sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                        - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                            - (h1.S2 * s3R - h2.S2 * s3L) / dV3 
                                            - rhoc * hc * gam2 * uc * vc / rmean + coord_lattice->gpu_cot[jj] / rmean * (pc + rhoc * hc * gam2 *wc * wc) 
                                            + self->gpu_sourceS2[real_loc] * decay_const,

                                // L(S3)
                                -(f1.S3 * s1R - f2.S3 * s1L) / dV1
                                        - (g1.S3 * s2R - g2.S3 * s2L) / dV2 
                                            - (h1.S3 * s3R - h2.S3 * s3L) / dV3 
                                                - rhoc * hc * gam2 * wc * (uc + vc * coord_lattice->gpu_cot[jj])/ rmean
                                            +     self->gpu_sourceS3[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        - (h1.tau* s3R - h2.tau* s3L) / dV3 
                                            + self->gpu_sourceTau[real_loc] * decay_const
                            } * dt;
                        #else
                            cons[aid] +=
                            Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        - (h1.D * s3R - h2.D * s3L) / dV3 
                                            + sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        - (h1.S1 * s3R - h2.S1 * s3L) / dV3 
                                        + rhoc * hc * gam2 * (vc * vc + wc * wc) / rmean + 2 * pc / rmean +
                                                sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                        - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                            - (h1.S2 * s3R - h2.S2 * s3L) / dV3 
                                            - rhoc * hc * gam2 * uc * vc / rmean + coord_lattice->cot[jj] / rmean * (pc + rhoc * hc * gam2 *wc * wc) 
                                            + sourceS2[real_loc] * decay_const,

                                // L(S3)
                                -(f1.S3 * s1R - f2.S3 * s1L) / dV1
                                        - (g1.S3 * s2R - g2.S3 * s2L) / dV2 
                                            - (h1.S3 * s3R - h2.S3 * s3L) / dV3 
                                                - rhoc * hc * gam2 * wc * (uc + vc * coord_lattice->cot[jj])/ rmean
                                                    + sourceS3[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        - (h1.tau* s3R - h2.tau* s3L) / dV3 
                                            + sourceTau[real_loc] * decay_const
                            } * dt;
                        #endif
                    
                    break;

                    } // end spherical case
            } // end switch
                
        }
        else
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            Primitive zleft_most, zright_most, zleft_mid, zright_mid;

            if (!is_periodic)
            {
                // Coordinate X
                xleft_most  = prim_buff[tza * bx * by + tya * bx + (txa - 2)];
                xleft_mid   = prim_buff[tza * bx * by + tya * bx + (txa - 1)];
                center      = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
                xright_mid  = prim_buff[tza * bx * by + tya * bx + (txa + 1)];
                xright_most = prim_buff[tza * bx * by + tya * bx + (txa + 2)];

                // Coordinate Y
                yleft_most  = prim_buff[tza * bx * by + (tya - 2) * bx + txa];
                yleft_mid   = prim_buff[tza * bx * by + (tya - 1) * bx + txa];
                yright_mid  = prim_buff[tza * bx * by + (tya + 1) * bx + txa];
                yright_most = prim_buff[tza * bx * by + (tya + 2) * bx + txa];

                // Coordinate z
                zleft_most  = prim_buff[(tza - 2) * bx * by + tya * bx + txa];
                zleft_mid   = prim_buff[(tza - 1) * bx * by + tya * bx + txa];
                zright_mid  = prim_buff[(tza + 1) * bx * by + tya * bx + txa];
                zright_most = prim_buff[(tza + 2) * bx * by + tya * bx + txa];
            }
            else
            {
                // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                /* TODO: Fix this */
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
            xprims_l.v3 =
                center.v3 + (real)0.5 * minmod(plm_theta * (center.v3 - xleft_mid.v3),
                                            (real)0.5 * (xright_mid.v3 - xleft_mid.v3),
                                            plm_theta * (xright_mid.v3 - center.v3));

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

            xprims_r.v3 = xright_mid.v3 -
                            (real)0.5 * minmod(plm_theta * (xright_mid.v3 - center.v3),
                                        (real)0.5 * (xright_most.v3 - center.v3),
                                        plm_theta * (xright_most.v3 - xright_mid.v3));

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
            yprims_l.v3 =
                center.v3 + (real)0.5 * minmod(plm_theta * (center.v3 - yleft_mid.v3),
                                            (real)0.5 * (yright_mid.v3 - yleft_mid.v3),
                                            plm_theta * (yright_mid.v3 - center.v3));
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
            yprims_r.v3 = yright_mid.v3 -
                            (real)0.5 * minmod(plm_theta * (yright_mid.v3 - center.v3),
                                        (real)0.5 * (yright_most.v3 - center.v3),
                                        plm_theta * (yright_most.v3 - yright_mid.v3));

            yprims_r.p = yright_mid.p -
                            (real)0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                        (real)0.5 * (yright_most.p - center.p),
                                        plm_theta * (yright_most.p - yright_mid.p));

            // Reconstructed right Primitive vector in z-direction at j+1/2
            // interfce
            zprims_l.rho =
                center.rho + (real)0.5 * minmod(plm_theta * (center.rho - zleft_mid.rho),
                                            (real)0.5 * (zright_mid.rho - zleft_mid.rho),
                                            plm_theta * (zright_mid.rho - center.rho));

            zprims_l.v1 =
                center.v1 + (real)0.5 * minmod(plm_theta * (center.v1 - zleft_mid.v1),
                                            (real)0.5 * (zright_mid.v1 - zleft_mid.v1),
                                            plm_theta * (zright_mid.v1 - center.v1));

            zprims_l.v2 =
                center.v2 + (real)0.5 * minmod(plm_theta * (center.v2 - zleft_mid.v2),
                                            (real)0.5 * (zright_mid.v2 - zleft_mid.v2),
                                            plm_theta * (zright_mid.v2 - center.v2));

            zprims_l.v3 =
                center.v3 + (real)0.5 * minmod(plm_theta * (center.v3 - zleft_mid.v3),
                                            (real)0.5 * (zright_mid.v3 - zleft_mid.v3),
                                            plm_theta * (zright_mid.v3 - center.v3));

            zprims_l.p =
                center.p + (real)0.5 * minmod(plm_theta * (center.p - zleft_mid.p),
                                        (real)0.5 * (zright_mid.p - zleft_mid.p),
                                        plm_theta * (zright_mid.p - center.p));

            zprims_r.rho =
                zright_mid.rho -
                (real)0.5 * minmod(plm_theta * (zright_mid.rho - center.rho),
                                (real)0.5 * (zright_most.rho - center.rho),
                                plm_theta * (zright_most.rho - zright_mid.rho));

            zprims_r.v1 = zright_mid.v1 -
                            (real)0.5 * minmod(plm_theta * (zright_mid.v1 - center.v1),
                                        (real)0.5 * (zright_most.v1 - center.v1),
                                        plm_theta * (zright_most.v1 - zright_mid.v1));

            zprims_r.v2 = zright_mid.v2 -
                            (real)0.5 * minmod(plm_theta * (zright_mid.v2 - center.v2),
                                        (real)0.5 * (zright_most.v2 - center.v2),
                                        plm_theta * (zright_most.v2 - zright_mid.v2));

            zprims_r.v3 = zright_mid.v3 -
                            (real)0.5 * minmod(plm_theta * (zright_mid.v3 - center.v3),
                                        (real)0.5 * (zright_most.v3 - center.v3),
                                        plm_theta * (zright_most.v3 - zright_mid.v3));

            zprims_r.p = zright_mid.p -
                            (real)0.5 * minmod(plm_theta * (zright_mid.p - center.p),
                                        (real)0.5 * (zright_most.p - center.p),
                                        plm_theta * (zright_most.p - zright_mid.p));

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            ux_l = self->prims2cons(xprims_l);
            ux_r = self->prims2cons(xprims_r);

            uy_l = self->prims2cons(yprims_l);
            uy_r = self->prims2cons(yprims_r);

            uz_l = self->prims2cons(zprims_l);
            uz_r = self->prims2cons(zprims_r);

            f_l = self->calc_Flux(xprims_l, 1);
            f_r = self->calc_Flux(xprims_r, 1);

            g_l = self->calc_Flux(yprims_l, 2);
            g_r = self->calc_Flux(yprims_r, 2);

            h_l = self->calc_Flux(zprims_l, 3);
            h_r = self->calc_Flux(zprims_r, 3);

            if (hllc)
            {
                f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h1 = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }
            else
            {
                f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h1 = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
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

            xprims_l.v3 = xleft_mid.v3 +
                            (real)0.5 * minmod(plm_theta * (xleft_mid.v3 - xleft_most.v3),
                                        (real)0.5 * (center.v3 - xleft_most.v3),
                                        plm_theta * (center.v3 - xleft_mid.v3));

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

            xprims_r.v3 =
                center.v3 - (real)0.5 * minmod(plm_theta * (center.v3 - xleft_mid.v3),
                                            (real)0.5 * (xright_mid.v3 - xleft_mid.v3),
                                            plm_theta * (xright_mid.v3 - center.v3));

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

            yprims_l.v3 = yleft_mid.v3 +
                            (real)0.5 * minmod(plm_theta * (yleft_mid.v3 - yleft_most.v3),
                                        (real)0.5 * (center.v3 - yleft_most.v3),
                                        plm_theta * (center.v3 - yleft_mid.v3));

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

            yprims_r.v3 =
                center.v3 - (real)0.5 * minmod(plm_theta * (center.v3 - yleft_mid.v3),
                                            (real)0.5 * (yright_mid.v3 - yleft_mid.v3),
                                            plm_theta * (yright_mid.v3 - center.v3));

            yprims_r.p =
                center.p - (real)0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                        (real)0.5 * (yright_mid.p - yleft_mid.p),
                                        plm_theta * (yright_mid.p - center.p));

            // Left side Primitive in z
            zprims_l.rho = zleft_mid.rho +
                            (real)0.5 * minmod(plm_theta * (zleft_mid.rho - zleft_most.rho),
                                        (real)0.5 * (center.rho - zleft_most.rho),
                                        plm_theta * (center.rho - zleft_mid.rho));

            zprims_l.v1 = zleft_mid.v1 +
                            (real)0.5 * minmod(plm_theta * (zleft_mid.v1 - zleft_most.v1),
                                        (real)0.5 * (center.v1 - zleft_most.v1),
                                        plm_theta * (center.v1 - zleft_mid.v1));

            zprims_l.v2 = zleft_mid.v2 +
                            (real)0.5 * minmod(plm_theta * (zleft_mid.v2 - zleft_most.v2),
                                        (real)0.5 * (center.v2 - zleft_most.v2),
                                        plm_theta * (center.v2 - zleft_mid.v2));

            zprims_l.v3 = zleft_mid.v3 +
                            (real)0.5 * minmod(plm_theta * (zleft_mid.v3 - zleft_most.v3),
                                        (real)0.5 * (center.v3 - zleft_most.v3),
                                        plm_theta * (center.v3 - zleft_mid.v3));

            zprims_l.p =
                zleft_mid.p + (real)0.5 * minmod(plm_theta * (zleft_mid.p - zleft_most.p),
                                            (real)0.5 * (center.p - zleft_most.p),
                                            plm_theta * (center.p - zleft_mid.p));

            // Right side Primitive in z
            zprims_r.rho =
                center.rho - (real)0.5 * minmod(plm_theta * (center.rho - zleft_mid.rho),
                                            (real)0.5 * (zright_mid.rho - zleft_mid.rho),
                                            plm_theta * (zright_mid.rho - center.rho));

            zprims_r.v1 =
                center.v1 - (real)0.5 * minmod(plm_theta * (center.v1 - zleft_mid.v1),
                                            (real)0.5 * (zright_mid.v1 - zleft_mid.v1),
                                            plm_theta * (zright_mid.v1 - center.v1));

            zprims_r.v2 =
                center.v2 - (real)0.5 * minmod(plm_theta * (center.v2 - zleft_mid.v2),
                                            (real)0.5 * (zright_mid.v2 - zleft_mid.v2),
                                            plm_theta * (zright_mid.v2 - center.v2));

            zprims_r.v3 =
                center.v3 - (real)0.5 * minmod(plm_theta * (center.v3 - zleft_mid.v3),
                                            (real)0.5 * (zright_mid.v3 - zleft_mid.v3),
                                            plm_theta * (zright_mid.v3 - center.v3));

            zprims_r.p =
                center.p - (real)0.5 * minmod(plm_theta * (center.p - zleft_mid.p),
                                        (real)0.5 * (zright_mid.p - zleft_mid.p),
                                        plm_theta * (zright_mid.p - center.p));
            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            ux_l = self->prims2cons(xprims_l);
            ux_r = self->prims2cons(xprims_r);
            uy_l = self->prims2cons(yprims_l);
            uy_r = self->prims2cons(yprims_r);
            uz_l = self->prims2cons(zprims_l);
            uz_r = self->prims2cons(zprims_r);

            f_l = self->calc_Flux(xprims_l, 1);
            f_r = self->calc_Flux(xprims_r, 1);
            g_l = self->calc_Flux(yprims_l, 2);
            g_r = self->calc_Flux(yprims_r, 2);
            h_l = self->calc_Flux(zprims_l, 3);
            h_r = self->calc_Flux(zprims_r, 3);

            // favl = (uy_r - uy_l) * (-K);
            
            if (hllc)
            {
                f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h2 = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
                
            }
            else
            {
                f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                h2 = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

                
            //Advance depending on geometry
            luint real_loc = kk * xpg * ypg + jj * xpg + ii;
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            real dx1 = coord_lattice->gpu_dx1[ii];
                            real dx2  = coord_lattice->gpu_dx2[jj];
                            real dz = coord_lattice->gpu_dx3[kk];
                            self->gpu_cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D )  / dx2 - (h1.D - h2.D)     / dz + self->gpu_sourceD   [real_loc] );
                            self->gpu_cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1)  / dx2 - (h1.S1 - h2.S3)   / dz + self->gpu_sourceS1  [real_loc] );
                            self->gpu_cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx1  - (g1.S2  - g2.S2) / dx2 - (h1.S2 - h2.S3)   / dz + self->gpu_sourceS2  [real_loc] );
                            self->gpu_cons[aid].S3  += (real)0.5 * dt * ( -(f1.S3 - f2.S3)   / dx1  - (g1.S3  - g2.S3) / dx2 - (h1.S3 - h2.S3)   / dz + self->gpu_sourceS3  [real_loc] );
                            self->gpu_cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau) / dx2 - (h1.tau - h2.tau) / dz + self->gpu_sourceTau [real_loc] );
                        #else
                            real dx1 = self->coord_lattice.dx1[ii];
                            real dx2  = self->coord_lattice.dx2[jj];
                            real dz = self->coord_lattice.dx3[kk];
                            cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D )  / dx2 - (h1.D - h2.D)     / dz + sourceD   [real_loc] );
                            cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1)  / dx2 - (h1.S1 - h2.S3)   / dz + sourceS1  [real_loc] );
                            cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx1  - (g1.S2  - g2.S2) / dx2 - (h1.S2 - h2.S3)   / dz + sourceS2  [real_loc] );
                            cons[aid].S3  += (real)0.5 * dt * ( -(f1.S3 - f2.S3)   / dx1  - (g1.S3  - g2.S3) / dx2 - (h1.S3 - h2.S3)   / dz + sourceS3  [real_loc] );
                            cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau) / dx2 - (h1.tau - h2.tau) / dz + sourceTau [real_loc] );
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
                        real s3R        = coord_lattice->gpu_x3_face_areas[kk + 1];
                        real s3L        = coord_lattice->gpu_x3_face_areas[kk + 0];
                        real rmean      = coord_lattice->gpu_x1mean[ii]           ;
                        real dV1        = coord_lattice->gpu_dV1[ii]              ;
                        real dV2        = rmean * coord_lattice->gpu_dV2[jj]      ;
                        real dV3        = rmean * coord_lattice->gpu_sin[jj] * coord_lattice->gpu_dx3[kk];
                        #else
                        real s1R    = self->coord_lattice.x1_face_areas[ii + 1];
                        real s1L    = self->coord_lattice.x1_face_areas[ii + 0];
                        real s2R    = self->coord_lattice.x2_face_areas[jj + 1];
                        real s2L    = self->coord_lattice.x2_face_areas[jj + 0];
                        real s3R    = self->coord_lattice.x3_face_areas[kk + 1];
                        real s3L    = self->coord_lattice.x3_face_areas[kk + 0];
                        real rmean  = self->coord_lattice.x1mean[ii];
                        real dV1    = self->coord_lattice.dV1[ii];
                        real dV2    = rmean * self->coord_lattice.dV2[jj];
                        real dV3    = rmean * self->coord_lattice.sin[jj] * self->coord_lattice.dx3[kk];
                        #endif
                        // // Grab central primitives
                        real rhoc = prim_buff[tza * bx * by + tya * bx + txa].rho;
                        real pc   = prim_buff[tza * bx * by + tya * bx + txa].p;
                        real uc   = prim_buff[tza * bx * by + tya * bx + txa].v1;
                        real vc   = prim_buff[tza * bx * by + tya * bx + txa].v2;
                        real wc   = prim_buff[tza * bx * by + tya * bx + txa].v3;

                        real hc    = (real)1.0 + gamma * pc/(rhoc * (gamma - (real)1.0));
                        real gam2  = (real)1.0/((real)1.0 - (uc * uc + vc * vc + wc * wc));

                        #if GPU_CODE
                            self->gpu_cons[aid] +=
                            Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        - (h1.D * s3R - h2.D * s3L) / dV3 
                                            + self->gpu_sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        - (h1.S1 * s3R - h2.S1 * s3L) / dV3 
                                        + rhoc * hc * gam2 * (vc * vc + wc * wc) / rmean + 2 * pc / rmean +
                                                self->gpu_sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                        - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                            - (h1.S2 * s3R - h2.S2 * s3L) / dV3 
                                            - rhoc * hc * gam2 * uc * vc / rmean + coord_lattice->gpu_cot[jj] / rmean * (pc + rhoc * hc * gam2 *wc * wc) 
                                            + self->gpu_sourceS2[real_loc] * decay_const,

                                // L(S3)
                                -(f1.S3 * s1R - f2.S3 * s1L) / dV1
                                        - (g1.S3 * s2R - g2.S3 * s2L) / dV2 
                                            - (h1.S3 * s3R - h2.S3 * s3L) / dV3 
                                                - rhoc * hc * gam2 * wc * (uc + vc * coord_lattice->gpu_cot[jj])/ rmean
                                            +     self->gpu_sourceS3[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        - (h1.tau* s3R - h2.tau* s3L) / dV3 
                                            + self->gpu_sourceTau[real_loc] * decay_const
                            } * dt * (real)0.5;
                        #else
                            cons[aid] +=
                            Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        - (h1.D * s3R - h2.D * s3L) / dV3 
                                            + sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        - (h1.S1 * s3R - h2.S1 * s3L) / dV3 
                                        + rhoc * hc * gam2 * (vc * vc + wc * wc) / rmean + 2 * pc / rmean +
                                                sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                        - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                            - (h1.S2 * s3R - h2.S2 * s3L) / dV3 
                                            - rhoc * hc * gam2 * uc * vc / rmean + self->coord_lattice.cot[jj] / rmean * (pc + rhoc * hc * gam2 *wc * wc) 
                                            + sourceS2[real_loc] * decay_const,

                                // L(S3)
                                -(f1.S3 * s1R - f2.S3 * s1L) / dV1
                                        - (g1.S3 * s2R - g2.S3 * s2L) / dV2 
                                            - (h1.S3 * s3R - h2.S3 * s3L) / dV3 
                                                - rhoc * hc * gam2 * wc * (uc + vc * self->coord_lattice.cot[jj])/ rmean
                                                    + sourceS3[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        - (h1.tau* s3R - h2.tau* s3L) / dV3 
                                            + sourceTau[real_loc] * decay_const
                            } * dt * (real)0.5;
                        #endif 
                    
                    break;

                    } // end spherical case
            } // end switch
        }// end else 

    });
}
//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD3D::simulate3D(
    const std::vector<std::vector<real>> sources,
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
    bool hllc)
{
    std::string tnow, tchunk, tstep;
    luint total_zones = nx * ny * nz;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + (real)0.5) / round_place
               : floor(tstart * round_place + (real)0.5) / round_place + chkpt_interval;

    std::string filename;

    this->sources     = sources;
    this->first_order = first_order;
    this->periodic    = boundary_condition == "periodic";
    this->hllc        = hllc;
    this->linspace    = linspace;
    this->plm_theta   = plm_theta;
    this->dt          = init_dt;
    this->bc          = boundary_cond_map.at(boundary_condition);
    this->geometry    = geometry_map.at(coord_system);
    this->xphysical_grid = (first_order) ? nx - 2: nx - 4;
    this->yphysical_grid = (first_order) ? ny - 2: ny - 4;
    this->zphysical_grid = (first_order) ? nz - 2: nz - 4;
    this->idx_active     = (first_order) ? 1     : 2;

    this->active_zones = xphysical_grid * yphysical_grid * zphysical_grid;

    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice3D(x1, x2, x3, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice3D(x1, x2, x3, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else
    {
        this->coord_lattice = CLattice3D(x1, x2, x3, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    if (coord_lattice.x2vertices[yphysical_grid] == PI){
        bipolar = true;
    }
    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.x1max      = x1[xphysical_grid - 1];
    setup.x1min      = x1[0];
    setup.x2max      = x2[yphysical_grid - 1];
    setup.x2min      = x2[0];
    setup.zmax      = x3[zphysical_grid - 1];
    setup.zmin      = x3[0];
    setup.nx        = nx;
    setup.ny        = ny;
    setup.nz        = nz;
    setup.linspace  = linspace;
    setup.ad_gamma  = gamma;

    cons.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);

    // Define the source terms
    sourceD   = sources[0];
    sourceS1  = sources[1];
    sourceS2  = sources[2];
    sourceS3  = sources[3];
    sourceTau = sources[4];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state3D[0].size(); i++)
    {
        auto D            = state3D[0][i];
        auto S1           = state3D[1][i];
        auto S2           = state3D[2][i];
        auto S3           = state3D[3][i];
        auto E            = state3D[4][i];
        auto S            = sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        cons[i]           = Conserved(D, S1, S2, S3, E);
        pressure_guess[i] = std::abs(S - D - E);
    }
    n = 0;
    // deallocate initial state vector
    std::vector<int> state3D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (tstart - engine_duration)));


    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr3d::PrimitiveData transfer_prims;

    // if (t == 0)
    // {
    //     config_ghosts2D(cons, nx, ny, first_order);
    // }
    // Copy the current SRHD instance over to the device

    SRHD3D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD3D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD3D));
    simbi::dual::DualSpace3D<Primitive, Conserved, SRHD3D> dualMem;
    dualMem.copyHostToDev(*this, device_self);
    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    luint tchunk_order_of_mag = 2;
    luint time_order_of_mag;

    const luint nxBlocks          = (nx + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;
    const luint nyBlocks          = (ny + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;
    const luint nzBlocks          = (nz + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;
    const luint physical_nxBlocks = (xphysical_grid + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;
    const luint physical_nyBlocks = (yphysical_grid + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;
    const luint physical_nzBlocks = (zphysical_grid + BLOCK_SIZE3D - 1) / BLOCK_SIZE3D;

    dim3 agridDim  = dim3(physical_nxBlocks, physical_nyBlocks, physical_nzBlocks); // active grid dimensions
    dim3 fgridDim  = dim3(nxBlocks, nyBlocks, nzBlocks);                            // full grid dimensions
    dim3 threadDim = dim3(BLOCK_SIZE3D, BLOCK_SIZE3D, BLOCK_SIZE3D);                // thread block dimensions

    const luint xblockdim         = xphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : xphysical_grid;
    const luint yblockdim         = yphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : yphysical_grid;
    const luint zblockdim         = zphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : zphysical_grid;
    const luint radius            = (first_order) ? 1 : 2;
    const luint shBlockSize       = BLOCK_SIZE3D + 2 * radius;
    const luint shBlockSpace      = shBlockSize * shBlockSize * shBlockSize;
    const luint shBlockBytes = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny, nz}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid, zphysical_grid}, 
                                                         {xblockdim, yblockdim, zblockdim}, shBlockBytes);

    if constexpr(BuildPlatform == Platform::GPU)
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
    else 
        cons2prim(fullP);

    simbi::gpu::api::deviceSynch();
    
    // Some benchmarking tools 
    luint      n   = 0;
    luint  nfold   = 0;
    luint  ncheck  = 0;
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
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts3D(fullP, device_self, nx, ny, nz, true, bc);
            } else {
                // First Half Step
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts3D(fullP, this, nx, ny, nz, true, bc);
            }
            
            t += dt; 

            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
                
                transfer_prims = vec2struct<sr3d::PrimitiveData, Primitive>(prims);
                writeToProd<sr3d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }

            n++;

            // std::cin.get();

            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
                adapt_dt(device_self, geometry, activeP);
            else 
                adapt_dt();
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
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts3D(fullP, device_self, nx, ny, nz, false, bc);

                // Final Half Step
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts3D(fullP, device_self, nx, ny, nz, false, bc);
            }  else {
                // First Half Step
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts3D(fullP, this, nx, ny, nz, false, bc);

                // Final Half Step
                advance(device_self, activeP, shBlockSize, radius, geometry, simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts3D(fullP, this, nx, ny, nz, false, bc);
            }   
            
            t += dt; 

            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
                
                transfer_prims = vec2struct<sr3d::PrimitiveData, Primitive>(prims);
                writeToProd<sr3d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 3, total_zones);
                t_interval += chkpt_interval;
            }
            n++;

            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
                adapt_dt(device_self, geometry, activeP);
            else 
                adapt_dt();
            // Update decay constant
            decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (t - engine_duration)));

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

    transfer_prims = vec2struct<sr3d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(5, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.v3;
    solution[4] = transfer_prims.p;

    return solution;
};
