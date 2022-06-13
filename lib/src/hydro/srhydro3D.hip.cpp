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
    coord_system(coord_system),
    inFailureState(false)
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
                D   = cons[idx].d;     // Relativistic Mass Density
                S1  = cons[idx].s1;   // X1-Momentum Denity
                S2  = cons[idx].s2;   // X2-Momentum Density
                S3  = cons[idx].s3;   // X2-Momentum Density
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
                    Ws = static_cast<real>(1.0) / sqrt(static_cast<real>(1.0) - v2);
                    rhos = D / Ws;
                    eps = (tau + D * (static_cast<real>(1.0) - Ws) + (static_cast<real>(1.0) - Ws * Ws) * p) / (D * Ws);
                    f = (gamma - static_cast<real>(1.0)) * rhos * eps - p;

                    h = static_cast<real>(1.0) + eps + p / rhos;
                    c2 = gamma * p / (h * rhos);
                    g = c2 * v2 - static_cast<real>(1.0);
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
                Ws = static_cast<real>(1.0) / sqrt(static_cast<real>(1.0) - (v1 * v1 + v2 * v2 + v3 * v3));

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
    const auto xpg = xphysical_grid;
    const auto ypg = yphysical_grid;
    const auto zpg = zphysical_grid;
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;
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
        real D    = conserved_buff[tid].d;
        real S1   = conserved_buff[tid].s1;
        real S2   = conserved_buff[tid].s2;
        real S3   = conserved_buff[tid].s3;
        real tau  = conserved_buff[tid].tau;
        real S    = sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        
        #if GPU_CODE
        real peq = self->gpu_pressure_guess[gid];
        #else 
        real peq = pressure_guess[gid];
        #endif

        real tol = D * tol_scale;
        #if GPU_CODE
        if (tid == 0) found_failure = self->inFailureState;
        simbi::gpu::api::synchronize();
        #else 
        found_failure = self->inFailureState;
        #endif
            
        while (!found_failure && workLeftToDo)
        {
            if (tid == 0 && self->inFailureState) 
                found_failure = true;

            do
            {
                pre = peq;
                et  = tau + D + pre;
                v2 = S * S / (et * et);
                W   = static_cast<real>(1.0) / sqrt(static_cast<real>(1.0) - v2);
                rho = D / W;

                eps = (tau + (static_cast<real>(1.0) - W) * D + (static_cast<real>(1.0) - W * W) * pre) / (D * W);

                h = static_cast<real>(1.0) + eps + pre / rho;
                c2 = self->gamma * pre / (h * rho);

                g = c2 * v2 - static_cast<real>(1.0);
                f = (self->gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                iter++;
                if (iter >= MAX_ITER)
                {
                    const auto kk  = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z: simbi::detail::get_height(gid, xpg, ypg);
                    const auto jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: simbi::detail::get_row(gid, xpg, ypg, kk);
                    const auto ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: simbi::detail::get_column(gid, xpg, ypg, kk);
                    printf("\nCons2Prim cannot converge\n");
                    printf("Density: %f, Pressure: %f, Vsq: %f, xindex: %lu, yindex: %lu, zindex: %lu\n", rho, peq, v2, ii, jj, kk);
                    found_failure        = true;
                    self->inFailureState = true;
                    simbi::gpu::api::synchronize();
                    break;
                }

            } while (std::abs(peq - pre) >= tol);

            real inv_et = static_cast<real>(1.0) / (tau + D + peq); 
            real vx = S1 * inv_et;
            real vy = S2 * inv_et;
            real vz = S3 * inv_et;
            
            #if GPU_CODE
                self->gpu_pressure_guess[gid] = peq;
                self->gpu_prims[gid]          = Primitive{rho, vx, vy, vz, peq};
            #else
                pressure_guess[gid] = peq;
                prims[gid]          = Primitive{rho, vx, vy, vz,  peq};
            #endif
            workLeftToDo = false;
        }
    });
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD3D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const luint nhat)
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
Conserved SRHD3D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real vz = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / sqrt(static_cast<real>(1.0) - (vx * vx + vy * vy + vz * vz));
    const real h = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - 1));

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

//     real D = state.d;
//     real S1 = state.s1;
//     real S2 = state.s2;
//     real tau = state.tau;
//     real E = tau + D;

//     switch (nhat)
//     {
//     case 1:
//         cofactor = static_cast<real>(1.0) / (a - aStar);
//         Dstar = cofactor * (a - v1) * D;
//         S1star = cofactor * (S1 * (a - v1) - pressure + pStar);
//         S2star = cofactor * (a - v1) * S2;
//         Estar = cofactor * (E * (a - v1) + pStar * aStar - pressure * v1);
//         tauStar = Estar - Dstar;

//         starStates = Conserved(Dstar, S1star, S2star, tauStar);

//         return starStates;
//     case 2:
//         cofactor = static_cast<real>(1.0) / (a - aStar);
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

                    h = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - 1.));
                    cs = sqrt(gamma * pressure / (rho * h));

                    plus_v1  = (v1 + cs) / (static_cast<real>(1.0) + v1 * cs);
                    plus_v2  = (v2 + cs) / (static_cast<real>(1.0) + v2 * cs);
                    plus_v3  = (v3 + cs) / (static_cast<real>(1.0) + v3 * cs);
                    minus_v1 = (v1 - cs) / (static_cast<real>(1.0) - v1 * cs);
                    minus_v2 = (v2 - cs) / (static_cast<real>(1.0) - v2 * cs);
                    minus_v3 = (v3 - cs) / (static_cast<real>(1.0) - v3 * cs);

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

void SRHD3D::adapt_dt(SRHD3D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, const luint bytes)
{
    #if GPU_CODE
    {
        luint psize = p.blockSize.x*p.blockSize.y;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                compute_dt<SRHD3D, Primitive><<<p.gridSize,p.blockSize, bytes>>>
                (dev, geometry, psize, dx1, dx2, dx3);
                dtWarpReduce<SRHD3D, Primitive, 4><<<p.gridSize,p.blockSize,bytes>>>
                (dev);
                break;
            
            case simbi::Geometry::SPHERICAL:
                compute_dt<SRHD3D, Primitive><<<p.gridSize,p.blockSize, bytes>>>
                (dev, geometry, psize, dlogx1, dx2, dx3, x1min, x1max, x2min, x2max, x3min, x3max);
                dtWarpReduce<SRHD3D, Primitive, 4><<<p.gridSize,p.blockSize,bytes>>>
                (dev);
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
Conserved SRHD3D::calc_Flux(const Primitive &prims,   luint nhat = 1)
{

    const real rho      = prims.rho;
    const real vx       = prims.v1;
    const real vy       = prims.v2;
    const real vz       = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / sqrt(static_cast<real>(1.0) - (vx * vx + vy * vy + vz*vz));

    const real h  = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));
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
    const real aLminus = aL < static_cast<real>(0.0) ? aL : static_cast<real>(0.0);
    const real aRplus  = aR > static_cast<real>(0.0) ? aR : static_cast<real>(0.0);

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
    if (static_cast<real>(0.0) <= aL)
    {
        return left_flux;
    }
    else if (static_cast<real>(0.0) >= aR)
    {
        return right_flux;
    }

    const real aLminus = aL < static_cast<real>(0.0) ? aL : static_cast<real>(0.0);
    const real aRplus  = aR > static_cast<real>(0.0) ? aR : static_cast<real>(0.0);

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aR - left_state * aL - right_flux + left_flux) / (aR - aL);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux 
        = (left_flux * aRplus - right_flux * aLminus + (right_state - left_state) * aRplus * aLminus) 
            / (aRplus - aLminus);

    //------ Mignone & Bodo subtract off the rest mass density
    const real e  = hll_state.tau + hll_state.d;
    const real s  = hll_state.momentum(nhat);
    const real fe = hll_flux.tau + hll_flux.d;
    const real fs = hll_flux.momentum(nhat);

    //------Calculate the contact wave velocity and pressure
    const real a = fe;
    const real b = -(e + fs);
    const real c = s;
    const real quad = -static_cast<real>(0.5) * (b + sgn(b) * sqrt(b * b - 4.0 * a * c));
    const real aStar = c * (static_cast<real>(1.0) / quad);
    const real pStar = -aStar * fe + fs;

    // return Conserved(0.0, 0.0, 0.0, 0.0);
    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D = left_state.d;
        const real S1 = left_state.s1;
        const real S2 = left_state.s2;
        const real S3 = left_state.s3;
        const real tau = left_state.tau;
        const real E = tau + D;
        const real cofactor = static_cast<real>(1.0) / (aL - aStar);
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
        const real D  = right_state.d;
        const real S1 = right_state.s1;
        const real S2 = right_state.s2;
        const real S3 = right_state.s3;
        const real tau = right_state.tau;
        const real E = tau + D;
        const real cofactor = static_cast<real>(1.0) / (aR - aStar);

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
    const luint bx,
    const luint by,
    const luint bz,
    const luint radius, 
    const simbi::Geometry geometry, 
    const simbi::MemSide user)
{
    auto *self = (BuildPlatform == Platform::GPU) ? dev : this;
    const luint xpg                 = this->xphysical_grid;
    const luint ypg                 = this->yphysical_grid;
    const luint zpg                 = this->zphysical_grid;
    const bool is_first_order       = this->first_order;
    const bool is_periodic          = this->periodic;
    const bool hllc                 = this->hllc;
    const real dt                   = this->dt;
    const real decay_const          = this->decay_const;
    const real plm_theta            = this->plm_theta;
    const real gamma                = this->gamma;
    const real dx1                  = this->dx1;
    const real dx2                  = this->dx2;
    const real dx3                  = this->dx3;
    const real dlogx1               = this->dlogx1;
    const real x1min                = this->x1min;
    const real x1max                = this->x1max;
    const real x2min                = this->x2min;
    const real x2max                = this->x2max;
    const real x3min                = this->x3min;
    const real x3max                = this->x3max;
    const luint nx                  = this->nx;
    const luint ny                  = this->ny;
    const luint nz                  = this->nz;
    const bool d_all_zeros          = this->d_all_zeros;
    const bool s1_all_zeros         = this->s1_all_zeros;
    const bool s2_all_zeros         = this->s2_all_zeros;
    const bool s3_all_zeros         = this->s3_all_zeros;
    const bool e_all_zeros          = this->e_all_zeros;
    const luint extent              = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.z * p.gridSize.z * p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const luint xextent             = p.blockSize.x;
    const luint yextent             = p.blockSize.y;
    const luint zextent             = p.blockSize.z;
    const CLattice3D *coord_lattice = &(self->coord_lattice);

    // Choice of column major striding by user
    const luint sx = (col_maj) ? 1  : bx;
    const luint sy = (col_maj) ? by :  1;
    const luint sz = (col_maj) ? bz :  1;

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
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x : 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y : 0;
        const luint tz  = (BuildPlatform == Platform::GPU) ? threadIdx.z : 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;
        const luint tza = (BuildPlatform == Platform::GPU) ? tz + radius : ka;

        Conserved ux_l, ux_r, uy_l, uy_r, uz_l, uz_r;
        Conserved f_l, f_r, g_l, g_r, h_l, h_r, frf, flf, grf, glf, hrf, hlf;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r, zprims_l, zprims_r;

        luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            luint tzl = zextent;

            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tza * bx * by + tya * bx + txa] = self->gpu_prims[aid];
            if (tz < radius)    
            {
                if (ka + zextent > nz - 1) tzl = nz - radius - ka + tz;
                prim_buff[(tza - radius) * bx * by + tya * bx + txa] = self->gpu_prims[(ka - radius) * nx * ny + ja * nx + ia];
                prim_buff[(tza + tzl   ) * bx * by + tya * bx + txa] = self->gpu_prims[(ka + tzl   ) * nx * ny + ja * nx + ia];

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if (radius == 2)
                {\
                    if (tzl == 1)
                    {
                        prim_buff[(tza + 1 - radius) * bx * by + tya * bx + txa] =  self->gpu_prims[(ka + 1 - radius) * nx * ny + ja * nx + ia];
                        prim_buff[(tza + 1 + tzl   ) * bx * by + tya * bx + txa] =  self->gpu_prims[(ka + 1 + tzl   ) * nx * ny + ja * nx + ia]; 
                    }
                    prim_buff[(tza + tzl - 1) * bx * by + tya * bx + txa] =  self->gpu_prims[(ka + tzl - 1) * nx * ny + ja * nx + ia]; 
                }  
            }
            if (ty < radius)    
            {
                if (ja + yextent > ny - 1) tyl = ny - radius - ja + ty;
                prim_buff[tza * bx * by + (tya - radius) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja - radius) * nx + ia];
                prim_buff[tza * bx * by + (tya + tyl   ) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja + tyl   ) * nx + ia];

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if (radius == 2)
                {
                    if (tyl == 1)
                    {
                        prim_buff[tza * bx * by + (tya + 1 - radius) * bx + txa] =  self->gpu_prims[ka * nx * ny + ((ja + 1 - radius) * nx) + ia];
                        prim_buff[tza * bx * by + (tya + 1 + tyl) * bx + txa]    =  self->gpu_prims[ka * nx * ny + ((ja + 1 + txl   ) * nx) + ia]; 
                    }
                    prim_buff[tza * bx * by + (tya + tyl - 1) * bx + txa]    =  self->gpu_prims[ka * nx * ny + ((ja + txl - 1) * nx) + ia]; 

                } 
            }
            if (tx < radius)
            {   
                if (ia + xextent > nx - 1) txl = nx - radius - ia + tx;
                // printf("ia: %lu, val: %lu, txl: %lu, txa + txl: %lu\n", ia, ia + xextent, txl, txa + txl);
                prim_buff[tza * bx * by + tya * bx + txa - radius] =  self->gpu_prims[ka * nx * ny + ja * nx + ia - radius];
                prim_buff[tza * bx * by + tya * bx + txa +    txl] =  self->gpu_prims[ka * nx * ny + ja * nx + ia + txl]; 

                // Sometimes there's a single active thread, so we have to load all 5 zones immediately
                if (radius == 2)
                {
                    if (txl == 1)
                    {
                        prim_buff[tza * bx * by + tya * bx + txa + 1 - radius] =  self->gpu_prims[ka * nx * ny + ja * nx + ia + 1 - radius];
                        prim_buff[tza * bx * by + tya * bx + txa + 1 +    txl] =  self->gpu_prims[ka * nx * ny + ja * nx + ia + 1 + txl]; 
                    }
                    prim_buff[tza * bx * by + tya * bx + txa + txl - 1] = self->gpu_prims[ka * nx * ny + ja * nx + ia + txl - 1]; 
                }
            }
            simbi::gpu::api::synchronize();
        #endif
        
        // if (ia == nx - 3)
        // {
        //     printf("[%lu, %lu, txl=%lu, xext: %lu] LL: %3.e, L: %3.e, C: %3.e, R: %3.e, RR: %3.e\n", ia, txa, txl, xextent, 
        //     prim_buff[tza * bx * by + tya * bx + txa - 2].rho,
        //     prim_buff[tza * bx * by + tya * bx + txa - 1].rho,
        //     prim_buff[tza * bx * by + tya * bx + txa - 0].rho,
        //     prim_buff[tza * bx * by + tya * bx + txa + 1].rho,
        //     prim_buff[tza * bx * by + tya * bx + txa + 2].rho);
        // }
        if (is_first_order)
        {

            xprims_l = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
            xprims_r = prim_buff[tza * bx * by + tya * bx + (txa + 1)];
            //j+1/2
            yprims_l = prim_buff[tza * bx * by + (tya + 0) * bx + txa];
            yprims_r = prim_buff[tza * bx * by + (tya + 1) * bx + txa];
            //j+1/2
            zprims_l = prim_buff[(tza + 0) * bx * by + tya * bx + txa];
            zprims_r = prim_buff[(tza + 1) * bx * by + tya * bx + txa];

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
            if (self->hllc)
            {
                frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hrf = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);

            } else {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hrf = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

            // Set up the left and right state interfaces for i-1/2
            xprims_l = prim_buff[tza * bx * by + tya * bx + (txa - 1)];
            xprims_r = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
            //j+1/2
            yprims_l = prim_buff[tza * bx * by + (tya - 1) * bx + txa]; 
            yprims_r = prim_buff[tza * bx * by + (tya + 0) * bx + txa]; 
            //k+1/2
            zprims_l = prim_buff[(tza - 1) * bx * by + tya * bx + txa]; 
            zprims_r = prim_buff[(tza - 0) * bx * by + tya * bx + txa]; 

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
                flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hlf = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);

            } else {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hlf = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }   
        }
        else
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            Primitive zleft_most, zright_most, zleft_mid, zright_mid;

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

            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprims_l =  center +  minmod((center - xleft_mid) * plm_theta,
                               (xright_mid - xleft_mid)  * static_cast<real>(0.5),
                                (xright_mid - center) * plm_theta) * static_cast<real>(0.5);

            // Reconstructed right Primitive vector in x
            xprims_r =  xright_mid - minmod((xright_mid - center) * plm_theta, 
                                            (xright_most - center) * static_cast<real>(0.5),
                                            (xright_most - xright_mid) * plm_theta) * static_cast<real>(0.5);

            // Reconstructed right Primitive vector in y-direction at j+1/2
            // interfce
            yprims_l = center + minmod((center - yleft_mid) * plm_theta,
                                        (yright_mid - yleft_mid) * static_cast<real>(0.5),
                                        (yright_mid - center) * plm_theta) * static_cast<real>(0.5);

            yprims_r = yright_mid - minmod((yright_mid - center) * plm_theta,
                                           (yright_most - center) * static_cast<real>(0.5),
                                           (yright_most - yright_mid) * plm_theta) * static_cast<real>(0.5);

            // Reconstructed right Primitive vector in z-direction at j+1/2
            // interfce
            zprims_l = center + minmod((center - zleft_mid) * plm_theta,
                                       (zright_mid - zleft_mid) * static_cast<real>(0.5),
                                       (zright_mid - center) * plm_theta) * static_cast<real>(0.5);

            zprims_r = zright_mid -  minmod((zright_mid - center) * plm_theta,
                                            (zright_most - center) * static_cast<real>(0.5),
                                            (zright_most - zright_mid) * plm_theta) * static_cast<real>(0.5);

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
                frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hrf = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }
            else
            {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hrf = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

            // Do the same thing, but for the left side interface [i - 1/2]

            // Left side Primitive in x
            xprims_l = xleft_mid +
                            minmod((xleft_mid - xleft_most) * plm_theta,
                                        (center - xleft_most) * static_cast<real>(0.5),
                                        (center - xleft_mid) * plm_theta) * static_cast<real>(0.5);

            // Right side Primitive in x
            xprims_r =
                center - minmod((center - xleft_mid) * plm_theta,
                                            (xright_mid - xleft_mid) * static_cast<real>(0.5),
                                            (xright_mid - center) * plm_theta) * static_cast<real>(0.5);

            // Left side Primitive in y
            yprims_l = yleft_mid +
                            minmod((yleft_mid - yleft_most) * plm_theta,
                                        (center - yleft_most) * static_cast<real>(0.5),
                                        (center - yleft_mid) * plm_theta) * static_cast<real>(0.5);

            // Right side Primitive in y
            yprims_r = center - minmod((center - yleft_mid) * plm_theta,
                                            (yright_mid - yleft_mid) * static_cast<real>(0.5),
                                            (yright_mid - center) * plm_theta) * static_cast<real>(0.5);

            // Left side Primitive in z
            zprims_l = zleft_mid +
                            minmod((zleft_mid - zleft_most) * plm_theta,
                                        (center - zleft_most) * static_cast<real>(0.5),
                                        (center - zleft_mid) * plm_theta) * static_cast<real>(0.5);

            // Right side Primitive in z
            zprims_r = center - minmod((center - zleft_mid) * plm_theta,
                                        (zright_mid - zleft_mid * static_cast<real>(0.5)),
                                        (zright_mid - center) * plm_theta) * static_cast<real>(0.5);
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
                flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hlf = self->calc_hllc_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }
            else
            {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                hlf = self->calc_hll_flux(uz_l, uz_r, h_l, h_r, zprims_l, zprims_r, 3);
            }

        }// end else 

        //Advance depending on geometry
        const luint real_loc =  kk * xpg * ypg + jj * xpg + ii;
        const auto step = (is_first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    #if GPU_CODE
                        real dx1 = coord_lattice->gpu_dx1[ii];
                        real dx2  = coord_lattice->gpu_dx2[jj];
                        real dx3 = coord_lattice->gpu_dx3[kk];
                        self->gpu_cons[aid].d   += step * dt * ( -(frf.d - flf.d)     / dx1 - (grf.d   - glf.d )  / dx2 - (hrf.d - hlf.d)     / dx3 + self->gpu_sourceD   [real_loc] );
                        self->gpu_cons[aid].s1  += step * dt * ( -(frf.s1 - flf.s1)   / dx1 - (grf.s1  - glf.s1)  / dx2 - (hrf.s1 - hlf.s3)   / dx3 + self->gpu_sourceS1  [real_loc] );
                        self->gpu_cons[aid].s2  += step * dt * ( -(frf.s2 - flf.s2)   / dx1  - (grf.s2  - glf.s2) / dx2 - (hrf.s2 - hlf.s3)   / dx3 + self->gpu_sourceS2  [real_loc] );
                        self->gpu_cons[aid].s3  += step * dt * ( -(frf.s3 - flf.s3)   / dx1  - (grf.s3  - glf.s3) / dx2 - (hrf.s3 - hlf.s3)   / dx3 + self->gpu_sourceS3  [real_loc] );
                        self->gpu_cons[aid].tau += step * dt * ( -(frf.tau - flf.tau) / dx1 - (grf.tau - glf.tau) / dx2 - (hrf.tau - hlf.tau) / dx3 + self->gpu_sourceTau [real_loc] );
                    #else
                        real dx1 = self->coord_lattice.dx1[ii];
                        real dx2  = self->coord_lattice.dx2[jj];
                        real dx3 = self->coord_lattice.dx3[kk];
                        cons[aid].d   += step * dt * ( -(frf.d - flf.d)     / dx1 - (grf.d   - glf.d )  / dx2 - (hrf.d - hlf.d)     / dx3 + sourceD   [real_loc] );
                        cons[aid].s1  += step * dt * ( -(frf.s1 - flf.s1)   / dx1 - (grf.s1  - glf.s1)  / dx2 - (hrf.s1 - hlf.s3)   / dx3 + sourceS1  [real_loc] );
                        cons[aid].s2  += step * dt * ( -(frf.s2 - flf.s2)   / dx1  -(grf.s2  - glf.s2)  / dx2 - (hrf.s2 - hlf.s3)   / dx3 + sourceS2  [real_loc] );
                        cons[aid].s3  += step * dt * ( -(frf.s3 - flf.s3)   / dx1  -(grf.s3  - glf.s3)  / dx2 - (hrf.s3 - hlf.s3)   / dx3 + sourceS3  [real_loc] );
                        cons[aid].tau += step * dt * ( -(frf.tau - flf.tau) / dx1 - (grf.tau - glf.tau) / dx2 - (hrf.tau - hlf.tau) / dx3 + sourceTau [real_loc] );
                    #endif
                    
                break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                const real rl           = (ii > 0 ) ? x1min * pow(10, (ii -static_cast<real>(0.5)) * dlogx1) :  x1min;
                const real rr           = (ii < xpg - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                const real tl           = (jj > 0 ) ? x2min + (jj - static_cast<real>(0.5)) * dx2 :  x2min;
                const real tr           = (jj < ypg - 1) ? tl + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                const real ql           = (kk > 0 ) ? x3min + (kk - static_cast<real>(0.5)) * dx3 :  x3min;
                const real qr           = (kk < zpg - 1) ? ql + dx3 * (kk == 0 ? 0.5 : 1.0) :  x3max; 
                const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real s1R          = rr * rr; 
                const real s1L          = rl * rl; 
                const real s2R          = std::sin(tr);
                const real s2L          = std::sin(tl);
                const real thmean       = static_cast<real>(0.5) * (tl + tr);
                const real sint         = std::sin(thmean);
                const real dV1          = rmean * rmean * (rr - rl);             
                const real dV2          = rmean * sint  * (tr - tl); 
                const real dV3          = rmean * sint  * (qr - ql); 
                const real cot          = std::cos(thmean) / sint;

                const real d_source  = (d_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceD[real_loc];
                const real s1_source = (s1_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS1[real_loc];
                const real s2_source = (s2_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS2[real_loc];
                const real s3_source = (s3_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS3[real_loc];
                const real e_source  = (e_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceTau[real_loc];

                // Grab central primitives
                const real rhoc = prim_buff[txa + tya * bx + tza * bx * by].rho;
                const real uc   = prim_buff[txa + tya * bx + tza * bx * by].v1;
                const real vc   = prim_buff[txa + tya * bx + tza * bx * by].v2;
                const real wc   = prim_buff[txa + tya * bx + tza * bx * by].v3;
                const real pc   = prim_buff[txa + tya * bx + tza * bx * by].p;

                const real hc   = static_cast<real>(1.0) + gamma * pc/(rhoc * (gamma - static_cast<real>(1.0)));
                const real gam2 = static_cast<real>(1.0)/(static_cast<real>(1.0) - (uc * uc + vc * vc + wc * wc));

                const Conserved geom_source  = {static_cast<real>(0.0), (rhoc * hc * gam2 * (vc * vc + wc * wc)) / rmean + pc * (s1R - s1L) / dV1, - (rhoc * hc * gam2 * uc * vc) / rmean + pc * (s2R - s2L)/dV2 , - rhoc * hc * gam2 * wc * (uc + vc * cot) / rmean, static_cast<real>(0.0)};
                const Conserved source_terms = Conserved{d_source, s1_source, s2_source, s3_source, e_source} * decay_const;

                #if GPU_CODE 
                    self->gpu_cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 + (hrf - hlf) / dV3 - geom_source - source_terms) * dt * step;
                #else
                    cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 + (hrf - hlf) / dV3 - geom_source - source_terms) * dt * step;
                #endif
                
                break;
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
        } // end switch

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
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

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

    dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    dx3     = (x3[zphysical_grid - 1] - x3[0]) / (zphysical_grid - 1);
    x1min   = x1[0];
    x1max   = x1[xphysical_grid - 1];
    x2min   = x2[0];
    x2max   = x2[yphysical_grid - 1];
    x3min   = x3[0];
    x3max   = x3[zphysical_grid - 1];

    d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    s3_all_zeros = std::all_of(sourceS3.begin(),  sourceS3.end(),  [](real i) {return i == 0;});
    e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});

    if (coord_lattice.x2vertices[yphysical_grid] == PI){
        bipolar = true;
    }
    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.x1max          = x1[xphysical_grid - 1];
    setup.x1min          = x1[0];
    setup.x2max          = x2[yphysical_grid - 1];
    setup.x2min          = x2[0];
    setup.zmax           = x3[zphysical_grid - 1];
    setup.zmin           = x3[0];
    setup.nx             = nx;
    setup.ny             = ny;
    setup.nz             = nz;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;
    
    cons.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);

    // Define the source terms
    sourceD   = sources[0];
    sourceS1  = sources[1];
    sourceS2  = sources[2];
    sourceS3  = sources[3];
    sourceTau = sources[4];

    d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    s3_all_zeros = std::all_of(sourceS3.begin(),  sourceS3.end(),  [](real i) {return i == 0;});
    e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state3D[0].size(); i++)
    {
        auto D            = state3D[0][i];
        auto S1           = state3D[1][i];
        auto S2           = state3D[2][i];
        auto S3           = state3D[3][i];
        auto E            = state3D[4][i];
        auto S            = sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        cons[i]           = Conserved{D, S1, S2, S3, E};
        pressure_guess[i] = std::abs(S - D - E);
    }
    n = 0;
    // deallocate initial state vector
    std::vector<int> state3D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (tstart - engine_duration)));


    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr3d::PrimitiveData transfer_prims;

    SRHD3D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD3D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD3D));
    simbi::dual::DualSpace3D<Primitive, Conserved, SRHD3D> dualMem;
    dualMem.copyHostToDev(*this, device_self);
    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // // Setup the system
    const luint xblockdim       = xphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : xphysical_grid;
    const luint yblockdim       = yphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : yphysical_grid;
    const luint zblockdim       = zphysical_grid > BLOCK_SIZE3D ? BLOCK_SIZE3D : zphysical_grid;
    const luint radius          = (periodic) ? 0 : (first_order) ? 1 : 2;
    const luint bx              = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint by              = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint bz              = (BuildPlatform == Platform::GPU) ? zblockdim + 2 * radius: nz;
    const luint shBlockSpace    = bx * by * bz;
    const luint shBlockBytes    = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny, nz}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid, zphysical_grid}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    
    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU)
        {
            config_ghosts3D(fullP, device_self, nx, ny, nz, first_order, bc);
        } else {
            config_ghosts3D(fullP, this, nx, ny, nz, first_order, bc);
        }
    }

    const auto dtShBytes = zblockdim * xblockdim * yblockdim * sizeof(Primitive) + zblockdim * xblockdim * yblockdim * sizeof(real);
    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, geometry, activeP, dtShBytes);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
        transfer_prims = vec2struct<sr3d::PrimitiveData, Primitive>(prims);
        writeToProd<sr3d::PrimitiveData, Primitive>(&transfer_prims, &prods);
        tnow = create_step_str(t_interval, tchunk);
        filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
        setup.t = t;
        setup.dt = dt;
        write_hdf5(data_directory, filename, prods, setup, 3, total_zones);
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

    // Simulate :)
    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            advance(self, activeP, bx, by, bz, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, true, bc);
            t += dt; 
            

            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\rIteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            else 
                adapt_dt();

            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));
        }
    } else {
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();

            // First half step
            advance(self, activeP, bx, by, bz,  radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, false, bc);

            // Final half step
            advance(self, activeP, bx, by, bz,  radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, false, bc); 
            t += dt; 

            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\rIteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            else 
                adapt_dt();

            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));
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

    transfer_prims = vec2struct<sr3d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(5, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.v3;
    solution[4] = transfer_prims.p;

    return solution;
};
