/*
 * C++ Source to perform 3D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "util/device_api.hpp"
#include "util/dual.hpp"
#include "util/parallel_for.hpp"
#include "util/printb.hpp"
#include "common/helpers.hip.hpp"
#include "srhydro3D.hip.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;


/* Define typedefs because I am lazy */
using Primitive           = sr3d::Primitive;
using Conserved           = sr3d::Conserved;
using Eigenvals           = sr3d::Eigenvals;
using dualType            = simbi::dual::DualSpace3D<Primitive, Conserved, SRHD3D>;
constexpr auto write2file = helpers::write_to_file<simbi::SRHD3D, sr3d::PrimitiveData, Primitive, dualType, 3>;

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
    state3D(state3D),
    nx(nx),
    ny(ny),
    nz(nz),
    gamma(gamma),
    x1(x1),
    x2(x2),
    x3(x3),
    cfl(cfl),
    coord_system(coord_system),
    inFailureState(false),
    nzones(state3D[0].size())
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
void SRHD3D::cons2prim(
    ExecutionPolicy<> p, 
    SRHD3D *dev, 
    simbi::MemSide user)
{
    const luint xpg    = xphysical_grid;
    const luint ypg    = yphysical_grid;
    // const luint zpg    = zphysical_grid;
    const luint radius = (first_order) ? 1 : 2;
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;
        #if GPU_CODE 
        auto* const conserved_buff = self->gpu_cons;
        #else 
        auto *const conserved_buff = &cons[0];
        #endif 

        luint iter  = 0;
        real D    = conserved_buff[gid].d;
        real S1   = conserved_buff[gid].s1;
        real S2   = conserved_buff[gid].s2;
        real S3   = conserved_buff[gid].s3;
        real tau  = conserved_buff[gid].tau;
        real S    = std::sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        
        #if GPU_CODE
        auto tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
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
            do
            {
                pre = peq;
                et  = tau + D + pre;
                v2 = S * S / (et * et);
                W   = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - v2);
                rho = D / W;

                eps = (tau + (static_cast<real>(1.0) - W) * D + (static_cast<real>(1.0) - W * W) * pre) / (D * W);

                h = static_cast<real>(1.0) + eps + pre / rho;
                c2 = self->gamma * pre / (h * rho);

                g = c2 * v2 - static_cast<real>(1.0);
                f = (self->gamma - static_cast<real>(1.0)) * rho * eps - pre;

                peq = pre - f / g;
                iter++;
                if (iter >= MAX_ITER || std::isnan(peq))
                {
                    const luint kk    = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z: simbi::detail::get_height(gid, xpg, ypg);
                    const luint jj    = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: simbi::detail::get_row(gid, xpg, ypg, kk);
                    const luint ii    = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: simbi::detail::get_column(gid, xpg, ypg, kk);
                    const lint ireal  = helpers::get_real_idx(ii, radius, self->xphysical_grid);
                    const lint jreal  = helpers::get_real_idx(jj, radius, self->yphysical_grid); 
                    const lint kreal  = helpers::get_real_idx(kk, radius, self->zphysical_grid); 
                    const real x1l    = self->get_x1face(ireal, self->geometry, 0);
                    const real x1r    = self->get_x1face(ireal, self->geometry, 1);
                    const real x2l    = self->get_x2face(jreal, 0);
                    const real x2r    = self->get_x2face(jreal, 1);
                    const real x3l    = self->get_x3face(kreal, 0);
                    const real x3r    = self->get_x3face(kreal, 1);
                    const real x1mean = helpers::calc_any_mean(x1l, x1r, self->x1cell_spacing);
                    const real x2mean = helpers::calc_any_mean(x2l, x2r, self->x2cell_spacing);
                    const real x3mean = helpers::calc_any_mean(x3l, x3r, self->x3cell_spacing);

                    printf("\nCons2Prim cannot converge\n");
                    printf("Density: %f, Pressure: %f, Vsq: %f, x1coord: %.2e, x2coord: %.2e, x3coord: %.2e\n", rho, peq, v2, x1mean, x2mean, x3mean);
                    found_failure        = true;
                    self->inFailureState = true;
                    simbi::gpu::api::synchronize();
                    break;
                }

            } while (std::abs(peq - pre) >= tol);

            real inv_et = static_cast<real>(1.0) / (tau + D + peq); 
            real v1 = S1 * inv_et;
            real v2 = S2 * inv_et;
            real v3 = S3 * inv_et;
            
            #if GPU_CODE
                self->gpu_pressure_guess[gid] = peq;
                self->gpu_prims[gid]          = Primitive{rho, v1, v2, v3, peq};
            #else
                pressure_guess[gid] = peq;
                prims[gid]          = Primitive{rho, v1, v2, v3,  peq};
            #endif
            workLeftToDo = false;
        }
    });
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD3D::calc_Eigenvals(
    const Primitive &primsL,
    const Primitive &primsR,
    const luint nhat)
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
            //--------Calc the wave speeds based on Mignone and Bodo (2005)
            const real sL = csL * csL * (static_cast<real>(1.0) / (gamma * gamma * (static_cast<real>(1.0) - csL * csL)));
            const real sR = csR  * csR  * (static_cast<real>(1.0) / (gamma * gamma * (static_cast<real>(1.0) - csR  * csR)));

            // Define temporaries to save computational cycles
            const real qfL   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sL);
            const real qfR   = static_cast<real>(1.0) / (static_cast<real>(1.0) + sR);
            const real sqrtR = std::sqrt(sR * (static_cast<real>(1.0)- vR  * vR  + sR));
            const real sqrtL = std::sqrt(sL * (static_cast<real>(1.0)- vL * vL + sL));

            const real lamLm = (vL - sqrtL) * qfL;
            const real lamRm = (vR  - sqrtR) * qfR;
            const real lamLp = (vL + sqrtL) * qfL;
            const real lamRp = (vR  + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

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
Conserved SRHD3D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real v1 = prims.v1;
    const real v2 = prims.v2;
    const real v3 = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (v1 * v1 + v2 * v2 + v3 * v3));
    const real h = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - 1));

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
void SRHD3D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real cfl_dt;
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
                const auto sint    = std::sin(x2[jj]);
                #pragma omp for nowait schedule(static) reduction(min:min_dt)
                for (luint ii = 0; ii < xphysical_grid; ii++)
                {
                    const auto shift_i  = ii + idx_active;
                    const auto aid      = shift_k * nx * ny + shift_j * nx + shift_i;
                    const auto rho      = prims[aid].rho;
                    const auto v1       = prims[aid].v1;
                    const auto v2       = prims[aid].v2;
                    const auto v3       = prims[aid].v3;
                    const auto pressure = prims[aid].p;
                    const auto h        = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - 1.));
                    const auto cs       = std::sqrt(gamma * pressure / (rho * h));

                    // Left/Right wave speeds
                    const auto plus_v1  = (v1 + cs) / (static_cast<real>(1.0) + v1 * cs);
                    const auto plus_v2  = (v2 + cs) / (static_cast<real>(1.0) + v2 * cs);
                    const auto plus_v3  = (v3 + cs) / (static_cast<real>(1.0) + v3 * cs);
                    const auto minus_v1 = (v1 - cs) / (static_cast<real>(1.0) - v1 * cs);
                    const auto minus_v2 = (v2 - cs) / (static_cast<real>(1.0) - v2 * cs);
                    const auto minus_v3 = (v3 - cs) / (static_cast<real>(1.0) - v3 * cs);

                    const auto x1l     = get_x1face(ii, geometry, 0);
                    const auto x1r     = get_x1face(ii, geometry, 1);
                    const auto dx1     = x1r - x1l; 
                    switch (geometry)
                    {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = std::min(
                                    {dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                     dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))),
                                     dx3 / (std::max(std::abs(plus_v3), std::abs(minus_v3)))});
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                        {
                            const auto rmean = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l) / (x1r * x1r * x1r - x1l * x1l *x1l);
                            const auto rproj = rmean * sint;
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
                            break;
                        }
                    case simbi::Geometry::CYLINDRICAL:
                        // TODO: Implement
                        break;
                    }
                        
                    min_dt = helpers::my_min(min_dt, cfl_dt);
                    
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
                deviceReduceKernel<SRHD3D, 3><<<p.gridSize,p.blockSize>>>(dev, active_zones);
                deviceReduceKernel<SRHD3D, 3><<<1,1024>>>(dev, p.gridSize.x * p.gridSize.y);
                break;
            
            case simbi::Geometry::SPHERICAL:
                compute_dt<SRHD3D, Primitive><<<p.gridSize,p.blockSize, bytes>>>
                (dev, geometry, psize, dlogx1, dx2, dx3, x1min, x1max, x2min, x2max, x3min, x3max);
                deviceReduceKernel<SRHD3D, 3><<<p.gridSize,p.blockSize>>>(dev, active_zones);
                deviceReduceKernel<SRHD3D, 3><<<1,1024>>>(dev, p.gridSize.x * p.gridSize.y);
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
Conserved SRHD3D::calc_Flux(const Primitive &prims, const luint nhat = 1)
{
    const real rho      = prims.rho;
    const real v1       = prims.v1;
    const real v2       = prims.v2;
    const real v3       = prims.v3;
    const real pressure = prims.p;
    const real lorentz_gamma = static_cast<real>(1.0) / std::sqrt(static_cast<real>(1.0) - (v1 * v1 + v2 * v2 + v3*v3));

    const real h  = static_cast<real>(1.0) + gamma * pressure / (rho * (gamma - static_cast<real>(1.0)));
    const real D  = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * v1;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * v2;
    const real S3 = rho * lorentz_gamma * lorentz_gamma * h * v3;
    const real tau =
                    rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return (nhat == 1) ? Conserved{D * v1, S1 * v1 + pressure, S2 * v1, S3 * v1,  (tau + pressure) * v1}
          :(nhat == 2) ? Conserved{D * v2, S1 * v2, S2 * v2 + pressure, S3 * v2,  (tau + pressure) * v2}
          :              Conserved{D * v3, S1 * v3, S2 * v3, S3 * v3 + pressure,  (tau + pressure) * v3};
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
    const real quad = -static_cast<real>(0.5) * (b + helpers::sgn(b) * std::sqrt(b * b - 4.0 * a * c));
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
    auto *self      = (BuildPlatform == Platform::GPU) ? dev : this;
    const luint xpg = this->xphysical_grid;
    const luint ypg = this->yphysical_grid;
    const luint zpg = this->zphysical_grid;

    #if GPU_CODE
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
    const luint xextent             = p.blockSize.x;
    const luint yextent             = p.blockSize.y;
    const luint zextent             = p.blockSize.z;
    #endif 
    const luint extent              = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.z * p.gridSize.z * p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const auto step                 = (first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
    // Choice of column major striding by user
    // const luint sx = (col_maj) ? 1  : bx;
    // const luint sy = (col_maj) ? by :  1;
    // const luint sz = (col_maj) ? bz :  1;

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

        Conserved uxL, uxR, uyL, uyR, uzL, uzR;
        Conserved fL, fR, gL, gR, hL, hR, frf, flf, grf, glf, hrf, hlf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR, zprimsL, zprimsR;

        luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            luint tzl = zextent;

            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tza * bx * by + tya * bx + txa] = self->gpu_prims[aid];
            if (tz < radius)    
            {
                if (blockIdx.z == p.gridSize.z - 1 && (ka + zextent > nz - radius + tz)) {
                    tzl = nz - radius - ka + tz;
                }
                prim_buff[(tza - radius) * bx * by + tya * bx + txa] = self->gpu_prims[(ka - radius) * nx * ny + ja * nx + ia];
                prim_buff[(tza + tzl   ) * bx * by + tya * bx + txa] = self->gpu_prims[(ka + tzl   ) * nx * ny + ja * nx + ia];
            }
            if (ty < radius)    
            {
                if (blockIdx.y == p.gridSize.y - 1 && (ja + yextent > ny - radius + ty)) {
                    tyl = ny - radius - ja + ty;
                }
                prim_buff[tza * bx * by + (tya - radius) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja - radius) * nx + ia];
                prim_buff[tza * bx * by + (tya + tyl   ) * bx + txa] = self->gpu_prims[ka * nx * ny + (ja + tyl   ) * nx + ia];
            }
            if (tx < radius)
            {   
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                prim_buff[tza * bx * by + tya * bx + txa - radius] =  self->gpu_prims[ka * nx * ny + ja * nx + ia - radius];
                prim_buff[tza * bx * by + tya * bx + txa +    txl] =  self->gpu_prims[ka * nx * ny + ja * nx + ia + txl]; 
            }
            simbi::gpu::api::synchronize();
        #endif
        
        if (self->first_order){
            xprimsL = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
            xprimsR  = prim_buff[tza * bx * by + tya * bx + (txa + 1)];
            //j+1/2
            yprimsL = prim_buff[tza * bx * by + (tya + 0) * bx + txa];
            yprimsR  = prim_buff[tza * bx * by + (tya + 1) * bx + txa];
            //j+1/2
            zprimsL = prim_buff[(tza + 0) * bx * by + tya * bx + txa];
            zprimsR  = prim_buff[(tza + 1) * bx * by + tya * bx + txa];

            uxL = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);

            uyL = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);

            uzL = self->prims2cons(zprimsL);
            uzR  = self->prims2cons(zprimsR);

            fL = self->calc_Flux(xprimsL, 1);
            fR  = self->calc_Flux(xprimsR, 1);

            gL = self->calc_Flux(yprimsL, 2);
            gR  = self->calc_Flux(yprimsR, 2);

            hL = self->calc_Flux(zprimsL, 3);
            hR  = self->calc_Flux(zprimsR, 3);

            // Calc HLL Flux at i+1/2 interface
            if (self->hllc)
            {
                frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = self->calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);

            } else {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = self->calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[tza * bx * by + tya * bx + (txa - 1)];
            xprimsR  = prim_buff[tza * bx * by + tya * bx + (txa + 0)];
            //j+1/2
            yprimsL = prim_buff[tza * bx * by + (tya - 1) * bx + txa]; 
            yprimsR  = prim_buff[tza * bx * by + (tya + 0) * bx + txa]; 
            //k+1/2
            zprimsL = prim_buff[(tza - 1) * bx * by + tya * bx + txa]; 
            zprimsR  = prim_buff[(tza - 0) * bx * by + tya * bx + txa]; 

            uxL = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);

            uyL = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);

            uzL = self->prims2cons(zprimsL);
            uzR  = self->prims2cons(zprimsR);

            fL = self->calc_Flux(xprimsL, 1);
            fR  = self->calc_Flux(xprimsR, 1);

            gL = self->calc_Flux(yprimsL, 2);
            gR  = self->calc_Flux(yprimsR, 2);

            hL = self->calc_Flux(zprimsL, 3);
            hR  = self->calc_Flux(zprimsR, 3);

            // Calc HLL Flux at i-1/2 interface
            if (self-> hllc)
            {
                flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = self->calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);

            } else {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = self->calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            }   
        } else{
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
            xprimsL = center     + helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center) * plm_theta) * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::minmod((xright_mid - center) * plm_theta, (xright_most - center) * static_cast<real>(0.5), (xright_most - xright_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsL = center     + helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center) * plm_theta) * static_cast<real>(0.5);  
            yprimsR  = yright_mid - helpers::minmod((yright_mid - center) * plm_theta, (yright_most - center) * static_cast<real>(0.5), (yright_most - yright_mid)*plm_theta) * static_cast<real>(0.5);
            zprimsL = center     + helpers::minmod((center - zleft_mid)*plm_theta, (zright_mid - zleft_mid)*static_cast<real>(0.5), (zright_mid - center) * plm_theta) * static_cast<real>(0.5);  
            zprimsR  = zright_mid - helpers::minmod((zright_mid - center) * plm_theta, (zright_most - center) * static_cast<real>(0.5), (zright_most - zright_mid)*plm_theta) * static_cast<real>(0.5);


            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);

            uyL = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);

            uzL = self->prims2cons(zprimsL);
            uzR  = self->prims2cons(zprimsR);

            fL = self->calc_Flux(xprimsL, 1);
            fR  = self->calc_Flux(xprimsR, 1);

            gL = self->calc_Flux(yprimsL, 2);
            gR  = self->calc_Flux(yprimsR, 2);

            hL = self->calc_Flux(zprimsL, 3);
            hR  = self->calc_Flux(zprimsR, 3);

            if (self->hllc) {
                frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = self->calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            } else {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hrf = self->calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            // Do the same thing, but for the left side interface [i - 1/2]
            xprimsL = xleft_mid + helpers::minmod((xleft_mid - xleft_most) * plm_theta, (center - xleft_most) * static_cast<real>(0.5), (center - xleft_mid)*plm_theta) * static_cast<real>(0.5);
            xprimsR  = center    - helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center)*plm_theta)*static_cast<real>(0.5);
            yprimsL = yleft_mid + helpers::minmod((yleft_mid - yleft_most) * plm_theta, (center - yleft_most) * static_cast<real>(0.5), (center - yleft_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsR  = center    - helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center)*plm_theta)*static_cast<real>(0.5);
            zprimsL = zleft_mid + helpers::minmod((zleft_mid - zleft_most) * plm_theta, (center - zleft_most) * static_cast<real>(0.5), (center - zleft_mid)*plm_theta) * static_cast<real>(0.5);
            zprimsR  = center    - helpers::minmod((center - zleft_mid)*plm_theta, (zright_mid - zleft_mid)*static_cast<real>(0.5), (zright_mid - center)*plm_theta)*static_cast<real>(0.5);


            // Calculate the left and right states using the reconstructed PLM Primitive
            uxL = self->prims2cons(xprimsL);
            uxR  = self->prims2cons(xprimsR);
            uyL = self->prims2cons(yprimsL);
            uyR  = self->prims2cons(yprimsR);
            uzL = self->prims2cons(zprimsL);
            uzR  = self->prims2cons(zprimsR);

            fL = self->calc_Flux(xprimsL, 1);
            fR  = self->calc_Flux(xprimsR, 1);
            gL = self->calc_Flux(yprimsL, 2);
            gR  = self->calc_Flux(yprimsR, 2);
            hL = self->calc_Flux(zprimsL, 3);
            hR  = self->calc_Flux(zprimsR, 3);

            if (self->hllc) {
                flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = self->calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            } else {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
                hlf = self->calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3);
            }

        }// end else 
        
        //Advance depending on geometry
        const luint real_loc =  kk * xpg * ypg + jj * xpg + ii;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    #if GPU_CODE
                    const real d_source  = (self->d_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceD[real_loc];
                    const real s1_source = (self->s1_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS1[real_loc];
                    const real s2_source = (self->s2_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS2[real_loc];
                    const real s3_source = (self->s3_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS3[real_loc];
                    const real e_source  = (self->e_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceTau[real_loc];
                    #else 
                    const real d_source  = (self->d_all_zeros)   ? static_cast<real>(0.0) : self->sourceD[real_loc];
                    const real s1_source = (self->s1_all_zeros)  ? static_cast<real>(0.0) : self->sourceS1[real_loc];
                    const real s2_source = (self->s2_all_zeros)  ? static_cast<real>(0.0) : self->sourceS2[real_loc];
                    const real s3_source = (self->s3_all_zeros)  ? static_cast<real>(0.0) : self->sourceS3[real_loc];
                    const real e_source  = (self->e_all_zeros)   ? static_cast<real>(0.0) : self->sourceTau[real_loc];
                    #endif 
                    const Conserved source_terms = Conserved{d_source, s1_source, s2_source, s3_source, e_source} * decay_const;
                    #if GPU_CODE 
                       self->gpu_cons[aid] -= ( (frf - flf ) / dx1 + (grf - glf) / dx2 + (hrf - hlf) / dx3 - source_terms) * dt * step;
                    #else
                        cons[aid] -= ( (frf  - flf ) / dx1 + (grf - glf) / dx2 + (hrf - hlf) / dx3 - source_terms) * dt * step;
                    #endif
                break;
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl           = (ii > 0 ) ? x1min * std::pow(10, (ii -static_cast<real>(0.5)) * dlogx1) :  x1min;
                    const real rr           = (ii < xpg - 1) ? rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
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

                    #if GPU_CODE
                    const real d_source  = (self->d_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceD[real_loc];
                    const real s1_source = (self->s1_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS1[real_loc];
                    const real s2_source = (self->s2_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS2[real_loc];
                    const real s3_source = (self->s3_all_zeros)  ? static_cast<real>(0.0) : self->gpu_sourceS3[real_loc];
                    const real e_source  = (self->e_all_zeros)   ? static_cast<real>(0.0) : self->gpu_sourceTau[real_loc];
                    #else 
                    const real d_source  = (self->d_all_zeros)   ? static_cast<real>(0.0) : self->sourceD[real_loc];
                    const real s1_source = (self->s1_all_zeros)  ? static_cast<real>(0.0) : self->sourceS1[real_loc];
                    const real s2_source = (self->s2_all_zeros)  ? static_cast<real>(0.0) : self->sourceS2[real_loc];
                    const real s3_source = (self->s3_all_zeros)  ? static_cast<real>(0.0) : self->sourceS3[real_loc];
                    const real e_source  = (self->e_all_zeros)   ? static_cast<real>(0.0) : self->sourceTau[real_loc];
                    #endif 
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
    real dlogt, 
    real plm_theta,
    real engine_duration, 
    real chkpt_interval,
    std::string data_directory, 
    std::string boundary_condition,
    bool first_order,
    bool linspace, 
    bool hllc)
{   
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

    std::string filename;

    // Define the source terms
    this->sourceD        = sources[0];
    this->sourceS1       = sources[1];
    this->sourceS2       = sources[2];
    this->sourceS3       = sources[3];
    this->sourceTau      = sources[4];
    this->total_zones    = nx * ny * nz;
    this->first_order     = first_order;
    this->periodic        = boundary_condition == "periodic";
    this->hllc            = hllc;
    this->dlogt           = dlogt;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->bc              = helpers::boundary_cond_map.at(boundary_condition);
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
    this->x1min           = x1[0];
    this->x1max           = x1[xphysical_grid - 1];
    this->x2min           = x2[0];
    this->x2max           = x2[yphysical_grid - 1];
    this->x3min           = x3[0];
    this->x3max           = x3[zphysical_grid - 1];

    this->d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
    this->s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
    this->s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
    this->s3_all_zeros = std::all_of(sourceS3.begin(),  sourceS3.end(),  [](real i) {return i == 0;});
    this->e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    // Stuff for moving mesh 
    // TODO: make happen at some point
    this->hubble_param = 0.0; //adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (x2max == 0.5 * M_PI){
        this->reflecting_theta = true;
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
    setup.xactive_zones  = xphysical_grid;
    setup.yactive_zones  = yphysical_grid;
    setup.zactive_zones  = zphysical_grid;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;
    
    cons.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);
    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state3D[0].size(); i++)
    {
        auto D            = state3D[0][i];
        auto S1           = state3D[1][i];
        auto S2           = state3D[2][i];
        auto S3           = state3D[3][i];
        auto E            = state3D[4][i];
        auto S            = std::sqrt(S1 * S1 + S2 * S2 + S3 * S3);
        cons[i]           = Conserved{D, S1, S2, S3, E};
        pressure_guess[i] = std::abs(S - D - E);
    }
    n = 0;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (tstart - engine_duration)));


    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr3d::PrimitiveData transfer_prims;

    SRHD3D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(SRHD3D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD3D));
    dualType dualMem;
    dualMem.copyHostToDev(*this, device_self);

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
        write2file(this, device_self, dualMem, setup, data_directory, t, t_interval, chkpt_interval, zphysical_grid);
        t_interval += chkpt_interval;
    }

    // Some benchmarking tools 
    luint      n   = 0;
    luint  nfold   = 0;
    luint  ncheck  = 0;
    real    zu_avg = 0;
    #if GPU_CODE
    anyGpuEvent_t t1, t2;
    anyGpuEventCreate(&t1);
    anyGpuEventCreate(&t2);
    float delta_t;
    #else 
    high_resolution_clock::time_point t1, t2;
    double delta_t;
    #endif

    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;
    // Simulate :)
    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            helpers::recordEvent(t1);
            advance(self, activeP, bx, by, bz, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, true, bc);
            helpers::recordEvent(t2);
            t += dt; 
            

            if (n >= nfold){
                anyGpuEventSynchronize(t2);
                helpers::recordDuration(delta_t, t1, t2);
                if (BuildPlatform == Platform::GPU) {
                    delta_t *= 1e-3;
                }
                ncheck += 1;
                zu_avg += total_zones / delta_t;
                if constexpr(BuildPlatform == Platform::GPU) {
                    const real gtx_emperical_bw       = total_zones * (sizeof(Primitive) + sizeof(Conserved)) * (1.0 + 4.0 * radius) / (delta_t * 1e9);
                    writefl("\riteration:{:>06} dt:{:>08.2e} time:{:>08.2e} zones/sec:{:>08.2e} ebw(%):{:>04.2f}", n, dt, t, total_zones/delta_t, static_cast<real>(100.0) * gtx_emperical_bw / gtx_theoretical_bw);
                } else {
                    writefl("\riteration:{:>06}    dt: {:>08.2e}    time: {:>08.2e}    zones/sec: {:>08.2e}", n, dt, t, total_zones/delta_t);
                }
                nfold += 100;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                write2file(this, device_self, dualMem, setup, data_directory, t, t_interval, chkpt_interval, zphysical_grid);
                if (dlogt != 0) {
                    t_interval *= std::pow(10, dlogt);
                } else {
                    t_interval += chkpt_interval;
                }
            }

            n++;
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            else 
                adapt_dt();

            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));
            if constexpr(BuildPlatform == Platform::GPU) {
                this->inFailureState = device_self->inFailureState;
            }
            if (inFailureState) {
                simbi::gpu::api::deviceSynch();
            }
            
        }
    } else {
        while (t < tend && !inFailureState)
        {
            helpers::recordEvent(t1);
            // First half step
            advance(self, activeP, bx, by, bz,  radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, false, bc);

            // Final half step
            advance(self, activeP, bx, by, bz,  radius, geometry, memside);
            cons2prim(fullP, self, memside);
            config_ghosts3D(fullP, self, nx, ny, nz, false, bc); 
            helpers::recordEvent(t2);
            t += dt; 

            if (n >= nfold){
                anyGpuEventSynchronize(t2);
                helpers::recordDuration(delta_t, t1, t2);
                if (BuildPlatform == Platform::GPU) {
                    delta_t *= 1e-3;
                }
                ncheck += 1;
                zu_avg += total_zones / delta_t;
                 if constexpr(BuildPlatform == Platform::GPU) {
                    const real gtx_emperical_bw       = total_zones * (sizeof(Primitive) + sizeof(Conserved)) * (1.0 + 4.0 * radius) / (delta_t * 1e9);
                    writefl("\riteration:{:>06} dt:{:>08.2e} time:{:>08.2e} zones/sec:{:>08.2e} ebw(%):{:>04.2f}", n, dt, t, total_zones/delta_t, static_cast<real>(100.0) * gtx_emperical_bw / gtx_theoretical_bw);
                } else {
                    writefl("\riteration:{:>06}    dt: {:>08.2e}    time: {:>08.2e}    zones/sec: {:>08.2e}", n, dt, t, total_zones/delta_t);
                }
                nfold += 100;
            }
            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                write2file(this, device_self, dualMem, setup, data_directory, t, t_interval, chkpt_interval, zphysical_grid);
                if (dlogt != 0) {
                    t_interval *= std::pow(10, dlogt);
                } else {
                    t_interval += chkpt_interval;
                }
            }
            n++;
            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
                adapt_dt(device_self, geometry, activeP, dtShBytes);
            else 
                adapt_dt();

            // Update decay constant
            decay_const = static_cast<real>(1.0) / (static_cast<real>(1.0) + exp(static_cast<real>(10.0) * (t - engine_duration)));
            if constexpr(BuildPlatform == Platform::GPU) {
                this->inFailureState = device_self->inFailureState;
            }
            if (inFailureState) {
                simbi::gpu::api::deviceSynch();
            }
            
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

    transfer_prims = helpers::vec2struct<sr3d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(5, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.v3;
    solution[4] = transfer_prims.p;

    return solution;
};
