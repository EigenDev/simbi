/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "util/device_api.hpp"
#include "util/dual.hpp"
#include "common/helpers.hpp"
#include "helpers.hip.hpp"
#include "srhydro2D.hip.hpp"
#include "util/printb.hpp"
#include "util/parallel_for.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace std::chrono;

// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(std::vector<std::vector<real>> state2D, luint nx, luint ny, real gamma,
               std::vector<real> x1, std::vector<real> x2, real CFL,
               std::string coord_system = "cartesian")
:

    nx(nx),
    ny(ny),
    nzones(state2D[0].size()),
    state2D(state2D),
    gamma(gamma),
    x1(x1),
    x2(x2),
    CFL(CFL),
    coord_system(coord_system)
{
    d_all_zeros  = false;
    s1_all_zeros = false;
    s2_all_zeros = false;
    e_all_zeros  = false;
}

// Destructor
SRHD2D::~SRHD2D() {}

/* Define typedefs because I am lazy */
typedef sr2d::Primitive Primitive;
typedef sr2d::Conserved Conserved;
typedef sr2d::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD2D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const luint nhat = 1)
{
    // Eigenvals lambda;

    // Separate the left and right Primitive
    const real rho_l = prims_l.rho;
    const real p_l = prims_l.p;
    const real h_l = 1. + gamma * p_l / (rho_l * (gamma - 1));

    const real rho_r = prims_r.rho;
    const real p_r = prims_r.p;
    const real h_r = 1. + gamma * p_r / (rho_r * (gamma - 1));

    const real cs_r = sqrt(gamma * p_r / (h_r * rho_r));
    const real cs_l = sqrt(gamma * p_l / (h_l * rho_l));

    const real v_l = prims_l.vcomponent(nhat);
    const real v_r = prims_r.vcomponent(nhat);

    //-----------Calculate wave speeds based on Shneider et al. 1992
    const real vbar  = (real)0.5 * (v_l + v_r);
    const real cbar  = (real)0.5 * (cs_l + cs_r);
    const real bl    = (vbar - cbar)/(1. - cbar*vbar);
    const real br    = (vbar + cbar)/(1. + cbar*vbar);
    const real aL    = my_min(bl, (v_l - cs_l)/(1. - v_l*cs_l));
    const real aR    = my_max(br, (v_r + cs_r)/(1. + v_r*cs_r));

    //--------Calc the wave speeds based on Mignone and Bodo (2005)
    // const real sL = cs_l * cs_l * (1. / (gamma * gamma * (1 - cs_l * cs_l)));
    // const real sR = cs_r * cs_r * (1. / (gamma * gamma * (1 - cs_r * cs_r)));

    // // Define temporaries to save computational cycles
    // const real qfL = 1. / (1. + sL);
    // const real qfR = 1. / (1. + sR);
    // const real sqrtR = sqrt(sR * (1 - v1_r * v1_r + sR));
    // const real sqrtL = sqrt(sL * (1 - v1_l * v1_l + sL));

    // const real lamLm = (v1_l - sqrtL) * qfL;
    // const real lamRm = (v1_r - sqrtR) * qfR;
    // const real lamLp = (v1_l + sqrtL) * qfL;
    // const real lamRp = (v1_r + sqrtR) * qfR;

    // const real aL = lamLm < lamRm ? lamLm : lamRm;
    // const real aR = lamLp > lamRp ? lamLp : lamRp;

    return Eigenvals(aL, aR, cs_l, cs_r);
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2cons(const Primitive &prims)
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1 - (vx * vx + vy * vy));
    const real h = 1. + gamma * pressure / (rho * (gamma - 1.));

    return Conserved{
        rho * lorentz_gamma, 
        rho * h * lorentz_gamma * lorentz_gamma * vx,
        rho * h * lorentz_gamma * lorentz_gamma * vy,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma};
};

Conserved SRHD2D::calc_intermed_statesSR2D(const Primitive &prims,
                                           const Conserved &state, real a,
                                           real aStar, real pStar,
                                           luint nhat = 1)
{
    real Dstar, S1star, S2star, tauStar, Estar, cofactor;
    Conserved starStates;

    real pressure = prims.p;
    real v1 = prims.v1;
    real v2 = prims.v2;

    real D = state.D;
    real S1 = state.S1;
    real S2 = state.S2;
    real tau = state.tau;
    real E = tau + D;

    switch (nhat)
    {
    case 1:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v1) * D;
        S1star = cofactor * (S1 * (a - v1) - pressure + pStar);
        S2star = cofactor * (a - v1) * S2;
        Estar = cofactor * (E * (a - v1) + pStar * aStar - pressure * v1);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    case 2:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v2) * D;
        S1star = cofactor * (a - v2) * S1;
        S2star = cofactor * (S2 * (a - v2) - pressure + pStar);
        Estar = cofactor * (E * (a - v2) + pStar * aStar - pressure * v2);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    }

    return starStates;
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------

// Adapt the CFL conditonal timestep
void SRHD2D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real dx1, cs, dx2, rho, pressure, v1, v2, rmean, h;
        real cfl_dt;
        luint shift_i, shift_j;
        real plus_v1, plus_v2, minus_v1, minus_v2;
        luint aid; // active index id

        // Compute the minimum timestep given CFL
        for (luint jj = 0; jj < yphysical_grid; jj++)
        {
            dx2 = coord_lattice.dx2[jj];
            shift_j = jj + idx_active;
            #pragma omp for schedule(static)
            for (luint ii = 0; ii < xphysical_grid; ii++)
            {
                shift_i  = ii + idx_active;
                aid      = shift_i + nx * shift_j;
                dx1      = coord_lattice.dx1[ii];
                rho      = prims[aid].rho;
                v1       = prims[aid].v1;
                v2       = prims[aid].v2;
                pressure = prims[aid].p;

                h = 1. + gamma * pressure / (rho * (gamma - 1.));
                cs = sqrt(gamma * pressure / (rho * h));

                plus_v1  = (v1 + cs) / (1. + v1 * cs);
                plus_v2  = (v2 + cs) / (1. + v2 * cs);
                minus_v1 = (v1 - cs) / (1. - v1 * cs);
                minus_v2 = (v2 - cs) / (1. - v2 * cs);

                if (coord_system == "cartesian")
                {

                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }
                else
                {
                    rmean = coord_lattice.x1mean[ii];
                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                rmean * dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }

                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
                
            } // end ii 
        } // end jj
    } // end parallel region

    dt = CFL * min_dt;
};

void SRHD2D::adapt_dt(SRHD2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p)
{
    #if GPU_CODE
    {
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            dtWarpReduce<SRHD2D, Primitive, 128><<<p.gridSize,p.blockSize>>>
            (dev, geometry, dx1, dx2);
            break;
        
        case simbi::Geometry::SPHERICAL:
            dtWarpReduce<SRHD2D, Primitive, 128><<<p.gridSize,p.blockSize>>>
            (dev, geometry, dlogx1, dx2, x1min, x1max, x2min, x2max);
            break;
        }
        
        simbi::gpu::api::deviceSynch();
        simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
    // std::cout << "dt out: " <<  dt << "\n";
    // std::cin.get();
    #endif
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2flux(const Primitive &prims, luint nhat = 1)
{
    const real vn = prims.vcomponent(nhat);
    const unsigned int kron = kronecker(nhat, 1);
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1. - (vx * vx + vy * vy));

    const real h = 1. + gamma * pressure / (rho * (gamma - 1));
    const real D = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const real tau = rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return Conserved{D * vn, S1 * vn + kron * pressure, S2 * vn + !kron * pressure, (tau + pressure) * vn};
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const luint nhat)
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
Conserved SRHD2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint nhat = 1)
{

    Conserved starStateR, starStateL;
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if (0 <= aL)
    {
        return left_flux;
    }
    else if (0 >= aR)
    {
        return right_flux;
    }

    const real aLminus = aL < 0 ? aL : 0;
    const real aRplus  = aR > 0 ? aR : 0;

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
    const real quad = -0.5 * (b + sgn(b) * sqrt(b * b - 4 * a * c));
    const real aStar = c * (1 / quad);
    const real pStar = -aStar * fe + fs;

    // Apply the low-Mach HLLC fix found in Fleichman et al 2020: 
    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
    const real cL       = lambda.csL;
    const real cR       = lambda.csR;
    const real ma_lim   = 0.20;

    //--------------Compute the L Star State----------
    real pressure = left_prims.p;
    real D        = left_state.D;
    real S1       = left_state.S1;
    real S2       = left_state.S2;
    real tau      = left_state.tau;
    real E        = tau + D;
    real cofactor = 1. / (aL - aStar);

    real vL =  left_prims.vcomponent(nhat);
    real vR = right_prims.vcomponent(nhat);
    unsigned int kron = kronecker(nhat, 1);
    // Left Star State in x-direction of coordinate lattice
    real Dstar    = cofactor * (aL - vL) * D;
    real S1star   = cofactor * (S1 * (aL - vL) +  kron * (-pressure + pStar) );
    real S2star   = cofactor * (S2 * (aL - vL) + !kron * (-pressure + pStar) );
    real Estar    = cofactor * (E * (aL - vL) + pStar * aStar - pressure * vL);
    real tauStar  = Estar - Dstar;
    starStateL    = Conserved(Dstar, S1star, S2star, tauStar);

    pressure = right_prims.p;
    D        = right_state.D;
    S1       = right_state.S1;
    S2       = right_state.S2;
    tau      = right_state.tau;
    E        = tau + D;
    cofactor = 1. / (aR - aStar);

    Dstar   = cofactor * (aR - vR) * D;
    S1star   = cofactor * (S1 * (aR - vR) +  kron * (-pressure + pStar) );
    S2star   = cofactor * (S2 * (aR - vR) + !kron * (-pressure + pStar) );
    Estar   = cofactor * (E * (aR - vR) + pStar * aStar - pressure * vR);
    tauStar = Estar - Dstar;
    starStateR = Conserved(Dstar, S1star, S2star, tauStar);

    const real ma_local = my_max(std::abs(vL / cL), std::abs(vR / cR));
    const real phi      = sin(my_min(1.0, ma_local / ma_lim) * PI * 0.5);
    const real aL_lm    = (phi != 0 ) ? phi * aL : aL;
    const real aR_lm    = (phi != 0 ) ? phi * aR : aR;

    return (left_flux + right_flux) * 0.5 + ( (starStateL - left_state) * aL_lm
        + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm ) * 0.5;

    // if (-aL <= (aStar - aL))
    // {
    //     const real pressure = left_prims.p;
    //     const real D = left_state.D;
    //     const real S1 = left_state.S1;
    //     const real S2 = left_state.S2;
    //     const real tau = left_state.tau;
    //     const real E = tau + D;
    //     const real cofactor = 1. / (aL - aStar);
    //     //--------------Compute the L Star State----------
    //     switch (nhat)
    //     {
    //     case 1:
    //     {
    //         const real v1 = left_prims.v1;
    //         // Left Star State in x-direction of coordinate lattice
    //         const real Dstar    = cofactor * (aL - v1) * D;
    //         const real S1star   = cofactor * (S1 * (aL - v1) - pressure + pStar);
    //         const real S2star   = cofactor * (aL - v1) * S2;
    //         const real Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
    //         const real tauStar  = Estar - Dstar;

    //         const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

    //         //---------Compute the L Star Flux
    //         return left_flux + (interstate_left - left_state) * aL;
    //     }

    //     case 2:
    //         const real v2 = left_prims.v2;
    //         // Start States in y-direction in the coordinate lattice
    //         const real Dstar   = cofactor * (aL - v2) * D;
    //         const real S1star  = cofactor * (aL - v2) * S1;
    //         const real S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
    //         const real Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

    //         //---------Compute the L Star Flux
    //         return left_flux + (interstate_left - left_state) * aL;
    //     }
    // }
    // else
    // {
    //     const real pressure = right_prims.p;
    //     const real D = right_state.D;
    //     const real S1 = right_state.S1;
    //     const real S2 = right_state.S2;
    //     const real tau = right_state.tau;
    //     const real E = tau + D;
    //     const real cofactor = 1. / (aR - aStar);

    //     /* Compute the L/R Star State */
    //     switch (nhat)
    //     {
    //     case 1:
    //     {
    //         const real v1 = right_prims.v1;
    //         const real Dstar = cofactor * (aR - v1) * D;
    //         const real S1star = cofactor * (S1 * (aR - v1) - pressure + pStar);
    //         const real S2star = cofactor * (aR - v1) * S2;
    //         const real Estar = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

    //         // Compute the intermediate right flux
    //         return right_flux + (interstate_right - right_state) * aR;
    //     }

    //     case 2:
    //         const real v2 = right_prims.v2;
    //         // Start States in y-direction in the coordinate lattice
    //         const real cofactor = 1. / (aR - aStar);
    //         const real Dstar = cofactor * (aR - v2) * D;
    //         const real S1star = cofactor * (aR - v2) * S1;
    //         const real S2star = cofactor * (S2 * (aR - v2) - pressure + pStar);
    //         const real Estar = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
    //         const real tauStar = Estar - Dstar;

    //         const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

    //         // Compute the intermediate right flux
    //         return right_flux + (interstate_right - right_state) * aR;
    //     }
    // }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================

void SRHD2D::cons2prim(
    ExecutionPolicy<> p, 
    SRHD2D *dev, 
    simbi::MemSide user)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (luint gid){
        real eps, pre, v2, et, c2, h, g, f, W, rho;
        #if GPU_CODE
        extern __shared__ Conserved  conserved_buff[];
        #else
        auto* const conserved_buff = &cons[0];
        #endif 

        const auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * threadIdx.y + threadIdx.x : gid;
        // load shared memory
        if constexpr(BuildPlatform == Platform::GPU)
            conserved_buff[tid] = self->gpu_cons[gid];
            
        simbi::gpu::api::synchronize();
        luint iter  = 0;
        real D    = conserved_buff[tid].D;
        real S1   = conserved_buff[tid].S1;
        real S2   = conserved_buff[tid].S2;
        real tau  = conserved_buff[tid].tau;
        real S    = sqrt(S1 * S1 + S2 * S2);

        #if GPU_CODE
        real peq = self->gpu_pressure_guess[gid];
        #else 
        real peq = self->pressure_guess[gid];
        #endif

        real tol = D * tol_scale;
        do
        {
            pre = peq;
            et  = tau + D + pre;
            v2 = S * S / (et * et);
            W   = 1 / sqrt(1 - v2);
            rho = D / W;

            eps = (tau + (1 - W) * D + (1 - W * W) * pre) / (D * W);

            h = 1 + eps + pre / rho;
            c2 = self->gamma * pre / (h * rho);

            g = c2 * v2 - 1;
            f = (self->gamma - 1) * rho * eps - pre;

            peq = pre - f / g;
            iter++;
            if (iter >= MAX_ITER)
            {
                printf("\nCons2Prim cannot converge\n");
                self->dt = INFINITY;
                return;
            }

        } while (std::abs(peq - pre) >= tol);

        real inv_et = 1 / (tau + D + peq);
        real vx     = S1 * inv_et;
        real vy     = S2 * inv_et;

        #if GPU_CODE
            self->gpu_pressure_guess[gid] = peq;
            self->gpu_prims[gid]          = Primitive{D * sqrt(1 - (vx * vx + vy * vy)), vx, vy, peq};
        #else
            self->pressure_guess[gid] = peq;
            self->prims[gid]          = Primitive{D * sqrt(1 - (vx * vx + vy * vy)), vx, vy, peq};
        #endif
        

    });
}

void SRHD2D::advance(
    SRHD2D *dev, 
    const ExecutionPolicy<> p,
    const luint bx,
    const luint by,
    const luint radius, 
    const simbi::Geometry geometry, 
    const simbi::MemSide user)
{
    auto *self = (BuildPlatform == Platform::GPU) ? dev : this;
    const luint xpg                   = this->xphysical_grid;
    const luint ypg                   = this->yphysical_grid;
    const luint extent                = (BuildPlatform == Platform::GPU) ? p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
    const luint xextent               = p.blockSize.x;
    const luint yextent               = p.blockSize.y;

    #if GPU_CODE
    const bool first_order         = this->first_order;
    const bool periodic            = this->periodic;
    const bool hllc                = this->hllc;
    const real dt                  = this->dt;
    const real decay_const         = this->decay_const;
    const real plm_theta           = this->plm_theta;
    const real gamma               = this->gamma;
    const luint nx                 = this->nx;
    const luint ny                 = this->ny;
    const real xmin                = this->x1min;
    const real xmax                = this->x1max;
    const real ymin                = this->x2min;
    const real ymax                = this->x2max;
    const real dx2                 = this->dx2;
    const real dlogx1              = this->dlogx1;
    const real dx1                 = this->dx1;
    const real imax                = this->xphysical_grid - 1;
    const real jmax                = this->yphysical_grid - 1;
    const bool d_all_zeros         = this->d_all_zeros;
    const bool s1_all_zeros        = this->s1_all_zeros;
    const bool s2_all_zeros        = this->s2_all_zeros;
    const bool e_all_zeros         = this->e_all_zeros;
    const real pow_dlogr           = pow(10, dlogx1);
    #endif

    // const CLattice2D *coord_lattice = &(self->coord_lattice);
    const luint nbs = (BuildPlatform == Platform::GPU) ? bx * by : nzones;

    // if on NVidia GPU, do column major striding, row-major otherwise
    const luint sx = (col_maj) ? 1  : bx;
    const luint sy = (col_maj) ? by :  1;

    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = &prims[0];
        #endif 

        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        if       constexpr(BuildPlatform == Platform::GPU) if ((ii >= xpg) || (jj >= ypg)) return;

        // printf("ii=%lu, jj=%lu\n", ii, jj);
        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + radius : ja;

        Conserved ux_l, ux_r, uy_l, uy_r;
        Conserved f_l, f_r, g_l, g_r, f1, f2, g1, g2;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r;

        // do column-major index if on GPU
        luint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        if  constexpr(BuildPlatform == Platform::GPU)
        {
            luint txl = xextent;
            luint tyl = yextent;

            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = self->gpu_prims[aid];
            if constexpr(col_maj)
            {
                if (tx < radius)
                {
                    if (ia + xextent > nx - 1) txl = nx - radius - ia + tx;
                    prim_buff[(txa - radius) * sy + tya] = self->gpu_prims[(ia - radius) * ny + ja];
                    prim_buff[(txa + txl   ) * sy + tya] = self->gpu_prims[(ia + txl   ) * ny + ja]; 
                
                }
                if (ty < radius)
                {   
                    if (ja + yextent > ny - 1) tyl = ny - radius - ja + ty;
                    prim_buff[txa * sy + tya - radius] =  self->gpu_prims[ia * ny + ja - radius];
                    prim_buff[txa * sy + tya +    tyl] =  self->gpu_prims[ia * ny + ja + tyl]; 
                }

            } else {
                if (ty < radius)
                {
                    if (ja + yextent > ny - 1) tyl = ny - radius - ja + ty;
                    prim_buff[(tya - radius) * sx + txa] = self->gpu_prims[(ja - radius) * nx + ia];
                    prim_buff[(tya + tyl   ) * sx + txa] = self->gpu_prims[(ja + tyl   ) * nx + ia]; 
                
                }
                if (tx < radius)
                {   
                    if (ia + xextent > nx - 1) txl = nx - radius - ia + tx;
                    prim_buff[tya * sx + txa - radius] =  self->gpu_prims[ja * nx + ia - radius];
                    prim_buff[tya * sx + txa +    txl] =  self->gpu_prims[ja * nx + ia + txl]; 
                }
            }
            
            simbi::gpu::api::synchronize();
        }

        if (first_order)
        {
            if (periodic)
            {
                xprims_l = prim_buff[txa * sy + tya * sx];
                xprims_r = roll(prim_buff, (txa + 1) * sy + tya * sx, nbs);

                yprims_l = prim_buff[txa * sy + tya * sx];
                yprims_r = roll(prim_buff, txa + (tya + 1) * sx, nbs);
            }
            else
            {
                xprims_l = prim_buff[(txa + 0) * sy + (tya + 0) * sx];
                xprims_r = prim_buff[(txa + 1) * sy + (tya + 0) * sx];
                //j+1/2
                yprims_l = prim_buff[(txa + 0) * sy + (tya + 0) * sx];
                yprims_r = prim_buff[(txa + 0) * sy + (tya + 1) * sx];
            }
            
            // i+1/2
            ux_l = self->prims2cons(xprims_l); 
            ux_r = self->prims2cons(xprims_r); 
            // j+1/2
            uy_l = self->prims2cons(yprims_l);  
            uy_r = self->prims2cons(yprims_r); 

            f_l = self->prims2flux(xprims_l, 1);
            f_r = self->prims2flux(xprims_r, 1);

            g_l = self->prims2flux(yprims_l, 2);
            g_r = self->prims2flux(yprims_r, 2);

            // Calc HLL Flux at i+1/2 interface
            if (hllc)
            {
                // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                //     f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // } else {
                //     f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // }
                
                // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                //     g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // } else {
                //     g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // }
                f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            // Set up the left and right state interfaces for i-1/2
            if (periodic)
            {
                xprims_l = roll(prim_buff,  txa - 1 + tya * bx, nbs);
                xprims_r = prim_buff[txa + tya * bx];

                yprims_l = roll(prim_buff, txa + (tya - 1) * bx, nbs);
                yprims_r = prim_buff[txa + tya * bx];
            }
            else
            {
                xprims_l = prim_buff[(txa - 1) * sy + (tya + 0) * sx];
                xprims_r = prim_buff[(txa - 0) * sy + (tya + 0) * sx];
                //j+1/2
                yprims_l = prim_buff[(txa - 0) * sy + (tya - 1) * sx]; 
                yprims_r = prim_buff[(txa + 0) * sy + (tya - 0) * sx]; 
            }

            // i+1/2
            ux_l = self->prims2cons(xprims_l); 
            ux_r = self->prims2cons(xprims_r); 
            // j+1/2
            uy_l = self->prims2cons(yprims_l);  
            uy_r = self->prims2cons(yprims_r); 

            f_l = self->prims2flux(xprims_l, 1);
            f_r = self->prims2flux(xprims_r, 1);

            g_l = self->prims2flux(yprims_l, 2);
            g_r = self->prims2flux(yprims_r, 2);

            // Calc HLL Flux at i-1/2 interface
            if (hllc)
            {
                // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                //     f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // } else {
                //     f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // }
                
                // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                //     g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // } else {
                //     g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                // }
                f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

            } else {
                f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            //Advance depending on geometry
            luint real_loc = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
            // printf("(%lu, %lu) f1: %f, f2: %f, g1: %f, g2: %f\n", ia, ja,  f1.tau, f2.tau, g1.tau, g2.tau);
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            // const real xl = (ii > 0) ? x1min + (ii - 0.5) * dx
                            // real dx = coord_lattice->gpu_dx1[ii];
                            // real dy = coord_lattice->gpu_dx2[jj];
                            const real d_source  = (d_all_zeros)   ? 0 : self->gpu_sourceD[real_loc];
                            const real s1_source = (s1_all_zeros)  ? 0 : self->gpu_sourceS1[real_loc];
                            const real s2_source = (s2_all_zeros)  ? 0 : self->gpu_sourceS2[real_loc];
                            const real e_source  = (e_all_zeros)   ? 0 : self->gpu_sourceTau[real_loc];
                            self->gpu_cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D ) / dx2 + d_source * decay_const);
                            self->gpu_cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1) / dx2 + d_source * decay_const);
                            self->gpu_cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx1  -(g1.S2  - g2.S2) / dx2 + d_source * decay_const);
                            self->gpu_cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau)/ dx2 + d_source * decay_const);
                        #else
                            real dx = coord_lattice.dx1[ii];
                            real dy = coord_lattice.dx2[jj];
                            cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + sourceD   [real_loc] );
                            cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + sourceS1  [real_loc] );
                            cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + sourceS2  [real_loc] );
                            cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + sourceTau [real_loc] );
                        #endif
                    

                    break;
                    }
                
                case simbi::Geometry::SPHERICAL:
                    {
                    #if GPU_CODE
                    const real rl           = (ii > 0)    ? xmin * pow(10, (ii - 0.5) * dlogx1) : xmin;
                    const real rr           = (ii < imax) ? rl   * pow_dlogr: xmax;
                    const real tl           = (jj > 0)    ? ymin + (jj - 0.5) * dx2: ymin;
                    const real tr           = (jj < jmax) ? tl + dx2 : ymax; 
                    const real rmean        = (real)0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                    const real s1R          = rr * rr; //coord_lattice->gpu_x1_face_areas[ii + 1];
                    const real s1L          = rl * rl; //coord_lattice->gpu_x1_face_areas[ii + 0];
                    const real s2R          = sin(tr); //coord_lattice->gpu_x2_face_areas[jj + 1];
                    const real s2L          = sin(tl); //coord_lattice->gpu_x2_face_areas[jj + 0];
                    const real thmean       = 0.5 * (tl + tr);
                    const real sint         = sin(thmean);
                    const real dV1          = rmean * rmean * (rr - rl);             
                    const real dV2          = rmean * sint * (tr - tl); 
                    const real cot          = cos(thmean) / sint ;

                    const real d_source  = (d_all_zeros)   ? 0 : self->gpu_sourceD[real_loc];
                    const real s1_source = (s1_all_zeros)  ? 0 : self->gpu_sourceS1[real_loc];
                    const real s2_source = (s2_all_zeros)  ? 0 : self->gpu_sourceS2[real_loc];
                    const real e_source  = (e_all_zeros)   ? 0 : self->gpu_sourceTau[real_loc];
                    #else
                    real s1R   = coord_lattice.x1_face_areas[ii + 1];
                    real s1L   = coord_lattice.x1_face_areas[ii + 0];
                    real s2R   = coord_lattice.x2_face_areas[jj + 1];
                    real s2L   = coord_lattice.x2_face_areas[jj + 0];
                    real rmean = coord_lattice.x1mean[ii];
                    real dV1   = coord_lattice.dV1[ii];
                    real dV2   = rmean * coord_lattice.dV2[jj];
                    #endif
                    // Grab central primitives
                    real rhoc = prim_buff[txa * sy + tya * sx].rho;
                    real pc   = prim_buff[txa * sy + tya * sx].p;
                    real uc   = prim_buff[txa * sy + tya * sx].v1;
                    real vc   = prim_buff[txa * sy + tya * sx].v2;

                    real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                    real gam2 = 1/(1 - (uc * uc + vc * vc));

                    #if GPU_CODE 
                        self->gpu_cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + d_source * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + (rhoc * hc * gam2 * vc * vc + (real)2.0 * pc) / rmean +
                                            s1_source * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc - pc * cot) / rmean 
                                            + s2_source * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + e_source * decay_const
                            } * dt;
                    #else
                        cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + self->sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + 2 * pc / rmean +
                                            self->sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice.cot[jj] / rmean) 
                                            + self->sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + self->sourceTau[real_loc] * decay_const
                            } * dt;
                    #endif
                    
                    break;
                    }
            } // end switch
                
        }
        else
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            if (!periodic)
            {
                // Coordinate X
                xleft_most  = prim_buff[(txa - 2) * sy + tya * sx];
                xleft_mid   = prim_buff[(txa - 1) * sy + tya * sx];
                center      = prim_buff[(txa + 0) * sy + tya * sx];
                xright_mid  = prim_buff[(txa + 1) * sy + tya * sx];
                xright_most = prim_buff[(txa + 2) * sy + tya * sx];

                // Coordinate Y
                yleft_most  = prim_buff[txa * sy + (tya - 2) * sx];
                yleft_mid   = prim_buff[txa * sy + (tya - 1) * sx];
                yright_mid  = prim_buff[txa * sy + (tya + 1) * sx];
                yright_most = prim_buff[txa * sy + (tya + 2) * sx];
            }
            else
            {
                // X Coordinate
                xleft_most   = roll(prim_buff, tya * sx + (txa - 2) * sy, nbs);
                xleft_mid    = roll(prim_buff, tya * sx + (txa - 1) * sy, nbs);
                center       =      prim_buff[ tya * sx + (txa + 0) * sy];
                xright_mid   = roll(prim_buff, tya * sx + (txa + 1) * sy, nbs);
                xright_most  = roll(prim_buff, tya * sx + (txa + 2) * sy, nbs);

                yleft_most   = roll(prim_buff, txa * sy + (tya - 2) * sx, nbs);
                yleft_mid    = roll(prim_buff, txa * sy + (tya - 1) * sx, nbs);
                yright_mid   = roll(prim_buff, txa * sy + (tya + 1) * sx, nbs);
                yright_most  = roll(prim_buff, txa * sy + (tya + 2) * sx, nbs);
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

                yprims_r.p = yright_mid.p -
                                (real)0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                            (real)0.5 * (yright_most.p - center.p),
                                            plm_theta * (yright_most.p - yright_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = self->prims2cons(xprims_l);
                ux_r = self->prims2cons(xprims_r);
                uy_l = self->prims2cons(yprims_l);
                uy_r = self->prims2cons(yprims_r);

                f_l = self->prims2flux(xprims_l, 1);
                f_r = self->prims2flux(xprims_r, 1);
                g_l = self->prims2flux(yprims_l, 2);
                g_r = self->prims2flux(yprims_r, 2);

                if (hllc)
                {
                    // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    //     f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // } else {
                    //     f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // }
                    
                    // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    //     g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // } else {
                    //     g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // }
                    f1 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                else
                {
                    f1 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
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

                yprims_r.p =
                    center.p - (real)0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                            (real)0.5 * (yright_mid.p - yleft_mid.p),
                                            plm_theta * (yright_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // Primitive
                ux_l = self->prims2cons(xprims_l);
                ux_r = self->prims2cons(xprims_r);
                uy_l = self->prims2cons(yprims_l);
                uy_r = self->prims2cons(yprims_r);

                f_l = self->prims2flux(xprims_l, 1);
                f_r = self->prims2flux(xprims_r, 1);
                g_l = self->prims2flux(yprims_l, 2);
                g_r = self->prims2flux(yprims_r, 2);

                
                if (hllc)
                {
                    // if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    //     f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // } else {
                    //     f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    // }
                    
                    // if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    //     g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // } else {
                    //     g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    // }
                    f2 = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);   
                }
                else
                {
                    f2 = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
            //Advance depending on geometry
            luint real_loc = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        #if GPU_CODE
                            // real dx = coord_lattice->gpu_dx1[ii];
                            // real dy = coord_lattice->gpu_dx2[jj];
                            self->gpu_cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx1 - (g1.D   - g2.D ) / dx2 + self->gpu_sourceD [real_loc] );
                            self->gpu_cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx1 - (g1.S1  - g2.S1) / dx2 + self->gpu_sourceS1[real_loc] );
                            self->gpu_cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx1 - (g1.S2  - g2.S2) / dx2 + self->gpu_sourceS2[real_loc] );
                            self->gpu_cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx1 - (g1.tau - g2.tau)/ dx2 + self->gpu_sourceTau [real_loc]);
                        #else
                            real dx = self->coord_lattice.dx1[ii];
                            real dy = self->coord_lattice.dx2[jj];
                            self->cons[aid].D   += (real)0.5 * dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + sourceD   [real_loc] );
                            self->cons[aid].S1  += (real)0.5 * dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + sourceS1  [real_loc] );
                            self->cons[aid].S2  += (real)0.5 * dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + sourceS2  [real_loc] );
                            self->cons[aid].tau += (real)0.5 * dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + sourceTau [real_loc] );
                        #endif
                    

                    break;
                    }
                
                case simbi::Geometry::SPHERICAL:
                    {
                    #if GPU_CODE
                    const real rl           = (ii > 0)    ? xmin * pow(10, (ii - 0.5) * dlogx1) : xmin;
                    const real rr           = (ii < imax) ? rl   * pow_dlogr: xmax;
                    const real tl           = (jj > 0)    ? ymin + (jj - 0.5) * dx2: ymin;
                    const real tr           = (jj < jmax) ? tl + dx2 : ymax; 
                    const real rmean        = (real)0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                    const real s1R          = rr * rr; //coord_lattice->gpu_x1_face_areas[ii + 1];
                    const real s1L          = rl * rl; //coord_lattice->gpu_x1_face_areas[ii + 0];
                    const real s2R          = sin(tr); //coord_lattice->gpu_x2_face_areas[jj + 1];
                    const real s2L          = sin(tl); //coord_lattice->gpu_x2_face_areas[jj + 0];
                    const real thmean       = 0.5 * (tl + tr);
                    const real sint         = sin(thmean);
                    const real dV1          = rmean * rmean * (rr - rl);             
                    const real dV2          = rmean * sint * (tr - tl); 
                    const real cot          = cos(thmean) / sint ;

                    const real d_source  = (d_all_zeros)   ? 0 : self->gpu_sourceD[real_loc];
                    const real s1_source = (s1_all_zeros)  ? 0 : self->gpu_sourceS1[real_loc];
                    const real s2_source = (s2_all_zeros)  ? 0 : self->gpu_sourceS2[real_loc];
                    const real e_source  = (e_all_zeros)   ? 0 : self->gpu_sourceTau[real_loc];
                    #else 
                    real s1R   = coord_lattice.x1_face_areas[ii + 1];
                    real s1L   = coord_lattice.x1_face_areas[ii + 0];
                    real s2R   = coord_lattice.x2_face_areas[jj + 1];
                    real s2L   = coord_lattice.x2_face_areas[jj + 0];
                    real rmean = coord_lattice.x1mean[ii];
                    real dV1   = coord_lattice.dV1[ii];
                    real dV2   = rmean * coord_lattice.dV2[jj];
                    #endif
                    // Grab central primitives
                    real rhoc = prim_buff[tya * sx + txa * sy].rho;
                    real pc   = prim_buff[tya * sx + txa * sy].p;
                    real uc   = prim_buff[tya * sx + txa * sy].v1;
                    real vc   = prim_buff[tya * sx + txa * sy].v2;

                    real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
                    real gam2 = 1 / (1 - (uc * uc + vc * vc));

                    #if GPU_CODE 
                        self->gpu_cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + d_source * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + rhoc * hc * gam2 * vc * vc / rmean + 2 * pc / rmean +
                                            s1_source * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc / rmean - pc * cot / rmean) 
                                            + s2_source* decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + e_source * decay_const
                            } * dt * (real)0.5;
                    #else
                        cons[aid] += Conserved{
                                // L(D)
                                -(f1.D * s1R - f2.D * s1L) / dV1 
                                    - (g1.D * s2R - g2.D * s2L) / dV2 
                                        + sourceD[real_loc] * decay_const,

                                // L(S1)
                                -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                    - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                        + (rhoc * hc * gam2 * vc * vc + 2 * pc ) / rmean +
                                            sourceS1[real_loc] * decay_const,

                                // L(S2)
                                -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                        - (rhoc * hc * gam2 * uc * vc - pc * coord_lattice.cot[jj]) / rmean 
                                            + sourceS2[real_loc] * decay_const,

                                // L(tau)
                                -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                    - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                        + sourceTau[real_loc] * decay_const
                            } * dt * (real)0.5;
                    #endif
                    
                    break;
                    }
            } // end switch
        }

    });
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
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
    std::string tnow, tchunk, tstep, filename;
    luint total_zones = nx * ny;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + (real)0.5) / round_place
               : floor(tstart * round_place + (real)0.5) / round_place + chkpt_interval;

    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->plm_theta = plm_theta;
    this->dt    = init_dt;

    if (first_order)
    {
        this->xphysical_grid = nx - 2;
        this->yphysical_grid = ny - 2;
        this->idx_active = 1;
        this->i_start = 1;
        this->j_start = 1;
        this->i_bound = nx - 1;
        this->j_bound = ny - 1;
    }
    else
    {
        this->xphysical_grid = nx - 4;
        this->yphysical_grid = ny - 4;
        this->idx_active = 2;
        this->i_start = 2;
        this->j_start = 2;
        this->i_bound = nx - 2;
        this->j_bound = ny - 2;
    }

    this->active_zones = xphysical_grid * yphysical_grid;

    //--------Config the System Enums
    config_system();
    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else
    {
        this->coord_lattice = CLattice2D(x1, x2, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    if (coord_lattice.x2vertices[yphysical_grid] == PI){
        bipolar = true;
    }
    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.xmax = x1[xphysical_grid - 1];
    setup.xmin = x1[0];
    setup.ymax = x2[yphysical_grid - 1];
    setup.ymin = x2[0];
    setup.nx = nx;
    setup.ny = ny;
    setup.linspace = linspace;

    cons.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);
    // Define the source terms
    sourceD    = sources[0];
    sourceS1   = sources[1];
    sourceS2   = sources[2];
    sourceTau  = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        auto D            = state2D[0][i];
        auto S1           = state2D[1][i];
        auto S2           = state2D[2][i];
        auto E            = state2D[3][i];
        auto S            = sqrt(S1 * S1 + S2 * S2);
        cons[i]           = Conserved(D, S1, S2, E);
        pressure_guess[i] = std::abs(S - D - E);
    }
    // deallocate initial state vector
    std::vector<int> state2D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1 / (1 + exp((real)10.0 * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    
    // Copy the current SRHD instance over to the device
    SRHD2D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(SRHD2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(SRHD2D));
    simbi::dual::DualSpace2D<Primitive, Conserved, SRHD2D> dualMem;
    dualMem.copyHostToDev(*this, device_self);

    if constexpr(BuildPlatform == Platform::GPU)
    {   
        dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
        dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
        dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
        x1min   = x1[0];
        x1max   = x1[xphysical_grid - 1];
        x2min   = x2[0];
        x2max   = x2[yphysical_grid - 1];

        d_all_zeros  = std::all_of(sourceD.begin(),   sourceD.end(),   [](real i) {return i == 0;});
        s1_all_zeros = std::all_of(sourceS1.begin(),  sourceS1.end(),  [](real i) {return i == 0;});
        s2_all_zeros = std::all_of(sourceS2.begin(),  sourceS2.end(),  [](real i) {return i == 0;});
        e_all_zeros  = std::all_of(sourceTau.begin(), sourceTau.end(), [](real i) {return i == 0;});
    }
    
    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // // Setup the system
    const luint xblockdim       = xphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : xphysical_grid;
    const luint yblockdim       = yphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : yphysical_grid;
    const luint radius          = (first_order) ? 1 : 2;
    const luint bx              = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint by              = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const luint shBlockSpace    = bx * by;
    const luint shBlockBytes    = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({yphysical_grid, xphysical_grid}, {yblockdim, xblockdim}, shBlockBytes);

    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU)
        {
            config_ghosts2DGPU(fullP, device_self, nx, ny, true);
        } else {
            config_ghosts2DGPU(fullP, this, nx, ny, true);
        }
    }

    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, geometry[coord_system], activeP);
    } else
    {
        cons2prim(fullP);
        adapt_dt();
    }
        
    simbi::gpu::api::deviceSynch();
    
    // Some benchmarking tools 
    luint      n   = 0;
    luint  nfold   = 0;
    luint  ncheck  = 0;
    real    zu_avg = 0;
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
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, true);
            } else {
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, true);
            }
            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                // simbi::util::writeln("Iteration: {0} \t dt: {1} \t time: {2} \t Zones/sec: {3}", n, dt, t, total_zones / delta_t.count());
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< total_zones / delta_t.count() << std::flush;
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
                
                transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
                writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
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
            {
                adapt_dt(device_self, geometry[coord_system], activeP);
            } else {
                adapt_dt();
            }
            // std::cin.get();

            // Update decay constant
            decay_const = 1 / (1 + exp((real)10.0 * (t - engine_duration)));
        }
    } else {
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            if constexpr(BuildPlatform == Platform::GPU)
            {
                // First Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, false);

                // Final Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Dev);
                cons2prim(fullP, device_self, simbi::MemSide::Dev);
                config_ghosts2DGPU(fullP, device_self, nx, ny, false);
            } else {
                // First Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, false);

                // Final Half Step
                advance(device_self, activeP, bx, by, radius, geometry[coord_system], simbi::MemSide::Host);
                cons2prim(fullP);
                config_ghosts2DGPU(fullP, this, nx, ny, false);
            }
            

            t += dt; 

            if (n >= nfold){
                ncheck += 1;
                simbi::gpu::api::deviceSynch();
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones/ delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< total_zones/ delta_t.count() << std::flush;
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
                
                transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);
                writeToProd<sr2d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            n++;

            // Update decay constant
            decay_const = 1 / (1 + exp((real)10.0 * (t - engine_duration)));

            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry[coord_system], activeP);
            } else {
                adapt_dt();
            }
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

    // cons2prim2D();
    transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(4, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
