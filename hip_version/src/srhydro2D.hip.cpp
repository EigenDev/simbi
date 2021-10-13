/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helpers.hpp"
#include "helpers.hip.hpp"
#include "srhydro2D.hip.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace std::chrono;

__device__
bool simbi::quirk_strong_shock(real pl, real pr)
{
    return abs(pr - pl) / my_min(pl, pr) > a;
}
// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(std::vector<std::vector<real>> state2D, int NX, int NY, real gamma,
               std::vector<real> x1, std::vector<real> x2, real CFL,
               std::string coord_system = "cartesian")
:

    NX(NX),
    NY(NY),
    nzones(state2D[0].size()),
    state2D(state2D),
    gamma(gamma),
    x1(x1),
    x2(x2),
    CFL(CFL),
    coord_system(coord_system)
{
}

// Destructor
SRHD2D::~SRHD2D() {}

/* Define typedefs because I am lazy */
typedef sr2d::Primitive Primitive;
typedef sr2d::Conserved Conserved;
typedef sr2d::Eigenvals Eigenvals;

//================================================
//              DUAL SPACE FOR 2D SRHD
//================================================
SRHD2D_DualSpace::SRHD2D_DualSpace(){}

SRHD2D_DualSpace::~SRHD2D_DualSpace()
{
    printf("\nFreeing Device Memory...\n");
    hipFree(host_u0);
    hipFree(host_prims);
    hipFree(host_clattice);
    hipFree(host_dV1);
    hipFree(host_dV2);
    hipFree(host_dx1);
    hipFree(host_dx2);
    hipFree(host_fas1);
    hipFree(host_fas2);
    hipFree(host_x1m);
    hipFree(host_cot);
    hipFree(host_source0);
    hipFree(host_sourceD);
    hipFree(host_sourceS1);
    hipFree(host_sourceS2);
    hipFree(host_pressure_guess);
    printf("Memory Freed.\n");
}
void SRHD2D_DualSpace::copyStateToGPU(
    const simbi::SRHD2D &host,
    simbi::SRHD2D *device
)
{
    int nx     = host.NX;
    int ny     = host.NY;
    int nxreal = host.xphysical_grid; 
    int nyreal = host.yphysical_grid;

    int nzones = nx * ny;
    int nzreal = nxreal * nyreal;

    // Precompute byes
    int cbytes  = nzones * sizeof(Conserved);
    int pbytes  = nzones * sizeof(Primitive);
    int rbytes  = nzones * sizeof(real);

    int rrbytes  = nzreal * sizeof(real);
    int r1bytes  = nxreal * sizeof(real);
    int r2bytes  = nyreal * sizeof(real);
    int fa1bytes = host.coord_lattice.x1_face_areas.size() * sizeof(real);
    int fa2bytes = host.coord_lattice.x2_face_areas.size() * sizeof(real);

    //--------Allocate the memory for pointer objects-------------------------
    hipMalloc((void **)&host_u0,              cbytes);
    hipMalloc((void **)&host_prims,           pbytes);
    hipMalloc((void **)&host_pressure_guess,  rbytes);
    hipMalloc((void **)&host_dx1,             r1bytes);
    hipMalloc((void **)&host_dx2,             r2bytes);
    hipMalloc((void **)&host_dV1,             r1bytes);
    hipMalloc((void **)&host_dV2,             r2bytes);
    hipMalloc((void **)&host_x1m,             r1bytes);
    hipMalloc((void **)&host_cot,             r2bytes);
    hipMalloc((void **)&host_fas1,            fa1bytes);
    hipMalloc((void **)&host_fas2,            fa2bytes);
    hipMalloc((void **)&host_source0,         rrbytes);
    hipMalloc((void **)&host_sourceD,         rrbytes);
    hipMalloc((void **)&host_sourceS1,        rrbytes);
    hipMalloc((void **)&host_sourceS2,        rrbytes);

    hipMalloc((void **)&host_dtmin,            rbytes);
    hipMalloc((void **)&host_clattice, sizeof(CLattice2D));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.u0.data(), cbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.u0 to host_u0");

    hipMemcpy(host_prims, host.prims.data()    , pbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.prims to host_prims");

    hipMemcpy(host_pressure_guess, host.pressure_guess.data() , rbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.pressure_guess to host_pre_guess");

    hipMemcpy(host_source0, host.source_tau.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.source0 to host_source0");

    hipMemcpy(host_sourceD, host.sourceD.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sourceD to host_sourceD");

    hipMemcpy(host_sourceS1, host.source_S1.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sourceS1 to host_sourceS1");

    hipMemcpy(host_sourceS2, host.source_S2.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sourceS2 to host_sourceS2");

    // copy pointer to allocated device storage to device class
    if ( hipMemcpy(&(device->gpu_cons), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_u0 -> device_sys_tate\n");
    };

    if( hipMemcpy(&(device->gpu_prims),     &host_prims, sizeof(Primitive *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_prims -> device_prims\n");
    };

    if( hipMemcpy(&(device->gpu_pressure_guess),  &host_pressure_guess, sizeof(real *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_pressure_guess -> device_gpu_pressure_guess\n");
    };

    hipMemcpy(&(device->gpu_sourceTau), &host_source0, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying source0 to device");

    hipMemcpy(&(device->gpu_sourceD), &host_sourceD, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying sourceD to device");

    hipMemcpy(&(device->gpu_sourceS1), &host_sourceS1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying sourceS1 to device");

    hipMemcpy(&(device->gpu_sourceS2), &host_sourceS2, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying sourceS1 to device");

    hipMemcpy(&(device->dt_min), &host_dtmin, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying min_dt to device");

    // ====================================================
    //          GEOMETRY DEEP COPY
    //=====================================================
    hipMemcpy(host_dx1, host.coord_lattice.dx1.data() , r1bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(host_dx2, host.coord_lattice.dx2.data() , r2bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(host_dV1,  host.coord_lattice.dV1.data(), r1bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dV1");

    hipMemcpy(host_dV2,  host.coord_lattice.dV2.data(), r2bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dV2");

    hipMemcpy(host_fas1, host.coord_lattice.x1_face_areas.data() , fa1bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1 face areas");

    hipMemcpy(host_fas2, host.coord_lattice.x2_face_areas.data() , fa2bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x2 face areas");

    hipMemcpy(host_x1m, host.coord_lattice.x1mean.data(), r1bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1mean");

    hipMemcpy(host_cot, host.coord_lattice.cot.data(), r2bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring cot");

    // Now copy pointer to device directly
    hipMemcpy(&(device->coord_lattice.gpu_dx1), &host_dx1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_dx2), &host_dx2, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx2");

    hipMemcpy(&(device->coord_lattice.gpu_dV1), &host_dV1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dV1");

    hipMemcpy(&(device->coord_lattice.gpu_dV2), &host_dV2, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dV2");

    hipMemcpy(&(device->coord_lattice.gpu_x1mean),&host_x1m, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1m");

    hipMemcpy(&(device->coord_lattice.gpu_cot),&host_cot, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring cot");

    hipMemcpy(&(device->coord_lattice.gpu_x1_face_areas), &host_fas1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1 face areas");

    hipMemcpy(&(device->coord_lattice.gpu_x2_face_areas), &host_fas2, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x2 face areas");

    hipMemcpy(&(device->dt),          &host.dt      ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->plm_theta),   &host.plm_theta,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->gamma),       &host.gamma   ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->CFL)  ,       &host.CFL     ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->NX),          &host.NX      ,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->NY),          &host.NY      ,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->i_bound),     &host.i_bound,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->i_start),     &host.i_start,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->j_bound),     &host.j_bound,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->j_start),     &host.j_start,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->idx_active),  &host.idx_active, sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->decay_const), &host.decay_const, sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->xphysical_grid),&host.xphysical_grid,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->yphysical_grid),&host.yphysical_grid,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->active_zones),  &host.active_zones,    sizeof(int),  hipMemcpyHostToDevice);
    
}

void SRHD2D_DualSpace::copyGPUStateToHost(
    const simbi::SRHD2D *device,
    simbi::SRHD2D &host
)
{
    const int nx     = host.NX;
    const int ny     = host.NY;
    const int cbytes = nx * ny * sizeof(Conserved); 
    const int pbytes = nx * ny * sizeof(Primitive);

    hipMemcpy(host.u0.data(), host_u0, cbytes, hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device conservatives to host");
    hipMemcpy(host.prims.data(),   host_prims , pbytes, hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device prims to host");
    
}
//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------

void SRHD2D::cons2prim2D()
{
    /**
   * Return a 2D matrix containing the primitive
   * variables density , pressure, and
   * three-velocity
   */

    real S1, S2, S, D, tau, tol;
    real W, v1, v2;

    // Define Newton-Raphson Vars
    real etotal, c2, f, g, p, peq;
    real Ws, rhos, eps, h;

    int aid  = 0;
    int iter = 0;
    int maximum_iteration = 50;
    for (int jj = 0; jj < NY; jj++)
    {
        for (int ii = 0; ii < NX; ii++)
        {
            aid = ii + NX * jj;
            D   = u0[aid].D;     // Relativistic Mass Density
            S1  = u0[aid].S1;   // X1-Momentum Denity
            S2  = u0[aid].S2;   // X2-Momentum Density
            tau = u0[aid].tau; // Energy Density
            S = sqrt(S1 * S1 + S2 * S2);

            peq = (n != 0.0) ? pressure_guess[aid] : abs(S - D - tau);

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
                Ws = 1.0 / sqrt(1.0 - v2);
                rhos = D / Ws;
                eps = (tau + D * (1. - Ws) + (1. - Ws * Ws) * p) / (D * Ws);
                f = (gamma - 1.0) * rhos * eps - p;

                h = 1. + eps + p / rhos;
                c2 = gamma * p / (h * rhos);
                g = c2 * v2 - 1.0;
                peq = p - f / g;
                iter++;

                if (iter > maximum_iteration)
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

            } while (abs(peq - p) >= tol);
        

            v1 = S1 / (tau + D + peq);
            v2 = S2 / (tau + D + peq);
            Ws = 1.0 / sqrt(1.0 - (v1 * v1 + v2 * v2));

            // Update the Gamma array
            // lorentz_gamma[ii + NX * jj] = Ws;

            // Update the pressure guess for the next time step
            pressure_guess[aid] = peq;
            prims[aid] = Primitive{D / Ws, v1, v2, peq};
        }
    }
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD2D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const unsigned int nhat = 1)
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

    switch (nhat)
    {
    case 1:
    {
        const real v1_l = prims_l.v1;
        const real v1_r = prims_r.v1;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const real vbar  = 0.5 * (v1_l + v1_r);
        const real cbar  = 0.5 * (cs_l + cs_r);
        const real bl    = (vbar - cbar)/(1. - cbar*vbar);
        const real br    = (vbar + cbar)/(1. + cbar*vbar);
        const real aL    = my_min(bl, (v1_l - cs_l)/(1. - v1_l*cs_l));
        const real aR    = my_max(br, (v1_r + cs_r)/(1. + v1_r*cs_r));

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

        return Eigenvals(aL, aR);
    }
    case 2:
        const real v2_r = prims_r.v2;
        const real v2_l = prims_l.v2;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const real vbar  = 0.5 * (v2_l + v2_r);
        const real cbar  = 0.5 * (cs_l + cs_r);
        const real bl    = (vbar - cbar)/(1. - cbar*vbar);
        const real br    = (vbar + cbar)/(1. + cbar*vbar);
        const real aL    = my_min(bl, (v2_l - cs_l)/(1. - v2_l*cs_l));
        const real aR    = my_max(br, (v2_r + cs_r)/(1. + v2_r*cs_r));

        // return Eigenvals(aL, aR);

        // Calc the wave speeds based on Mignone and Bodo (2005)
        // real sL = cs_l * cs_l * (1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
        // real sR = cs_r * cs_r * (1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

        // // Define some temporaries to save a few cycles
        // const real qfL = 1. / (1. + sL);
        // const real qfR = 1. / (1. + sR);
        // const real sqrtR = sqrt(sR * (1 - v2_r * v2_r + sR));
        // const real sqrtL = sqrt(sL * (1 - v2_l * v2_l + sL));

        // const real lamLm = (v2_l - sqrtL) * qfL;
        // const real lamRm = (v2_r - sqrtR) * qfR;
        // const real lamLp = (v2_l + sqrtL) * qfL;
        // const real lamRp = (v2_r + sqrtR) * qfR;
        // const real aL = lamLm < lamRm ? lamLm : lamRm;
        // const real aR = lamLp > lamRp ? lamLp : lamRp;

        return Eigenvals(aL, aR);
    }
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
                                           int nhat = 1)
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
__device__ void simbi::warp_reduce_min(volatile real smem[BLOCK_SIZE2D][BLOCK_SIZE2D])
{
    for (int stridey = BLOCK_SIZE2D /2; stridey >= 1; stridey /=  2)
    {
        for (int stridex = BLOCK_SIZE2D/2; stridex >= 1; stridex /= 2)
        {
            smem[threadIdx.y][threadIdx.x] = smem[threadIdx.y+stridey][threadIdx.x+stridex] 
                < smem[threadIdx.y][threadIdx.x] ? smem[threadIdx.y+stridey][threadIdx.x+stridex] 
                : smem[threadIdx.y][threadIdx.x];
        }
    }

}

// Adapt the CFL conditonal timestep
__global__ void simbi::adapt_dtGPU(
    SRHD2D *s, 
    const simbi::Geometry geometry)
{
    real cfl_dt, rmean;
    real gamma = s->gamma;
    __shared__ volatile real dt_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];
    __shared__ Primitive prim_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ii  = blockDim.x * blockIdx.x + threadIdx.x;
    const int jj  = blockDim.y * blockIdx.y + threadIdx.y;
    const int ia  = ii + s->idx_active;
    const int ja  = jj + s->idx_active;
    const int aid = ja * s-> NX + ia;
    const int nx  = s->NX;

    const CLattice2D *coord_lattice = &(s->coord_lattice);

    if ( (ii < s->xphysical_grid) && (jj < s->yphysical_grid))
    {   
        prim_buff[ty][tx] = s->gpu_prims[aid];
        __syncthreads();

        real dx1  = s->coord_lattice.gpu_dx1[ii];
        real dx2  = s->coord_lattice.gpu_dx2[jj];
        real rho  = prim_buff[ty][tx].rho;
        real p    = prim_buff[ty][tx].p;
        real v1   = prim_buff[ty][tx].v1;
        real v2   = prim_buff[ty][tx].v2;

        real h  = 1. + gamma * p / (rho * (gamma - 1.));
        real cs = sqrt(gamma * p / (rho * h));

        real plus_v1  = (v1 + cs) / (1. + v1 * cs);
        real plus_v2  = (v2 + cs) / (1. + v2 * cs);
        real minus_v1 = (v1 - cs) / (1. - v1 * cs);
        real minus_v2 = (v2 - cs) / (1. - v2 * cs);

        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                cfl_dt = my_min(dx1 / (my_max(abs(plus_v1), abs(minus_v1))),
                                dx2 / (my_max(abs(plus_v2), abs(minus_v2))));
                break;
            
            case simbi::Geometry::SPHERICAL:
                // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                rmean = coord_lattice->gpu_x1mean[ii];
                cfl_dt = my_min(dx1 / (my_max(abs(plus_v1), abs(minus_v1))),
                                rmean * dx2 / (my_max(abs(plus_v2), abs(minus_v2))));
                break;
        } // end switch

        dt_buff[threadIdx.y][threadIdx.x] = s->CFL * cfl_dt;
        __syncthreads();

        if ((threadIdx.x < BLOCK_SIZE2D / 2) && (threadIdx.y < BLOCK_SIZE2D / 2))
        {
            warp_reduce_min(dt_buff);
        }
        if((threadIdx.x == 0) && (threadIdx.y == 0) )
        {
            // printf("min dt: %f", s->dt_min[0]);
            s->dt_min[blockIdx.x + blockIdx.y * blockDim.x] = dt_buff[threadIdx.y][threadIdx.x]; // dt_min[0] == minimum
            // printf("min dt: %f\n", s->dt_min[0]);
            s->dt = s->dt_min[0];
        }
        
    }
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
GPU_CALLABLE_MEMBER
Conserved SRHD2D::prims2flux(const Primitive &prims, unsigned int nhat = 1)
{

    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1. - (vx * vx + vy * vy));

    const real h = 1. + gamma * pressure / (rho * (gamma - 1));
    const real D = rho * lorentz_gamma;
    const real S1 = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const real S2 = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const real tau =
                    rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return (nhat == 1) ? Conserved(D * vx, S1 * vx + pressure, S2 * vx,
                                   (tau + pressure) * vx)
                       : Conserved(D * vy, S1 * vy, S2 * vy + pressure,
                                   (tau + pressure) * vy);
};

GPU_CALLABLE_MEMBER
Conserved SRHD2D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const unsigned int nhat)
{
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    // Calculate plus/minus alphas
    const real aLminus = aL < 0.0 ? aL : 0.0;
    const real aRplus  = aR > 0.0 ? aR : 0.0;

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
    const unsigned int nhat = 1)
{

    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const real aL = lambda.aL;
    const real aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if (0.0 <= aL)
    {
        return left_flux;
    }
    else if (0.0 >= aR)
    {
        return right_flux;
    }

    const real aLminus = aL < 0.0 ? aL : 0.0;
    const real aRplus  = aR > 0.0 ? aR : 0.0;

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
    const real quad = -0.5 * (b + sgn(b) * sqrt(b * b - 4.0 * a * c));
    const real aStar = c * (1.0 / quad);
    const real pStar = -aStar * fe + fs;

    // return Conserved(0.0, 0.0, 0.0, 0.0);
    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D = left_state.D;
        const real S1 = left_state.S1;
        const real S2 = left_state.S2;
        const real tau = left_state.tau;
        const real E = tau + D;
        const real cofactor = 1. / (aL - aStar);
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
            const real Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
            const real tauStar  = Estar - Dstar;

            const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

            //---------Compute the L Star Flux
            return left_flux + (interstate_left - left_state) * aL;
        }

        case 2:
            const real v2 = left_prims.v2;
            // Start States in y-direction in the coordinate lattice
            const real Dstar   = cofactor * (aL - v2) * D;
            const real S1star  = cofactor * (aL - v2) * S1;
            const real S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
            const real Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
            const real tauStar = Estar - Dstar;

            const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

            //---------Compute the L Star Flux
            return left_flux + (interstate_left - left_state) * aL;
        }
    }
    else
    {
        const real pressure = right_prims.p;
        const real D = right_state.D;
        const real S1 = right_state.S1;
        const real S2 = right_state.S2;
        const real tau = right_state.tau;
        const real E = tau + D;
        const real cofactor = 1. / (aR - aStar);

        /* Compute the L/R Star State */
        switch (nhat)
        {
        case 1:
        {
            const real v1 = right_prims.v1;
            const real Dstar = cofactor * (aR - v1) * D;
            const real S1star = cofactor * (S1 * (aR - v1) - pressure + pStar);
            const real S2star = cofactor * (aR - v1) * S2;
            const real Estar = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
            const real tauStar = Estar - Dstar;

            const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

            // Compute the intermediate right flux
            return right_flux + (interstate_right - right_state) * aR;
        }

        case 2:
            const real v2 = right_prims.v2;
            // Start States in y-direction in the coordinate lattice
            const real cofactor = 1. / (aR - aStar);
            const real Dstar = cofactor * (aR - v2) * D;
            const real S1star = cofactor * (aR - v2) * S1;
            const real S2star = cofactor * (S2 * (aR - v2) - pressure + pStar);
            const real Estar = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
            const real tauStar = Estar - Dstar;

            const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

            // Compute the intermediate right flux
            return right_flux + (interstate_right - right_state) * aR;
        }
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================

//=====================================================================
//                          KERNEL CALLS
//=====================================================================
__global__ void simbi::shared_gpu_cons2prim(SRHD2D *s){
    __shared__ Conserved  conserved_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];

    real eps, p, v2, et, c2, h, g, f, W, rho;
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    int jj = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int nx = s->NX;
    int iter = 0;
    if ((ii < s->NX) && (jj < s->NY)){
        int gid = jj * nx + ii;
        // load shared memory
        conserved_buff[ty][tx] = s->gpu_cons[gid];
        __syncthreads();
        real D    = conserved_buff[ty][tx].D;
        real S1   = conserved_buff[ty][tx].S1;
        real S2   = conserved_buff[ty][tx].S2;
        real tau  = conserved_buff[ty][tx].tau;
        real S    = sqrt(S1 * S1 + S2 * S2);

        real peq = s->gpu_pressure_guess[gid];

        real tol = D * 1.e-12;
        do
        {
            p = peq;
            et = tau + D + p;
            v2 = S * S / (et * et);
            W = 1.0 / sqrt(1.0 - v2);
            rho = D / W;

            eps = (tau + (1.0 - W) * D + (1. - W * W) * p) / (D * W);

            h = 1. + eps + p / rho;
            c2 = s->gamma * p / (h * rho);

            g = c2 * v2 - 1.0;
            f = (s->gamma - 1.0) * rho * eps - p;

            peq = p - f / g;
            iter++;
            if (iter >= MAX_ITER)
            {
                printf("\nCons2Prim cannot converge\n");
                s->dt = INFINITY;
                return;
            }

        } while (abs(peq - p) >= tol);

        real v1 = S1 / (tau + D + peq);
        real v2 = S2 / (tau + D + peq);

        s->gpu_pressure_guess[gid] = peq;
        s->gpu_prims[gid]          = Primitive{D * sqrt(1 - (v1 * v1 + v2 * v2)), v1, v2, peq};
    }
}

__global__ void simbi::shared_gpu_advance(
    SRHD2D *s,  
    const int sh_block_size,
    const int sh_block_space,
    const int radius, 
    const simbi::Geometry geometry)
{
    const int ii  = blockDim.x * blockIdx.x + threadIdx.x;
    const int jj  = blockDim.y * blockIdx.y + threadIdx.y;

    const int txa = threadIdx.x + radius;
    const int tya = threadIdx.y + radius;

    const int nx                    = s->NX;
    const int ny                    = s->NY;
    const int bs                    = sh_block_size;
    const int xpg                   = s->xphysical_grid;
    const real decay_constant       = s->decay_const;
    const CLattice2D *coord_lattice = &(s->coord_lattice);
    const real dt                   = s->dt;
    const real plm_theta            = s->plm_theta;
    const real gamma                = s->gamma;

    extern __shared__ Primitive prim_buff[];

    Conserved ux_l, ux_r, uy_l, uy_r;
    Conserved f_l, f_r, g_l, g_r, f1, f2, g1, g2;
    Primitive xprims_l, xprims_r, yprims_l, yprims_r;

    int ia  = ii + radius;
    int ja  = jj + radius;
    const bool xinbounds = ia + BLOCK_SIZE2D < nx - 1;
    const bool yinbounds = ja + BLOCK_SIZE2D < ny - 1;
    int aid = ja * nx + ia;


    // Check if we are at phyical grid boundary. If so, the active thread index
    // there becomes the ghost zone
    if ((ii >= s->xphysical_grid) || (jj >= s->yphysical_grid)){
        prim_buff[tya * bs + txa] = s->gpu_prims[ja * nx + nx - 1]; 
        return;
    }
    // Load Shared memory into buffer for active zones plus ghosts
    prim_buff[tya * bs + txa] = s->gpu_prims[aid];
    if (threadIdx.y < radius)    
    {
        prim_buff[(tya - radius      ) * bs + txa] = s->gpu_prims[(ja - radius) * nx + ia];
        prim_buff[(tya + BLOCK_SIZE2D) * bs + txa] = yinbounds ? s->gpu_prims[(ja + BLOCK_SIZE2D) * nx + ia] :  s->gpu_prims[(ny - 1) * nx + ia]; 
    
    }
    if (threadIdx.x < radius)
    {   
        prim_buff[tya * bs + txa - radius      ] =  s->gpu_prims[(ja * nx) + ia - radius];
        prim_buff[tya * bs + txa + BLOCK_SIZE2D] =  xinbounds ? s->gpu_prims[ja * nx + ia + BLOCK_SIZE2D] :  s->gpu_prims[ja * nx + nx - 1]; 
    }
    __syncthreads();

    if (s->first_order)
    {
        if (s->periodic)
        {
            xprims_l = prim_buff[txa + tya * bs];
            xprims_r = roll(prim_buff, (txa + 1) + tya * bs, sh_block_space);

            yprims_l = prim_buff[txa + tya * bs];
            yprims_r = roll(prim_buff, txa + (tya + 1) * bs, sh_block_space);
        }
        else
        {
            xprims_l = prim_buff[tya * bs + (txa + 0)];
            xprims_r = prim_buff[tya * bs + (txa + 1)];
            //j+1/2
            yprims_l = prim_buff[(tya + 0) * bs + txa];
            yprims_r = prim_buff[(tya + 1) * bs + txa];
        }
        
        // i+1/2
        ux_l = s->prims2cons(xprims_l); 
        ux_r = s->prims2cons(xprims_r); 
        // j+1/2
        uy_l = s->prims2cons(yprims_l);  
        uy_r = s->prims2cons(yprims_r); 

        f_l = s->prims2flux(xprims_l, 1);
        f_r = s->prims2flux(xprims_r, 1);

        g_l = s->prims2flux(yprims_l, 2);
        g_r = s->prims2flux(yprims_r, 2);

        // Calc HLL Flux at i+1/2 interface
        if (s-> hllc)
        {
            if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            } else {
                f1 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            }
            
            if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                g1 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                g1 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

        } else {
            f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            g1 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
        }

        // Set up the left and right state interfaces for i-1/2
        if (s->periodic)
        {
            xprims_l = roll(prim_buff,  txa - 1 + tya * bs, sh_block_space);
            xprims_r = prim_buff[txa + tya * bs];

            yprims_l = roll(prim_buff, txa + (tya - 1) * bs, sh_block_space);
            yprims_r = prim_buff[txa + tya * bs];
        }
        else
        {
            xprims_l = prim_buff[tya * bs + (txa - 1)];
            xprims_r = prim_buff[tya * bs + (txa + 0)];
            //j+1/2
            yprims_l = prim_buff[(tya - 1) * bs + txa]; 
            yprims_r = prim_buff[(tya + 0) * bs + txa]; 
        }

        // i+1/2
        ux_l = s->prims2cons(xprims_l); 
        ux_r = s->prims2cons(xprims_r); 
        // j+1/2
        uy_l = s->prims2cons(yprims_l);  
        uy_r = s->prims2cons(yprims_r); 

        f_l = s->prims2flux(xprims_l, 1);
        f_r = s->prims2flux(xprims_r, 1);

        g_l = s->prims2flux(yprims_l, 2);
        g_r = s->prims2flux(yprims_r, 2);

        // Calc HLL Flux at i-1/2 interface
        if (s-> hllc)
        {
            if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            } else {
                f2 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            }
            
            if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                g2 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                g2 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

        } else {
            f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
            g2 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
        }

        //Advance depending on geometry
        int real_loc = jj * xpg + ii;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                real dx = coord_lattice->gpu_dx1[ii];
                real dy = coord_lattice->gpu_dx2[jj];
                s->gpu_cons[aid].D   += dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + s->gpu_sourceD [real_loc] );
                s->gpu_cons[aid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + s->gpu_sourceS1[real_loc]);
                s->gpu_cons[aid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx  -(g1.S2  - g2.S2) / dy + s->gpu_sourceS2[real_loc]);
                s->gpu_cons[aid].tau += dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + s->gpu_sourceTau [real_loc] );

                break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                
                real s1R        = coord_lattice->gpu_x1_face_areas[ii + 1];
                real s1L        = coord_lattice->gpu_x1_face_areas[ii + 0];
                real s2R        = coord_lattice->gpu_x2_face_areas[jj + 1];
                real s2L        = coord_lattice->gpu_x2_face_areas[jj + 0];
                real rmean      = coord_lattice->gpu_x1mean[ii];
                real dV1        = coord_lattice->gpu_dV1[ii];
                real dV2        = rmean * coord_lattice->gpu_dV2[jj];
                // Grab central primitives
                real rhoc = prim_buff[tya * bs + txa].rho;
                real pc   = prim_buff[tya * bs + txa].p;
                real uc   = prim_buff[tya * bs + txa].v1;
                real vc   = prim_buff[tya * bs + txa].v2;

                real hc   = 1.0 + gamma * pc/(rhoc * (gamma - 1.0));
                real gam2  = 1.0/(1.0 - (uc * uc + vc * vc));

                s->gpu_cons[aid] += Conserved{
                            // L(D)
                            -(f1.D * s1R - f2.D * s1L) / dV1 
                                - (g1.D * s2R - g2.D * s2L) / dV2 
                                    + s->gpu_sourceD[real_loc] * decay_constant,

                            // L(S1)
                            -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                    + rhoc * hc * gam2 * vc * vc / rmean + 2.0 * pc / rmean +
                                        s->gpu_sourceS1[real_loc] * decay_constant,

                            // L(S2)
                            -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                    - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[jj] / rmean) 
                                        + s->gpu_sourceS2[real_loc] * decay_constant,

                            // L(tau)
                            -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                    + s->gpu_sourceTau[real_loc] * decay_constant
                        } * dt;
                break;
                }
        }
            
    }
    else
    {
        Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
        Primitive yleft_most, yright_most, yleft_mid, yright_mid;
        if (!(s->periodic))
            {
                // Coordinate X
                xleft_most  = prim_buff[(txa - 2) + bs * tya];
                xleft_mid   = prim_buff[(txa - 1) + bs * tya];
                center      = prim_buff[ txa      + bs * tya];
                xright_mid  = prim_buff[(txa + 1) + bs * tya];
                xright_most = prim_buff[(txa + 2) + bs * tya];

                // Coordinate Y
                yleft_most  = prim_buff[txa + bs * (tya - 2)];
                yleft_mid   = prim_buff[txa + bs * (tya - 1)];
                yright_mid  = prim_buff[txa + bs * (tya + 1)];
                yright_most = prim_buff[txa + bs * (tya + 2)];
            }
            else
            {
                // X Coordinate
                xleft_most   = roll(prim_buff, tya * bs + txa - 2, sh_block_space);
                xleft_mid    = roll(prim_buff, tya * bs + txa - 1, sh_block_space);
                center       = prim_buff[tya * bs + txa];
                xright_mid   = roll(prim_buff, tya * bs + txa + 1, sh_block_space);
                xright_most  = roll(prim_buff, tya * bs + txa + 2, sh_block_space);

                yleft_most   = roll(prim_buff, txa +  bs * (tya - 2), sh_block_space);
                yleft_mid    = roll(prim_buff, txa +  bs * (tya - 1), sh_block_space);
                yright_mid   = roll(prim_buff, txa +  bs * (tya + 1), sh_block_space);
                yright_most  = roll(prim_buff, txa +  bs * (tya + 2), sh_block_space);
            }
            // Reconstructed left X Primitive vector at the i+1/2 interface
            xprims_l.rho =
                center.rho + 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                            0.5 * (xright_mid.rho - xleft_mid.rho),
                                            plm_theta * (xright_mid.rho - center.rho));

            xprims_l.v1 =
                center.v1 + 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                            0.5 * (xright_mid.v1 - xleft_mid.v1),
                                            plm_theta * (xright_mid.v1 - center.v1));

            xprims_l.v2 =
                center.v2 + 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                            0.5 * (xright_mid.v2 - xleft_mid.v2),
                                            plm_theta * (xright_mid.v2 - center.v2));

            xprims_l.p =
                center.p + 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                        0.5 * (xright_mid.p - xleft_mid.p),
                                        plm_theta * (xright_mid.p - center.p));

            // Reconstructed right Primitive vector in x
            xprims_r.rho =
                xright_mid.rho -
                0.5 * minmod(plm_theta * (xright_mid.rho - center.rho),
                                0.5 * (xright_most.rho - center.rho),
                                plm_theta * (xright_most.rho - xright_mid.rho));

            xprims_r.v1 = xright_mid.v1 -
                            0.5 * minmod(plm_theta * (xright_mid.v1 - center.v1),
                                        0.5 * (xright_most.v1 - center.v1),
                                        plm_theta * (xright_most.v1 - xright_mid.v1));

            xprims_r.v2 = xright_mid.v2 -
                            0.5 * minmod(plm_theta * (xright_mid.v2 - center.v2),
                                        0.5 * (xright_most.v2 - center.v2),
                                        plm_theta * (xright_most.v2 - xright_mid.v2));

            xprims_r.p = xright_mid.p -
                            0.5 * minmod(plm_theta * (xright_mid.p - center.p),
                                        0.5 * (xright_most.p - center.p),
                                        plm_theta * (xright_most.p - xright_mid.p));

            // Reconstructed right Primitive vector in y-direction at j+1/2
            // interfce
            yprims_l.rho =
                center.rho + 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                            0.5 * (yright_mid.rho - yleft_mid.rho),
                                            plm_theta * (yright_mid.rho - center.rho));

            yprims_l.v1 =
                center.v1 + 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                            0.5 * (yright_mid.v1 - yleft_mid.v1),
                                            plm_theta * (yright_mid.v1 - center.v1));

            yprims_l.v2 =
                center.v2 + 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                            0.5 * (yright_mid.v2 - yleft_mid.v2),
                                            plm_theta * (yright_mid.v2 - center.v2));

            yprims_l.p =
                center.p + 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                        0.5 * (yright_mid.p - yleft_mid.p),
                                        plm_theta * (yright_mid.p - center.p));

            yprims_r.rho =
                yright_mid.rho -
                0.5 * minmod(plm_theta * (yright_mid.rho - center.rho),
                                0.5 * (yright_most.rho - center.rho),
                                plm_theta * (yright_most.rho - yright_mid.rho));

            yprims_r.v1 = yright_mid.v1 -
                            0.5 * minmod(plm_theta * (yright_mid.v1 - center.v1),
                                        0.5 * (yright_most.v1 - center.v1),
                                        plm_theta * (yright_most.v1 - yright_mid.v1));

            yprims_r.v2 = yright_mid.v2 -
                            0.5 * minmod(plm_theta * (yright_mid.v2 - center.v2),
                                        0.5 * (yright_most.v2 - center.v2),
                                        plm_theta * (yright_most.v2 - yright_mid.v2));

            yprims_r.p = yright_mid.p -
                            0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                        0.5 * (yright_most.p - center.p),
                                        plm_theta * (yright_most.p - yright_mid.p));

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            ux_l = s->prims2cons(xprims_l);
            ux_r = s->prims2cons(xprims_r);

            uy_l = s->prims2cons(yprims_l);
            uy_r = s->prims2cons(yprims_r);

            f_l = s->prims2flux(xprims_l, 1);
            f_r = s->prims2flux(xprims_r, 1);

            g_l = s->prims2flux(yprims_l, 2);
            g_r = s->prims2flux(yprims_r, 2);

            // favr = (uy_r - uy_l) * (-K);

            if (s->hllc)
            {
                if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                } else {
                    f1 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                }
                
                if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    g1 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    g1 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                // f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
            else
            {
                f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g1 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            // Do the same thing, but for the left side interface [i - 1/2]

            // Left side Primitive in x
            xprims_l.rho = xleft_mid.rho +
                            0.5 * minmod(plm_theta * (xleft_mid.rho - xleft_most.rho),
                                        0.5 * (center.rho - xleft_most.rho),
                                        plm_theta * (center.rho - xleft_mid.rho));

            xprims_l.v1 = xleft_mid.v1 +
                            0.5 * minmod(plm_theta * (xleft_mid.v1 - xleft_most.v1),
                                        0.5 * (center.v1 - xleft_most.v1),
                                        plm_theta * (center.v1 - xleft_mid.v1));

            xprims_l.v2 = xleft_mid.v2 +
                            0.5 * minmod(plm_theta * (xleft_mid.v2 - xleft_most.v2),
                                        0.5 * (center.v2 - xleft_most.v2),
                                        plm_theta * (center.v2 - xleft_mid.v2));

            xprims_l.p =
                xleft_mid.p + 0.5 * minmod(plm_theta * (xleft_mid.p - xleft_most.p),
                                            0.5 * (center.p - xleft_most.p),
                                            plm_theta * (center.p - xleft_mid.p));

            // Right side Primitive in x
            xprims_r.rho =
                center.rho - 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                            0.5 * (xright_mid.rho - xleft_mid.rho),
                                            plm_theta * (xright_mid.rho - center.rho));

            xprims_r.v1 =
                center.v1 - 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                            0.5 * (xright_mid.v1 - xleft_mid.v1),
                                            plm_theta * (xright_mid.v1 - center.v1));

            xprims_r.v2 =
                center.v2 - 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                            0.5 * (xright_mid.v2 - xleft_mid.v2),
                                            plm_theta * (xright_mid.v2 - center.v2));

            xprims_r.p =
                center.p - 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                        0.5 * (xright_mid.p - xleft_mid.p),
                                        plm_theta * (xright_mid.p - center.p));

            // Left side Primitive in y
            yprims_l.rho = yleft_mid.rho +
                            0.5 * minmod(plm_theta * (yleft_mid.rho - yleft_most.rho),
                                        0.5 * (center.rho - yleft_most.rho),
                                        plm_theta * (center.rho - yleft_mid.rho));

            yprims_l.v1 = yleft_mid.v1 +
                            0.5 * minmod(plm_theta * (yleft_mid.v1 - yleft_most.v1),
                                        0.5 * (center.v1 - yleft_most.v1),
                                        plm_theta * (center.v1 - yleft_mid.v1));

            yprims_l.v2 = yleft_mid.v2 +
                            0.5 * minmod(plm_theta * (yleft_mid.v2 - yleft_most.v2),
                                        0.5 * (center.v2 - yleft_most.v2),
                                        plm_theta * (center.v2 - yleft_mid.v2));

            yprims_l.p =
                yleft_mid.p + 0.5 * minmod(plm_theta * (yleft_mid.p - yleft_most.p),
                                            0.5 * (center.p - yleft_most.p),
                                            plm_theta * (center.p - yleft_mid.p));

            // Right side Primitive in y
            yprims_r.rho =
                center.rho - 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                            0.5 * (yright_mid.rho - yleft_mid.rho),
                                            plm_theta * (yright_mid.rho - center.rho));

            yprims_r.v1 =
                center.v1 - 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                            0.5 * (yright_mid.v1 - yleft_mid.v1),
                                            plm_theta * (yright_mid.v1 - center.v1));

            yprims_r.v2 =
                center.v2 - 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                            0.5 * (yright_mid.v2 - yleft_mid.v2),
                                            plm_theta * (yright_mid.v2 - center.v2));

            yprims_r.p =
                center.p - 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                        0.5 * (yright_mid.p - yleft_mid.p),
                                        plm_theta * (yright_mid.p - center.p));

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            ux_l = s->prims2cons(xprims_l);
            ux_r = s->prims2cons(xprims_r);
            uy_l = s->prims2cons(yprims_l);
            uy_r = s->prims2cons(yprims_r);

            f_l = s->prims2flux(xprims_l, 1);
            f_r = s->prims2flux(xprims_r, 1);
            g_l = s->prims2flux(yprims_l, 2);
            g_r = s->prims2flux(yprims_r, 2);

            
            if (s->hllc)
            {
                if (quirk_strong_shock(xprims_l.p, xprims_r.p) ){
                    f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                } else {
                    f2 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                }
                
                if (quirk_strong_shock(yprims_l.p, yprims_r.p)){
                    g2 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                } else {
                    g2 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }
                // f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                // g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                
            }
            else
            {
                f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                g2 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
        //Advance depending on geometry
        int real_loc = jj * xpg + ii;
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            {
            real dx = coord_lattice->gpu_dx1[ii];
            real dy = coord_lattice->gpu_dx2[jj];
            s->gpu_cons[aid].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + s->gpu_sourceD   [real_loc] );
            s->gpu_cons[aid].S1  += 0.5 * dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + s->gpu_sourceS1  [real_loc] );
            s->gpu_cons[aid].S2  += 0.5 * dt * ( -(f1.S2 - f2.S2)   / dx - (g1.S2  - g2.S2) / dy + s->gpu_sourceS2  [real_loc] );
            s->gpu_cons[aid].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + s->gpu_sourceTau [real_loc] );

            break;
            }
        
        case simbi::Geometry::SPHERICAL:
            {
            real s1R        = coord_lattice->gpu_x1_face_areas[ii + 1];
            real s1L        = coord_lattice->gpu_x1_face_areas[ii + 0];
            real s2R        = coord_lattice->gpu_x2_face_areas[jj + 1];
            real s2L        = coord_lattice->gpu_x2_face_areas[jj + 0];
            real rmean      = coord_lattice->gpu_x1mean[ii];
            real dV1        = coord_lattice->gpu_dV1[ii];
            real dV2        = rmean * coord_lattice->gpu_dV2[jj];
            // Grab central primitives
            real rhoc = center.rho;
            real pc   = center.p;
            real uc   = center.v1;
            real vc   = center.v2;

            real hc   = 1.0 + gamma * pc/(rhoc * (gamma - 1.0));
            real gam2  = 1.0/(1.0 - (uc * uc + vc * vc));

            // printf("ii: %d, jj: %d, dt: %f\n", ii, jj, dt);
            s->gpu_cons[aid] +=
                Conserved{
                    // L(D)
                    -(f1.D * s1R - f2.D * s1L) / dV1 
                        - (g1.D * s2R - g2.D * s2L) / dV2 
                            + s->gpu_sourceD[real_loc] * decay_constant,

                    // L(S1)
                    -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                        - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                            + rhoc * hc * gam2 * vc * vc / rmean + 2 * pc / rmean +
                                    s->gpu_sourceS1[real_loc] * decay_constant,

                    // L(S2)
                    -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                            - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                            - (rhoc * hc * gam2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[jj] / rmean ) 
                                + s->gpu_sourceS2[real_loc] * decay_constant,

                    // L(tau)
                    -(f1.tau * s1R - f2.tau * s1L) / dV1 
                        - (g1.tau * s2R - g2.tau * s2L) / dV2 
                            + s->gpu_sourceTau[real_loc] * decay_constant
                } * dt * 0.5;
            break;
            }
        } // end switch
            
    }


    
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
    std::vector<std::vector<double>> &sources,
    double tstart,
    double tend,
    double init_dt,
    double plm_theta,
    double engine_duration,
    double chkpt_interval,
    std::string data_directory,
    bool first_order,
    bool periodic,
    bool linspace,
    bool hllc)
{
    std::string tnow, tchunk, tstep, filename;
    int total_zones = NX * NY;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;

    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->plm_theta = plm_theta;
    this->dt    = init_dt;

    if (first_order)
    {
        this->xphysical_grid = NX - 2;
        this->yphysical_grid = NY - 2;
        this->idx_active = 1;
        this->i_start = 1;
        this->j_start = 1;
        this->i_bound = NX - 1;
        this->j_bound = NY - 1;
    }
    else
    {
        this->xphysical_grid = NX - 4;
        this->yphysical_grid = NY - 4;
        this->idx_active = 2;
        this->i_start = 2;
        this->j_start = 2;
        this->i_bound = NX - 2;
        this->j_bound = NY - 2;
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
    setup.NX = NX;
    setup.NY = NY;
    setup.ad_gamma = gamma;
    setup.linspace = linspace;

    u0.resize(nzones);
    prims.resize(nzones);
    pressure_guess.resize(nzones);
    // Define the source terms
    sourceD    = sources[0];
    source_S1  = sources[1];
    source_S2  = sources[2];
    source_tau = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        u0[i] =
            Conserved(state2D[0][i], state2D[1][i], state2D[2][i], state2D[3][i]);
    }
    // deallocate initial state vector
    std::vector<int> state2D;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1.0 / (1.0 + exp(10.0 * (tstart - engine_duration)));

    // Set the Primitive from the initial conditions and initialize the pressure
    // guesses
    cons2prim2D();

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    if (t == 0)
    {
        config_ghosts2D(u0, NX, NY, first_order);
    }

    // Copy the current SRHD instance over to the device
    simbi::SRHD2D *device_self;
    hipMalloc((void**)&device_self,    sizeof(SRHD2D));
    hipMemcpy(device_self,  this,      sizeof(SRHD2D), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed when copying current sim state to device");
    SRHD2D_DualSpace dualMem;
    dualMem.copyStateToGPU(*this, device_self);
    hipCheckErrors("Error in copying host state to device");

    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // Setup the system
    const int nxBlocks = (NX + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int nyBlocks = (NY + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nxBlocks = (xphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nyBlocks = (yphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;

    dim3 agridDim  = dim3(physical_nxBlocks, physical_nyBlocks);    // active grid dimensions
    dim3 fgridDim  = dim3(nxBlocks, nyBlocks);                      // full grid dimensions
    dim3 threadDim = dim3(BLOCK_SIZE2D, BLOCK_SIZE2D);              // thread block dimensions
    // Some benchmarking tools 
    int      n   = 0;
    int  nfold   = 0;
    int  ncheck  = 0;
    double zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> delta_t;

    // Simulate :)
    if (first_order)
    {  
        const int radius = 1;
        const int shBlockSize  = BLOCK_SIZE2D + 2 * radius;
        const int shBlockSpace = shBlockSize * shBlockSize;
        const unsigned shBlockBytes = shBlockSpace * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, threadDim, 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, threadDim, shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[this->coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   fgridDim, threadDim, 0, 0, device_self, NX, NY, first_order);
            t += dt; 
            
            // hipDeviceSynchronize();

            if (n >= nfold){
                hipDeviceSynchronize();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
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
                dualMem.copyGPUStateToHost(device_self, *this);
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
            hipLaunchKernelGGL(adapt_dtGPU, dim3(physical_nxBlocks, physical_nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, geometry[coord_system]);
            hipMemcpy(&dt, &(device_self->dt),  sizeof(real), hipMemcpyDeviceToHost);

            // Update decay constant
            decay_const = 1.0 / (1.0 + exp(10.0 * (t - engine_duration)));
            hipMemcpy(&(device_self->decay_const), &decay_const,  sizeof(real), hipMemcpyHostToDevice);
        }
    } else {
        const int radius = 2;
        const int shBlockSize  = BLOCK_SIZE2D + 2 * radius;
        const int shBlockSpace = shBlockSize * shBlockSize;
        const unsigned shBlockBytes = shBlockSpace * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, threadDim, 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, threadDim, shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   fgridDim, threadDim, 0, 0, device_self, NX, NY, false);

            // Final Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, threadDim, 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, threadDim, shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   fgridDim, threadDim, 0, 0, device_self, NX, NY, false);

            t += dt; 
            
            // hipDeviceSynchronize();

            if (n >= nfold){
                hipDeviceSynchronize();
                ncheck += 1;
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
                dualMem.copyGPUStateToHost(device_self, *this);
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
            decay_const = 1.0 / (1.0 + exp(10.0 * (t - engine_duration)));
            hipMemcpy(&(device_self->decay_const), &decay_const,  sizeof(real), hipMemcpyHostToDevice);

            //Adapt the timestep
            hipLaunchKernelGGL(adapt_dtGPU, dim3(physical_nxBlocks, physical_nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, geometry[coord_system]);
            hipMemcpy(&dt, &(device_self->dt),  sizeof(real), hipMemcpyDeviceToHost);
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    hipFree(device_self);

    cons2prim2D();
    transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(4, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
