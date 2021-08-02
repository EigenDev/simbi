/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helper_functions.hpp"
#include "srhd_2d.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace std::chrono;

// Calculate a static PI
constexpr real pi() { return std::atan(1)*4; }
constexpr real K = 0.0;
constexpr real a = 1e-3;

GPU_CALLABLE_MEMBER
bool simbi::strong_shock(real pl, real pr){
    return abs(pr - pl) / min(pl, pr) > a;
}
// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(std::vector<std::vector<real>> state2D, int nx, int ny, real gamma,
               std::vector<real> x1, std::vector<real> x2, real Cfl,
               std::string coord_system = "cartesian")
{
    auto nzones = state2D[0].size();

    this->NX = nx;
    this->NY = ny;
    this->nzones = nzones;
    this->state2D = state2D;
    this->gamma = gamma;
    this->x1 = x1;
    this->x2 = x2;
    this->CFL = Cfl;
    this->coord_system = coord_system;
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
    hipFree(host_x1c);
    hipFree(host_x1m);
    hipFree(host_x2c);
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
    hipMalloc((void **)&host_x1c,             r1bytes);
    hipMalloc((void **)&host_x1m,             r1bytes);
    hipMalloc((void **)&host_x2c,             r2bytes);
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
    if ( hipMemcpy(&(device->gpu_state2D), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
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

    hipMemcpy(host_x1c, host.coord_lattice.x1ccenters.data(), r1bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1centers");

    hipMemcpy(host_x2c, host.coord_lattice.x2ccenters.data(), r2bytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x2centers");

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

    hipMemcpy(&(device->coord_lattice.gpu_x1ccenters), &host_x1c, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1c");

    hipMemcpy(&(device->coord_lattice.gpu_x2ccenters), &host_x2c, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x2c");

    hipMemcpy(&(device->coord_lattice.gpu_x1_face_areas), &host_fas1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1 face areas");

    hipMemcpy(&(device->coord_lattice.gpu_x2_face_areas), &host_fas2, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x2 face areas");

    hipMemcpy(&(device->dt),          &host.dt      ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->theta),       &host.theta   ,  sizeof(real), hipMemcpyHostToDevice);
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

std::vector<Primitive> SRHD2D::cons2prim2D(const std::vector<Conserved> &u_state2D)
{
    /**
   * Return a 2D matrix containing the primitive
   * variables density , pressure, and
   * three-velocity
   */

    real S1, S2, S, D, tau, tol;
    real W, v1, v2;

    std::vector<Primitive> prims;
    prims.reserve(nzones);

    // Define Newton-Raphson Vars
    real etotal, c2, f, g, p, peq;
    real Ws, rhos, eps, h;

    int iter = 0;
    int maximum_iteration = 50;
    for (int jj = 0; jj < NY; jj++)
    {
        for (int ii = 0; ii < NX; ii++)
        {
            D   = u_state2D [ii + NX * jj].D;     // Relativistic Mass Density
            S1  = u_state2D [ii + NX * jj].S1;   // X1-Momentum Denity
            S2  = u_state2D [ii + NX * jj].S2;   // X2-Momentum Density
            tau = u_state2D [ii + NX * jj].tau; // Energy Density
            S = sqrt(S1 * S1 + S2 * S2);

            peq = (n != 0.0) ? pressure_guess[ii + NX * jj] : abs(S - D - tau);

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
            pressure_guess[ii + NX * jj] = peq;

            prims.push_back(Primitive(D / Ws, v1, v2, peq));
        }
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
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
        const real aL = min(bl, (v1_l - cs_l)/(1. - v1_l*cs_l));
        const real aR = max(br, (v1_r + cs_r)/(1. + v1_r*cs_r));

        return Eigenvals(aL, aR);

        //--------Calc the wave speeds based on Mignone and Bodo (2005)
        // const real sL = cs_l * cs_l * (1. / (gamma * gamma * (1 - cs_l * cs_l)));
        // const real sR = cs_r * cs_r * (1. / (gamma * gamma * (1 - cs_r * cs_r)));

        // Define temporaries to save computational cycles
        // const real qfL = 1. / (1. + sL);
        // const real qfR = 1. / (1. + sR);
        // const real sqrtR = sqrt(sL * (1 - v1_l * v1_l + sL));
        // const real sqrtL = sqrt(sR * (1 - v1_r * v1_r + sL));

        // const real lamLm = (v1_l - sqrtL) * qfL;
        // const real lamRm = (v1_r - sqrtR) * qfR;
        // const real lamRp = (v1_l + sqrtL) * qfL;
        // const real lamLp = (v1_r + sqrtR) * qfR;

        // const real aL = lamLm < lamRm ? lamLm : lamRm;
        // const real aR = lamLp > lamRp ? lamLp : lamRp;

        // return Eigenvals(aL, aR);
    }
    case 2:
        const real v2_r = prims_r.v2;
        const real v2_l = prims_l.v2;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const real vbar  = 0.5 * (v2_l + v2_r);
        const real cbar  = 0.5 * (cs_l + cs_r);
        const real bl    = (vbar - cbar)/(1. - cbar*vbar);
        const real br    = (vbar + cbar)/(1. + cbar*vbar);
        const real aL = min(bl, (v2_l - cs_l)/(1. - v2_l*cs_l));
        const real aR = max(br, (v2_r + cs_r)/(1. + v2_r*cs_r));

        return Eigenvals(aL, aR);

        // Calc the wave speeds based on Mignone and Bodo (2005)
        // real sL = cs_l * cs_l * (1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
        // real sR = cs_r * cs_r * (1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

        // Define some temporaries to save a few cycles
        // const real qfL = 1. / (1. + sL);
        // const real qfR = 1. / (1. + sR);
        // const real sqrtR = sqrt(sL * (1 - v2_l * v2_l + sL));
        // const real sqrtL = sqrt(sR * (1 - v2_r * v2_r + sL));

        // const real lamLm = (v2_l - sqrtL) * qfL;
        // const real lamRm = (v2_r - sqrtR) * qfR;
        // const real lamRp = (v2_l + sqrtL) * qfL;
        // const real lamLp = (v2_r + sqrtR) * qfR;
        // const real aL = lamLm < lamRm ? lamLm : lamRm;
        // const real aR = lamLp > lamRp ? lamLp : lamRp;

        // return Eigenvals(aL, aR);
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------

Conserved SRHD2D::calc_stateSR2D(const Primitive &prims)
{
    const real rho = prims.rho;
    const real vx = prims.v1;
    const real vy = prims.v2;
    const real pressure = prims.p;
    const real lorentz_gamma = 1. / sqrt(1 - (vx * vx + vy * vy));
    const real h = 1. + gamma * pressure / (rho * (gamma - 1.));

    return Conserved{
        rho * lorentz_gamma, rho * h * lorentz_gamma * lorentz_gamma * vx,
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

// Adapt the CFL conditonal timestep
real SRHD2D::adapt_dt(const std::vector<Primitive> &prims)
{

    real r_left, r_right, left_cell, right_cell, lower_cell, upper_cell;
    real dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg, h;
    real min_dt, cfl_dt;
    int shift_i, shift_j;
    real plus_v1, plus_v2, minus_v1, minus_v2;

    min_dt = 0;
    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < yphysical_grid; jj++)
    {
        shift_j  = jj + idx_active;
        x2_right = coord_lattice.x2vertices[jj + 1];
        x2_left  = coord_lattice.x2vertices[jj];
        dx2 = x2_right - x2_left;
        for (int ii = 0; ii < xphysical_grid; ii++)
        {
            
            shift_i = ii + idx_active;

            r_right = coord_lattice.x1vertices[ii + 1];
            r_left = coord_lattice.x1vertices[ii];

            dx1 = r_right - r_left;
            rho = prims[shift_i + NX * shift_j].rho;
            v1  = prims[shift_i + NX * shift_j].v1;
            v2  = prims[shift_i + NX * shift_j].v2;
            pressure = prims[shift_i + NX * shift_j].p;

            h = 1. + gamma * pressure / (rho * (gamma - 1.));
            cs = sqrt(gamma * pressure / (rho * h));

            plus_v1 = (v1 + cs) / (1. + v1 * cs);
            plus_v2 = (v2 + cs) / (1. + v2 * cs);
            minus_v1 = (v1 - cs) / (1. - v1 * cs);
            minus_v2 = (v2 - cs) / (1. - v2 * cs);

            if (coord_system == "cartesian")
            {

                cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                             dx2 / (max(abs(plus_v2), abs(minus_v2))));
            }
            else
            {
                // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                volAvg = coord_lattice.x1mean[ii];
                // std::cout << volAvg << "\n";
                // std::cout << dx1 << "\n";
                // std::cout << dx2 << "\n";
                // std::cout << volAvg * dx2 << "\n";
                // std::cin.get();
                cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                             volAvg * dx2 / (max(abs(plus_v2), abs(minus_v2))));
            }

            if ((ii > 0) || (jj > 0))
            {
                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
            }
            else
            {
                min_dt = cfl_dt;
            }
        }
    }
    return CFL * min_dt;
};

__device__ void warp_reduce_min(volatile real smem[BLOCK_SIZE2D][BLOCK_SIZE2D])
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
__global__ void adapt_dtGPU(
    SRHD2D *s, 
    const simbi::Geometry geometry)
{
    real r_left, r_right, left_cell, right_cell, dr, cs;
    real cfl_dt;
    real h, rho, p, v1, v2, dx1, dx2, rmean;
    real plus_v1 , plus_v2 , minus_v1, minus_v2;

    real gamma = s->gamma;
    real min_dt = INFINITY;
    int neighbor_tx, neighbor_ty, neighbor_tid;

    __shared__ volatile real dt_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];
    __shared__ Primitive   prim_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ii  = blockDim.x * blockIdx.x + threadIdx.x;
    const int jj  = blockDim.y * blockIdx.y + threadIdx.y;
    const int ia  = ii + s->idx_active;
    const int ja  = jj + s->idx_active;
    const int gid = jj * s-> NX + ii;
    const int nx  = s->NX;

    const int shift_i = ii + s->idx_active;
    const int shift_j = jj + s->idx_active;

    const CLattice2D *coord_lattice = &(s->coord_lattice);

    if ( (ii < s->xphysical_grid) && (jj < s->yphysical_grid))
    {   

        dx1  = s->coord_lattice.gpu_dx1[ii];
        dx2  = s->coord_lattice.gpu_dx2[jj];
        rho  = s->gpu_prims[ja * nx + ia].rho;
        p    = s->gpu_prims[ja * nx + ia].p;
        v1   = s->gpu_prims[ja * nx + ia].v1;
        v2   = s->gpu_prims[ja * nx + ia].v2;

        h  = 1. + gamma * p / (rho * (gamma - 1.));
        cs = sqrt(gamma * p / (rho * h));

        plus_v1  = (v1 + cs) / (1. + v1 * cs);
        plus_v2  = (v2 + cs) / (1. + v2 * cs);
        minus_v1 = (v1 - cs) / (1. - v1 * cs);
        minus_v2 = (v2 - cs) / (1. - v2 * cs);

        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                            dx2 / (max(abs(plus_v2), abs(minus_v2))));
            break;
        
        case simbi::Geometry::SPHERICAL:
            // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
            rmean = coord_lattice->gpu_x1mean[neighbor_tx];
            cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                        rmean * dx2 / (max(abs(plus_v2), abs(minus_v2))));
            break;
        }

        min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;

        min_dt *= s->CFL;

        dt_buff[threadIdx.y][threadIdx.x] = min_dt;

        __syncthreads();

        // if ((threadIdx.x < BLOCK_SIZE2D / 2) && (threadIdx.y < BLOCK_SIZE2D / 2))
        // {
        //     warp_reduce_min(dt_buff);
        // }
        // if((threadIdx.x == 0) && (threadIdx.y == 0) )
        // {
        //     // printf("min dt: %d\n", dt_buff[threadIdx.y][threadIdx.x]);
        //     s->dt = dt_buff[threadIdx.y][threadIdx.x]; // dt_min[0] == minimum
        // }
        
    }
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
Conserved SRHD2D::calc_Flux(const Primitive &prims, unsigned int nhat = 1)
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

std::vector<Conserved> SRHD2D::u_dot2D(const std::vector<Conserved> &u_state)
{
    int xcoordinate, ycoordinate;

    std::vector<Conserved> L;
    L.reserve(active_zones);

    Conserved ux_l, ux_r, uy_l, uy_r;
    Conserved f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    Primitive xprims_l, xprims_r, yprims_l, yprims_r;

    Primitive xleft_most, xleft_mid, xright_mid, xright_most;
    Primitive yleft_most, yleft_mid, yright_mid, yright_most;
    Primitive center;

    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    const int i_start = idx_active;
    const int j_start = idx_active;
    const int i_bound = x_bound;
    const int j_bound = y_bound;

    switch (geometry[coord_system])
    {
    case Geometry::CARTESIAN:
    {
        real dx = (x1[xphysical_grid - 1] - x1[0]) / xphysical_grid;
        real dy = (x2[yphysical_grid - 1] - x2[0]) / yphysical_grid;
        if (first_order)
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l = u_state[ii + NX * jj];
                    ux_r = u_state[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l = u_state[ii + NX * jj];
                    uy_r = u_state[(ii + 1) + NX * jj];

                    xprims_l = prims[ii + jj * NX];
                    xprims_r = prims[(ii + 1) + jj * NX];

                    yprims_l = prims[ii + jj * NX];

                    yprims_r = prims[ii + (jj + 1) * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l = u_state[(ii - 1) + NX * jj];
                    ux_r = u_state[ii + NX * jj];

                    // j-1/2
                    uy_l = u_state[(ii - 1) + NX * jj];
                    uy_r = u_state[ii + NX * jj];

                    xprims_l = prims[(ii - 1) + jj * NX];
                    xprims_r = prims[ii + jj * NX];

                    yprims_l = prims[ii + (jj - 1) * NX];
                    yprims_r = prims[ii + jj * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    L.push_back(Conserved{(f1 - f2) * -1.0 / dx - (g1 - g2) / dy});
                }
            }
        }
        else
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    if (periodic)
                    {
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Poplate this later */
                    }
                    else
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;
                        ycoordinate = jj - 2;

                        // Coordinate X
                        xleft_most = prims[(ii - 2) + NX * jj];
                        xleft_mid = prims[(ii - 1) + NX * jj];
                        center = prims[ii + NX * jj];
                        xright_mid = prims[(ii + 1) + NX * jj];
                        xright_most = prims[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most = prims[ii + NX * (jj - 2)];
                        yleft_mid = prims[ii + NX * (jj - 1)];
                        yright_mid = prims[ii + NX * (jj + 1)];
                        yright_most = prims[ii + NX * (jj + 2)];
                    }

                    // Reconstructed left X Primitive vector at the i+1/2 interface
                    xprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Reconstructed right Primitive vector in x
                    xprims_r.rho =
                        xright_mid.rho -
                        0.5 * minmod(theta * (xright_mid.rho - center.rho),
                                     0.5 * (xright_most.rho - center.rho),
                                     theta * (xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 -
                                  0.5 * minmod(theta * (xright_mid.v1 - center.v1),
                                               0.5 * (xright_most.v1 - center.v1),
                                               theta * (xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 -
                                  0.5 * minmod(theta * (xright_mid.v2 - center.v2),
                                               0.5 * (xright_most.v2 - center.v2),
                                               theta * (xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p -
                                 0.5 * minmod(theta * (xright_mid.p - center.p),
                                              0.5 * (xright_most.p - center.p),
                                              theta * (xright_most.p - xright_mid.p));

                    // Reconstructed right Primitive vector in y-direction at j+1/2
                    // interfce
                    yprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    yprims_r.rho =
                        yright_mid.rho -
                        0.5 * minmod(theta * (yright_mid.rho - center.rho),
                                     0.5 * (yright_most.rho - center.rho),
                                     theta * (yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 -
                                  0.5 * minmod(theta * (yright_mid.v1 - center.v1),
                                               0.5 * (yright_most.v1 - center.v1),
                                               theta * (yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 -
                                  0.5 * minmod(theta * (yright_mid.v2 - center.v2),
                                               0.5 * (yright_most.v2 - center.v2),
                                               theta * (yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p -
                                 0.5 * minmod(theta * (yright_mid.p - center.p),
                                              0.5 * (yright_most.p - center.p),
                                              theta * (yright_most.p - yright_mid.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Left side Primitive in x
                    xprims_l.rho = xleft_mid.rho +
                                   0.5 * minmod(theta * (xleft_mid.rho - xleft_most.rho),
                                                0.5 * (center.rho - xleft_most.rho),
                                                theta * (center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 +
                                  0.5 * minmod(theta * (xleft_mid.v1 - xleft_most.v1),
                                               0.5 * (center.v1 - xleft_most.v1),
                                               theta * (center.v1 - xleft_mid.v1));

                    xprims_l.v2 = xleft_mid.v2 +
                                  0.5 * minmod(theta * (xleft_mid.v2 - xleft_most.v2),
                                               0.5 * (center.v2 - xleft_most.v2),
                                               theta * (center.v2 - xleft_mid.v2));

                    xprims_l.p =
                        xleft_mid.p + 0.5 * minmod(theta * (xleft_mid.p - xleft_most.p),
                                                   0.5 * (center.p - xleft_most.p),
                                                   theta * (center.p - xleft_mid.p));

                    // Right side Primitive in x
                    xprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Left side Primitive in y
                    yprims_l.rho = yleft_mid.rho +
                                   0.5 * minmod(theta * (yleft_mid.rho - yleft_most.rho),
                                                0.5 * (center.rho - yleft_most.rho),
                                                theta * (center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 +
                                  0.5 * minmod(theta * (yleft_mid.v1 - yleft_most.v1),
                                               0.5 * (center.v1 - yleft_most.v1),
                                               theta * (center.v1 - yleft_mid.v1));

                    yprims_l.v2 = yleft_mid.v2 +
                                  0.5 * minmod(theta * (yleft_mid.v2 - yleft_most.v2),
                                               0.5 * (center.v2 - yleft_most.v2),
                                               theta * (center.v2 - yleft_mid.v2));

                    yprims_l.p =
                        yleft_mid.p + 0.5 * minmod(theta * (yleft_mid.p - yleft_most.p),
                                                   0.5 * (center.p - yleft_most.p),
                                                   theta * (center.p - yleft_mid.p));

                    // Right side Primitive in y
                    yprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    L.push_back(Conserved{(f1 - f2) * -1.0 / dx - (g1 - g2) / dy});
                }
            }
        }

        break;
    }
    case Geometry::SPHERICAL:
        //=======================================================================================================================================================
        //                                  SPHERICAL
        //=======================================================================================================================================================
        real right_cell, left_cell, lower_cell, upper_cell, ang_avg;
        real r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        real theta_right, theta_left, ycoordinate, xcoordinate;
        real upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        if (first_order)
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 1;
                upper_tsurface = coord_lattice.x2_face_areas[ycoordinate + 1];
                lower_tsurface = coord_lattice.x2_face_areas[ycoordinate];
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l = u_state[ii + NX * jj];
                    ux_r = u_state[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l = u_state[ii + NX * jj];
                    uy_r = u_state[ii + NX * (jj + 1)];

                    xprims_l = prims[ii + jj * NX];
                    xprims_r = prims[(ii + 1) + jj * NX];
                    yprims_l = prims[ii + jj * NX];
                    yprims_r = prims[ii + (jj + 1) * NX];

                    // Get central values for spherical source terms
                    rhoc = xprims_l.rho;
                    pc   = xprims_l.p;
                    uc   = xprims_l.v1;
                    vc   = xprims_l.v2;

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i,j +1/2 interface
                    if (hllc)
                    {
                        f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l = u_state[(ii - 1) + NX * jj];
                    ux_r = u_state[ii + NX * jj];

                    // j-1/2
                    uy_l = u_state[ii + NX * (jj - 1)];
                    uy_r = u_state[ii + NX * jj];

                    xprims_l = prims[(ii - 1) + jj * NX];
                    xprims_r = prims[ii + jj * NX];
                    yprims_l = prims[ii + (jj - 1) * NX];
                    yprims_r = prims[ii + jj * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);
                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i,j - 1/2 interface
                    if (hllc)
                    {
                        f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    right_rsurface = coord_lattice.x1_face_areas[xcoordinate + 1];
                    left_rsurface  = coord_lattice.x1_face_areas[xcoordinate];
                    volAvg         = coord_lattice.x1mean[xcoordinate];
                    deltaV1        = coord_lattice.dV1[xcoordinate];
                    deltaV2        = volAvg * coord_lattice.dV2[ycoordinate];

                    L.push_back(Conserved{
                        // L(D)
                        -(f1.D * right_rsurface - f2.D * left_rsurface) / deltaV1 
                            - (g1.D * upper_tsurface - g2.D * lower_tsurface) / deltaV2 
                                + sourceD[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S1)
                        -(f1.S1 * right_rsurface - f2.S1 * left_rsurface) / deltaV1 
                            - (g1.S1 * upper_tsurface - g2.S1 * lower_tsurface) / deltaV2 
                                + rhoc * vc * vc / volAvg + 2 * pc / volAvg +
                                     source_S1[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S2)
                        -(f1.S2 * right_rsurface - f2.S2 * left_rsurface) / deltaV1
                             - (g1.S2 * upper_tsurface - g2.S2 * lower_tsurface) / deltaV2 
                                - (rhoc * uc * vc / volAvg - pc * coord_lattice.cot[ycoordinate] / (volAvg)) 
                                    + source_S2[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(tau)
                        -(f1.tau * right_rsurface - f2.tau * left_rsurface) / deltaV1 
                            - (g1.tau * upper_tsurface - g2.tau * lower_tsurface) / deltaV2 
                                + source_tau[xcoordinate + xphysical_grid * ycoordinate] * decay_const
                    });
                }
            }
        }
        else
        {
            bool at_north_pole = false; 
            bool at_south_pole = false; 
            bool at_adjn_pole  = false;
            bool at_adjs_pole  = false;
            bool zero_flux     = false;
            bool zero_flux_north     = false;
            bool zero_flux_south     = false;
            auto null_flux = Conserved{0.0, 0.0, 0.0, 0.0};

            // Left/Right artificial viscosity
            Conserved favl, favr;
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 2;
                upper_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 1)];
                lower_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 0)];
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    if (!periodic)
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;

                        // Coordinate X
                        xleft_most  = prims[(ii - 2) + NX * jj];
                        xleft_mid   = prims[(ii - 1) + NX * jj];
                        center      = prims[ ii      + NX * jj];
                        xright_mid  = prims[(ii + 1) + NX * jj];
                        xright_most = prims[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most  = prims[ii + NX * (jj - 2)];
                        yleft_mid   = prims[ii + NX * (jj - 1)];
                        yright_mid  = prims[ii + NX * (jj + 1)];
                        yright_most = prims[ii + NX * (jj + 2)];
                    }
                    else
                    {
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Fix this */
                    }
                    // Reconstructed left X Primitive vector at the i+1/2 interface
                    xprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Reconstructed right Primitive vector in x
                    xprims_r.rho =
                        xright_mid.rho -
                        0.5 * minmod(theta * (xright_mid.rho - center.rho),
                                     0.5 * (xright_most.rho - center.rho),
                                     theta * (xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 -
                                  0.5 * minmod(theta * (xright_mid.v1 - center.v1),
                                               0.5 * (xright_most.v1 - center.v1),
                                               theta * (xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 -
                                  0.5 * minmod(theta * (xright_mid.v2 - center.v2),
                                               0.5 * (xright_most.v2 - center.v2),
                                               theta * (xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p -
                                 0.5 * minmod(theta * (xright_mid.p - center.p),
                                              0.5 * (xright_most.p - center.p),
                                              theta * (xright_most.p - xright_mid.p));

                    // Reconstructed right Primitive vector in y-direction at j+1/2
                    // interfce
                    yprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    yprims_r.rho =
                        yright_mid.rho -
                        0.5 * minmod(theta * (yright_mid.rho - center.rho),
                                     0.5 * (yright_most.rho - center.rho),
                                     theta * (yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 -
                                  0.5 * minmod(theta * (yright_mid.v1 - center.v1),
                                               0.5 * (yright_most.v1 - center.v1),
                                               theta * (yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 -
                                  0.5 * minmod(theta * (yright_mid.v2 - center.v2),
                                               0.5 * (yright_most.v2 - center.v2),
                                               theta * (yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p -
                                 0.5 * minmod(theta * (yright_mid.p - center.p),
                                              0.5 * (yright_most.p - center.p),
                                              theta * (yright_most.p - yright_mid.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    favr = (uy_r - uy_l) * (-K);

                    if (hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
                            g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        } else {
                            g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        }
                        // f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        // g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Do the same thing, but for the left side interface [i - 1/2]

                    // Left side Primitive in x
                    xprims_l.rho = xleft_mid.rho +
                                   0.5 * minmod(theta * (xleft_mid.rho - xleft_most.rho),
                                                0.5 * (center.rho - xleft_most.rho),
                                                theta * (center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 +
                                  0.5 * minmod(theta * (xleft_mid.v1 - xleft_most.v1),
                                               0.5 * (center.v1 - xleft_most.v1),
                                               theta * (center.v1 - xleft_mid.v1));

                    xprims_l.v2 = xleft_mid.v2 +
                                  0.5 * minmod(theta * (xleft_mid.v2 - xleft_most.v2),
                                               0.5 * (center.v2 - xleft_most.v2),
                                               theta * (center.v2 - xleft_mid.v2));

                    xprims_l.p =
                        xleft_mid.p + 0.5 * minmod(theta * (xleft_mid.p - xleft_most.p),
                                                   0.5 * (center.p - xleft_most.p),
                                                   theta * (center.p - xleft_mid.p));

                    // Right side Primitive in x
                    xprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Left side Primitive in y
                    yprims_l.rho = yleft_mid.rho +
                                   0.5 * minmod(theta * (yleft_mid.rho - yleft_most.rho),
                                                0.5 * (center.rho - yleft_most.rho),
                                                theta * (center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 +
                                  0.5 * minmod(theta * (yleft_mid.v1 - yleft_most.v1),
                                               0.5 * (center.v1 - yleft_most.v1),
                                               theta * (center.v1 - yleft_mid.v1));

                    yprims_l.v2 = yleft_mid.v2 +
                                  0.5 * minmod(theta * (yleft_mid.v2 - yleft_most.v2),
                                               0.5 * (center.v2 - yleft_most.v2),
                                               theta * (center.v2 - yleft_mid.v2));

                    yprims_l.p =
                        yleft_mid.p + 0.5 * minmod(theta * (yleft_mid.p - yleft_most.p),
                                                   0.5 * (center.p - yleft_most.p),
                                                   theta * (center.p - yleft_mid.p));

                    // Right side Primitive in y
                    yprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);
                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);
                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    favl = (uy_r - uy_l) * (-K);
                    
                    if (hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
                            g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        } else {
                            g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        }
                        // f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        // g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        
                    }
                    else
                    {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    rhoc = center.rho;
                    pc   = center.p;
                    uc   = center.v1;
                    vc   = center.v2;

                    // Compute the surface areas
                    // upper_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 1)*xphysical_grid + xcoordinate];
                    // lower_tsurface = coord_lattice.x2_face_areas[ycoordinate * xphysical_grid + xcoordinate];
                    // right_rsurface = coord_lattice.x1_face_areas[ycoordinate * (xphysical_grid + 1) + xcoordinate + 1];
                    // left_rsurface  = coord_lattice.x1_face_areas[ycoordinate * (xphysical_grid + 1) + xcoordinate];
                    right_rsurface = coord_lattice.x1_face_areas[xcoordinate + 1];
                    left_rsurface  = coord_lattice.x1_face_areas[xcoordinate];

                    volAvg         = coord_lattice.x1mean[xcoordinate];
                    deltaV1        = coord_lattice.dV1[xcoordinate];
                    deltaV2        = volAvg * coord_lattice.dV2[ycoordinate];

                    L.push_back(Conserved{
                        // L(D)
                        -(f1.D * right_rsurface - f2.D * left_rsurface) / deltaV1 
                            - (g1.D * upper_tsurface - g2.D * lower_tsurface) / deltaV2 
                                + sourceD[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S1)
                        -(f1.S1 * right_rsurface - f2.S1 * left_rsurface) / deltaV1 
                            - (g1.S1 * upper_tsurface - g2.S1 * lower_tsurface) / deltaV2 
                                + rhoc * vc * vc / volAvg + 2.0 * pc / volAvg 
                                    + source_S1[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S2)
                        -(f1.S2 * right_rsurface - f2.S2 * left_rsurface) / deltaV1 
                            - (g1.S2 * upper_tsurface - g2.S2 * lower_tsurface) / deltaV2
                                -(rhoc * uc * vc / volAvg - pc * coord_lattice.cot[ycoordinate] / volAvg) 
                                    + source_S2[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(tau)
                        -(f1.tau * right_rsurface - f2.tau * left_rsurface) / deltaV1 
                            - (g1.tau * upper_tsurface - g2.tau * lower_tsurface) / deltaV2
                                + source_tau[xcoordinate + xphysical_grid * ycoordinate] * decay_const
                                    });
                }
            }
        }

        break;
    }

    return L;
};

//=====================================================================
//                          KERNEL CALLS
//=====================================================================
__global__ void simbi::shared_gpu_cons2prim(SRHD2D *s){
    __shared__ Conserved  conserved_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];
    __shared__ Primitive  primitive_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];

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
        conserved_buff[ty][tx] = s->gpu_state2D[gid];
        primitive_buff[ty][tx] = s->gpu_prims[gid];
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
                // exit(EXIT_FAILURE);
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
    const int bs                    = sh_block_size;
    const int ibound                = s->i_bound;
    const int istart                = s->i_start;
    const int jbound                = s->j_bound;
    const int jstart                = s->j_start;
    const int xpg                   = s->xphysical_grid;
    const real decay_constant       = s->decay_const;
    const CLattice2D *coord_lattice = &(s->coord_lattice);
    const real dt                   = s->dt;
    const real plm_theta            = s->theta;
    const real gamma                = s->gamma;


    extern __shared__ Conserved smem_buff[];
    Conserved *cons_buff = smem_buff;
    Primitive *prim_buff = (Primitive *)&cons_buff[sh_block_space];

    int xcoordinate, ycoordinate;
    Conserved ux_l, ux_r, uy_l, uy_r;
    Conserved f_l, f_r, g_l, g_r, f1, f2, g1, g2;
    Primitive xprims_l, xprims_r, yprims_l, yprims_r;

    if ((ii < s->NX) && (jj < s->NY))
    {   
        // printf("txa: %d, tya: %d\n", txa, tya);
        int gid = jj * nx + ii;

        // printf("center D: %f\n", cons_buff[tya * bs + txa].D);
        if (s->first_order)
        {
            cons_buff[tya * bs + txa] = s->gpu_state2D[gid];
            prim_buff[tya * bs + txa] = s->gpu_prims[gid];
            if (threadIdx.x < radius)
            {
                cons_buff[tya * bs + txa - radius      ] = s->gpu_state2D[(jj * nx) + (ii - radius)      ];
                cons_buff[tya * bs + txa + BLOCK_SIZE2D] = s->gpu_state2D[(jj * nx) + (ii + BLOCK_SIZE2D)];
                prim_buff[tya * bs + txa - radius      ] = s->gpu_prims[(jj * nx) + ii - radius      ];
                prim_buff[tya * bs + txa + BLOCK_SIZE2D] = s->gpu_prims[(jj * nx) + ii + BLOCK_SIZE2D];  
            }
            if (threadIdx.y < radius)
            {
                cons_buff[(tya - radius      ) * bs + txa] = s->gpu_state2D[(jj - radius) * nx       + ii];
                cons_buff[(tya + BLOCK_SIZE2D) * bs + txa] = s->gpu_state2D[(jj + BLOCK_SIZE2D) * nx + ii];
                prim_buff[(tya - radius      ) * bs + txa] = s->gpu_prims[(jj - radius) * nx         + ii];
                prim_buff[(tya + BLOCK_SIZE2D) * bs + txa] = s->gpu_prims[(jj + BLOCK_SIZE2D) * nx   + ii];  
            }
            __syncthreads();

            if ( ( (unsigned)(jj - jstart) < (jbound - jstart) ) 
                  && ((unsigned)(ii - istart) < (ibound - istart) ))
            {
                if (s->periodic)
                {
                    xcoordinate = ii;
                    ycoordinate = jj;
                    // Set up the left and right state interfaces for i+1/2
                    // u_l   = cons_buff[txa];
                    // u_r   = roll(cons_buff, txa + 1, sh_block_size);
                }
                else
                {
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l = cons_buff[tya * bs + (txa + 0)];
                    ux_r = cons_buff[tya * bs + (txa + 1)];
                    // j+1/2
                    uy_l = cons_buff[(tya + 0) * bs + txa]; 
                    uy_r = cons_buff[(tya + 1) * bs + txa]; 

                    xprims_l = prim_buff[tya * bs + (txa + 0)];
                    xprims_r = prim_buff[tya * bs + (txa + 1)];
                    //j+1/2
                    yprims_l = prim_buff[(tya + 0) * bs + txa];
                    yprims_r = prim_buff[(tya + 1) * bs + txa];
                }
                
                f_l = s->calc_Flux(xprims_l);
                f_r = s->calc_Flux(xprims_r);

                g_l = s->calc_Flux(yprims_l, 2);
                g_r = s->calc_Flux(yprims_r, 2);

                // Calc HLL Flux at i+1/2 interface
                if (s-> hllc)
                {
                    f1 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                } else {
                    f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                // Set up the left and right state interfaces for i-1/2
                if (s->periodic)
                {
                    xcoordinate = ii;
                }
                else
                {
                    // i+1/2
                    ux_l = cons_buff[tya * bs + (txa - 1)];
                    ux_r = cons_buff[tya * bs + (txa - 0)];
                    // j+1/2
                    uy_l = cons_buff[(tya - 1) * bs + txa]; 
                    uy_r = cons_buff[(tya - 0) * bs + txa]; 

                    xprims_l = prim_buff[tya * bs + (txa - 1)];
                    xprims_r = prim_buff[tya * bs + (txa + 0)];
                    //j+1/2
                    yprims_l = prim_buff[(tya - 1) * bs + txa]; 
                    yprims_r = prim_buff[(tya + 0) * bs + txa]; 
                }

                f_l = s->calc_Flux(xprims_l);
                f_r = s->calc_Flux(xprims_r);

                g_l = s->calc_Flux(yprims_l, 2);
                g_r = s->calc_Flux(yprims_r, 2);

                // Calc HLL Flux at i-1/2 interface
                if (s-> hllc)
                {
                    f2 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = s->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                } else {
                    f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = s->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                }

                //Advance depending on geometry
                int real_loc = ycoordinate * xpg + xcoordinate;
                switch (geometry)
                {
                    case simbi::Geometry::CARTESIAN:
                        {
                        real dx = coord_lattice->gpu_dx1[xcoordinate];
                        real dy = coord_lattice->gpu_dx2[ycoordinate];
                        s->gpu_state2D[gid].D   += dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + s->gpu_sourceD [real_loc] );
                        s->gpu_state2D[gid].S1  += dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + s->gpu_sourceS1[real_loc]);
                        s->gpu_state2D[gid].S2  += dt * ( -(f1.S2 - f2.S2)   / dx  - (g1.S2  - g2.S2) / dy + s->gpu_sourceS2[real_loc]);
                        s->gpu_state2D[gid].tau += dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + s->gpu_sourceTau [real_loc] );

                        break;
                        }
                    
                    case simbi::Geometry::SPHERICAL:
                        {
                        real s1R        = coord_lattice->gpu_x1_face_areas[xcoordinate + 1];
                        real s1L        = coord_lattice->gpu_x1_face_areas[xcoordinate + 0];
                        real s2R        = coord_lattice->gpu_x2_face_areas[ycoordinate + 1];
                        real s2L        = coord_lattice->gpu_x2_face_areas[ycoordinate + 0];
                        real rmean      = coord_lattice->gpu_x1mean[xcoordinate];
                        real dV1        = coord_lattice->gpu_dV1[xcoordinate];
                        real dV2        = rmean * coord_lattice->gpu_dV2[ycoordinate];
                        // Grab central primitives
                        real rhoc = prim_buff[tya * bs + txa].rho;
                        real pc   = prim_buff[tya * bs + txa].p;
                        real uc   = prim_buff[tya * bs + txa].v1;
                        real vc   = prim_buff[tya * bs + txa].v2;

                        real hc   = 1.0 + gamma * pc/(rhoc * (gamma - 1.0));
                        real wc2  = 1.0/(1.0 - (uc * uc + vc * vc));

                        s->gpu_state2D[gid] +=
                        Conserved{
                            // L(D)
                            -(f1.D * s1R - f2.D * s1L) / dV1 
                                - (g1.D * s2R - g2.D * s2L) / dV2 
                                    + s->gpu_sourceD[real_loc] * decay_constant,

                            // L(S1)
                            -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                    + rhoc * hc * wc2 * vc * vc / rmean + 2 * pc / rmean +
                                            s->gpu_sourceS1[real_loc] * decay_constant,

                            // L(S2)
                            -(f1.S2 * s1R - f2.S2 * s1R) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                    - (rhoc *hc * wc2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[ycoordinate] / (rmean)) 
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
        }
        else
        {
            prim_buff[tya * bs + txa] = s->gpu_prims[gid];
            if (threadIdx.x < radius)
            {
                prim_buff[tya * bs + txa - radius      ] = s->gpu_prims[(jj * nx) + ii - radius      ];
                prim_buff[tya * bs + txa + BLOCK_SIZE2D] = s->gpu_prims[(jj * nx) + ii + BLOCK_SIZE2D];  
            }
            if (threadIdx.y < radius)
            {
                prim_buff[(tya - radius      ) * bs + txa] = s->gpu_prims[(jj - radius) * nx         + ii];
                prim_buff[(tya + BLOCK_SIZE2D) * bs + txa] = s->gpu_prims[(jj + BLOCK_SIZE2D) * nx   + ii];  
            }
            __syncthreads();

            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            if ( ( (unsigned)(jj - jstart) < (jbound - jstart) ) 
                  && ((unsigned)(ii - istart) < (ibound - istart) ))
            {
                if (!(s->periodic))
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;

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
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Fix this */
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
                    ux_l = s->calc_stateSR2D(xprims_l);
                    ux_r = s->calc_stateSR2D(xprims_r);

                    uy_l = s->calc_stateSR2D(yprims_l);
                    uy_r = s->calc_stateSR2D(yprims_r);

                    f_l = s->calc_Flux(xprims_l);
                    f_r = s->calc_Flux(xprims_r);

                    g_l = s->calc_Flux(yprims_l, 2);
                    g_r = s->calc_Flux(yprims_r, 2);

                    // favr = (uy_r - uy_l) * (-K);

                    if (s->hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f1 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f1 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
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
                    ux_l = s->calc_stateSR2D(xprims_l);
                    ux_r = s->calc_stateSR2D(xprims_r);
                    uy_l = s->calc_stateSR2D(yprims_l);
                    uy_r = s->calc_stateSR2D(yprims_r);

                    f_l = s->calc_Flux(xprims_l);
                    f_r = s->calc_Flux(xprims_r);
                    g_l = s->calc_Flux(yprims_l, 2);
                    g_r = s->calc_Flux(yprims_r, 2);

                    // favl = (uy_r - uy_l) * (-K);
                    
                    if (s->hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f2 = s->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f2 = s->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
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
                int real_loc = ycoordinate * xpg + xcoordinate;
                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    {
                    real dx = coord_lattice->gpu_dx1[xcoordinate];
                    real dy = coord_lattice->gpu_dx2[ycoordinate];
                    s->gpu_state2D[gid].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx - (g1.D   - g2.D ) / dy + s->gpu_sourceD [real_loc]   );
                    s->gpu_state2D[gid].S1  += 0.5 * dt * ( -(f1.S1 - f2.S1)   / dx - (g1.S1  - g2.S1) / dy + s->gpu_sourceS1[real_loc]   );
                    s->gpu_state2D[gid].S2  += 0.5 * dt * ( -(f1.S2 - f2.S2)   / dx - (g1.S2  - g2.S2) / dy + s->gpu_sourceS2[real_loc]   );
                    s->gpu_state2D[gid].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx - (g1.tau - g2.tau)/ dy + s->gpu_sourceTau [real_loc] );

                    break;
                    }
                
                case simbi::Geometry::SPHERICAL:
                    {
                    real s1R        = coord_lattice->gpu_x1_face_areas[xcoordinate + 1];
                    real s1L        = coord_lattice->gpu_x1_face_areas[xcoordinate + 0];
                    real s2R        = coord_lattice->gpu_x2_face_areas[ycoordinate + 1];
                    real s2L        = coord_lattice->gpu_x2_face_areas[ycoordinate + 0];
                    real rmean      = coord_lattice->gpu_x1mean[xcoordinate];
                    real dV1        = coord_lattice->gpu_dV1[xcoordinate];
                    real dV2        = rmean * coord_lattice->gpu_dV2[ycoordinate];
                    // Grab central primitives
                    real rhoc = center.rho;
                    real pc   = center.p;
                    real uc   = center.v1;
                    real vc   = center.v2;

                    real hc   = 1.0 + gamma * pc/(rhoc * (gamma - 1.0));
                    real wc2  = 1.0/(1.0 - (uc * uc + vc * vc));

                    // printf("ii: %d, jj: %d, dt: %f\n", ii, jj, dt);
                    s->gpu_state2D[gid] +=
                        Conserved{
                            // L(D)
                            -(f1.D * s1R - f2.D * s1L) / dV1 
                                - (g1.D * s2R - g2.D * s2L) / dV2 
                                    + s->gpu_sourceD[0] * decay_constant,

                            // L(S1)
                            -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                    + rhoc * hc * wc2 * vc * vc / rmean + 2 * pc / rmean +
                                            s->gpu_sourceS1[real_loc] * decay_constant,

                            // L(S2)
                            -(f1.S2 * s1R - f2.S2 * s1R) / dV1
                                    - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                    - (rhoc * hc * wc2 * uc * vc / rmean - pc * coord_lattice->gpu_cot[ycoordinate] / (rmean)) 
                                        + s->gpu_sourceS2[real_loc] * decay_constant,

                            // L(tau)
                            -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                    + s->gpu_sourceTau[real_loc] * decay_constant
                        } * dt * 0.5;
                    break;
                    }
                }
                
            }
        }

    }
    
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<real>> SRHD2D::simulate2D(
    std::vector<real> lorentz_gamma, 
    const std::vector<std::vector<real>> sources,
    float tstart = 0., 
    float tend = 0.1, 
    real dt = 1.e-4, 
    real theta = 1.5,
    real engine_duration = 10, 
    real chkpt_interval = 0.1,
    std::string data_directory = "data/", 
    bool first_order = true,
    bool periodic = false, 
    bool linspace = true, 
    bool hllc = false)
{

    int i_real, j_real;
    std::string tnow, tchunk, tstep;
    int total_zones = NX * NY;
    
    real round_place = 1 / chkpt_interval;
    real t = tstart;
    real t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;

    std::string filename;

    this->sources = sources;
    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this->theta = theta;
    this->dt    = dt;

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
    this->xvertices.resize(x1.size() + 1);
    this->yvertices.resize(x2.size() + 1);

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

    if (coord_lattice.x2vertices[yphysical_grid] == pi()){
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

    std::vector<Conserved> u, u1, udot, udot1;
    u.resize(nzones);
    u1.resize(nzones);
    udot.reserve(active_zones);
    udot1.resize(nzones);
    prims.reserve(nzones);

    // Define the source terms
    sourceD    = sources[0];
    source_S1  = sources[1];
    source_S2  = sources[2];
    source_tau = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        u[i] =
            Conserved(state2D[0][i], state2D[1][i], state2D[2][i], state2D[3][i]);
    }
    n = 0;
    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1.0 / (1.0 + exp(10.0 * (tstart - engine_duration)));

    // Set the Primitive from the initial conditions and initialize the pressure
    // guesses
    pressure_guess.resize(nzones);
    prims = cons2prim2D(u);

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    if (t == 0)
    {
        config_ghosts2D(u, NX, NY, first_order);
    }
    
    u0  = u;

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
    int time_order_of_mag, num_zeros;

    // Setup the system
    const int nxBlocks = (NX + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int nyBlocks = (NY + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nxBlocks = (xphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;
    const int physical_nyBlocks = (yphysical_grid + BLOCK_SIZE2D - 1) / BLOCK_SIZE2D;

    // Some benchmarking tools 
    real avg_dt  = 0;
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
        const unsigned shBlockBytes = shBlockSpace * sizeof(Conserved) + shBlockSpace * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[this->coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, NX, NY, first_order);
            t += dt; 
            
            hipDeviceSynchronize();

            if (n >= nfold){
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
                nfold += 1000;
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
                
                transfer_prims = vecs2struct(prims);
                toWritePrim(&transfer_prims, &prods);
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
        }
    } else {
        const int radius = 2;
        const int shBlockSize  = BLOCK_SIZE2D + 2 * radius;
        const int shBlockSpace = shBlockSize * shBlockSize;
        const unsigned shBlockBytes = shBlockSpace * sizeof(Conserved) + shBlockSpace * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, NX, NY, first_order);

            // Final Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), shBlockBytes, 0, device_self, shBlockSize, shBlockSpace, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts2DGPU,   dim3(nxBlocks, nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, NX, NY, first_order);

            t += dt; 
            hipDeviceSynchronize();

            if (n >= nfold){
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += NX * NY / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< NX * NY / delta_t.count() << std::flush;
                nfold += 1000;
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
                
                transfer_prims = vecs2struct(prims);
                toWritePrim(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            n++;
            //Adapt the timestep
            // hipLaunchKernelGGL(adapt_dtGPU, dim3(physical_nxBlocks, physical_nyBlocks), dim3(BLOCK_SIZE2D, BLOCK_SIZE2D), 0, 0, device_self, geometry[coord_system]);
            // hipMemcpy(&dt, &(device_self->dt),  sizeof(real), hipMemcpyDeviceToHost);
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    hipFree(device_self);

    prims = cons2prim2D(u0);
    transfer_prims = vecs2struct(prims);

    std::vector<std::vector<real>> solution(4, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
