/*
 * C++ Library to perform extensive hydro calculations
 * to be later wrapped and plotted in Python
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helper_functions.hpp"
#include "srhd_1d.hpp"
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
SRHD::SRHD() {}

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
{
}

//================================================
//              DUAL SPACE FOR 1D SRHD
//================================================
SRHD_DualSpace::SRHD_DualSpace(){}

SRHD_DualSpace::~SRHD_DualSpace()
{
    printf("\nFreeing Device Memory...\n");
    hipFree(host_u0);
    hipFree(host_prims);
    hipFree(host_clattice);
    hipFree(host_dV);
    hipFree(host_dx1);
    hipFree(host_fas);
    hipFree(host_x1c);
    hipFree(host_x1m);
    hipFree(host_source0);
    hipFree(host_sourceD);
    hipFree(host_sourceS);
    hipFree(host_pressure_guess);
    printf("Memory Freed.\n");
}
void SRHD_DualSpace::copyStateToGPU(
    const simbi::SRHD &host,
    simbi::SRHD *device
)
{
    int nz     = host.Nx;
    int nzreal = host.pgrid_size; 

    // Precompute byes
    int cbytes  = nz * sizeof(Conserved);
    int pbytes  = nz * sizeof(Primitive);
    int rbytes  = nz * sizeof(real);

    int rrbytes = nzreal * sizeof(real);
    int fabytes = host.coord_lattice.face_areas.size() * sizeof(real);

    

    //--------Allocate the memory for pointer objects-------------------------
    hipMalloc((void **)&host_u0,              cbytes);
    hipMalloc((void **)&host_prims,           pbytes);
    hipMalloc((void **)&host_pressure_guess,  rbytes);
    hipMalloc((void **)&host_dx1,             rrbytes);
    hipMalloc((void **)&host_dV ,             rrbytes);
    hipMalloc((void **)&host_x1c,             rrbytes);
    hipMalloc((void **)&host_x1m,             rrbytes);
    hipMalloc((void **)&host_fas,             fabytes);
    hipMalloc((void **)&host_source0,         rrbytes);
    hipMalloc((void **)&host_sourceD,         rrbytes);
    hipMalloc((void **)&host_sourceS,         rrbytes);

    hipMalloc((void **)&host_dtmin,            rbytes);
    hipMalloc((void **)&host_clattice, sizeof(CLattice1D));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.sys_state.data(), cbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sys_state to host_u0");

    hipMemcpy(host_prims, host.prims.data()    , pbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.prims to host_prims");

    hipMemcpy(host_pressure_guess, host.pressure_guess.data() , rbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.pressure_guess to host_pre_guess");

    hipMemcpy(host_source0, host.source0.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.source0 to host_source0");

    hipMemcpy(host_sourceD, host.sourceD.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sourceD to host_sourceD");

    hipMemcpy(host_sourceS, host.sourceS.data() , rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.sourceS to host_sourceS");

    // copy pointer to allocated device storage to device class
    if ( hipMemcpy(&(device->gpu_sys_state), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
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

    hipMemcpy(&(device->gpu_source0), &host_source0, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying source0 to device");

    hipMemcpy(&(device->gpu_sourceD), &host_sourceD, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying sourceD to device");

    hipMemcpy(&(device->gpu_sourceS), &host_sourceS, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying sourceS to device");

    hipMemcpy(&(device->dt_min), &host_dtmin, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at copying min_dt to device");
    // ====================================================
    //          GEOMETRY DEEP COPY
    //=====================================================
    hipMemcpy(host_dx1, host.coord_lattice.dx1.data() ,       rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(host_dV,  host.coord_lattice.dV.data(),         rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dV");

    hipMemcpy(host_fas, host.coord_lattice.face_areas.data() , fabytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring face areas");

    hipMemcpy(host_x1c, host.coord_lattice.x1ccenters.data(), rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1centers");

    hipMemcpy(host_x1m, host.coord_lattice.x1mean.data(),     rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1mean");

    // Now copy pointer to device directly
    hipMemcpy(&(device->coord_lattice.gpu_dx1), &host_dx1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_dV), &host_dV, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_x1mean),&host_x1m, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_x1ccenters), &host_x1c, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_face_areas), &host_fas, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->dt),        &host.dt      ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->gamma),     &host.gamma   ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->CFL)  ,     &host.CFL     ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->Nx),        &host.Nx      ,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->pgrid_size),&host.pgrid_size,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->i_bound),   &host.i_bound,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->i_start),   &host.i_start,   sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->idx_shift), &host.idx_shift, sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->decay_constant), &host.decay_constant, sizeof(real), hipMemcpyHostToDevice);
    
}

void SRHD_DualSpace::copyGPUStateToHost(
    const simbi::SRHD *device,
    simbi::SRHD &host
)
{
    const int nz     = host.Nx;
    const int cbytes = nz * sizeof(Conserved); 
    const int pbytes = nz * sizeof(Primitive); 

    hipMemcpy(host.sys_state.data(), host_u0,        cbytes, hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device conservatives to host");
    hipMemcpy(host.prims.data(),     host_prims ,    pbytes, hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device prims to host");
    
}

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
void SRHD::cons2prim1D(const std::vector<Conserved> &u_state)
{
    real rho, S, D, tau, pmin;
    real v, W, tol, f, g, peq, h;
    real eps, rhos, p, v2, et, c2;
    int iter = 0;

    for (int ii = 0; ii < Nx; ii++)
    {
        D = u_state[ii].D;
        S = u_state[ii].S;
        tau = u_state[ii].tau;

        peq = n != 0 ? pressure_guess[ii] : abs(abs(S) - tau - D);

        tol = D * 1.e-12;

        iter = 0;
        do
        {
            p = peq;
            et = tau + D + p;
            v2 = S * S / (et * et);
            W = 1.0 / sqrt(1.0 - v2);
            rho = D / W;

            eps = (tau + (1.0 - W) * D + (1. - W * W) * p) / (D * W);

            h = 1. + eps + p / rho;
            c2 = gamma * p / (h * rho);

            g = c2 * v2 - 1.0;
            f = (gamma - 1.0) * rho * eps - p;

            peq = p - f / g;
            iter++;
            if (iter >= MAX_ITER)
            {
                std::cout << "\n";
                std::cout << "Cons2Prim cannot converge"
                          << "\n";
                exit(EXIT_FAILURE);
            }

        } while (abs(peq - p) >= tol);

        v = S / (tau + D + peq);

        W = 1. / sqrt(1 - v * v);
        pressure_guess[ii] = peq;
        prims[ii] = Primitive{D / W, v, peq};
    }
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals SRHD::calc_eigenvals(const Primitive &prims_l,
                               const Primitive &prims_r)
{

    // Initialize your important variables
    real v_r, v_l, p_r, p_l, cs_r, cs_l;
    real rho_l, rho_r, h_l, h_r, aL, aR;
    real sL, sR, minlam_l, minlam_r, pluslam_l, pluslam_r;
    real vbar, cbar;
    Eigenvals lambda;

    // Compute L/R Sound Speeds
    rho_l = prims_l.rho;
    p_l = prims_l.p;
    v_l = prims_l.v;
    h_l = 1. + gamma * p_l / (rho_l * (gamma - 1.));
    cs_l = sqrt(gamma * p_l / (rho_l * h_l));

    rho_r = prims_r.rho;
    p_r = prims_r.p;
    v_r = prims_r.v;
    h_r = 1. + gamma * p_r / (rho_r * (gamma - 1.));
    cs_r = sqrt(gamma * p_r / (rho_r * h_r));

    // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
    vbar = 0.5 * (v_l + v_r);
    cbar = 0.5 * (cs_r + cs_l);
    real br = (vbar + cbar) / (1 + vbar * cbar);
    real bl = (vbar - cbar) / (1 - vbar * cbar);

    lambda.aL = min(bl, (v_l - cs_l) / (1 - v_l * cs_l));
    lambda.aR = max(br, (v_r + cs_r) / (1 + v_l * cs_l));

    // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
    // sL          = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
    // sR          = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
    // minlam_l    = (v_l - sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    // minlam_r    = (v_r - sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);
    // pluslam_l   = (v_l + sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    // pluslam_r   = (v_r + sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);

    // lambda.aL = (minlam_l < minlam_r)   ? minlam_l : minlam_r;
    // lambda.aR = (pluslam_l > pluslam_r) ? pluslam_l : pluslam_r;

    return lambda;
};

// Adapt the CFL conditonal timestep
real SRHD::adapt_dt(const std::vector<Primitive> &prims)
{

    real r_left, r_right, left_cell, right_cell, dr, cs;
    real min_dt, cfl_dt;
    real h, rho, p, v, vPLus, vMinus;

    min_dt = 0;

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < pgrid_size; ii++)
    {
        dr  = coord_lattice.dx1[ii];
        rho = prims[ii + idx_shift].rho;
        p   = prims[ii + idx_shift].p;
        v   = prims[ii + idx_shift].v;

        h = 1. + gamma * p / (rho * (gamma - 1.));
        cs = sqrt(gamma * p / (rho * h));

        vPLus = (v + cs) / (1 + v * cs);
        vMinus = (v - cs) / (1 - v * cs);

        cfl_dt = dr / (std::max(abs(vPLus), abs(vMinus)));

        if (ii > 0)
        {
            min_dt = std::min(min_dt, cfl_dt);
        }
        else
        {
            min_dt = cfl_dt;
        }
    }

    return CFL * min_dt;
};

__device__ void warp_reduce_min(volatile real smem[BLOCK_SIZE])
{

    for (int stride = BLOCK_SIZE /2; stride >= 1; stride /=  2)
    {
        smem[threadIdx.x] = smem[threadIdx.x+stride] < smem[threadIdx.x] ? 
						smem[threadIdx.x+stride] : smem[threadIdx.x];;
    }

}

// Adapt the CFL conditonal timestep
__global__ void adapt_dtGPU(SRHD *s, int nBlocks, real *host_dt)
{
    real r_left, r_right, left_cell, right_cell, dr, cs;
    real cfl_dt;
    real h, rho, p, v, vPLus, vMinus;

    real gamma = s->gamma;
    real min_dt = INFINITY;
    int neighbor_tid;

    __shared__ volatile real dt_buff[BLOCK_SIZE];
    __shared__  Primitive prim_buff[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockDim.x*blockIdx.x + threadIdx.x;

    if (gid < s->pgrid_size)
    {
        prim_buff[tid] = s->gpu_prims[gid + s->idx_shift];

        // for(unsigned int stride = (blockDim.x/2); stride > 32 ; stride /=2){
        //     __syncthreads();

        //     if(tid < stride)
        //     {
        //         tbuff[thread_id] = min(minChunk[thread_id],minChunk[thread_id + stride]);
        //     }
        // }
        // tail part
        int mult = 0;
        for(int ii=1; mult + tid < s->pgrid_size; ii++) 
        {
            neighbor_tid = tid + mult;
            dr  = s->coord_lattice.gpu_dx1[neighbor_tid];
            rho = s->gpu_prims[neighbor_tid + s->idx_shift].rho;
            p   = s->gpu_prims[neighbor_tid + s->idx_shift].p;
            v   = s->gpu_prims[neighbor_tid + s->idx_shift].v;

            h = 1. + gamma * p / (rho * (gamma - 1.));
            cs = sqrt(gamma * p / (rho * h));

            vPLus  = (v + cs) / (1 + v * cs);
            vMinus = (v - cs) / (1 - v * cs);

            cfl_dt = dr / (max(abs(vPLus), abs(vMinus)));

            min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;

            mult = ii * BLOCK_SIZE;
        }

        // previously reduced MIN part
        // mult = 0;
        // int ii;
        // real val;
        // for(ii = 1; mult+threadIdx.x < nBlocks; ii++)
        // {
        //     val = s->dt_min[threadIdx.x + mult];

        //     min_dt = val < min_dt ? val : min_dt;
        //     mult = ii * BLOCK_SIZE;
        // }

        min_dt *= s->CFL;

        dt_buff[threadIdx.x] = min_dt;

        __syncthreads();

        if (threadIdx.x < BLOCK_SIZE / 2)
        {
            warp_reduce_min(dt_buff);
        }
        if(threadIdx.x == 0)
        {
            s->dt = dt_buff[threadIdx.x]; // dt_min[0] == minimum
            // *host_dt = dt_buff[threadIdx.x];
        }
        
    }
};
//----------------------------------------------------------------------------------------------------
//              STATE ARRAY CALCULATIONS
//----------------------------------------------------------------------------------------------------

// Get the (3,1) state array for computation. Used for Higher Order
// Reconstruction
GPU_CALLABLE_MEMBER
Conserved SRHD::calc_state(real rho, real v, real pressure)
{

    Conserved state;
    real W, h;

    h = 1. + gamma * pressure / (rho * (gamma - 1.));
    W = 1. / sqrt(1 - v * v);

    state.D = rho * W;
    state.S = rho * h * W * W * v;
    state.tau = rho * h * W * W - pressure - rho * W;

    return state;
};

GPU_CALLABLE_MEMBER
Conserved SRHD::calc_hll_state(const Conserved &left_state,
                               const Conserved &right_state,
                               const Conserved &left_flux,
                               const Conserved &right_flux,
                               const Primitive &left_prims,
                               const Primitive &right_prims)
{
    real aL, aR;
    Conserved hll_states;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    hll_states.D =
        (aR * right_state.D - aL * left_state.D - right_flux.D + left_flux.D) /
        (aR - aL);

    hll_states.S =
        (aR * right_state.S - aL * left_state.S - right_flux.S + left_flux.S) /
        (aR - aL);

    hll_states.tau = (aR * right_state.tau - aL * left_state.tau -
                      right_flux.tau + left_flux.tau) /
                     (aR - aL);

    return hll_states;
}

Conserved SRHD::calc_intermed_state(const Primitive &prims,
                                    const Conserved &state, const real a,
                                    const real aStar, const real pStar)
{
    real pressure, v, S, D, tau, E, Estar;
    real DStar, Sstar, tauStar;
    Eigenvals lambda;
    Conserved star_state;

    pressure = prims.p;
    v = prims.v;

    D = state.D;
    S = state.S;
    tau = state.tau;
    E = tau + D;

    DStar = ((a - v) / (a - aStar)) * D;
    Sstar = (1. / (a - aStar)) * (S * (a - v) - pressure + pStar);
    Estar = (1. / (a - aStar)) * (E * (a - v) + pStar * aStar - pressure * v);
    tauStar = Estar - DStar;

    star_state.D = DStar;
    star_state.S = Sstar;
    star_state.tau = tauStar;

    return star_state;
}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
GPU_CALLABLE_MEMBER
Conserved SRHD::calc_flux(real rho, real v, real pressure)
{

    Conserved flux;

    // The Flux components
    real mom, energy_dens, zeta, D, S, tau, h, W;

    W = 1. / sqrt(1 - v * v);
    h = 1. + gamma * pressure / (rho * (gamma - 1.));
    D = rho * W;
    S = rho * h * W * W * v;
    tau = rho * h * W * W - pressure - W * rho;

    mom = D * v;
    energy_dens = S * v + pressure;
    zeta = (tau + pressure) * v;

    flux.D = mom;
    flux.S = energy_dens;
    flux.tau = zeta;

    return flux;
};

GPU_CALLABLE_MEMBER
Conserved
SRHD::calc_hll_flux(const Primitive &left_prims, const Primitive &right_prims,
                    const Conserved &left_state, const Conserved &right_state,
                    const Conserved &left_flux, const Conserved &right_flux)
{
    Conserved hll_flux;
    real aLm, aRp;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    // Grab the necessary wave speeds
    real aR = lambda.aR;
    real aL = lambda.aL;

    aLm = (aL < 0.0) ? aL : 0.0;
    aRp = (aR > 0.0) ? aR : 0.0;

    // Compute the HLL Flux component-wise
    return (left_flux * aRp - right_flux * aLm +
            (right_state - left_state) * aLm * aRp) /
           (aRp - aLm);
};

GPU_CALLABLE_MEMBER
Conserved
SRHD::calc_hllc_flux(const Primitive &left_prims, const Primitive &right_prims,
                     const Conserved &left_state, const Conserved &right_state,
                     const Conserved &left_flux, const Conserved &right_flux)
{

    Conserved hllc_flux;
    Conserved hll_flux;

    Conserved starState;
    Conserved hll_state;
    real aL, aR, aStar, pStar;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    if (0.0 <= aL)
    {
        return left_flux;
    }
    else if (0.0 >= aR)
    {
        return right_flux;
    }

    hll_flux = calc_hll_flux(left_prims, right_prims, left_state, right_state,
                             left_flux, right_flux);

    hll_state = calc_hll_state(left_state, right_state, left_flux, right_flux,
                               left_prims, right_prims);

    real e = hll_state.tau + hll_state.D;
    real s = hll_state.S;
    real fs = hll_flux.S;
    real fe = hll_flux.tau + hll_flux.D;
    
    real a = fe;
    real b = - (e + fs);
    real c = s;
    real disc = sqrt( b*b - 4*a*c);
    real quad = -0.5*(b + sgn(b)*disc);
    aStar = c/quad;
    pStar = -fe * aStar + fs;

    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D = left_state.D;
        const real S = left_state.S;
        const real tau = left_state.tau;
        const real E = tau + D;
        const real cofactor = 1. / (aL - aStar);
        //--------------Compute the L Star State----------
        const real v = left_prims.v;
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aL - v) * D;
        const real Sstar   = cofactor * (S * (aL - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aL - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto interstate_left = Conserved(Dstar, Sstar, tauStar);

        //---------Compute the L Star Flux
        return left_flux + (interstate_left - left_state) * aL;
    }
    else
    {
        const real pressure = right_prims.p;
        const real D = right_state.D;
        const real S = right_state.S;
        const real tau = right_state.tau;
        const real E = tau + D;
        const real cofactor = 1. / (aR - aStar);
        //--------------Compute the L Star State----------
        const real v = left_prims.v;
        // Left Star State in x-direction of coordinate lattice
        const real Dstar    = cofactor * (aR - v) * D;
        const real Sstar   = cofactor * (S * (aR - v) - pressure + pStar);
        const real Estar    = cofactor * (E * (aR - v) + pStar * aStar - pressure * v);
        const real tauStar  = Estar - Dstar;

        const auto interstate_right = Conserved(Dstar, Sstar, tauStar);

        //---------Compute the R Star Flux
        return right_flux + (interstate_right - right_state) * aR;
    }
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

std::vector<Conserved> SRHD::u_dot1D(std::vector<Conserved> &u_state)
{

    int coordinate;
    Conserved u_l, u_r;
    Conserved f_l, f_r, f1, f2;
    Primitive prims_l, prims_r;
    std::vector<Conserved> L(Nx);

    real rmean, dV, sL, sR, pc, dx;

    if (first_order)
    {
        for (int ii = i_start; ii < i_bound; ii++)
        {
            if (periodic)
            {
                coordinate = ii;
                // Set up the left and right state interfaces for i+1/2
                u_l.D = u_state[ii].D;
                u_l.S = u_state[ii].S;
                u_l.tau = u_state[ii].tau;

                u_r = roll(u_state, ii + 1);
            }
            else
            {
                coordinate = ii - 1;
                // Set up the left and right state interfaces for i+1/2
                u_l.D = u_state[ii].D;
                u_l.S = u_state[ii].S;
                u_l.tau = u_state[ii].tau;

                u_r.D = u_state[ii + 1].D;
                u_r.S = u_state[ii + 1].S;
                u_r.tau = u_state[ii + 1].tau;
            }

            prims_l.rho = prims[ii].rho;
            prims_l.v = prims[ii].v;
            prims_l.p = prims[ii].p;

            prims_r.rho = prims[ii + 1].rho;
            prims_r.v = prims[ii + 1].v;
            prims_r.p = prims[ii + 1].p;

            f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
            f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

            // Calc HLL Flux at i+1/2 interface
            if (hllc)
            {
                f1 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            // Set up the left and right state interfaces for i-1/2
            if (periodic)
            {
                u_l = roll(u_state, ii - 1);

                u_r.D = u_state[ii].D;
                u_r.S = u_state[ii].S;
                u_r.tau = u_state[ii].tau;
            }
            else
            {
                u_l.D = u_state[ii - 1].D;
                u_l.S = u_state[ii - 1].S;
                u_l.tau = u_state[ii - 1].tau;

                u_r.D = u_state[ii].D;
                u_r.S = u_state[ii].S;
                u_r.tau = u_state[ii].tau;
            }

            prims_l.rho = prims[ii - 1].rho;
            prims_l.v = prims[ii - 1].v;
            prims_l.p = prims[ii - 1].p;

            prims_r.rho = prims[ii].rho;
            prims_r.v = prims[ii].v;
            prims_r.p = prims[ii].p;

            f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
            f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

            // Calc HLL Flux at i-1/2 interface
            if (hllc)
            {
                f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            
            switch (geometry[coord_system])
            {
            case simbi::Geometry::CARTESIAN:
                dx = coord_lattice.dx1[coordinate];
                L[coordinate].D   = -(f1.D - f2.D) / dx + sourceD[coordinate];
                L[coordinate].S   = -(f1.S - f2.S) / dx + sourceS[coordinate];
                L[coordinate].tau = -(f1.tau - f2.tau) / dx + source0[coordinate];
                break;
            
            case simbi::Geometry::SPHERICAL:
                pc = prims[ii].p;
                sL = coord_lattice.face_areas[coordinate + 0];
                sR = coord_lattice.face_areas[coordinate + 1];
                dV = coord_lattice.dV[coordinate];
                rmean = coord_lattice.x1mean[coordinate];

                L[coordinate].D = -(sR * f1.D - sL * f2.D) / dV +
                                sourceD[coordinate] * decay_constant;

                L[coordinate].S = -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                                sourceS[coordinate] * decay_constant;

                L[coordinate].tau = -(sR * f1.tau - sL * f2.tau) / dV +
                                    source0[coordinate] * decay_constant;
                break;
            }
            
        }
    }
    else
    {
        Primitive left_most, right_most, left_mid, right_mid, center;
        for (int ii = i_start; ii < i_bound; ii++)
        {
            if (periodic)
            {
                // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                coordinate = ii;
                left_most = roll(prims, ii - 2);
                left_mid = roll(prims, ii - 1);
                center = prims[ii];
                right_mid = roll(prims, ii + 1);
                right_most = roll(prims, ii + 2);
            }
            else
            {
                coordinate = ii - 2;
                left_most = prims[ii - 2];
                left_mid = prims[ii - 1];
                center = prims[ii];
                right_mid = prims[ii + 1];
                right_most = prims[ii + 2];
            }

            // Compute the reconstructed primitives at the i+1/2 interface

            // Reconstructed left primitives vector
            prims_l.rho =
                center.rho + 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                            0.5 * (right_mid.rho - left_mid.rho),
                                            theta * (right_mid.rho - center.rho));

            prims_l.v = center.v + 0.5 * minmod(theta * (center.v - left_mid.v),
                                                0.5 * (right_mid.v - left_mid.v),
                                                theta * (right_mid.v - center.v));

            prims_l.p = center.p + 0.5 * minmod(theta * (center.p - left_mid.p),
                                                0.5 * (right_mid.p - left_mid.p),
                                                theta * (right_mid.p - center.p));

            // Reconstructed right primitives vector
            prims_r.rho = right_mid.rho -
                            0.5 * minmod(theta * (right_mid.rho - center.rho),
                                        0.5 * (right_most.rho - center.rho),
                                        theta * (right_most.rho - right_mid.rho));

            prims_r.v =
                right_mid.v - 0.5 * minmod(theta * (right_mid.v - center.v),
                                            0.5 * (right_most.v - center.v),
                                            theta * (right_most.v - right_mid.v));

            prims_r.p =
                right_mid.p - 0.5 * minmod(theta * (right_mid.p - center.p),
                                            0.5 * (right_most.p - center.p),
                                            theta * (right_most.p - right_mid.p));

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
            u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

            f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
            f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

            if (hllc)
            {
                f1 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            // Do the same thing, but for the right side interface [i - 1/2]
            prims_l.rho =
                left_mid.rho + 0.5 * minmod(theta * (left_mid.rho - left_most.rho),
                                            0.5 * (center.rho - left_most.rho),
                                            theta * (center.rho - left_mid.rho));

            prims_l.v =
                left_mid.v + 0.5 * minmod(theta * (left_mid.v - left_most.v),
                                            0.5 * (center.v - left_most.v),
                                            theta * (center.v - left_mid.v));

            prims_l.p =
                left_mid.p + 0.5 * minmod(theta * (left_mid.p - left_most.p),
                                            0.5 * (center.p - left_most.p),
                                            theta * (center.p - left_mid.p));

            prims_r.rho =
                center.rho - 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                            0.5 * (right_mid.rho - left_mid.rho),
                                            theta * (right_mid.rho - center.rho));

            prims_r.v = center.v - 0.5 * minmod(theta * (center.v - left_mid.v),
                                                0.5 * (right_mid.v - left_mid.v),
                                                theta * (right_mid.v - center.v));

            prims_r.p = center.p - 0.5 * minmod(theta * (center.p - left_mid.p),
                                                0.5 * (right_mid.p - left_mid.p),
                                                theta * (right_mid.p - center.p));

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
            u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

            f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
            f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

            if (hllc)
            {
                f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            switch (geometry[coord_system])
            {
            case simbi::Geometry::CARTESIAN:
                dx = coord_lattice.dx1[coordinate];
                L[coordinate].D = -(f1.D - f2.D) / dx + sourceD[coordinate];
                L[coordinate].S = -(f1.S - f2.S) / dx + sourceS[coordinate];
                L[coordinate].tau = -(f1.tau - f2.tau) / dx + source0[coordinate];
                break;
            
            case simbi::Geometry::SPHERICAL:
                pc = prims[ii].p;
                sL = coord_lattice.face_areas[coordinate + 0];
                sR = coord_lattice.face_areas[coordinate + 1];
                dV = coord_lattice.dV[coordinate];
                rmean = coord_lattice.x1mean[coordinate];

                L[coordinate].D = -(sR * f1.D - sL * f2.D) / dV +
                                sourceD[coordinate] * decay_constant;

                L[coordinate].S = -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                                sourceS[coordinate] * decay_constant;

                L[coordinate].tau = -(sR * f1.tau - sL * f2.tau) / dV +
                                    source0[coordinate] * decay_constant;
                break;
            }
        }
    }
    return L;
};
//=====================================================================
//                          KERNEL CALLS
//=====================================================================
__global__ void simbi::shared_gpu_cons2prim(SRHD *s, int n){
    __shared__ Conserved  conserved_buff[BLOCK_SIZE];
    __shared__ Primitive  primitive_buff[BLOCK_SIZE];

    real eps, p, v2, et, c2, h, g, f, W, rho;
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;
    int iter = 0;
    if (ii < s->Nx){
        // load shared memory
        conserved_buff[tx] = s->gpu_sys_state[ii];
        primitive_buff[tx] = s->gpu_prims[ii];
        real D   = conserved_buff[tx].D;
        real S   = conserved_buff[tx].S;
        real tau = conserved_buff[tx].tau;

        real peq = s->gpu_pressure_guess[ii];

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
                printf("\n");
                printf("Cons2Prim cannot converge");
                printf("\n");
                // exit(EXIT_FAILURE);
            }

        } while (abs(peq - p) >= tol);

        real v = S / (tau + D + peq);

        s->gpu_pressure_guess[ii] = peq;
        s->gpu_prims[ii] = Primitive{D * sqrt(1 - v * v), v, peq};

    }
}

__global__ void simbi::gpu_advance(
    SRHD *s,  
    const int n, 
    const simbi::Geometry geometry)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    const int ibound                = s->i_bound;
    const int istart                = s->i_start;
    const real decay_constant       = s->decay_constant;
    const CLattice1D *coord_lattice = &(s->coord_lattice);
    const Primitive  *prims         = s->gpu_prims;
    const real dt                   = s->dt;
    const real plm_theta            = s->theta;

    int coordinate;
    Conserved u_l, u_r;
    Conserved f_l, f_r, f1, f2;
    Primitive prims_l, prims_r;
    real rmean, dV, sL, sR, pc, dx;

    
    Conserved *u_state = s->gpu_sys_state;
    if (ii < s-> Nx)
    {
        if (s->first_order)
        {
            if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l.D   = u_state[ii].D;
                    u_l.S   = u_state[ii].S;
                    u_l.tau = u_state[ii].tau;

                    u_r = roll(u_state, ii + 1, n);
                }
                else
                {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l.D   = u_state[ii].D;
                    u_l.S   = u_state[ii].S;
                    u_l.tau = u_state[ii].tau;

                    u_r.D   = u_state[ii + 1].D;
                    u_r.S   = u_state[ii + 1].S;
                    u_r.tau = u_state[ii + 1].tau;
                }

                prims_l.rho = s->gpu_prims[ii].rho;
                prims_l.v   = s->gpu_prims[ii].v;
                prims_l.p   = s->gpu_prims[ii].p;

                prims_r.rho = s->gpu_prims[ii + 1].rho;
                prims_r.v   = s->gpu_prims[ii + 1].v;
                prims_r.p   = s->gpu_prims[ii + 1].p;

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i+1/2 interface
                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Set up the left and right state interfaces for i-1/2
                if (s->periodic)
                {
                    u_l = roll(u_state, ii - 1, n);

                    u_r.D   = u_state[ii].D;
                    u_r.S   = u_state[ii].S;
                    u_r.tau = u_state[ii].tau;
                }
                else
                {
                    u_l.D   = u_state[ii - 1].D;
                    u_l.S   = u_state[ii - 1].S;
                    u_l.tau = u_state[ii - 1].tau;

                    u_r.D   = u_state[ii].D;
                    u_r.S   = u_state[ii].S;
                    u_r.tau = u_state[ii].tau;
                }

                prims_l.rho = s->gpu_prims[ii - 1].rho;
                prims_l.v   = s->gpu_prims[ii - 1].v;
                prims_l.p   = s->gpu_prims[ii - 1].p;

                prims_r.rho = s->gpu_prims[ii].rho;
                prims_r.v   = s->gpu_prims[ii].v;
                prims_r.p   = s->gpu_prims[ii].p;

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i-1/2 interface
                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[coordinate];
                    s->gpu_sys_state[ii].D   += dt * -(f1.D - f2.D)     / dx + s->gpu_sourceD[coordinate];
                    s->gpu_sys_state[ii].S   += dt * -(f1.S - f2.S)     / dx + s->gpu_sourceS[coordinate];
                    s->gpu_sys_state[ii].tau += dt * -(f1.tau - f2.tau) / dx + s->gpu_source0[coordinate];

                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = s->gpu_prims[ii].p;
                    sL = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_sys_state[ii].D += dt * ( 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].S += dt * (
                        -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].tau += dt *(
                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    );
                    break;
                }
                
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    coordinate = ii;
                    left_most  = roll(prims, ii - 2, n);
                    left_mid   = roll(prims, ii - 1, n);
                    center     = s->gpu_prims[ii];
                    right_mid  = roll(prims, ii + 1, n);
                    right_most = roll(prims, ii + 2, n);
                }
                else
                {
                    coordinate = ii - 2;
                    left_most  = s->gpu_prims[ii - 2];
                    left_mid   = s->gpu_prims[ii - 1];
                    center     = s->gpu_prims[ii];
                    right_mid  = s->gpu_prims[ii + 1];
                    right_most = s->gpu_prims[ii + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho =
                    center.rho + 0.5 * minmod(s->theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                s->theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5 * minmod(s->theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    s->theta * (right_mid.v - center.v));

                prims_l.p = center.p + 0.5 * minmod(s->theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    s->theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                                0.5 * minmod(s->theta * (right_mid.rho - center.rho),
                                            0.5 * (right_most.rho - center.rho),
                                            s->theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - 0.5 * minmod(s->theta * (right_mid.v - center.v),
                                                0.5 * (right_most.v - center.v),
                                                s->theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - 0.5 * minmod(s->theta * (right_mid.p - center.p),
                                                0.5 * (right_most.p - center.p),
                                                s->theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = s->calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho =
                    left_mid.rho + 0.5 * minmod(s->theta * (left_mid.rho - left_most.rho),
                                                0.5 * (center.rho - left_most.rho),
                                                s->theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + 0.5 * minmod(s->theta * (left_mid.v - left_most.v),
                                                0.5 * (center.v - left_most.v),
                                                s->theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + 0.5 * minmod(s->theta * (left_mid.p - left_most.p),
                                                0.5 * (center.p - left_most.p),
                                                s->theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - 0.5 * minmod(s->theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                s->theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5 * minmod(s->theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    s->theta * (right_mid.v - center.v));

                prims_r.p = center.p - 0.5 * minmod(s->theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    s->theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = s->calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[coordinate];
                    s->gpu_sys_state[ii].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx +  s->gpu_sourceD[coordinate] );
                    s->gpu_sys_state[ii].S   += 0.5 * dt * ( -(f1.S - f2.S)     / dx +  s->gpu_sourceS[coordinate] );
                    s->gpu_sys_state[ii].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx  + s->gpu_source0[coordinate] );
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = s->gpu_prims[ii].p;
                    sL = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_sys_state[ii].D += 0.5 * dt * (
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].S += 0.5 * dt * (
                        -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].tau +=  0.5 * dt * (
                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    );
                    break;
                }
            }
        }

    }
    
}


__global__ void simbi::shared_gpu_advance(
    SRHD *s,  
    const int sh_block_size,
    const int radius, 
    const simbi::Geometry geometry)
{
    int ii  = blockDim.x * blockIdx.x + threadIdx.x;
    int txa = threadIdx.x + radius;
    int nx  = s->Nx;

    extern __shared__ Conserved smem_buff[];
    Conserved *cons_buff = smem_buff;
    Primitive *prim_buff = (Primitive *)&cons_buff[sh_block_size];

    const int ibound                = s->i_bound;
    const int istart                = s->i_start;
    const real decay_constant       = s->decay_constant;
    const CLattice1D *coord_lattice = &(s->coord_lattice);
    const Primitive  *prims         = s->gpu_prims;
    const real dt                   = s->dt;
    const real plm_theta            = s->theta;

    int coordinate;
    Conserved u_l, u_r;
    Conserved f_l, f_r, f1, f2;
    Primitive prims_l, prims_r;
    real rmean, dV, sL, sR, pc, dx;

    if (ii < s-> Nx)
    {
        cons_buff[txa] = s->gpu_sys_state[ii];
        prim_buff[txa] = s->gpu_prims[ii];
        if (threadIdx.x < radius)
        {
            cons_buff[txa - radius]     = s->gpu_sys_state[ii - radius];
            cons_buff[txa + BLOCK_SIZE] = s->gpu_sys_state[ii + BLOCK_SIZE];
            prim_buff[txa - radius]     = s->gpu_prims[ii - radius];
            prim_buff[txa + BLOCK_SIZE] = s->gpu_prims[ii + BLOCK_SIZE];  
        }
        __syncthreads();
        if (s->first_order)
        {
            if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l   = cons_buff[txa];
                    u_r   = roll(cons_buff, txa + 1, sh_block_size);
                }
                else
                {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l = cons_buff[txa];
                    u_r = cons_buff[txa + 1];
                }

                prims_l = prim_buff[txa];
                prims_r = prim_buff[txa + 1];

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i+1/2 interface
                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Set up the left and right state interfaces for i-1/2
                if (s->periodic)
                {
                    u_l = roll(cons_buff, txa - 1, sh_block_size);
                    u_r = cons_buff[txa];
                }
                else
                {
                    u_l   = cons_buff[txa - 1];
                    u_r   = cons_buff[txa];
                }

                prims_l = prim_buff[txa - 1];
                prims_r = prim_buff[txa];

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                // Calc HLL Flux at i-1/2 interface
                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[coordinate];
                    s->gpu_sys_state[ii].D   += dt * -(f1.D - f2.D)     / dx + s->gpu_sourceD[coordinate];
                    s->gpu_sys_state[ii].S   += dt * -(f1.S - f2.S)     / dx + s->gpu_sourceS[coordinate];
                    s->gpu_sys_state[ii].tau += dt * -(f1.tau - f2.tau) / dx + s->gpu_source0[coordinate];

                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = prim_buff[txa].p;
                    sL = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_sys_state[ii].D += dt * ( 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].S += dt * (
                        -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].tau += dt *(
                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    );
                    break;
                }
                
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    coordinate = ii;
                    left_most  = roll(prim_buff, txa - 2, sh_block_size);
                    left_mid   = roll(prim_buff, txa - 1, sh_block_size);
                    center     = prim_buff[txa];
                    right_mid  = roll(prim_buff, txa + 1, sh_block_size);
                    right_most = roll(prim_buff, txa + 2, sh_block_size);
                }
                else
                {
                    coordinate = ii - 2;
                    left_most  = prim_buff[txa - 2];
                    left_mid   = prim_buff[txa - 1];
                    center     = prim_buff[txa];
                    right_mid  = prim_buff[txa + 1];
                    right_most = prim_buff[txa + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho =
                    center.rho + 0.5 * minmod(s->theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                s->theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5 * minmod(s->theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    s->theta * (right_mid.v - center.v));

                prims_l.p = center.p + 0.5 * minmod(s->theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    s->theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                                0.5 * minmod(s->theta * (right_mid.rho - center.rho),
                                            0.5 * (right_most.rho - center.rho),
                                            s->theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - 0.5 * minmod(s->theta * (right_mid.v - center.v),
                                                0.5 * (right_most.v - center.v),
                                                s->theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - 0.5 * minmod(s->theta * (right_mid.p - center.p),
                                                0.5 * (right_most.p - center.p),
                                                s->theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = s->calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho =
                    left_mid.rho + 0.5 * minmod(s->theta * (left_mid.rho - left_most.rho),
                                                0.5 * (center.rho - left_most.rho),
                                                s->theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + 0.5 * minmod(s->theta * (left_mid.v - left_most.v),
                                                0.5 * (center.v - left_most.v),
                                                s->theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + 0.5 * minmod(s->theta * (left_mid.p - left_most.p),
                                                0.5 * (center.p - left_most.p),
                                                s->theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - 0.5 * minmod(s->theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                s->theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5 * minmod(s->theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    s->theta * (right_mid.v - center.v));

                prims_r.p = center.p - 0.5 * minmod(s->theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    s->theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->calc_state(prims_l.rho, prims_l.v, prims_l.p);
                u_r = s->calc_state(prims_r.rho, prims_r.v, prims_r.p);

                f_l = s->calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                f_r = s->calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[coordinate];
                    s->gpu_sys_state[ii].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx +  s->gpu_sourceD[coordinate] );
                    s->gpu_sys_state[ii].S   += 0.5 * dt * ( -(f1.S - f2.S)     / dx +  s->gpu_sourceS[coordinate] );
                    s->gpu_sys_state[ii].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx  + s->gpu_source0[coordinate] );
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = s->gpu_prims[ii].p;
                    sL = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_sys_state[ii].D += 0.5 * dt * (
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].S += 0.5 * dt * (
                        -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant
                    );

                    s->gpu_sys_state[ii].tau +=  0.5 * dt * (
                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    );
                    break;
                }
            }
        }

    }
    
}

std::vector<std::vector<real>>
SRHD::simulate1D(std::vector<real> &lorentz_gamma, std::vector<std::vector<real>> &sources,
                 real tstart = 0.0, real tend = 0.1, real dt = 1.e-4,
                 real theta = 1.5, real engine_duration = 10,
                 real chkpt_interval = 0.1, std::string data_directory = "data/",
                 bool first_order = true, bool periodic = false,
                 bool linspace = true, bool hllc = false)
{
    this->periodic = periodic;
    this->first_order = first_order;
    this->theta = theta;
    this->linspace = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this->sourceD = sources[0];
    this->sourceS = sources[1];
    this->source0 = sources[2];
    this->hllc = hllc;
    this->engine_duration = engine_duration;
    this->t    = tstart;
    this->dt   = dt;
    this->tend = tend;
    // Define the swap vector for the integrated state
    this->Nx = lorentz_gamma.size();

    if (periodic)
    {
        this->idx_shift = 0;
        this->i_start   = 0;
        this->i_bound   = Nx;
    }
    else
    {
        if (first_order)
        {
            this->idx_shift  = 1;
            this->pgrid_size = Nx - 2;
            this->i_start    = 1;
            this->i_bound    = Nx - 1;
        }
        else
        {
            this->idx_shift  = 2;
            this->pgrid_size = Nx - 4;
            this->i_start    = 2;
            this->i_bound    = Nx - 2;
        }
    }
    config_system();
    int i_real;
    n = 0;
    std::vector<Conserved> u, u1, udot, udot1;
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / chkpt_interval;
    real t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;
    DataWriteMembers setup;
    setup.xmax = r[pgrid_size - 1];
    setup.xmin = r[0];
    setup.xactive_zones = pgrid_size;
    setup.NX = Nx;

    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    sr1d::PrimitiveArray transfer_prims;

    u.resize(Nx);
    prims.resize(Nx);
    pressure_guess.resize(Nx);
    // Copy the state array into real & profile variables
    for (size_t ii = 0; ii < Nx; ii++)
    {
        u[ii] = Conserved{state[0][ii],
                          state[1][ii],
                          state[2][ii]};
    }

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

    cons2prim1D(u);
    n++;

    sys_state = u;

    // Copy the current SRHD instance over to the device
    simbi::SRHD *device_self;
    hipMalloc((void**)&device_self,    sizeof(SRHD));
    hipMemcpy(device_self,  this,      sizeof(SRHD), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed when copying current sim state to device");
    SRHD_DualSpace dualMem;
    dualMem.copyStateToGPU(*this, device_self);
    hipCheckErrors("Error in copying host state to device");

    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag, num_zeros;

    // Setup the system
    const int nBlocks = (this->Nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int physical_nBlocks = (this->pgrid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
        const int shBlockSize  = BLOCK_SIZE + 2 * radius;
        const unsigned shBlockBytes = shBlockSize * sizeof(Conserved) + shBlockSize * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nBlocks), dim3(BLOCK_SIZE), 0, 0, device_self, this->Nx);
            hipLaunchKernelGGL(shared_gpu_advance, dim3(nBlocks), dim3(BLOCK_SIZE), shBlockBytes, 0, device_self, shBlockSize, radius, geometry[this->coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU, dim3(1), dim3(1), 0, 0, device_self, Nx, first_order);
            t += dt; 
            n++;
            hipDeviceSynchronize();

            if (n >= nfold){
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += Nx / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< Nx / delta_t.count() << std::flush;
                nfold += 1000;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                dualMem.copyGPUStateToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
                writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", pgrid_size);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, Nx);
                t_interval += chkpt_interval;
            }

            // Adapt the timestep
            hipLaunchKernelGGL(adapt_dtGPU, dim3(physical_nBlocks), dim3(BLOCK_SIZE), 0, 0, device_self, nBlocks, &(dt));
            hipMemcpy(&dt, &(device_self->dt),  sizeof(real), hipMemcpyDeviceToHost);
        }
    } else {
        const int radius = 2;
        const int shBlockSize  = BLOCK_SIZE + 2 * radius;
        const unsigned shBlockBytes = shBlockSize * sizeof(Conserved) + shBlockSize * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nBlocks), dim3(BLOCK_SIZE), 0, 0, device_self, this->Nx);
            hipLaunchKernelGGL(shared_gpu_advance, dim3(nBlocks), dim3(BLOCK_SIZE), shBlockBytes, 0, device_self, shBlockSize, radius, geometry[this->coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU, dim3(1), dim3(1), 0, 0, device_self, Nx, first_order);

            // Final Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nBlocks), dim3(BLOCK_SIZE), 0, 0, device_self, this->Nx);
            hipLaunchKernelGGL(shared_gpu_advance, dim3(nBlocks), dim3(BLOCK_SIZE), shBlockBytes, 0, device_self, shBlockSize, radius, geometry[this->coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU, dim3(1), dim3(1), 0, 0, device_self, Nx, first_order);

            t += dt; 
            n++;
            hipDeviceSynchronize();

            if (n >= nfold){
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += Nx / delta_t.count();
                std::cout << std::fixed << std::setprecision(3) << std::scientific;
                    std::cout << "\r"
                        << "Iteration: " << std::setw(5) << n 
                        << "\t"
                        << "dt: " << std::setw(5) << dt 
                        << "\t"
                        << "Time: " << std::setw(10) <<  t
                        << "\t"
                        << "Zones/sec: "<< Nx / delta_t.count() << std::flush;
                nfold += 1000;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                dualMem.copyGPUStateToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
                writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", pgrid_size);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, Nx);
                t_interval += chkpt_interval;
            }

            //Adapt the timestep
            hipLaunchKernelGGL(adapt_dtGPU, dim3(physical_nBlocks), dim3(BLOCK_SIZE), 0, 0, device_self, nBlocks, &(dt));
            hipMemcpy(&dt, &(device_self->dt),  sizeof(real), hipMemcpyDeviceToHost);
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    hipFree(device_self);
    cons2prim1D(sys_state);

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(Nx, 0));
    for (size_t ii = 0; ii < Nx; ii++)
    {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};