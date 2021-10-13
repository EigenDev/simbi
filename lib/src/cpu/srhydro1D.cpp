/*
 * C++ Library to perform extensive hydro calculations
 * to be later wrapped and plotted in Python
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "common/helpers.hpp"
#include "helpers.hip.hpp"
#include "srhydro1D.hip.hpp"
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
    int nz     = host.nx;
    int nzreal = host.active_zones; 

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
    hipMalloc((void **)&host_x1m,             rrbytes);
    hipMalloc((void **)&host_fas,             fabytes);
    hipMalloc((void **)&host_source0,         rrbytes);
    hipMalloc((void **)&host_sourceD,         rrbytes);
    hipMalloc((void **)&host_sourceS,         rrbytes);

    hipMalloc((void **)&host_dtmin,            rbytes);
    hipMalloc((void **)&host_clattice, sizeof(CLattice1D));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.cons.data(), cbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring host.cons to host_u0");

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

    hipMemcpy(host_x1m, host.coord_lattice.x1mean.data(),     rrbytes, hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring x1mean");

    // Now copy pointer to device directly
    hipMemcpy(&(device->coord_lattice.gpu_dx1), &host_dx1, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_dV), &host_dV, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_x1mean),&host_x1m, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->coord_lattice.gpu_face_areas), &host_fas, sizeof(real *), hipMemcpyHostToDevice);
    hipCheckErrors("Memcpy failed at transferring dx1");

    hipMemcpy(&(device->dt),        &host.dt      ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->gamma),     &host.gamma   ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->CFL)  ,     &host.CFL     ,  sizeof(real), hipMemcpyHostToDevice);
    hipMemcpy(&(device->nx),        &host.nx      ,  sizeof(int),  hipMemcpyHostToDevice);
    hipMemcpy(&(device->active_zones),&host.active_zones,  sizeof(int),  hipMemcpyHostToDevice);
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
    const int nz     = host.nx;
    const int cbytes = nz * sizeof(Conserved); 
    const int pbytes = nz * sizeof(Primitive); 

    hipMemcpy(host.cons.data(), host_u0,        cbytes, hipMemcpyDeviceToHost);
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
void SRHD::cons2prim1D()
{
    real rho, S, D, tau;
    real v, W, tol, f, g, peq, h;
    real eps, p, v2, et, c2;
    int iter = 0;
    for (int ii = 0; ii < nx; ii++)
    {
        D   = cons[ii].D;
        S   = cons[ii].S;
        tau = cons[ii].tau;

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
        pressure_guess[ii] = peq;
        prims[ii] = Primitive{D * std::sqrt(1 - v * v), v, peq};
    }
};

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
    const real h_l   = 1. + gamma * p_l / (rho_l * (gamma - 1.));
    const real cs_l  = sqrt(gamma * p_l / (rho_l * h_l));

    const real rho_r = prims_r.rho;
    const real p_r   = prims_r.p;
    const real v_r   = prims_r.v;
    const real h_r   = 1. + gamma * p_r / (rho_r * (gamma - 1.));
    const real cs_r  = sqrt(gamma * p_r / (rho_r * h_r));

    // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
    const real vbar = 0.5 * (v_l + v_r);
    const real cbar = 0.5 * (cs_r + cs_l);
    const real br = (vbar + cbar) / (1 + vbar * cbar);
    const real bl = (vbar - cbar) / (1 - vbar * cbar);

    const real aL = my_min(bl, (v_l - cs_l) / (1 - v_l * cs_l));
    const real aR = my_max(br, (v_r + cs_r) / (1 + v_r * cs_r));


    // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
    // const real sL = cs_l*cs_l/(gamma*gamma*(1.0 - cs_l*cs_l));
    // const real sR = cs_r*cs_r/(gamma*gamma*(1.0 - cs_r*cs_r));
    // // Define temporaries to save computational cycles
    // const real qfL = 1. / (1. + sL);
    // const real qfR = 1. / (1. + sR);
    // const real sqrtR = sqrt(sR * (1.0 - v_r * v_r + sR));
    // const real sqrtL = sqrt(sL * (1.0 - v_l * v_l + sL));

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
    double min_dt = INFINITY;
    #pragma omp parallel 
    {
        double dr, cs, cfl_dt;
        double h, rho, p, v, vPLus, vMinus;

        // Compute the minimum timestep given CFL
        #pragma omp for schedule(static)
        for (int ii = 0; ii < active_zones; ii++)
        {
            dr  = coord_lattice.dx1[ii];
            rho = prims[ii + idx_shift].rho;
            p   = prims[ii + idx_shift].p;
            v   = prims[ii + idx_shift].v;

            h = 1. + gamma * p / (rho * (gamma - 1.));
            cs = sqrt(gamma * p / (rho * h));

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
    real min_dt = INFINITY;
    dtWarpReduce<SRHD, Primitive, 128><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
    hipDeviceSynchronize();
    hipMemcpy(&dt, &(dev->dt),  sizeof(real), hipMemcpyDeviceToHost);
    
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
    const real h = 1. + gamma * pre / (rho * (gamma - 1.));
    const real W = 1. / sqrt(1 - v * v);

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
    const real Sstar = (1. / (a - aStar)) * (S * (a - v) - pressure + pStar);
    const real Estar = (1. / (a - aStar)) * (E * (a - v) + pStar * aStar - pressure * v);
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

    const real W = 1. / sqrt(1 - v * v);
    const real h = 1. + gamma * pre / (rho * (gamma - 1.));
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
    const real aLm = aL < 0.0 ? aL : 0.0;
    const real aRp = aR > 0.0 ? aR : 0.0;

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

    if (0.0 <= aL)
    {
        return left_flux;
    }
    else if (0.0 >= aR)
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
    const real disc = sqrt( b*b - 4.0*a*c);
    const real quad = -0.5*(b + sgn(b)*disc);
    const real aStar = c/quad;
    const real pStar = -fe * aStar + fs;

    if (-aL <= (aStar - aL))
    {
        const real pressure = left_prims.p;
        const real D        = left_state.D;
        const real S        = left_state.S;
        const real tau      = left_state.tau;
        const real E        = tau + D;
        const real cofactor = 1. / (aL - aStar);
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
        const real cofactor  = 1. / (aR - aStar);
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

//=====================================================================
//                          KERNEL CALLS
//=====================================================================
__global__ void simbi::shared_gpu_cons2prim(SRHD *s){
    __shared__ Conserved  conserved_buff[BLOCK_SIZE];
    __shared__ Primitive  primitive_buff[BLOCK_SIZE];

    real eps, p, v2, et, c2, h, g, f, W, rho;
    const real gamma = s->gamma;
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;
    int iter = 0;
    if (ii < s->nx){
        // load shared memory
        conserved_buff[tx] = s->gpu_cons[ii];
        primitive_buff[tx] = s->gpu_prims[ii];
        const real D       = conserved_buff[tx].D;
        const real S       = conserved_buff[tx].S;
        const real tau     = conserved_buff[tx].tau;

        real peq = s->gpu_pressure_guess[ii];

        const real tol = D * 1.e-12;
        do
        {
            p = peq;
            et = tau + D + p;
            v2 = S * S / (et * et);
            W = 1.0 / sqrt(1.0 - v2);
            rho = D / W;

            eps = (tau + (1.0 - W) * D + (1. - W * W) * p) / (D * W);

            h  = 1. + eps + p / rho;
            c2 = gamma * p / (h * rho); 

            g = c2 * v2 - 1.0;
            f = (gamma - 1.0) * rho * eps - p;

            peq = p - f / g;
            if (iter >= MAX_ITER)
            {
                printf("\n");
                printf("Cons2Prim cannot converge");
                printf("\n");
                // exit(EXIT_FAILURE);
            }
            iter++;

        } while (abs(peq - p) >= tol);

        real v = S / (tau + D + peq);

        s->gpu_pressure_guess[ii] = peq;
        s->gpu_prims[ii] = Primitive{D * sqrt(1 - v * v), v, peq};

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

    extern __shared__ Primitive prim_buff[];

    const int ibound                = s->i_bound;
    const int istart                = s->i_start;
    const real decay_constant       = s->decay_constant;
    const CLattice1D *coord_lattice = &(s->coord_lattice);
    const real dt                   = s->dt;
    const real plm_theta            = s->plm_theta;
    const int nx                    = s->nx;

    int coordinate;
    Conserved u_l, u_r;
    Conserved f_l, f_r, f1, f2;
    Primitive prims_l, prims_r;
    real rmean, dV, sL, sR, pc, dx;
    if (ii < s->active_zones)
    {
        const int ia = ii + radius;
        const bool inbounds = ia + BLOCK_SIZE < nx - 1;
        prim_buff[txa] = s->gpu_prims[ia];

        if(!inbounds)
        {
            prim_buff[txa] = s->gpu_prims[nx - 1];
        }
        if (threadIdx.x < radius)
        {
            prim_buff[txa - radius]     = s->gpu_prims[ia - radius];
            prim_buff[txa + BLOCK_SIZE] = inbounds ? s->gpu_prims[ia + BLOCK_SIZE] : s->gpu_prims[nx - 1];  
        }
        __syncthreads();

        if (s->first_order)
        {
            // if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    prims_l = prim_buff[txa];
                    prims_r = roll(prim_buff, txa + 1, sh_block_size);
                }
                else
                {
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    prims_l = prim_buff[txa];
                    prims_r = prim_buff[txa + 1];
                }
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);
                

                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

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
                    prims_l = roll(prim_buff, txa - 1, sh_block_size);
                    prims_r = prim_buff[txa];
                }
                else
                {
                    prims_l = prim_buff[txa - 1];
                    prims_r = prim_buff[txa];
                }

                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);

                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

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
                    s->gpu_cons[ia].D   += dt * ( -(f1.D - f2.D)     / dx + s->gpu_sourceD[coordinate] );
                    s->gpu_cons[ia].S   += dt * ( -(f1.S - f2.S)     / dx + s->gpu_sourceS[coordinate] );
                    s->gpu_cons[ia].tau += dt * ( -(f1.tau - f2.tau) / dx + s->gpu_source0[coordinate] );

                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = prim_buff[txa].p;
                    sL = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + 2.0 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    } * dt;
                    break;
                }
                
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            // if ( (unsigned)(ii - istart) < (ibound - istart))
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
                    coordinate = ii;
                    left_most  = prim_buff[txa - 2];
                    left_mid   = prim_buff[txa - 1];
                    center     = prim_buff[txa + 0];
                    right_mid  = prim_buff[txa + 1];
                    right_most = prim_buff[txa + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho =
                    center.rho + 0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                plm_theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_l.p = center.p + 0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                                0.5 * minmod(plm_theta * (right_mid.rho - center.rho),
                                            0.5 * (right_most.rho - center.rho),
                                            plm_theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - 0.5 * minmod(plm_theta * (right_mid.v - center.v),
                                                0.5 * (right_most.v - center.v),
                                                plm_theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - 0.5 * minmod(plm_theta * (right_mid.p - center.p),
                                                0.5 * (right_most.p - center.p),
                                                plm_theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);

                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

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
                    left_mid.rho + 0.5 * minmod(plm_theta * (left_mid.rho - left_most.rho),
                                                0.5 * (center.rho - left_most.rho),
                                                plm_theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + 0.5 * minmod(plm_theta * (left_mid.v - left_most.v),
                                                0.5 * (center.v - left_most.v),
                                                plm_theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + 0.5 * minmod(plm_theta * (left_mid.p - left_most.p),
                                                0.5 * (center.p - left_most.p),
                                                plm_theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - 0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                plm_theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_r.p = center.p - 0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);

                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

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
                    s->gpu_cons[ia].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx +  s->gpu_sourceD[coordinate] );
                    s->gpu_cons[ia].S   += 0.5 * dt * ( -(f1.S - f2.S)     / dx +  s->gpu_sourceS[coordinate] );
                    s->gpu_cons[ia].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx  + s->gpu_source0[coordinate] );
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc    = prim_buff[txa].p;
                    sL    = coord_lattice->gpu_face_areas[coordinate + 0];
                    sR    = coord_lattice->gpu_face_areas[coordinate + 1];
                    dV    = coord_lattice->gpu_dV[coordinate];
                    rmean = coord_lattice->gpu_x1mean[coordinate];

                    s->gpu_cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[coordinate] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + 2.0 * pc / rmean +
                        s->gpu_sourceS[coordinate] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[coordinate] * decay_constant
                    } * dt * 0.5;
                    break;
                }
            }
        }

    }
    
}

std::vector<std::vector<real>>
SRHD::simulate1D(
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
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;
    DataWriteMembers setup;
    setup.xmax = r[active_zones - 1];
    setup.xmin = r[0];
    setup.xactive_zones = active_zones;
    setup.NX = nx;
    setup.linspace = linspace;

    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    sr1d::PrimitiveArray transfer_prims;

    cons.resize(nx);
    prims.resize(nx);
    pressure_guess.resize(nx);
    // Copy the state array into real & profile variables
    for (size_t ii = 0; ii < nx; ii++)
    {
        cons[ii] = Conserved{state[0][ii],
                          state[1][ii],
                          state[2][ii]};
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

    cons2prim1D();
    // Check if user input a valid initial timestep. 
    // if not, adapt it
    adapt_dt();
    dt = init_dt < dt ? init_dt : dt;

    dt_arr.resize(nx);


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
    int time_order_of_mag;

    // Setup the system
    const int nBlocks = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int physical_nBlocks = (active_zones + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dim3 fgridDim   = dim3(nBlocks);
    const dim3 agridDim   = dim3(physical_nBlocks);
    const dim3 threadDim  = dim3(BLOCK_SIZE);
    // Some benchmarking tools 
    int   nfold   = 0;
    int   ncheck  = 0;
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
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, threadDim, 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, threadDim, shBlockBytes, 0, device_self, shBlockSize, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU,   dim3(1), dim3(1), 0, 0, device_self, nx, first_order);
            t += dt; 
            
            hipDeviceSynchronize();

            if (n >= nfold){
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
                filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            // Adapt the timestep
            adapt_dt(device_self, physical_nBlocks);
        }
    } else {
        const int radius = 2;
        const int shBlockSize  = BLOCK_SIZE + 2 * radius;
        const unsigned shBlockBytes = shBlockSize * sizeof(Conserved) + shBlockSize * sizeof(Primitive);
        while (t < tend)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, threadDim, 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, threadDim, shBlockBytes, 0, device_self, shBlockSize, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU, dim3(1), dim3(1), 0, 0, device_self, nx, first_order);

            // Final Half Step
            hipLaunchKernelGGL(shared_gpu_cons2prim, fgridDim, dim3(BLOCK_SIZE), 0, 0, device_self);
            hipLaunchKernelGGL(shared_gpu_advance,   agridDim, dim3(BLOCK_SIZE), shBlockBytes, 0, device_self, shBlockSize, radius, geometry[coord_system]);
            hipLaunchKernelGGL(config_ghosts1DGPU,   dim3(1), dim3(1), 0, 0, device_self, nx, first_order);

            t += dt; 
            hipDeviceSynchronize();

            if (n >= nfold){
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
                nfold += 1;
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
                filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            n++;
            //Adapt the timestep
            adapt_dt(device_self, physical_nBlocks);
        }

    }
    
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";

    hipFree(device_self);
    cons2prim1D();

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (size_t ii = 0; ii < nx; ii++)
    {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }

    return final_prims;
};