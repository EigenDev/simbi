/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "euler1D.hpp" 
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include "util/parallel_for.hpp"
#include "util/exec_policy.hpp"
#include "util/dual.hpp"
#include "util/device_api.hpp"
#include "util/printb.hpp"
#include "helpers.hip.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;


// Default Constructor 
Newtonian1D::Newtonian1D () {}

// Overloaded Constructor
Newtonian1D::Newtonian1D(
    std::vector< std::vector<real> > init_state, 
    real gamma, 
    real cfl, 
    std::vector<real> r,
    std::string coord_system = "cartesian") :

    init_state(init_state),
    gamma(gamma),
    r(r),
    coord_system(coord_system),
    cfl(cfl),
    inFailureState(false)
    {

    }

// Destructor 
Newtonian1D::~Newtonian1D() {}


// Typedefs because I'm lazy
typedef hydro1d::Conserved Conserved;
typedef hydro1d::Primitive Primitive;
typedef hydro1d::Eigenvals Eigenvals;
//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------
/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
void Newtonian1D::cons2prim(ExecutionPolicy<> p, Newtonian1D *dev, simbi::MemSide user){
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
     simbi::parallel_for(p, (luint)0, nx, [=] GPU_LAMBDA (luint ii){
         #if GPU_CODE
        __shared__ Conserved  conserved_buff[BLOCK_SIZE];
        #else 
        auto* const conserved_buff = &cons[0];
        #endif 

        luint tx = (BuildPlatform == Platform::GPU) ? threadIdx.x : ii;
        // Compile time thread selection
        #if GPU_CODE
            conserved_buff[tx] = self->gpu_cons[ii];
        #endif
        simbi::gpu::api::synchronize();

        real rho, pre, v;
        rho = conserved_buff[tx].rho;
        v   = conserved_buff[tx].m/rho;
        pre = (self->gamma - static_cast<real>(1.0))*(conserved_buff[tx].e_dens - static_cast<real>(0.5) * rho * v * v);

        #if GPU_CODE
            self->gpu_prims[ii] = Primitive{rho, v, pre};
        #else
            prims[ii]  = Primitive{rho, v, pre};
        #endif
        // workLeftToDo = false;
    });
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------

GPU_CALLABLE_MEMBER
Eigenvals Newtonian1D::calc_eigenvals(const Primitive &left_prim, const Primitive &right_prim)
{
    // Separate the left and right state components
    const real rho_l    = left_prim.rho;
    const real v_l      = left_prim.v;
    const real p_l      = left_prim.p;
    const real rho_r    = right_prim.rho;
    const real v_r      = right_prim.v;
    const real p_r      = right_prim.p;

    const real cs_r = std::sqrt(gamma * p_r/rho_r);
    const real cs_l = std::sqrt(gamma * p_l/rho_l);

    switch (sim_solver)
    {
    case SOLVER::HLLE:
        {
        const real aR = my_max(my_max(v_l + cs_l, v_r + cs_r), static_cast<real>(0.0)); 
        const real aL = my_min(my_min(v_l - cs_l, v_r - cs_r), static_cast<real>(0.0));
        return Eigenvals{aL, aR};
        }
    case SOLVER::HLLC:
        real cbar   = static_cast<real>(0.5)*(cs_l + cs_r);
        real rhoBar = static_cast<real>(0.5)*(rho_l + rho_r);
        real pStar  = static_cast<real>(0.5)*(p_l + p_r) + static_cast<real>(0.5)*(v_l - v_r)*cbar*rhoBar;

        // Steps to Compute HLLC as described in Toro et al. 2019
        real z      = (gamma - 1.)/(2.*gamma);
        real num    = cs_l + cs_r - ( gamma-1.)/2 *(v_r - v_l);
        real denom  = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
        real p_term = num/denom;
        real qL, qR;

        pStar = pow(p_term, (1./z));

        if (pStar <= p_l){
            qL = 1.;
        } else {
            qL = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));
        }

        if (pStar <= p_r){
            qR = 1.;
        } else {
            qR = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));
        }

        real aL = v_l - qL*cs_l;
        real aR = v_r + qR*cs_r;

        real aStar = ( (p_r - p_l + rho_l*v_l*(aL - v_l) - rho_r*v_r*(aR - v_r))/
                        (rho_l*(aL - v_l) - rho_r*(aR - v_r) ) );

        return Eigenvals{aL, aR, aStar, pStar};
    }

};

// Adapt the cfl conditonal timestep
void Newtonian1D::adapt_dt(){
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real dx1, cs, cfl_dt;
        real v, pre, rho;
        luint shift_i;

        // Compute the minimum timestep given cfl
        #pragma omp for schedule(static) reduction(min:min_dt)
        for (luint ii = 0; ii < active_zones; ii++){
            shift_i = ii + idx_active;
            dx1      = coord_lattice.dx1[ii];

            rho = prims[shift_i].rho;
            v   = prims[shift_i].v;
            pre = prims[shift_i].p;

            cs = std::sqrt(gamma * pre/rho);
            cfl_dt = dx1/(std::max({std::abs(v + cs), std::abs(v - cs)}));

            min_dt = std::min(min_dt, cfl_dt);
    
        }
    }

    dt = cfl * min_dt;
};

void Newtonian1D::adapt_dt(Newtonian1D *dev, luint blockSize, luint tblock)
{   
    #if GPU_CODE
    {
        compute_dt<Newtonian1D, Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        dtWarpReduce<Newtonian1D, Primitive, 16><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        simbi::gpu::api::deviceSynch();
        simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
    #endif
};
//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state tensor for computation. Used for Higher Order Reconstruction
GPU_CALLABLE_MEMBER
Conserved Newtonian1D::prims2cons(const Primitive &prim)
{
    real energy = prim.p/(gamma - static_cast<real>(1.0)) + static_cast<real>(0.5) * prim.rho * prim.v * prim.v;

    return Conserved{prim.rho, prim.rho * prim.v, energy};
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
GPU_CALLABLE_MEMBER
Conserved Newtonian1D::prims2flux(const Primitive &prim)
{
    real energy = prim.p/(gamma - static_cast<real>(1.0)) + static_cast<real>(0.5) * prim.rho * prim.v * prim.v;

    return Conserved{
        prim.rho * prim.v,
        prim.rho * prim.v * prim.v + prim.p,
        (energy + prim.p)*prim.v

    };
};

GPU_CALLABLE_MEMBER
Conserved Newtonian1D::calc_hll_flux(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux)
{
    Eigenvals lambda;
    lambda = calc_eigenvals(left_prims, right_prims);
    real am = lambda.aL;
    real ap = lambda.aR;

    // Compute the HLL Flux component-wise
    return (left_flux * ap - right_flux * am + (right_state - left_state) * am * ap)  / (ap - am) ;

};

GPU_CALLABLE_MEMBER
Conserved Newtonian1D::calc_hllc_flux(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux)
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    real aL = lambda.aL; 
    real aR = lambda.aR; 
    real ap = std::max(static_cast<real>(0.0), aR);
    real am = std::min(static_cast<real>(0.0), aL);
    if (0.0 <= aL){
        return left_flux;
    } 
    else if (0.0 >= aR){
        return right_flux;
    }

    real aStar = lambda.aStar;
    real pStar = lambda.pStar;

    auto hll_flux = (left_flux * ap + right_flux * am - (right_state - left_state) * am * ap)  / (am + ap) ;

    auto hll_state = (right_state * aR - left_state * aL - right_flux + left_flux)/(aR - aL);
    
    if (- aL <= (aStar - aL)){
        real pressure = left_prims.p;
        real v        = left_prims.v;
        real rho      = left_state.rho;
        real m        = left_state.m;
        real energy   = left_state.e_dens;
        real cofac    = 1./(aL - aStar);

        real rhoStar = cofac * (aL - v)*rho;
        real mstar   = cofac * (m*(aL - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aL - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the luintermediate left flux
        return left_flux + (star_state - left_state) * aL;
    } else {
        real pressure = right_prims.p;
        real v        = right_prims.v;
        real rho      = right_state.rho;
        real m        = right_state.m;
        real energy   = right_state.e_dens;
        real cofac    = 1./(aR - aStar);

        real rhoStar = cofac * (aR - v)*rho;
        real mstar   = cofac * (m*(aR - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aR - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the luintermediate right flux
        return right_flux + (star_state - right_state) * aR;
    }
    
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

void Newtonian1D::advance(
    const luint radius, 
    const simbi::Geometry geometry,
    const ExecutionPolicy<> p,
    Newtonian1D *dev,  
    const luint sh_block_size,
    const simbi::MemSide user
)
{
    auto *self = (user == simbi::MemSide::Host) ? this : dev;
    #if GPU_CODE
    const real dt                   = this->dt;
    const real plm_theta            = this->plm_theta;
    const auto nx                   = this->nx;
    const real decay_constant       = this->decay_constant;
    const CLattice1D *coord_lattice = &(self->coord_lattice);
    #endif 

    const lint bx                   = (BuildPlatform == Platform::GPU) ? sh_block_size : this->nx;
    const lint pseudo_radius        = (first_order) ? 1 : 2;
    simbi::parallel_for(p, (luint)0, active_zones, [=] GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = &prims[0];
        #endif 

        Conserved u_l, u_r;
        Conserved f_l, f_r, frf, flf;
        Primitive prims_l, prims_r;
        real dx1, rmean, dV, sL, sR, pc;

        lint ia = ii + radius;
        lint txa = (BuildPlatform == Platform::GPU) ?  threadIdx.x + pseudo_radius : ia;
        #if GPU_CODE
            luint txl = BLOCK_SIZE;
            // Check if the active index exceeds the active zones
            // if it does, then this thread buffer will take on the
            // ghost index at the very end
            prim_buff[txa] = self->gpu_prims[ia];
            if (threadIdx.x < pseudo_radius)
            {
                if (ia + BLOCK_SIZE > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[txa - pseudo_radius] = self->gpu_prims[mod(ia - pseudo_radius, nx)];
                prim_buff[txa + txl]           = self->gpu_prims[(ia + txl) % nx];
            }
            simbi::gpu::api::synchronize();
        #endif

        if (self->first_order)
        {
            prims_l = prim_buff[(txa + 0) % bx];
            prims_r = prim_buff[(txa + 1) % bx];
            
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            // Calc HLL Flux at i+1/2 luinterface
            if (self->hllc)
            {
                frf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                frf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            // Set up the left and right state luinterfaces for i-1/2
            prims_l = prim_buff[mod(txa - 1, bx)];
            prims_r = prim_buff[(txa + 0) % bx];
            
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            // Calc HLL Flux at i-1/2 luinterface
            if (self->hllc)
            {
                flf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                flf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;

            left_most   = prim_buff[mod(txa - 2, bx)];
            left_mid    = prim_buff[mod(txa - 1, bx)];
            center      = prim_buff[(txa + 0)  % bx];
            right_mid   = prim_buff[(txa + 1)  % bx];
            right_most  = prim_buff[(txa + 2)  % bx];

            // Compute the reconstructed primitives at the i+1/2 luinterface

            // Reconstructed left primitives vector
            prims_l = center + minmod((center - left_mid) * plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center) * plm_theta) * static_cast<real>(0.5);
            prims_r = right_mid - minmod((right_mid - center)*plm_theta, (right_most - center)*static_cast<real>(0.5), (right_most- right_mid) * plm_theta) * static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            if (self->hllc)
            {
                frf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                frf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }

            // Do the same thing, but for the right side luinterface [i - 1/2]
            prims_l = left_mid + minmod((left_mid - left_most) * plm_theta, (center - left_most)*static_cast<real>(0.5), (center - left_mid)*plm_theta)*static_cast<real>(0.5);
            prims_r = center   - minmod((center - left_mid)*plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            u_l = self->prims2cons(prims_l);
            u_r = self->prims2cons(prims_r);
            f_l = self->prims2flux(prims_l);
            f_r = self->prims2flux(prims_r);

            if (self->hllc)
            {
                flf = self->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
            else
            {
                flf = self->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
            }
        }
        const auto step = (self->first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            { 
            #if GPU_CODE
                dx1 = coord_lattice->gpu_dx1[ii];
                self->gpu_cons[ia] -= ( (frf - flf)/ dx1) * dt * step;
            #else
                dx1 = self->coord_lattice.dx1[ii];
                cons[ia] -= ( (frf - flf)/ dx1) * dt * step;
            #endif
            
            break;  
            }
        case simbi::Geometry::SPHERICAL:
            {
            #if GPU_CODE
                pc    = prim_buff[txa].p;
                sL    = coord_lattice->gpu_face_areas[ii + 0];
                sR    = coord_lattice->gpu_face_areas[ii + 1];
                dV    = coord_lattice->gpu_dV[ii];
                rmean = coord_lattice->gpu_x1mean[ii];

                const auto geom_sources = Conserved{0.0, (real)2.0 * pc / rmean, 0.0};
                // const auto sources = Conserved{self->gpu_sourceD[ii], self->gpu_sourceS[ii],self->gpu_source0[ii]} * decay_constant;
                self->gpu_cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources) * dt * step;
            #else
                pc    = prim_buff[txa].p;
                sL    = self->coord_lattice.face_areas[ii + 0];
                sR    = self->coord_lattice.face_areas[ii + 1];
                dV    = self->coord_lattice.dV[ii];
                rmean = self->coord_lattice.x1mean[ii];

                const auto geom_sources = Conserved{0.0, (real)2.0 * pc / rmean, 0.0};
                // const auto sources = Conserved{sourceD[ii], sourceS[ii],source0[ii]} * decay_constant;
                cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources) * dt * step;
            #endif
            break;
            }
        case simbi::Geometry::CYLINDRICAL:
            // TODO: Implement Cylindrical coordinates at some point
            break;
        } // end switch
    }); // end parallel region
    
};


 std::vector<std::vector<real> > Newtonian1D::simulate1D(
    std::vector<std::vector<real>> &sources,
    real tstart,
    real tend,
    real init_dt,
    real plm_theta,
    real engine_duration,
    real chkpt_luinterval,
    std::string data_directory,
    std::string boundary_condition,
    bool first_order,
    bool linspace,
    bool hllc)
{
    this->periodic        = boundary_condition == "periodic";
    this->first_order     = first_order;
    this->plm_theta       = plm_theta;
    this->linspace        = linspace;
    this->sourceRho       = sources[0];
    this->sourceMom       = sources[1];
    this->sourceE         = sources[2];
    this->hllc            = hllc;
    this->engine_duration = engine_duration;
    this->t               = tstart;
    this->dt              = init_dt;
    // Define the swap vector for the luintegrated state
    this->nx = init_state[0].size();
    this->bc            = boundary_cond_map.at(boundary_condition);
    this->geometry      = geometry_map.at(coord_system);
    this->idx_active    = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones  = (periodic) ? nx: (first_order) ? nx - 2 : nx - 4;
    
    // TODO: invoke mesh motion later
    this->mesh_motion = false;
    if (hllc){
        this->sim_solver = simbi::SOLVER::HLLC;
    } else {
        this->sim_solver = simbi::SOLVER::HLLE;
    }

    n = 0;
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / chkpt_luinterval;
    real t_luinterval =
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_luinterval;
    DataWriteMembers setup;
    setup.x1max          = r[active_zones - 1];
    setup.x1min          = r[0];
    setup.xactive_zones  = active_zones;
    setup.nx             = nx;
    setup.linspace       = linspace;
    setup.coord_system   = coord_system;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;


    cons.resize(nx);
    prims.resize(nx);
    // Copy the state array luinto real & profile variables
    for (size_t ii = 0; ii < nx; ii++)
    {
        cons[ii] = Conserved{init_state[0][ii], init_state[1][ii], init_state[2][ii]};
    }
    cons_n = cons;
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
    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    hydro1d::PrimitiveData transfer_prims;

    // Tools for file string formatting
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // Some benchmarking tools 
    luint   nfold   = 0;
    luint   ncheck  = 0;
    real     zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<real> delta_t;

    // Copy the current SRHD instance over to the device
    Newtonian1D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(Newtonian1D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(Newtonian1D));
    simbi::dual::DualSpace1D<Primitive, Conserved, Newtonian1D> dualMem;
    dualMem.copyHostToDev(*this, device_self);

    const auto xblockdim          = nx > BLOCK_SIZE ? BLOCK_SIZE : nx;
    const luint radius            = (periodic) ? 0 : (first_order) ? 1 : 2;
    const luint pseudo_radius     = (first_order) ? 1 : 2;
    const luint shBlockSize       = BLOCK_SIZE + 2 * pseudo_radius;
    const luint shBlockBytes      = shBlockSize * sizeof(Primitive);
    const auto fullP              = simbi::ExecutionPolicy({nx}, {xblockdim}, shBlockBytes);
    const auto activeP            = simbi::ExecutionPolicy({active_zones}, {xblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, activeP.gridSize.x, xblockdim);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;

    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, true, bc);

            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                // std::cout << std::fixed << std::setprecision(3) << std::scientific;
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, nx/delta_t.count());
                nfold += 100;
            }

            /* Write to a File every tenth of a second */
            if (t >= t_luinterval)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<hydro1d::PrimitiveData, Primitive>(prims);
                writeToProd<hydro1d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_luinterval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_luinterval += chkpt_luinterval;
            }
            n++;
            
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, activeP.gridSize.x,xblockdim);
            } else {
                adapt_dt();
            }
        }
    } else {
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            cons2prim(fullP, self, memside);
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, false, bc);

            // Final Half Step
            cons2prim(fullP, self, memside);
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, false, bc);
            simbi::gpu::api::deviceSynch();
            t += dt; 
            

            if (n >= nfold){
                // simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += nx / delta_t.count();
                // std::cout << std::fixed << std::setprecision(3) << std::scientific;
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, nx/delta_t.count());
                nfold += 100;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_luinterval && t != INFINITY)
            {
                if constexpr(BuildPlatform == Platform::GPU) dualMem.copyDevToHost(device_self, *this);
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<hydro1d::PrimitiveData, Primitive>(prims);
                writeToProd<hydro1d::PrimitiveData, Primitive>(&transfer_prims, &prods);
                tnow = create_step_str(t_luinterval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_luinterval += chkpt_luinterval;
            }
            n++;
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, activeP.gridSize.x, xblockdim);
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

    transfer_prims = vec2struct<hydro1d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(3, std::vector<real>(nx));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v;
    solution[2] = transfer_prims.p;
    
    return solution;

 };
