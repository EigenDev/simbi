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



// Typedefs because I'm lazy
using Conserved = hydro1d::Conserved;
using Primitive = hydro1d::Primitive;
using Eigenvals = hydro1d::Eigenvals;
using dualType  = dual::DualSpace1D<Primitive, Conserved, Newtonian1D>;
constexpr auto write2file = helpers::write_to_file<simbi::Newtonian1D, hydro1d::PrimitiveData, Primitive, dualType, 1>;
// Default Constructor 
Newtonian1D::Newtonian1D () {}

// Overloaded Constructor
Newtonian1D::Newtonian1D(
    std::vector< std::vector<real> > state, 
    real gamma, 
    real cfl, 
    std::vector<real> x1,
    std::string coord_system = "cartesian") 
    :
    state(state),
    gamma(gamma),
    cfl(cfl),
    x1(x1),
    coord_system(coord_system),
    inFailureState(false),
    nx(state[0].size())
    {

    }

// Destructor 
Newtonian1D::~Newtonian1D() {}

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
    const real rhoL    = left_prim.rho;
    const real vL      = left_prim.v;
    const real pL      = left_prim.p;
    const real rhoR   = right_prim.rho;
    const real vR     = right_prim.v;
    const real pR     = right_prim.p;

    const real csR = std::sqrt(gamma * pR/rhoR);
    const real csL = std::sqrt(gamma * pL/rhoL);

    switch (sim_solver)
    {
    case SOLVER::HLLE:
        {
        const real aR = helpers::my_max(helpers::my_max(vL + csL, vR+ csR), static_cast<real>(0.0)); 
        const real aL = helpers::my_min(helpers::my_min(vL - csL, vR- csR), static_cast<real>(0.0));
        return Eigenvals{aL, aR};
        }
    case SOLVER::HLLC:
        real cbar   = static_cast<real>(0.5)*(csL + csR);
        real rhoBar = static_cast<real>(0.5)*(rhoL + rhoR);
        real pStar  = static_cast<real>(0.5)*(pL + pR) + static_cast<real>(0.5)*(vL - vR)*cbar*rhoBar;

        // Steps to Compute HLLC as described in Toro et al. 2019
        real z      = (gamma - 1.)/(2.*gamma);
        real num    = csL + csR- ( gamma-1.)/2 *(vR- vL);
        real denom  = csL * std::pow(pL, -z) + csR * std::pow(pR, -z);
        real p_term = num/denom;
        real qL, qR;

        pStar = std::pow(p_term, (1./z));

        if (pStar <= pL){
            qL = 1.;
        } else {
            qL = std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pL - 1.));
        }

        if (pStar <= pR){
            qR = 1.;
        } else {
            qR = std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pR- 1.));
        }

        real aL = vL - qL*csL;
        real aR = vR + qR*csR;

        real aStar = ( (pR- pL + rhoL*vL*(aL - vL) - rhoR*vR*(aR - vR))/
                        (rhoL*(aL - vL) - rhoR*(aR - vR) ) );

        return Eigenvals{aL, aR, aStar, pStar};
    }

};

// Adapt the cfl conditonal timestep
void Newtonian1D::adapt_dt(){
    real min_dt = INFINITY;
    // #pragma omp parallel 
    {
        // Compute the minimum timestep given cfl
        #pragma omp parallel for schedule(static) reduction(min:min_dt)
        for (luint ii = 0; ii < active_zones; ii++){
            const auto shift_i  = ii + idx_active;
            const real rho      = prims[shift_i].rho;
            const real v        = prims[shift_i].v;
            const real pre      = prims[shift_i].p;
            const real cs       = std::sqrt(gamma * pre/rho);
            const real vPLus    = v + cs;
            const real vMinus   = v - cs;
            const real x1l      = get_xface(ii, geometry, 0);
            const real x1r      = get_xface(ii, geometry, 1);
            const real dx1      = x1r - x1l;
            const real vfaceL   = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
            const real vfaceR   = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
            const real cfl_dt   = dx1 / (helpers::my_max(std::abs(vPLus - vfaceR), std::abs(vMinus - vfaceL)));
            min_dt              = std::min(min_dt, cfl_dt);
        }
    }

    dt = cfl * min_dt;
};

void Newtonian1D::adapt_dt(Newtonian1D *dev, luint blockSize, luint tblock)
{   
    #if GPU_CODE
    {
        compute_dt<Newtonian1D, Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        deviceReduceKernel<Newtonian1D, 1><<<blockSize, BLOCK_SIZE>>>(dev, active_zones);
        deviceReduceKernel<Newtonian1D, 1><<<1,1024>>>(dev, blockSize);
        simbi::gpu::api::deviceSynch();
        this->dt = dev->dt;
        // dtWarpReduce<Newtonian1D, Primitive, 16><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev);
        // simbi::gpu::api::deviceSynch();
        // simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
    #endif
};
//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state array for computation. Used for Higher Order Reconstruction
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
    lambda  = calc_eigenvals(left_prims, right_prims);
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
    real ap = helpers::my_max(static_cast<real>(0.0), aR);
    real am = helpers::my_min(static_cast<real>(0.0), aL);
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
    #endif 

    const lint bx                   = (BuildPlatform == Platform::GPU) ? sh_block_size : this->nx;
    const lint pseudo_radius        = (first_order) ? 1 : 2;
    simbi::parallel_for(p, (luint)0, active_zones, [=] GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = &prims[0];
        #endif 

        Conserved uL, uR;
        Conserved fL, fR, frf, flf;
        Primitive primsL, primsR;

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
                prim_buff[txa - pseudo_radius] = self->gpu_prims[helpers::mod(ia - pseudo_radius, nx)];
                prim_buff[txa + txl]           = self->gpu_prims[(ia + txl) % nx];
            }
            simbi::gpu::api::synchronize();
        #endif

        const real x1l    = self->get_xface(ii, geometry, 0);
        const real x1r    = self->get_xface(ii, geometry, 1);
        const real vfaceL = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
        const real vfaceR = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
        if (self->first_order)
        {
            primsL = prim_buff[(txa + 0) % bx];
            primsR = prim_buff[(txa + 1) % bx];
            
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            // Calc HLL Flux at i+1/2 luinterface
            if (self->hllc)
            {
                frf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                frf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }

            // Set up the left and right state luinterfaces for i-1/2
            primsL = prim_buff[helpers::mod(txa - 1, bx)];
            primsR = prim_buff[(txa + 0) % bx];
            
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            // Calc HLL Flux at i-1/2 luinterface
            if (self->hllc)
            {
                flf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                flf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;

            left_most   = prim_buff[helpers::mod(txa - 2, bx)];
            left_mid    = prim_buff[helpers::mod(txa - 1, bx)];
            center      = prim_buff[(txa + 0)  % bx];
            right_mid   = prim_buff[(txa + 1)  % bx];
            right_most  = prim_buff[(txa + 2)  % bx];

            // Compute the reconstructed primitives at the i+1/2 luinterface

            // Reconstructed left primitives vector
            primsL = center    + helpers::minmod((center - left_mid) * plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center) * plm_theta) * static_cast<real>(0.5);
            primsR = right_mid - helpers::minmod((right_mid - center)*plm_theta, (right_most - center)*static_cast<real>(0.5), (right_most- right_mid) * plm_theta) * static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            if (self->hllc)
            {
                frf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                frf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }

            // Do the same thing, but for the right side luinterface [i - 1/2]
            primsL = left_mid + helpers::minmod((left_mid - left_most) * plm_theta, (center - left_most)*static_cast<real>(0.5), (center - left_mid)*plm_theta)*static_cast<real>(0.5);
            primsR = center   - helpers::minmod((center - left_mid)*plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = self->prims2cons(primsL);
            uR = self->prims2cons(primsR);
            fL = self->prims2flux(primsL);
            fR = self->prims2flux(primsR);

            if (self->hllc)
            {
                flf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                flf = self->calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }
        }
        const auto step = (self->first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            { 
            #if GPU_CODE
                self->gpu_cons[ia] -= ( (frf - flf)/ self->dx1) * dt * step;
            #else
                cons[ia] -= ( (frf - flf)/ self->dx1) * dt * step;
            #endif
            break;  
            }
        case simbi::Geometry::SPHERICAL:
            {
                const real rlf    = x1l + vfaceL * step * dt; 
                const real rrf    = x1r + vfaceR * step * dt;
                const real rmean  = static_cast<real>(0.75) * (rrf * rrf * rrf * rrf - rlf * rlf * rlf * rlf) / (rrf * rrf * rrf - rlf * rlf * rlf);
                const real sR     = rrf * rrf; 
                const real sL     = rlf * rlf; 
                const real dV     = rmean * rmean * (rrf - rlf);    
                const real factor = (self->mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;

                #if GPU_CODE
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{self->gpu_sourceRho[ii], self->gpu_sourceMom[ii],self->gpu_sourceE[ii]} * decay_constant;
                    self->gpu_cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                #else 
                    const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                    const auto sources = Conserved{sourceRho[ii], sourceMom[ii], sourceE[ii]} * decay_constant;
                    cons[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                #endif 
                
                break;
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
    this->dlogt           = dlogt;
    // Define the swap vector for the integrated state
    this->bc              = helpers::boundary_cond_map.at(boundary_condition);
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = (periodic) ? nx: (first_order) ? nx - 2 : nx - 4;
    this->dlogx1          = std::log10(x1[active_zones - 1]/ x1[0]) / (active_zones - 1);
    this->dx1             = (x1[active_zones - 1] - x1[0]) / (active_zones - 1);
    this->x1min           = x1[0];
    this->x1max           = x1[active_zones - 1];
    this->xcell_spacing   = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->total_zones     = nx;
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
    real round_place = 1 / chkpt_interval;
    real t_interval =
        t == 0 ? floor(tstart * round_place + static_cast<real>(0.5)) / round_place
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;
    DataWriteMembers setup;
    setup.x1max          = x1[active_zones - 1];
    setup.x1min          = x1[0];
    setup.xactive_zones  = active_zones;
    setup.nx             = nx;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;



    cons.resize(nx);
    prims.resize(nx);
    // Copy the state array luinto real & profile variables
    for (size_t ii = 0; ii < nx; ii++)
    {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
    }
    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    hydro1d::PrimitiveData transfer_prims;

    // Some benchmarking tools 
    luint   nfold   = 0;
    luint   ncheck  = 0;
    real     zu_avg = 0;
    #if GPU_CODE
    anyGpuEvent_t t1, t2;
    anyGpuEventCreate(&t1);
    anyGpuEventCreate(&t2);
    float delta_t;
    #else 
    high_resolution_clock::time_point t1, t2;
    double delta_t;
    #endif

    // Copy the current SRHD instance over to the device
    Newtonian1D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(Newtonian1D));
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
            helpers::recordEvent(t1);
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, true, bc);
            helpers::recordEvent(t2);
            t += dt; 
            
             if (n >= nfold){
                anyGpuEventSynchronize(t2);
                helpers::recordDuration(delta_t, t1, t2);
                if (BuildPlatform == Platform::GPU) {
                    delta_t *= 1e-3;
                }
                ncheck += 1;
                zu_avg += nx / delta_t;
                 if constexpr(BuildPlatform == Platform::GPU) {
                    // Calculation derived from: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
                    constexpr real gtx_theoretical_bw = 1875e6 * (192.0 / 8.0) * 2 / 1e9;
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
                write2file(this, device_self, dualMem, setup, data_directory, t, t_interval, chkpt_interval, active_zones);
                if (dlogt != 0) {
                    t_interval *= std::pow(10, dlogt);
                } else {
                    t_interval += chkpt_interval;
                }
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
            helpers::recordEvent(t1);
            // First Half Step
            cons2prim(fullP, self, memside);
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, false, bc);

            // Final Half Step
            cons2prim(fullP, self, memside);
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            if (!periodic) config_ghosts1D(fullP, self, nx, false, bc);
            helpers::recordEvent(t2);
            t += dt; 
            

             if (n >= nfold){
                anyGpuEventSynchronize(t2);
                helpers::recordDuration(delta_t, t1, t2);
                if (BuildPlatform == Platform::GPU) {
                    delta_t *= 1e-3;
                }
                ncheck += 1;
                zu_avg += nx / delta_t;
                 if constexpr(BuildPlatform == Platform::GPU) {
                    // Calculation derived from: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
                    constexpr real gtx_theoretical_bw = 1875e6 * (192.0 / 8.0) * 2 / 1e9;
                    const real gtx_emperical_bw       = total_zones * (sizeof(Primitive) + sizeof(Conserved)) * (1.0 + 4.0 * radius) / (delta_t * 1e9);
                    writefl("\riteration:{:>06} dt:{:>08.2e} time:{:>08.2e} zones/sec:{:>08.2e} ebw(%):{:>04.2f}", n, dt, t, total_zones/delta_t, static_cast<real>(100.0) * gtx_emperical_bw / gtx_theoretical_bw);
                } else {
                    writefl("\riteration:{:>06}    dt: {:>08.2e}    time: {:>08.2e}    zones/sec: {:>08.2e}", n, dt, t, total_zones/delta_t);
                }
                nfold += 100;
            }
            
            /* Write to a File every tenth of a second */
            if (t >= t_interval && t != INFINITY)
            {
                write2file(this, device_self, dualMem, setup, data_directory, t, t_interval, chkpt_interval, active_zones);
                if (dlogt != 0) {
                    t_interval *= std::pow(10, dlogt);
                } else {
                    t_interval += chkpt_interval;
                }
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
    if (ncheck > 0) {
         writeln("Average zone update/sec for:{:>5} iterations was {:>5.2e} zones/sec", n, zu_avg/ncheck);
    }

    if constexpr(BuildPlatform == Platform::GPU)
    {
        dualMem.copyDevToHost(device_self, *this);
        simbi::gpu::api::gpuFree(device_self);
    }

    transfer_prims = helpers::vec2struct<hydro1d::PrimitiveData, Primitive>(prims);
    std::vector<std::vector<real>> solution(3, std::vector<real>(nx));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v;
    solution[2] = transfer_prims.p;
    
    return solution;

 };
