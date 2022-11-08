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
#include "util/parallel_for.hpp"
#include "util/exec_policy.hpp"
#include "util/device_api.hpp"
#include "util/printb.hpp"
#include "util/logger.hpp"
#include "common/helpers.hip.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;



// Typedefs because I'm lazy
using Conserved = hydro1d::Conserved;
using Primitive = hydro1d::Primitive;
using Eigenvals = hydro1d::Eigenvals;
constexpr auto write2file = helpers::write_to_file<hydro1d::PrimitiveSOA, 1, Newtonian1D>;

// Overloaded Constructor
Newtonian1D::Newtonian1D(
    std::vector< std::vector<real> > state, 
    real gamma, 
    real cfl, 
    std::vector<real> x1,
    std::string coord_system = "cartesian") 
    :
    HydroBase(
        state,
        gamma,
        cfl,
        x1,
        coord_system
    )
    
{
}

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
    auto* const conserved_buff = cons.data();
    auto* const primitive_buff = prims.data();
     simbi::parallel_for(p, (luint)0, nx, [=] GPU_LAMBDA (luint ii){ 
        real rho = conserved_buff[ii].rho;
        real v   = conserved_buff[ii].m / rho;
        real pre = (self->gamma - static_cast<real>(1.0))*(conserved_buff[ii].e_dens - static_cast<real>(0.5) * rho * v * v);
        primitive_buff[ii]  = Primitive{rho, v, pre};
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
    case Solver::HLLE:
        {
        const real aR = helpers::my_max(helpers::my_max(vL + csL, vR+ csR), static_cast<real>(0.0)); 
        const real aL = helpers::my_min(helpers::my_min(vL - csL, vR- csR), static_cast<real>(0.0));
        return Eigenvals{aL, aR};
        }
    case Solver::HLLC:
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
        compute_dt<Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(dev, prims.data(), dt_min.data());
        deviceReduceKernel<1><<<blockSize, BLOCK_SIZE>>>(dev, dt_min.data(), active_zones);
        deviceReduceKernel<1><<<1,1024>>>(dev, dt_min.data(), blockSize);
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
    const auto pseudo_radius        = this->pseudo_radius;
    #endif 

    const lint bx = (BuildPlatform == Platform::GPU) ? sh_block_size : this->nx;
    auto* const prim_data   = prims.data();
    auto* const cons_data   = cons.data();
    auto* const dens_source = sourceRho.data();
    auto* const mom_source  = sourceMom.data();
    auto* const erg_source  = sourceE.data();
    simbi::parallel_for(p, (luint)0, active_zones, [=] GPU_LAMBDA (luint ii) {
        #if GPU_CODE
        extern __shared__ Primitive prim_buff[];
        #else 
        auto* const prim_buff = prim_data;
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
            prim_buff[txa] = prim_data[ia];
            if (threadIdx.x < pseudo_radius)
            {
                if (ia + BLOCK_SIZE > nx - 1) txl = nx - radius - ia + threadIdx.x;
                prim_buff[txa - pseudo_radius] = prim_data[helpers::mod(ia - pseudo_radius, nx)];
                prim_buff[txa + txl]           = prim_data[(ia + txl) % nx];
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
            if (self->hllc) {
                frf = self->calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            } else {
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
        const auto sources = Conserved{dens_source[ii], mom_source[ii], erg_source[ii]} * self->decay_constant;
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            { 
                cons_data[ia] -= ( (frf - flf)/ self->dx1) * dt * step;
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
                const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                cons_data[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
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
    int  chkpt_idx,
    std::string data_directory,
    std::string boundary_condition,
    bool first_order,
    bool linspace,
    bool hllc)
{
    anyDisplayProps();
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->init_chkpt_idx  = chkpt_idx;
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
    this->total_zones     = nx;
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->checkpoint_zones= active_zones;
    // TODO: invoke mesh motion later
    this->mesh_motion = false;
    if (hllc){
        this->sim_solver = simbi::Solver::HLLC;
    } else {
        this->sim_solver = simbi::Solver::HLLE;
    }

    n = 0;
    // Write some info about the setup for writeup later
    real round_place = 1 / this->chkpt_interval;
    this->t_interval =
        t == 0 ? 0
               : dlogt !=0 ? tstart
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + this->chkpt_interval;
    this->setup.x1max          = x1[active_zones - 1];
    this->setup.x1min          = x1[0];
    this->setup.xactive_zones  = active_zones;
    this->setup.nx             = nx;
    this->setup.linspace       = linspace;
    this->setup.ad_gamma       = gamma;
    this->setup.first_order    = first_order;
    this->setup.coord_system   = coord_system;
    this->setup.boundarycond   = boundary_condition;
    this->setup.regime         = "classical";
    this->setup.x1             = x1;


    dt_min.resize(active_zones);
    cons.resize(nx);
    prims.resize(nx);
    // Copy the state array luinto real & profile variables
    for (size_t ii = 0; ii < nx; ii++)
    {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
    }
    
    // Copy the current Newtonian1D instance over to the device
    Newtonian1D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(Newtonian1D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(Newtonian1D));
    cons.copyToGpu();
    prims.copyToGpu();
    dt_min.copyToGpu();
    sourceRho.copyToGpu();
    sourceMom.copyToGpu();
    sourceE.copyToGpu();

    const auto xblockdim     = nx > BLOCK_SIZE ? BLOCK_SIZE : nx;
    this->radius             = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->pseudo_radius      = (first_order) ? 1 : 2;
    const luint shBlockSize  = BLOCK_SIZE + 2 * pseudo_radius;
    const luint shBlockBytes = shBlockSize * sizeof(Primitive);
    const auto fullP         = simbi::ExecutionPolicy({nx}, {xblockdim}, shBlockBytes);
    const auto activeP       = simbi::ExecutionPolicy({active_zones}, {xblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, activeP.gridSize.x, xblockdim);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    // Save initial condition
    if (t == 0) {
        write2file(*this, setup, data_directory, t, t_interval, this->chkpt_interval, active_zones);
        t_interval += this->chkpt_interval;
    }

    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;

    while (t < tend & !inFailureState)
    {
        simbi::detail::with_logger(*this, [&](){
            advance(radius, geometry, activeP, self, shBlockSize, memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts1D_t(fullP, cons, nx, first_order, bc, (conserved_t*)nullptr);
        });

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(device_self, activeP.gridSize.x,xblockdim);
        } else {
            adapt_dt();
        }
    }

    if (detail::logger::ncheck > 0) {
         writeln("Average zone update/sec for:{:>5} iterations was {:>5.2e} zones/sec", detail::logger::n, detail::logger::zu_avg/ detail::logger::ncheck);
    }

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }
    
    return final_prims;

 };
