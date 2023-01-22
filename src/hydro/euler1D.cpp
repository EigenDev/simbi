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
void Newtonian1D::cons2prim(const ExecutionPolicy<> &p){
    auto* const conserved_buff = cons.data();
    auto* const primitive_buff = prims.data();
     simbi::parallel_for(p, (luint)0, nx, [=] GPU_LAMBDA (luint ii){ 
        real rho = conserved_buff[ii].rho;
        real v   = conserved_buff[ii].m / rho;
        real pre = (gamma - 1)*(conserved_buff[ii].e_dens - static_cast<real>(0.5) * rho * v * v);
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
        // real cbar   = static_cast<real>(0.5)*(csL + csR);
        // real rhoBar = static_cast<real>(0.5)*(rhoL + rhoR);
        // real pStar  = static_cast<real>(0.5)*(pL + pR) + static_cast<real>(0.5)*(vL - vR)*cbar*rhoBar;

        // Steps to Compute HLLC as described in Toro et al. 2019
        const real num    = csL + csR- ( gamma-1.)/2 *(vR- vL);
        const real denom  = csL * std::pow(pL, -hllc_z) + csR * std::pow(pR, -hllc_z);
        const real p_term = num/denom;
        const real pStar  = std::pow(p_term, (1./hllc_z));

        const real qL = 
            (pStar <= pL) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pL - 1.));

        const real qR = 
            (pStar <= pR) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pR- 1.));

        const real aL = vL - qL*csL;
        const real aR = vR + qR*csR;

        const real aStar = ( (pR- pL + rhoL*vL*(aL - vL) - rhoR*vR*(aR - vR))/
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

void Newtonian1D::adapt_dt(luint blockSize, luint tblock)
{   
    #if GPU_CODE
        compute_dt<Primitive><<<dim3(blockSize), dim3(BLOCK_SIZE)>>>(this, prims.data(), dt_min.data());
        deviceReduceWarpAtomicKernel<1><<<blockSize, BLOCK_SIZE>>>(this, dt_min.data(), active_zones);
        gpu::api::deviceSynch();
        // deviceReduceKernel<1><<<blockSize, BLOCK_SIZE>>>(this, dt_min.data(), active_zones);
        // deviceReduceKernel<1><<<1,1024>>>(this, dt_min.data(), blockSize);
    #endif
};
//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state array for computation. Used for Higher Order Reconstruction
GPU_CALLABLE_MEMBER
Conserved Newtonian1D::prims2cons(const Primitive &prim)
{
    const real rho = prim.rho;
    const real v   = prim.v;
    const real pre = prim.p;
    real energy    = pre / (gamma - 1) + static_cast<real>(0.5) * rho * v * v;
    return Conserved{rho, rho * v, energy};
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
GPU_CALLABLE_MEMBER
Conserved Newtonian1D::prims2flux(const Primitive &prim)
{
    const real rho = prim.rho;
    const real v   = prim.v;
    const real pre = prim.p;
    real energy    = pre / (gamma - 1) + static_cast<real>(0.5) * rho * v * v;

    return Conserved{
        rho * v,
        rho * v * v + pre,
        (energy + pre) * v
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
    const Eigenvals lambda  = calc_eigenvals(left_prims, right_prims);
    const real am = lambda.aL;
    const real ap = lambda.aR;

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
    const Eigenvals lambda = calc_eigenvals(left_prims, right_prims);
    const real aL    = lambda.aL; 
    const real aR    = lambda.aR; 
    const real aStar = lambda.aStar;
    const real pStar = lambda.pStar;
    if (0.0 <= aL){
        return left_flux;
    } 
    else if (0.0 >= aR){
        return right_flux;
    }
    
    const real ap = helpers::my_max(static_cast<real>(0.0), aR);
    const real am = helpers::my_min(static_cast<real>(0.0), aL);
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
    const ExecutionPolicy<> &p,
    const luint xstride)
{
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

        const real x1l    = get_xface(ii, geometry, 0);
        const real x1r    = get_xface(ii, geometry, 1);
        const real vfaceL = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
        const real vfaceR = 0.0; // (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;
        if (first_order)
        {
            primsL = prim_buff[(txa + 0) % xstride];
            primsR = prim_buff[(txa + 1) % xstride];
            
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            // Calc HLL Flux at i+1/2 interface
            if (hllc) {
                frf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            } else {
                frf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }

            // Set up the left and right state interfaces for i-1/2
            primsL = prim_buff[helpers::mod(txa - 1, xstride)];
            primsR = prim_buff[(txa + 0) % xstride];
            
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            // Calc HLL Flux at i-1/2 interface
            if (hllc)
            {
                flf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                flf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;

            left_most   = prim_buff[helpers::mod(txa - 2, xstride)];
            left_mid    = prim_buff[helpers::mod(txa - 1, xstride)];
            center      = prim_buff[(txa + 0)  % xstride];
            right_mid   = prim_buff[(txa + 1)  % xstride];
            right_most  = prim_buff[(txa + 2)  % xstride];

            // Compute the reconstructed primitives at the i+1/2 interface

            // Reconstructed left primitives vector
            primsL = center    + helpers::minmod((center - left_mid) * plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center) * plm_theta) * static_cast<real>(0.5);
            primsR = right_mid - helpers::minmod((right_mid - center)*plm_theta, (right_most - center)*static_cast<real>(0.5), (right_most- right_mid) * plm_theta) * static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            if (hllc)
            {
                frf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                frf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }

            // Do the same thing, but for the right side interface [i - 1/2]
            primsL = left_mid + helpers::minmod((left_mid - left_most) * plm_theta, (center - left_most)*static_cast<real>(0.5), (center - left_mid)*plm_theta)*static_cast<real>(0.5);
            primsR = center   - helpers::minmod((center - left_mid)*plm_theta, (right_mid - left_mid)*static_cast<real>(0.5), (right_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // primitives
            uL = prims2cons(primsL);
            uR = prims2cons(primsR);
            fL = prims2flux(primsL);
            fR = prims2flux(primsR);

            if (hllc)
            {
                flf = calc_hllc_flux(primsL, primsR, uL, uR, fL, fR);
            }
            else
            {
                flf = calc_hll_flux(primsL, primsR, uL, uR, fL, fR);
            }
        }
        const auto d_source = den_source_all_zeros    ? 0.0 : dens_source[ii];
        const auto m_source = mom1_source_all_zeros   ? 0.0 : mom_source[ii];
        const auto e_source = energy_source_all_zeros ? 0.0 : erg_source[ii];
        const auto sources = Conserved{d_source, m_source, e_source} * time_constant;
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            { 
                cons_data[ia] -= ( (frf - flf) * invdx1) * dt * step;
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
                const real factor = (mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;
                const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                cons_data[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                break;
            }
        case simbi::Geometry::CYLINDRICAL:
            {
                const real rlf    = x1l + vfaceL * step * dt; 
                const real rrf    = x1r + vfaceR * step * dt;
                const real rmean  = (2.0 / 3.0) * (rrf * rrf * rrf - rlf * rlf * rlf) / (rrf * rrf - rlf * rlf);
                const real sR     = rrf * rrf; 
                const real sL     = rlf * rlf; 
                const real dV     = rmean * rmean * (rrf - rlf);    
                const real factor = (mesh_motion) ? dV : 1;         
                const real pc     = prim_buff[txa].p;
                const auto geom_sources = Conserved{0.0, pc * (sR - sL) / dV, 0.0};
                cons_data[ia] -= ( (frf * sR - flf * sL) / dV - geom_sources - sources) * step * dt * factor;
                break;
            }
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
    std::vector<std::string> boundary_conditions,
    bool first_order,
    bool linspace,
    bool hllc,
    bool constant_sources,
    std::vector<std::vector<real>> boundary_sources)
{
    anyDisplayProps();
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->init_chkpt_idx  = chkpt_idx;
    this->periodic        = boundary_conditions[0] == "periodic";
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
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = (periodic) ? nx: (first_order) ? nx - 2 : nx - 4;
    this->dlogx1          = std::log10(x1[active_zones - 1]/ x1[0]) / (active_zones - 1);
    this->dx1             = (x1[active_zones - 1] - x1[0]) / (active_zones - 1);
    this->invdx1          = 1 / dx1;
    this->x1min           = x1[0];
    this->x1max           = x1[active_zones - 1];
    this->total_zones     = nx;
    this->x1cell_spacing  = (linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
    this->checkpoint_zones= active_zones;
    this->den_source_all_zeros    = std::all_of(sourceRho.begin(), sourceRho.end(), [](real i) {return i==0;});
    this->mom1_source_all_zeros   = std::all_of(sourceMom.begin(), sourceMom.end(), [](real i) {return i==0;});
    this->energy_source_all_zeros = std::all_of(sourceE.begin(), sourceE.end(), [](real i) {return i==0;});
    
    // TODO: invoke mesh motion later
    this->mesh_motion = false;
    if (hllc){
        this->sim_solver = simbi::Solver::HLLC;
    } else {
        this->sim_solver = simbi::Solver::HLLE;
    }

    inflow_zones.resize(2);
    for (size_t i = 0; i < 2; i++)
    {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = Conserved{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2]};
    }
    
    n = 0;
    // Write some info about the setup for writeup later
    real round_place = 1 / this->chkpt_interval;
    this->t_interval =
        t == 0 ? 0
               : dlogt !=0 ? tstart
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + this->chkpt_interval;
    setup.x1max          = x1[active_zones - 1];
    setup.x1min          = x1[0];
    setup.xactive_zones  = active_zones;
    setup.nx             = nx;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.regime         = "classical";
    setup.x1             = x1;
    setup.mesh_motion    = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions = 1;


    dt_min.resize(active_zones);
    cons.resize(nx);
    prims.resize(nx);
    // Copy the state array into real & profile variables
    for (size_t ii = 0; ii < nx; ii++) {
        cons[ii] = Conserved{state[0][ii], state[1][ii], state[2][ii]};
    }
    
    cons.copyToGpu();
    prims.copyToGpu();
    dt_min.copyToGpu();
    sourceRho.copyToGpu();
    sourceMom.copyToGpu();
    sourceE.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();

    const auto xblockdim     = nx > BLOCK_SIZE ? BLOCK_SIZE : nx;
    this->radius             = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->pseudo_radius      = (first_order) ? 1 : 2;
    this->step               = (first_order) ? 1 : static_cast<real>(0.5);
    const luint shBlockSize  = BLOCK_SIZE + 2 * pseudo_radius;
    const luint shBlockBytes = shBlockSize * sizeof(Primitive);
    const auto fullP         = simbi::ExecutionPolicy(nx, xblockdim);
    const auto activeP       = simbi::ExecutionPolicy(active_zones, xblockdim, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP);
        adapt_dt(activeP.gridSize.x, xblockdim);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }
    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);

    // Save initial condition
    if (t == 0) {
        write2file(*this, setup, data_directory, t, t_interval, this->chkpt_interval, active_zones);
        t_interval += this->chkpt_interval;
    }
    const auto xstride = (BuildPlatform == Platform::GPU) ? shBlockSize : nx;

    simbi::detail::logger::with_logger(*this, tend, [&](){
        advance(activeP, xstride);
        cons2prim(fullP);
        if (!periodic) {
            config_ghosts1D_t(fullP, cons, nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
        }   
        
        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP.gridSize.x,xblockdim);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
    });

    std::vector<std::vector<real>> final_prims(3, std::vector<real>(nx, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v;
        final_prims[2][ii] = prims[ii].p;
    }
    
    return final_prims;

 };
