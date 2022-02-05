/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "euler2D.hpp" 
#include <cmath>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include "util/parallel_for.hpp"
#include "util/dual.hpp"
#include "util/printb.hpp"
#include "helpers.hip.hpp"


using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;

// Default Constructor 
Newtonian2D::Newtonian2D () {}

// Overloaded Constructor
Newtonian2D::Newtonian2D(
    std::vector<std::vector<real> > init_state, 
    luint nx,
    luint ny,
    real gamma, 
    std::vector<real> x1, 
    std::vector<real> x2, 
    real cfl, 
    std::string coord_system = "cartesian")
:
    init_state(init_state),
    nx(nx),
    ny(ny),
    nzones(init_state[0].size()),
    gamma(gamma),
    x1(x1),
    x2(x2),
    cfl(cfl),
    coord_system(coord_system),
    inFailureState(false)
{}

// Destructor 
Newtonian2D::~Newtonian2D() {}


// Typedefs because I'm lazy
typedef hydro2d::Conserved Conserved;
typedef hydro2d::Primitive Primitive;
typedef hydro2d::Eigenvals Eigenvals;
//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

/**
 * Return a 1 + 2D matrix containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
void Newtonian2D::cons2prim()
{
    #pragma omp parallel
    {
        real rho, energy;
        real v1, v2, pre;
        for (luint jj = 0; jj < ny; jj++)
        {  
            #pragma omp for nowait schedule(static)
            for (luint ii = 0; ii < nx; ii++)
            {   
                luint gid = jj * nx + ii;
                rho     = cons[gid].rho;
                v1      = cons[gid].m1/rho;
                v2      = cons[gid].m2/rho;

                pre = (gamma - 1.0)*(cons[gid].e_dens - 0.5 * rho * (v1 * v1 + v2 * v2));
                prims [gid] = Primitive{rho, v1, v2, pre};
            }
        }
    }
};

void Newtonian2D::cons2prim(
    ExecutionPolicy<> p, 
    Newtonian2D *dev, 
    simbi::MemSide user
)
{
    #if GPU_CODE
    const auto gamma = this->gamma;
    #endif 
    auto* self = (user == simbi::MemSide::Dev) ? dev : this;
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (const luint gid) {
        #if GPU_CODE
        extern __shared__ Conserved cons_buff[];
        #else 
        auto* const cons_buff = &cons[0];
        #endif 
        const auto tid = (BuildPlatform == Platform::GPU) ? blockDim.x * threadIdx.y + threadIdx.x : gid;

        #if GPU_CODE
        cons_buff[tid] = self->gpu_cons[gid];
        simbi::gpu::api::synchronize();
        #endif 

        const real rho     = cons_buff[tid].rho;
        const real v1      = cons_buff[tid].m1/rho;
        const real v2      = cons_buff[tid].m2/rho;
        const real rho_chi = cons_buff[tid].chi;
        const real pre     = (gamma - (real)1.0)*(cons_buff[tid].e_dens - (real)0.5 * rho * (v1 * v1 + v2 * v2));

        #if GPU_CODE
        self->gpu_prims[gid] = Primitive{rho, v1, v2, pre, rho_chi / rho};
        #else
        prims[gid] = Primitive{rho, v1, v2, pre, rho_chi / rho};
        #endif

    });
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
GPU_CALLABLE_MEMBER
Eigenvals Newtonian2D::calc_eigenvals(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
{   
    if (hllc)
    {
        const real v_l   = left_prims.vcomponent(ehat);
        const real v_r   = right_prims.vcomponent(ehat);
        const real p_r   = right_prims.p;
        const real p_l   = left_prims.p;
        const real rho_l = left_prims.rho;
        const real rho_r = right_prims.rho;

        const real cs_r = std::sqrt(gamma * p_r/rho_r);
        const real cs_l = std::sqrt(gamma * p_l/rho_l);

        // Calculate the mean velocities of sound and fluid
        const real cbar   = 0.5*(cs_l + cs_r);
        const real rhoBar = 0.5*(rho_l + rho_r);
        const real z      = (gamma - 1.)/(2.0*gamma);
        const real num    = cs_l + cs_r - (gamma - 1.) * 0.5 *(v_r - v_r);
        const real denom  = cs_l/std::pow(p_l,z) + cs_r/std::pow(p_r, z);
        const real p_term = num/denom;
        const real pStar  = std::pow(p_term, (1./z));

        const real qL = 
            (pStar <= p_l) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));

        const real qR = 
            (pStar <= p_r) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));

        const real aL = v_r - qL*cs_l;
        const real aR = v_r + qR*cs_r;

        const real aStar = 
            ( (p_r - p_l + rho_l*v_l*(aL - v_l) - rho_r*v_r*(aR - v_r) )/ 
                (rho_l*(aL - v_l) - rho_r*(aR - v_r) ) );

        return Eigenvals(aL, aR, aStar, pStar);

    } else {
        const real v_l   = left_prims.vcomponent(ehat);
        const real v_r   = right_prims.vcomponent(ehat);
        const real p_r   = right_prims.p;
        const real p_l   = left_prims.p;
        const real rho_l = left_prims.rho;
        const real rho_r = right_prims.rho;
        const real cs_r  = std::sqrt(gamma * p_r/rho_r);
        const real cs_l  = std::sqrt(gamma * p_l/rho_l);

        const real aL = my_min(v_l - cs_l, v_r - cs_r);
        const real aR = my_max(v_l + cs_l, v_r + cs_r);

        return Eigenvals(aL, aR);

    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state tensor for computation. 
// It is being doing poluintwise in this case as opposed to over
// the entire array since we are in c++
GPU_CALLABLE_MEMBER
Conserved Newtonian2D::prims2cons(const Primitive &prims)
 {
    const real rho = prims.rho;
    const real vx  = prims.v1;
    const real vy  = prims.v2;
    const real pre = prims.p;
    const real et  = pre/(gamma - 1.0) + 0.5 * rho * (vx*vx + vy*vy);

    return Conserved{rho, rho*vx, rho*vy, et};
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the cfl conditonal timestep
void Newtonian2D::adapt_dt()
{
    real min_dt = INFINITY;
    #pragma omp parallel
    {
        real dx1, cs, dx2, rho, pressure, v1, v2, rmean;
        real cfl_dt;
        luint shift_i, shift_j, aid;

        for (luint jj = 0; jj < yphysical_grid; jj++)
        {
            dx2     = coord_lattice.dx2[jj];
            shift_j = jj + idx_active;
            #pragma omp for schedule(static) reduction(min:min_dt)
            for (luint ii = 0; ii < xphysical_grid; ii++)
            {
                shift_i  = ii + idx_active;
                dx1      = coord_lattice.dx1[ii];
                aid      = shift_j * nx + shift_i;
                rho      = prims[aid].rho;
                v1       = prims[aid].v1;
                v2       = prims[aid].v2;
                pressure = prims[aid].p;

                cs       = std::sqrt(gamma * pressure / rho );

                switch (geometry[coord_system])
                {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = 
                        my_min( dx1/(my_max(std::abs(v1 + cs), std::abs(v1 - cs))), dx2/(my_max(std::abs(v2 + cs), std::abs(v2 - cs))) );

                    break;
                
                case simbi::Geometry::SPHERICAL:
                    rmean = coord_lattice.x1mean[ii];
                    cfl_dt = 
                        my_min( dx1/(my_max(std::abs(v1 + cs), std::abs(v1 - cs))), rmean*dx2/(my_max(std::abs(v2 + cs), std::abs(v2 - cs))) );

                    break;
                }

                min_dt = my_min(min_dt, cfl_dt);
            } // end ii
        } // end jj
    }// end parallel region
    dt = cfl * min_dt;
};

void Newtonian2D::adapt_dt(Newtonian2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, luint bytes)
{
    #if GPU_CODE
    {
        luint psize = p.blockSize.x*p.blockSize.y;
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            compute_dt<Newtonian2D, Primitive><<<p.gridSize,p.blockSize, bytes>>>
            (dev, geometry, psize, dx1, dx2);
            dtWarpReduce<Newtonian2D, Primitive, 128><<<p.gridSize,p.blockSize,bytes>>>
            (dev);
            break;
        
        case simbi::Geometry::SPHERICAL:
            compute_dt<Newtonian2D, Primitive><<<p.gridSize,p.blockSize, bytes>>>
            (dev, geometry, psize, dlogx1, dx2, x1min, x1max, x2min, x2max);
            dtWarpReduce<Newtonian2D, Primitive, 128><<<p.gridSize,p.blockSize,bytes>>>
            (dev);
            break;
        }
        
        simbi::gpu::api::deviceSynch();
        simbi::gpu::api::copyDevToHost(&dt, &(dev->dt),  sizeof(real));
    }
    #endif
}
//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 2D Flux array (4,1). Either return F or G depending on directional flag
GPU_CALLABLE_MEMBER
Conserved Newtonian2D::prims2flux(const Primitive &prims, const luint ehat)
{
    const auto vn  = prims.vcomponent(ehat);
    const auto rho = prims.rho;
    const auto vx  = prims.v1;
    const auto vy  = prims.v2;
    const auto pre = prims.p;
    const auto et  = pre/(gamma - 1.0) + 0.5 * rho * (vx*vx + vy*vy);
    
    const auto dens  = rho*vn;
    const auto momx  = rho*vx*vn + pre * kronecker(1, ehat);
    const auto momy  = rho*vy*vn + pre * kronecker(2, ehat);
    const auto edens = (et + pre)*vn;

    return Conserved{dens, momx, momy, edens};

};

GPU_CALLABLE_MEMBER
Conserved Newtonian2D::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
                                        
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);
    real am = my_min((real)0.0, lambda.aL);
    real ap = my_max((real)0.0, lambda.aR);
    
    // Compute the HLL Flux 
    return  ( left_flux * ap - right_flux * am 
                + (right_state - left_state ) * am * ap )  /
                    (ap - am);
};

GPU_CALLABLE_MEMBER
Conserved Newtonian2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const luint ehat)
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims, ehat);

    const real aL    = lambda.aL;
    const real aR    = lambda.aR;
    const real aStar = lambda.aStar;
    const real pStar = lambda.pStar;

    // Quick checks before moving on with rest of computation
    if (0.0 <= aL){
        return left_flux;
    } else if (0.0 >= aR){
        return right_flux;
    }

    if (-aL <= (aStar - aL) ){
        const auto pre      = left_prims.p;
        const auto v1       = left_prims.v1;
        const auto v2       = left_prims.v2;
        const auto rho      = left_prims.rho;
        const auto m1       = left_state.m1;
        const auto m2       = left_state.m2;
        const auto energy   = left_state.e_dens;
        const auto cofac    = 1./(aL - aStar);

        switch (ehat)
        {
        case 1:
            {
                const auto rhoStar = cofac * (aL - v1) * rho;
                const auto m1star  = cofac * (m1*(aL - v1) - pre + pStar);
                const auto m2star  = cofac * (aL - v1) * m2;
                const auto eStar   = cofac * (energy*(aL - v1) + pStar*aStar - pre*v1);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return left_flux + (starstate - left_state)*aL;
            }
        
        case 2:
                const auto rhoStar = cofac * (aL - v2) * rho;
                const auto m1star  = cofac * (aL - v2) * m1; 
                const auto m2star  = cofac * (m2 * (aL - v2) - pre + pStar);
                const auto eStar   = cofac * (energy*(aL - v2) + pStar*aStar - pre*v2);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return left_flux + (starstate - left_state)*aL;
        }

    } else {
        const auto pre      = right_prims.p;
        const auto v1       = right_prims.v1;
        const auto v2       = right_prims.v2;
        const auto rho      = right_prims.rho;
        const auto m1       = right_state.m1;
        const auto m2       = right_state.m2;
        const auto energy   = right_state.e_dens;
        const auto cofac    = 1./(aR - aStar);

        switch (ehat)
        {
        case 1:
            {
                const auto rhoStar = cofac * (aR - v1) * rho;
                const auto m1star  = cofac * (m1*(aR - v1) - pre + pStar);
                const auto m2star  = cofac * (aR - v1) * m2;
                const auto eStar   = cofac * (energy*(aR - v1) + pStar*aStar - pre*v1);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return right_flux + (starstate - right_state)*aR;
            }
        
        case 2:
                const auto rhoStar = cofac * (aR - v2) * rho;
                const auto m1star  = cofac * (aR - v2) * m1; 
                const auto m2star  = cofac * (m2 * (aR - v2) - pre + pStar);
                const auto eStar   = cofac * (energy*(aR - v2) + pStar*aStar - pre*v2);

                const auto starstate = Conserved{rhoStar, m1star, m2star, eStar};

                return right_flux + (starstate - right_state)*aR;
        }


    }
    
};

//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

void Newtonian2D::advance(
    Newtonian2D *dev, 
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
    const luint extent                = (BuildPlatform == Platform::GPU) ? 
                                            p.blockSize.x * p.blockSize.y * p.gridSize.x * p.gridSize.y : active_zones;
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
    const real dx2                 = this->dx2;
    const real dlogx1              = this->dlogx1;
    const real dx1                 = this->dx1;
    const real imax                = this->xphysical_grid - 1;
    const real jmax                = this->yphysical_grid - 1;
    const bool rho_all_zeros       = this->rho_all_zeros;
    const bool m1_all_zeros        = this->m1_all_zeros;
    const bool m2_all_zeros        = this->m2_all_zeros;
    const bool e_all_zeros         = this->e_all_zeros;
    const real x1min                = this->x1min;
    const real x1max                = this->x1max;
    const real x2min                = this->x2min;
    const real x2max                = this->x2max;
    const real pow_dlogr           = pow(10, dlogx1);
    const auto nzones              = nx * ny;
    #endif

    // const CLattice2D *coord_lattice = &(self->coord_lattice);
    const luint nbs = (BuildPlatform == Platform::GPU) ? bx * by : nzones;

    // if on NVidia GPU, do column major striding, row-major otherwise
    const luint sx           = (col_maj) ? 1  : bx;
    const luint sy           = (col_maj) ? by :  1;
    const auto pseudo_radius = (first_order) ? 1 : 2;
    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = &prims[0];
        #endif 

        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        #if GPU_CODE 
        if ((ii >= xpg) || (jj >= ypg)) return;
        #endif

        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const luint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + pseudo_radius : ia;
        const luint tya = (BuildPlatform == Platform::GPU) ? ty + pseudo_radius : ja;

        Conserved ux_l, ux_r, uy_l, uy_r;
        Conserved f_l, f_r, g_l, g_r, frf, flf, grf, glf;
        Primitive xprims_l, xprims_r, yprims_l, yprims_r;

        const luint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;

            // Load Shared memory luinto buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = self->gpu_prims[aid];
            if (ty < pseudo_radius)
            {
                if (ja + yextent > ny - 1) tyl = ny - radius - ja + ty;
                prim_buff[(tya - pseudo_radius) * sx + txa] = self->gpu_prims[((ja - pseudo_radius) * nx + ia)];
                prim_buff[(tya + tyl   ) * sx + txa]        = self->gpu_prims[((ja + tyl   ) * nx + ia)]; 
            
            }
            if (tx < pseudo_radius)
            {   
                if (ia + xextent > nx - 1) txl = nx - radius - ia + tx;
                prim_buff[tya * sx + txa - pseudo_radius] =  self->gpu_prims[(ja * nx + ia - pseudo_radius)];
                prim_buff[tya * sx + txa +    txl]        =  self->gpu_prims[(ja * nx + ia + txl)]; 
            }
            
            simbi::gpu::api::synchronize();
        #endif

        if (first_order)
        {
            xprims_l = prim_buff[((txa + 0) * sy + (tya + 0) * sx) % nbs];
            xprims_r = prim_buff[((txa + 1) * sy + (tya + 0) * sx) % nbs];
            //j+1/2
            yprims_l = prim_buff[((txa + 0) * sy + (tya + 0) * sx) % nbs];
            yprims_r = prim_buff[((txa + 0) * sy + (tya + 1) * sx) % nbs];
            
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

            // Calc HLL Flux at i+1/2 luinterface
            if (hllc) {
                frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            } else {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            // Set up the left and right state luinterfaces for i-1/2
            xprims_l = prim_buff[( (txa - 1) * sy + (tya + 0) * sx ) % nbs];
            xprims_r = prim_buff[( (txa - 0) * sy + (tya + 0) * sx ) % nbs];
            //j+1/2
            yprims_l = prim_buff[( (txa - 0) * sy + (tya - 1) * sx ) % nbs]; 
            yprims_r = prim_buff[( (txa + 0) * sy + (tya - 0) * sx ) % nbs]; 

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

            // Calc HLL Flux at i-1/2 luinterface
            if (hllc)
            {
                flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

            } else {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }   
        }
        else 
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            // Coordinate X
            xleft_most  = prim_buff[((txa - 2) * sy + tya * sx)];
            xleft_mid   = prim_buff[((txa - 1) * sy + tya * sx)];
            center      = prim_buff[((txa + 0) * sy + tya * sx)];
            xright_mid  = prim_buff[((txa + 1) * sy + tya * sx)];
            xright_most = prim_buff[((txa + 2) * sy + tya * sx)];

            // Coordinate Y
            yleft_most  = prim_buff[(txa * sy + (tya - 2) * sx)];
            yleft_mid   = prim_buff[(txa * sy + (tya - 1) * sx)];
            yright_mid  = prim_buff[(txa * sy + (tya + 1) * sx)];
            yright_most = prim_buff[(txa * sy + (tya + 2) * sx)];

            // Reconstructed left X Primitive vector at the i+1/2 luinterface
            xprims_l = center     + minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*(real)0.5, (xright_mid - center) * plm_theta) * (real)0.5; 
            xprims_r = xright_mid - minmod((xright_mid - center) * plm_theta, (xright_most - center) * (real)0.5, (xright_most - xright_mid)*plm_theta) * (real)0.5;
            yprims_l = center     + minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*(real)0.5, (yright_mid - center) * plm_theta) * (real)0.5;  
            yprims_r = yright_mid - minmod((yright_mid - center) * plm_theta, (yright_most - center) * (real)0.5, (yright_most - yright_mid)*plm_theta) * (real)0.5;


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
                frf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
            else
            {
                frf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                grf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }

            // Do the same thing, but for the left side luinterface [i - 1/2]
            xprims_l = xleft_mid + minmod((xleft_mid - xleft_most) * plm_theta, (center - xleft_most) * (real)0.5, (center - xleft_mid)*plm_theta) * (real)0.5;
            xprims_r = center    - minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*(real)0.5, (xright_mid - center)*plm_theta)*(real)0.5;
            yprims_l = yleft_mid + minmod((yleft_mid - yleft_most) * plm_theta, (center - yleft_most) * (real)0.5, (center - yleft_mid)*plm_theta) * (real)0.5;
            yprims_r = center    - minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*(real)0.5, (yright_mid - center)*plm_theta)*(real)0.5;

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
                flf = self->calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
            else
            {
                flf = self->calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                glf = self->calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
            }
        }

        //Advance depending on geometry
        luint real_loc = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    #if GPU_CODE
                        const real rho_source  = (rho_all_zeros)   ? (real)0.0 : self->gpu_sourceRho[real_loc];
                        const real m1_source = (m1_all_zeros)  ? (real)0.0 : self->gpu_sourceM1[real_loc];
                        const real m2_source = (m2_all_zeros)  ? (real)0.0 : self->gpu_sourceM2[real_loc];
                        const real e_source  = (e_all_zeros)   ? (real)0.0 : self->gpu_sourceE[real_loc];
                        const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
                        self->gpu_cons[aid]   -= ( (frf - flf) / dx1 + (grf - glf)/ dx2 - source_terms) * (real)0.5 * dt;
                    #else
                        const real rho_source = (rho_all_zeros)   ? (real)0.0 : sourceRho[real_loc];
                        const real m1_source  = (m1_all_zeros)  ? (real)0.0   : sourceM1[real_loc];
                        const real m2_source  = (m2_all_zeros)  ? (real)0.0   : sourceM2[real_loc];
                        const real e_source   = (e_all_zeros)   ? (real)0.0   : sourceE[real_loc];
                        const real dx1 = self->coord_lattice.dx1[ii];
                        const real dx2  = self->coord_lattice.dx2[jj];
                        const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
                        cons[aid] -= ( (frf - flf) / dx1 + (grf - glf)/dx2 - source_terms) * (real)0.5 * dt;
                    #endif
                

                break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                #if GPU_CODE
                const real rl           = (ii > 0 ) ? x1min * pow(10, (ii -(real) 0.5) * dlogx1) :  x1min;
                const real rr           = (ii < xpg - 1) ? rl * pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                const real tl           = (jj > 0 ) ? x2min + (jj - (real)0.5) * dx2 :  x2min;
                const real tr           = (jj < ypg - 1) ? tl + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                const real rmean        = (real)0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real s1R          = rr * rr; 
                const real s1L          = rl * rl; 
                const real s2R          = std::sin(tr);
                const real s2L          = std::sin(tl);
                const real thmean       = (real)0.5 * (tl + tr);
                const real sint         = std::sin(thmean);
                const real dV1          = rmean * rmean * (rr - rl);             
                const real dV2          = rmean * sint * (tr - tl); 
                const real cot          = std::cos(thmean) / sint;

                const real rho_source  = (rho_all_zeros) ? (real)0.0 : self->gpu_sourceRho[real_loc];
                const real m1_source   = (m1_all_zeros)  ? (real)0.0 : self->gpu_sourceM1[real_loc];
                const real m2_source   = (m2_all_zeros)  ? (real)0.0 : self->gpu_sourceM2[real_loc];
                const real e_source    = (e_all_zeros)   ? (real)0.0 : self->gpu_sourceE[real_loc];
                #else
                const real s1R   = coord_lattice.x1_face_areas[ii + 1];
                const real s1L   = coord_lattice.x1_face_areas[ii + 0];
                const real s2R   = coord_lattice.x2_face_areas[jj + 1];
                const real s2L   = coord_lattice.x2_face_areas[jj + 0];
                const real rmean = coord_lattice.x1mean[ii];
                const real dV1   = coord_lattice.dV1[ii];
                const real dV2   = rmean * coord_lattice.dV2[jj];
                const real cot   = coord_lattice.cot[jj];

                const real rho_source  = (rho_all_zeros) ? (real)0.0 : sourceRho[real_loc];
                const real m1_source   = (m1_all_zeros)  ? (real)0.0 : sourceM1[real_loc];
                const real m2_source   = (m2_all_zeros)  ? (real)0.0 : sourceM2[real_loc];
                const real e_source    = (e_all_zeros)   ? (real)0.0 : sourceE[real_loc];
                #endif

                // Grab central primitives
                const real rhoc = prim_buff[txa * sy + tya * sx].rho;
                const real uc   = prim_buff[txa * sy + tya * sx].v1;
                const real vc   = prim_buff[txa * sy + tya * sx].v2;
                const real pc   = prim_buff[txa * sy + tya * sx].p;

                const Conserved geom_source  = {(real)0.0, (rhoc * vc * vc + (real)2.0 * pc) / rmean, - (rhoc  * uc * vc - pc * cot) / rmean , (real)0.0};
                const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
                const auto step = (first_order) ? (real)1.0 : (real)0.5;

                #if GPU_CODE 
                    self->gpu_cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 - geom_source - source_terms) * dt * step;
                #else
                    cons[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 - geom_source - source_terms) * dt * step;
                #endif
                
                break;
                }
        } // end switch

    });
}



//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
std::vector<std::vector<real> > Newtonian2D::simulate2D(
    const std::vector<std::vector<real>> sources,
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

    this->first_order    = first_order;
    this->periodic       = periodic;
    this->hllc           = hllc;
    this->linspace       = linspace;
    this->plm_theta      = plm_theta;
    this->dt             = init_dt;
    this->xphysical_grid = (periodic) ? nx : (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid = (periodic) ? ny : (first_order) ? ny - 2 : ny - 4;
    this->idx_active     = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones   = xphysical_grid * yphysical_grid;

    //--------Config the System Enums
    config_system();
    // sim_geom = geometry[coord_system];

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

    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.x1max     = x1[xphysical_grid - 1];
    setup.x1min     = x1[0];
    setup.x2max     = x2[yphysical_grid - 1];
    setup.x2min     = x2[0];
    setup.nx       = nx;
    setup.ny       = ny;
    setup.linspace = linspace;
    setup.ad_gamma = gamma;

    cons.resize(nzones);
    prims.resize(nzones);
    // Define the source terms
    sourceRho = sources[0];
    sourceM1  = sources[1];
    sourceM2  = sources[2];
    sourceE   = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < init_state[0].size(); i++)
    {
        auto rho      = init_state[0][i];
        auto m1       = init_state[1][i];
        auto m2       = init_state[2][i];
        auto e        = init_state[3][i];
        auto rho_chi  = init_state[4][i];
        cons[i]    = Conserved(rho, m1, m2, e, rho_chi);
    }

    // deallocate initial state vector
    std::vector<int> init_state;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1 / (1 + std::exp((real)10.0 * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;
    
    // Copy the current SRHD instance over to the device
    // if compiling for CPU, these functions do nothing
    Newtonian2D *device_self;
    simbi::gpu::api::gpuMalloc(&device_self, sizeof(Newtonian2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(Newtonian2D));
    simbi::dual::DualSpace2D<Primitive, Conserved, Newtonian2D> dualMem;
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

        rho_all_zeros  = std::all_of(sourceRho.begin(), sourceRho.end(),   [](real i) {return i == 0;});
        m1_all_zeros   = std::all_of(sourceM1.begin(),  sourceM1.end(),  [](real i) {return i == 0;});
        m2_all_zeros   = std::all_of(sourceM1.begin(),  sourceM2.end(),  [](real i) {return i == 0;});
        e_all_zeros    = std::all_of(sourceE.begin(),  sourceE.end(), [](real i) {return i == 0;});
    }
    
    // Some variables to handle file automatic file string
    // formatting 
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag;

    // // Setup the system
    const luint xblockdim       = xphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : xphysical_grid;
    const luint yblockdim       = yphysical_grid > BLOCK_SIZE2D ? BLOCK_SIZE2D : yphysical_grid;
    const luint radius          = (periodic) ? 0 : (first_order) ? 1 : 2;
    const luint pseudo_radius   = (first_order) ? 1 : 2;
    const luint bx              = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * pseudo_radius: nx;
    const luint by              = (BuildPlatform == Platform::GPU) ? yblockdim + 2 * pseudo_radius: ny;
    const luint shBlockSpace    = bx * by;
    const luint shBlockBytes    = shBlockSpace * sizeof(Primitive);
    const auto fullP            = simbi::ExecutionPolicy({nx, ny}, {xblockdim, yblockdim}, shBlockBytes);
    const auto activeP          = simbi::ExecutionPolicy({xphysical_grid, yphysical_grid}, {xblockdim, yblockdim}, shBlockBytes);

    if (t == 0)
    {
        if constexpr(BuildPlatform == Platform::GPU)
        {
            if (!periodic) config_ghosts2DGPU(fullP, device_self, nx, ny, first_order);
        } else {
            if (!periodic) config_ghosts2DGPU(fullP, this, nx, ny, first_order);
        }
    }
    const auto dtShBytes = xblockdim * yblockdim * sizeof(Primitive) + xblockdim * yblockdim * sizeof(real);
    if constexpr(BuildPlatform == Platform::GPU)
    {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, geometry[coord_system], activeP, dtShBytes);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    // Some benchmarking tools 
    luint      n   = 0;
    luint  nfold   = 0;
    luint  ncheck  = 0;
    real    zu_avg = 0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<real> delta_t;

    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;

    // Simulate :)
    if (first_order)
    {  
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            advance(self, activeP, bx, by, radius, geometry[coord_system], memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts2DGPU(fullP, self, nx, ny, true);
            t += dt; 
            
            if (n >= nfold){
                simbi::gpu::api::deviceSynch();
                ncheck += 1;
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones / delta_t.count();
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            // Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry[coord_system], activeP, dtShBytes);
            } else {
                adapt_dt();
            }
            decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (t - engine_duration)));
        }
    } else {
        while (t < tend && !inFailureState)
        {
            t1 = high_resolution_clock::now();
            // First Half Step
            advance(self, activeP, bx, by, radius, geometry[coord_system], memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts2DGPU(fullP, self, nx, ny, false);

            // Final Half Step
            advance(self, activeP, bx, by, radius, geometry[coord_system], memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts2DGPU(fullP, self, nx, ny, false);

            t += dt; 

            if (n >= nfold){
                ncheck += 1;
                simbi::gpu::api::deviceSynch();
                t2 = high_resolution_clock::now();
                delta_t = t2 - t1;
                zu_avg += total_zones/ delta_t.count();
                writefl("\r Iteration: {} \t dt: {} \t Time: {} \t Zones/sec: {}", n, dt, t, total_zones/delta_t.count());
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
                tnow     = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t  = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            n++;

            // Update decay constant
            decay_const = (real)1.0 / ((real)1.0 + exp((real)10.0 * (t - engine_duration)));

            simbi::gpu::api::copyDevToHost(&inFailureState, &(device_self->inFailureState),  sizeof(bool));
            //Adapt the timestep
            if constexpr(BuildPlatform == Platform::GPU)
            {
                adapt_dt(device_self, geometry[coord_system], activeP, dtShBytes);
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

    transfer_prims = vec2struct<sr2d::PrimitiveData, Primitive>(prims);

    std::vector<std::vector<real>> solution(5, std::vector<real>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;
    solution[4] = transfer_prims.chi;
 

    return solution;

 };