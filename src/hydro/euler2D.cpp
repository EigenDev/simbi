/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/
#include <cmath>
#include <chrono>
#include "euler2D.hpp" 
#include "util/parallel_for.hpp"
#include "util/printb.hpp"
#include "common/helpers.hip.hpp"
#include "util/timer.hpp"

using namespace simbi;
using namespace simbi::util;
using namespace std::chrono;
constexpr auto write2file = helpers::write_to_file<hydro2d::PrimitiveSOA, 2, Newtonian2D>;


// Typedefs because I'm lazy
using Conserved = hydro2d::Conserved;
using Primitive = hydro2d::Primitive;
using Eigenvals = hydro2d::Eigenvals;

// Default Constructor 
Newtonian2D::Newtonian2D () {}

// Overloaded Constructor
Newtonian2D::Newtonian2D(
    std::vector<std::vector<real> > state, 
    luint nx,
    luint ny,
    real gamma, 
    std::vector<real> x1, 
    std::vector<real> x2, 
    real cfl, 
    std::string coord_system = "cartesian")
:
    HydroBase(
        state, 
        nx,
        ny,
        gamma, 
        x1, 
        x2, 
        cfl, 
        coord_system)
{

}

// Destructor 
Newtonian2D::~Newtonian2D() {}

// Typedefs because I'm lazy
typedef hydro2d::Conserved Conserved;
typedef hydro2d::Primitive Primitive;
typedef hydro2d::Eigenvals Eigenvals;
//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------
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
    auto* const cons_data = cons.data();
    auto* const prim_data = prims.data();
    simbi::parallel_for(p, (luint)0, nzones, [=] GPU_LAMBDA (const luint gid) {
        const real rho     = cons_data[gid].rho;
        const real v1      = cons_data[gid].m1 / rho;
        const real v2      = cons_data[gid].m2 / rho;
        const real rho_chi = cons_data[gid].chi;
        const real pre     = (gamma - static_cast<real>(1.0))*(cons_data[gid].e_dens - static_cast<real>(0.5) * rho * (v1 * v1 + v2 * v2));

        prim_data[gid] = Primitive{rho, v1, v2, pre, rho_chi / rho};
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
        const real vL   = left_prims.vcomponent(ehat);
        const real vR   = right_prims.vcomponent(ehat);
        const real pR   = right_prims.p;
        const real pL   = left_prims.p;
        const real rhoL = left_prims.rho;
        const real rhoR = right_prims.rho;

        const real csR = std::sqrt(gamma * pR/rhoR);
        const real csL = std::sqrt(gamma * pL/rhoL);

        // Calculate the mean velocities of sound and fluid
        // const real cbar   = 0.5*(csL + csR);
        // const real rhoBar = 0.5*(rhoL + rhoR);
        const real z      = (gamma - 1.)/(2.0*gamma);
        const real num    = csL + csR- (gamma - 1.) * 0.5 *(vR- vR);
        const real denom  = csL * std::pow(pL, - z) + csR * std::pow(pR, - z);
        const real p_term = num/denom;
        const real pStar  = std::pow(p_term, (1./z));

        const real qL = 
            (pStar <= pL) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pL - 1.));

        const real qR = 
            (pStar <= pR) ? 1. : std::sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/pR- 1.));

        const real aL = vR- qL*csL;
        const real aR = vR+ qR*csR;

        const real aStar = 
            ( (pR- pL + rhoL*vL*(aL - vL) - rhoR*vR*(aR - vR) )/ 
                (rhoL*(aL - vL) - rhoR*(aR - vR) ) );

        return Eigenvals(aL, aR, aStar, pStar);

    } else {
        const real vL   = left_prims.vcomponent(ehat);
        const real vR  = right_prims.vcomponent(ehat);
        const real pR  = right_prims.p;
        const real pL   = left_prims.p;
        const real rhoL = left_prims.rho;
        const real rhoR = right_prims.rho;
        const real csR = std::sqrt(gamma * pR/rhoR);
        const real csL  = std::sqrt(gamma * pL/rhoL);

        const real aL = helpers::my_min(vL - csL, vR - csR);
        const real aR = helpers::my_max(vL + csL, vR + csR);

        return Eigenvals(aL, aR);

    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state array for computation. 
// It is being doing poluintwise in this case as opposed to over
// the entire array since we are in c++
GPU_CALLABLE_MEMBER
Conserved Newtonian2D::prims2cons(const Primitive &prims)
 {
    const real rho = prims.rho;
    const real v1  = prims.v1;
    const real v2  = prims.v2;
    const real pre = prims.p;
    const real et  = pre/(gamma - 1.0) + 0.5 * rho * (v1*v1 + v2*v2);

    return Conserved{rho, rho*v1, rho*v2, et};
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
        real cfl_dt;
        for (luint jj = 0; jj < yphysical_grid; jj++)
        {
            const auto shift_j = jj + idx_active;
            #pragma omp for schedule(static) reduction(min:min_dt)
            for (luint ii = 0; ii < xphysical_grid; ii++)
            {
                const auto shift_i  = ii + idx_active;
                const auto aid      = shift_j * nx + shift_i;
                const auto rho      = prims[aid].rho;
                const auto v1       = prims[aid].v1;
                const auto v2       = prims[aid].v2;
                const auto pressure = prims[aid].p;
                const auto cs       = std::sqrt(gamma * pressure / rho );

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = helpers::my_min( dx1/(helpers::my_max(std::abs(v1 + cs), std::abs(v1 - cs))), dx2/(helpers::my_max(std::abs(v2 + cs), std::abs(v2 - cs))) );
                    break;
                case simbi::Geometry::SPHERICAL:
                    {
                        const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2,  x2min);
                        const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dtheta = tr - tl;
                        const real x1l    = get_x1face(ii, geometry, 0);
                        const real x1r    = get_x1face(ii, geometry, 1);
                        const real dr     = x1r - x1l;
                        const real rmean  = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                        cfl_dt = helpers::my_min( dr/(helpers::my_max(std::abs(v1 + cs), std::abs(v1 - cs))), rmean*dtheta/(helpers::my_max(std::abs(v2 + cs), std::abs(v2 - cs))) );

                    break;
                    }
                case simbi::Geometry::CYLINDRICAL:
                    // TODO: Implement Cylindrical coordinates at some point
                    break;
                }
                min_dt = std::min(min_dt, cfl_dt);
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
            compute_dt<Primitive><<<p.gridSize,p.blockSize, bytes>>>(dev, prims.data(),dt_min.data(), geometry, psize, dx1, dx2);
            deviceReduceKernel<2><<<p.gridSize,p.blockSize>>>(dev, dt_min.data(), active_zones);
            deviceReduceKernel<2><<<1,1024>>>(dev, dt_min.data(), p.gridSize.x * p.gridSize.y);
            break;
        
        case simbi::Geometry::SPHERICAL:
            compute_dt<Primitive><<<p.gridSize,p.blockSize, bytes>>> (dev, prims.data(), dt_min.data(), geometry, psize, dlogx1, dx2, x1min, x1max, x2min, x2max);
            deviceReduceKernel<2><<<p.gridSize,p.blockSize>>>(dev, dt_min.data(),  active_zones);
            deviceReduceKernel<2><<<1,1024>>>(dev, dt_min.data(), p.gridSize.x * p.gridSize.y);
            break;
        case simbi::Geometry::CYLINDRICAL:
            // TODO: Implement Cylindrical coordinates at some point
            break;
        }
        simbi::gpu::api::deviceSynch();
        this->dt = dev->dt;
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
    const auto v1  = prims.v1;
    const auto v2  = prims.v2;
    const auto pre = prims.p;
    const auto et  = pre/(gamma - 1.0) + 0.5 * rho * (v1*v1 + v2*v2);
    
    const auto dens  = rho*vn;
    const auto momx  = rho*v1*vn + pre * kronecker(1, ehat);
    const auto momy  = rho*v2*vn + pre * kronecker(2, ehat);
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
    real am = helpers::my_min(static_cast<real>(0.0), lambda.aL);
    real ap = helpers::my_max(static_cast<real>(0.0), lambda.aR);
    
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

    #if GPU_CODE
    const luint xextent               = p.blockSize.x;
    const luint yextent               = p.blockSize.y;
    const bool first_order         = this->first_order;
    // const bool periodic            = this->periodic;
    const bool hllc                = this->hllc;
    const real dt                  = this->dt;
    // const real decay_const         = this->decay_const;
    const real plm_theta           = this->plm_theta;
    // const real gamma               = this->gamma;
    const lint nx                  = this->nx;
    const lint ny                  = this->ny;
    const real dx2                 = this->dx2;
    const real dlogx1              = this->dlogx1;
    const real dx1                 = this->dx1;
    const real x1min               = this->x1min;
    const real x1max               = this->x1max;
    const real x2min               = this->x2min;
    const real x2max               = this->x2max;
    const real pow_dlogr           = std::pow(10, dlogx1);
    // const auto nzones              = nx * ny;
    #endif
    // const luint nbs                = (BuildPlatform == Platform::GPU) ? bx * by : nzones;

    // Compile-time choice of coloumn major indexing
    const lint sx            = (col_maj) ? 1  : bx;
    const lint sy            = (col_maj) ? by :  1;
    const auto pseudo_radius = (first_order) ? 1 : 2;
    const auto step          = (first_order) ? static_cast<real>(1.0) : static_cast<real>(0.5);

    auto* const prim_data    = prims.data();
    auto* const cons_data    = cons.data();
    auto* const dens_source  = sourceRho.data();
    auto* const mom1_source  = sourceM1.data();
    auto* const mom2_source  = sourceM2.data();
    auto* const erg_source   = sourceE.data();
    simbi::parallel_for(p, (luint)0, extent, [=] GPU_LAMBDA (const luint idx){
       #if GPU_CODE 
        extern __shared__ Primitive prim_buff[];
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : idx % xpg;
        const luint jj  = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : idx / xpg;
        #if GPU_CODE 
        if ((ii >= xpg) || (jj >= ypg)) return;
        #endif

        const lint ia  = ii + radius;
        const lint ja  = jj + radius;
        const lint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x: 0;
        const lint ty  = (BuildPlatform == Platform::GPU) ? threadIdx.y: 0;
        const lint txa = (BuildPlatform == Platform::GPU) ? tx + pseudo_radius : ia;
        const lint tya = (BuildPlatform == Platform::GPU) ? ty + pseudo_radius : ja;

        Conserved uxL, uxR, uyL, uyR;
        Conserved fL, fR, gL, gR, frf, flf, grf, glf;
        Primitive xprimsL, xprimsR, yprimsL, yprimsR;

        const lint aid = (col_maj) ? ia * ny + ja : ja * nx + ia;
        // Load Shared memory luinto buffer for active zones plus ghosts
        #if GPU_CODE
            luint txl = xextent;
            luint tyl = yextent;
            // Load Shared memory into buffer for active zones plus ghosts
            prim_buff[tya * sx + txa * sy] = prim_data[aid];
            if (ty < pseudo_radius)
            {
                if (blockIdx.y == p.gridSize.y - 1 && (ja + yextent > ny - radius + ty)) {
                    tyl = ny - radius - ja + ty;
                }
                prim_buff[(tya - pseudo_radius) * sx + txa] = prim_data[helpers::mod(ja - pseudo_radius, ny) * nx + ia];
                prim_buff[(tya + tyl) * sx + txa]           = prim_data[(ja + tyl) % ny * nx + ia]; 
            }
            if (tx < pseudo_radius)
            {   
                if (blockIdx.x == p.gridSize.x - 1 && (ia + xextent > nx - radius + tx)) {
                    txl = nx - radius - ia + tx;
                }
                prim_buff[tya * sx + txa - pseudo_radius] =  prim_data[ja * nx + helpers::mod(ia - pseudo_radius, nx)];
                prim_buff[tya * sx + txa +    txl]        =  prim_data[ja * nx +    (ia + txl) % nx]; 
            }
            simbi::gpu::api::synchronize();
        #endif

        if (self->first_order)
        {
            xprimsL = prim_buff[(txa + 0)      * sy + (tya + 0) * sx];
            xprimsR = prim_buff[(txa + 1) % bx * sy + (tya + 0) * sx];
            //j+1/2
            yprimsL = prim_buff[(txa + 0) * sy + (tya + 0)      * sx];
            yprimsR = prim_buff[(txa + 0) * sy + (tya + 1) % by * sx];
            
            // i+1/2
            uxL = self->prims2cons(xprimsL); 
            uxR = self->prims2cons(xprimsR); 
            // j+1/2
            uyL = self->prims2cons(yprimsL);  
            uyR = self->prims2cons(yprimsR); 

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);

            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            // Calc HLL Flux at i+1/2 luinterface
            if (self->hllc) {
                frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            } else {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }

            xprimsL = prim_buff[helpers::mod(txa - 1, bx) * sy + (tya + 0) * sx];
            xprimsR = prim_buff[            (txa - 0)     * sy + (tya + 0) * sx];
            //j+1/2
            yprimsL = prim_buff[(txa - 0) * sy + helpers::mod(tya - 1, by) * sx]; 
            yprimsR = prim_buff[(txa + 0) * sy +             (tya - 0)     * sx]; 

            // i+1/2
            uxL = self->prims2cons(xprimsL); 
            uxR = self->prims2cons(xprimsR); 
            // j+1/2
            uyL = self->prims2cons(yprimsL);  
            uyR = self->prims2cons(yprimsR); 

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);
            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            // Calc HLL Flux at i-1/2 luinterface
            if (self->hllc)
            {
                flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);

            } else {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }   
        }
        else 
        {
            Primitive xleft_most, xright_most, xleft_mid, xright_mid, center;
            Primitive yleft_most, yright_most, yleft_mid, yright_mid;
            // Coordinate X
            xleft_most  = prim_buff[(helpers::mod(txa - 2, bx)    * sy + tya * sx)];
            xleft_mid   = prim_buff[(helpers::mod(txa - 1, bx)    * sy + tya * sx)];
            center      = prim_buff[(            (txa + 0)        * sy + tya * sx)];
            xright_mid  = prim_buff[(            (txa + 1) % bx   * sy + tya * sx)];
            xright_most = prim_buff[(            (txa + 2) % bx   * sy + tya * sx)];

            // Coordinate Y
            yleft_most  = prim_buff[(txa * sy + helpers::mod(tya - 2, by)  * sx)];
            yleft_mid   = prim_buff[(txa * sy + helpers::mod(tya - 1, by)  * sx)];
            yright_mid  = prim_buff[(txa * sy +             (tya + 1) % by * sx)];
            yright_most = prim_buff[(txa * sy +             (tya + 2) % by * sx)];

            // Reconstructed left X Primitive vector at the i+1/2 luinterface
            xprimsL = center     + helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center) * plm_theta) * static_cast<real>(0.5); 
            xprimsR = xright_mid - helpers::minmod((xright_mid - center) * plm_theta, (xright_most - center) * static_cast<real>(0.5), (xright_most - xright_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsL = center     + helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center) * plm_theta) * static_cast<real>(0.5);  
            yprimsR = yright_mid - helpers::minmod((yright_mid - center) * plm_theta, (yright_most - center) * static_cast<real>(0.5), (yright_most - yright_mid)*plm_theta) * static_cast<real>(0.5);


            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = self->prims2cons(xprimsL);
            uxR = self->prims2cons(xprimsR);
            uyL = self->prims2cons(yprimsL);
            uyR = self->prims2cons(yprimsR);

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);
            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            if (hllc)
            {
                frf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }
            else
            {
                frf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                grf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }

            // Do the same thing, but for the left side luinterface [i - 1/2]
            xprimsL = xleft_mid + helpers::minmod((xleft_mid - xleft_most) * plm_theta, (center - xleft_most) * static_cast<real>(0.5), (center - xleft_mid)*plm_theta) * static_cast<real>(0.5);
            xprimsR = center    - helpers::minmod((center - xleft_mid)*plm_theta, (xright_mid - xleft_mid)*static_cast<real>(0.5), (xright_mid - center)*plm_theta)*static_cast<real>(0.5);
            yprimsL = yleft_mid + helpers::minmod((yleft_mid - yleft_most) * plm_theta, (center - yleft_most) * static_cast<real>(0.5), (center - yleft_mid)*plm_theta) * static_cast<real>(0.5);
            yprimsR = center    - helpers::minmod((center - yleft_mid)*plm_theta, (yright_mid - yleft_mid)*static_cast<real>(0.5), (yright_mid - center)*plm_theta)*static_cast<real>(0.5);

            // Calculate the left and right states using the reconstructed PLM
            // Primitive
            uxL = self->prims2cons(xprimsL);
            uxR = self->prims2cons(xprimsR);
            uyL = self->prims2cons(yprimsL);
            uyR = self->prims2cons(yprimsR);

            fL = self->prims2flux(xprimsL, 1);
            fR = self->prims2flux(xprimsR, 1);
            gL = self->prims2flux(yprimsL, 2);
            gR = self->prims2flux(yprimsR, 2);

            
            if (self->hllc)
            {
                flf = self->calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }
            else
            {
                flf = self->calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1);
                glf = self->calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2);
            }
        }

        //Advance depending on geometry
        luint real_loc  = (col_maj) ? ii * ypg + jj : jj * xpg + ii;
        const real rho_source = dens_source[real_loc];
        const real m1_source  = mom1_source[real_loc];
        const real m2_source  = mom2_source[real_loc];
        const real e_source   = erg_source[real_loc];
        switch (geometry)
        {
            case simbi::Geometry::CARTESIAN:
                {
                    const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
                    cons_data[aid] -= ( (frf - flf) / dx1 + (grf - glf)/dx2 - source_terms) * step * dt;
                break;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                const real rl           = (ii > 0 ) ? x1min * std::pow(10, (ii -static_cast<real>(0.5)) * dlogx1) :  x1min;
                const real rr           = (ii < xpg - 1) ? rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                const real tl           = (jj > 0 ) ? x2min + (jj - static_cast<real>(0.5)) * dx2 :  x2min;
                const real tr           = (jj < ypg - 1) ? tl + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
                const real s1R          = rr * rr; 
                const real s1L          = rl * rl; 
                const real s2R          = std::sin(tr);
                const real s2L          = std::sin(tl);
                const real thmean       = static_cast<real>(0.5) * (tl + tr);
                const real sint         = std::sin(thmean);
                const real dV1          = rmean * rmean * (rr - rl);             
                const real dV2          = rmean * sint * (tr - tl); 
                // const real cot          = std::cos(thmean) / sint;

                // Grab central primitives
                const real rhoc = prim_buff[tya * bx + txa].rho;
                const real uc   = prim_buff[tya * bx + txa].v1;
                const real vc   = prim_buff[tya * bx + txa].v2;
                const real pc   = prim_buff[tya * bx + txa].p;
                
                const Conserved geom_source  = {static_cast<real>(0.0), (rhoc * vc * vc) / rmean + pc * (s1R - s1L) / dV1, - (rhoc * uc * vc) / rmean + pc * (s2R - s2L)/dV2 , static_cast<real>(0.0)};
                const Conserved source_terms = {rho_source, m1_source, m2_source, e_source};
                cons_data[aid] -= ( (frf * s1R - flf * s1L) / dV1 + (grf * s2R - glf * s2L) / dV2 - geom_source - source_terms) * dt * step;
                break;
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement Cylindrical coordinates at some point
                break;
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
    real round_place = 1 / chkpt_interval;
    this->t = tstart;
    this->t_interval =
        t == 0 ? 0
               : dlogt !=0 ? tstart
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;

    // Define the simulation members
    this->chkpt_interval  = chkpt_interval;
    this->data_directory  = data_directory;
    this->tstart          = tstart;
    this->init_chkpt_idx  = chkpt_idx;
    this->total_zones     = nx * ny;
    this->sourceRho       = sources[0];
    this->sourceM1        = sources[1];
    this->sourceM2        = sources[2];
    this->sourceE         = sources[3];
    this->first_order     = first_order;
    this->periodic        = boundary_condition == "periodic";
    this->hllc            = hllc;
    this->dlogt           = dlogt;
    this->linspace        = linspace;
    this->plm_theta       = plm_theta;
    this->xphysical_grid  = (periodic) ? nx : (first_order) ? nx - 2 : nx - 4;
    this->yphysical_grid  = (periodic) ? ny : (first_order) ? ny - 2 : ny - 4;
    this->idx_active      = (periodic) ? 0 : (first_order) ? 1 : 2;
    this->active_zones    = xphysical_grid * yphysical_grid;
    this->quirk_smoothing = quirk_smoothing;
    this->bc              = helpers::boundary_cond_map.at(boundary_condition);
    this->geometry        = helpers::geometry_map.at(coord_system);
    this->checkpoint_zones= yphysical_grid;
    this->dx2     = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
    this->dlogx1  = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
    this->dx1     = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
    this->x1min   = x1[0];
    this->x1max   = x1[xphysical_grid - 1];
    this->x2min   = x2[0];
    this->x2max   = x2[yphysical_grid - 1];

    this->rho_all_zeros  = std::all_of(sourceRho.begin(), sourceRho.end(),   [](real i) {return i == 0;});
    this->m1_all_zeros   = std::all_of(sourceM1.begin(),  sourceM1.end(),  [](real i) {return i == 0;});
    this->m2_all_zeros   = std::all_of(sourceM2.begin(),  sourceM2.end(),  [](real i) {return i == 0;});
    this->e_all_zeros    = std::all_of(sourceE.begin(),   sourceE.end(), [](real i) {return i == 0;});
    // Stuff for moving mesh
    this->hubble_param = 0.0; ///adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);

    if (x2max == 0.5 * M_PI){
        this->reflecting_theta = true;
    }
    // Write some info about the setup for writeup later
    setup.x1max          = x1[xphysical_grid - 1];
    setup.x1min          = x1[0];
    setup.x2max          = x2[yphysical_grid - 1];
    setup.x2min          = x2[0];
    setup.nx             = nx;
    setup.ny             = ny;
    setup.xactive_zones  = xphysical_grid;
    setup.yactive_zones  = yphysical_grid;
    setup.linspace       = linspace;
    setup.ad_gamma       = gamma;
    setup.first_order    = first_order;
    setup.coord_system   = coord_system;
    setup.boundarycond   = boundary_condition;
    setup.regime         = "classical";
    setup.x1             = x1;
    setup.x2             = x2;

    cons.resize(nzones);
    prims.resize(nzones);
    dt_min.resize(active_zones);
    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state[0].size(); i++)
    {
        auto rho      = state[0][i];
        auto m1       = state[1][i];
        auto m2       = state[2][i];
        auto e        = state[3][i];
        auto rho_chi  = state[4][i];
        cons[i]    = Conserved(rho, m1, m2, e, rho_chi);
    }

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_constant = 1 / (1 + std::exp(static_cast<real>(10.0) * (tstart - engine_duration)));

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    
    // Copy the current SRHD instance over to the device
    // if compiling for CPU, these functions do nothing
    Newtonian2D *device_self;
    simbi::gpu::api::gpuMallocManaged(&device_self, sizeof(Newtonian2D));
    simbi::gpu::api::copyHostToDevice(device_self, this, sizeof(Newtonian2D));
    cons.copyToGpu();
    prims.copyToGpu();
    dt_min.copyToGpu();
    sourceRho.copyToGpu();
    sourceM1.copyToGpu();
    sourceM2.copyToGpu();
    sourceE.copyToGpu();

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
        m2_all_zeros   = std::all_of(sourceM2.begin(),  sourceM2.end(),  [](real i) {return i == 0;});
        e_all_zeros    = std::all_of(sourceE.begin(),  sourceE.end(), [](real i) {return i == 0;});
    }

    // Setup the system
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
        if (!periodic) config_ghosts2D(fullP, cons.data(), nx, ny, first_order, bc);
    }
    const auto dtShBytes = xblockdim * yblockdim * sizeof(Primitive) + xblockdim * yblockdim * sizeof(real);
    
    // Determine the memory side and state position
    const auto memside = (BuildPlatform == Platform::GPU) ? simbi::MemSide::Dev : simbi::MemSide::Host;
    const auto self    = (BuildPlatform == Platform::GPU) ? device_self : this;

    if constexpr(BuildPlatform == Platform::GPU) {
        cons2prim(fullP, device_self, simbi::MemSide::Dev);
        adapt_dt(device_self, geometry, activeP, dtShBytes);
    } else {
        cons2prim(fullP);
        adapt_dt();
    }

    // Save initial condition
    if (t == 0) {
        write2file(*this, setup, data_directory, t, t_interval, chkpt_interval, yphysical_grid);
        t_interval += chkpt_interval;
    }
    
    // Simulate :)
    while (t < tend & !inFailureState)
    {
        simbi::detail::with_timer(*this, [&](){
            advance(self, activeP, bx, by, radius, geometry, memside);
            cons2prim(fullP, self, memside);
            if (!periodic) config_ghosts2D(fullP, cons.data(), nx, ny, first_order, bc);
        });

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(device_self, geometry, activeP, dtShBytes);
        } else {
            adapt_dt();
        }
        t += dt;
    }
    
    // if (ncheck > 0) {
    //      writeln("Average zone update/sec for:{:>5} iterations was {:>5.2e} zones/sec", n, zu_avg/ncheck);
    // }
    std::vector<std::vector<real>> final_prims(5, std::vector<real>(nzones, 0));
    for (luint ii = 0; ii < nx; ii++) {
        final_prims[0][ii] = prims[ii].rho;
        final_prims[1][ii] = prims[ii].v1;
        final_prims[2][ii] = prims[ii].v2;
        final_prims[3][ii] = prims[ii].p;
        final_prims[4][ii] = prims[ii].chi;
    }

    return final_prims;

 };