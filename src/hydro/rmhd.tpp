/*
 * C++ Source to perform RMHD Calculations
 * Marcus DuPont
 * New York University
 * 11/14/2023
 * Compressible Hydro Simulation
 */
#include <cmath>                  // for max, min
#include "util/device_api.hpp"    // for syncrohonize, devSynch, ...
#include "util/parallel_for.hpp"  // for parallel_for
#include "util/printb.hpp"        // for writeln
#include "util/logger.hpp"        // for logger

using namespace simbi;
using namespace simbi::util;

// Default Constructor
template<int dim>
RMHD<dim>::RMHD() {
    
}

// Overloaded Constructor
template<int dim>
RMHD<dim>::RMHD(
    std::vector<std::vector<real>> &state, 
    const InitialConditions &init_conditions)
:
    HydroBase(
        state,
        init_conditions
    )
{
}

// Destructor
template<int dim>
RMHD<dim>::~RMHD() {}


// Helpers
template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x1face(const lint ii, const int side) const
{
    switch (x1_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x1l = helpers::my_max<real>(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
            if (side == 0) {
                return x1l;
            }
            return helpers::my_min<real>(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
        }
    default:
        {
            const real x1l = helpers::my_max<real>(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
            if (side == 0) {
                return x1l;
            }
            return helpers::my_min<real>(x1l * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
        }
    }
}


template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x2face(const lint ii, const int side) const
{
    switch (x2_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x2l = helpers::my_max<real>(x2min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return x2l;
            }
            return helpers::my_min<real>(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }
    default:
        {
            const real x2l = helpers::my_max<real>(x2min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx2),  x2min);
            if (side == 0) {
                return x2l;
            }
            return helpers::my_min<real>(x2l * std::pow(10, dlogx2 * (ii == 0 ? 0.5 : 1.0)), x2max);
        }
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x3face(const lint ii, const int side) const
{
    switch (x3_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x3l = helpers::my_max<real>(x3min  + (ii - static_cast<real>(0.5)) * dx3,  x3min);
            if (side == 0) {
                return x3l;
            }
            return helpers::my_min<real>(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
        }
    default:
        {
            const real x3l = helpers::my_max<real>(x3min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx3),  x3min);
            if (side == 0) {
                return x3l;
            }
            return helpers::my_min<real>(x3l * std::pow(10, dlogx3 * (ii == 0 ? 0.5 : 1.0)), x3max);
        }
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x1_differential(const lint ii) const {
    const real x1l   = get_x1face(ii, 0);
    const real x1r   = get_x1face(ii, 1);
    const real xmean = helpers::get_cell_centroid(x1r, x1l, geometry);
    switch (geometry)
    {
    case Geometry::SPHERICAL:
        return xmean * xmean * (x1r - x1l);
    default:
        return xmean * (x1r - x1l);
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x2_differential(const lint ii) const {
    if constexpr(dim == 1) {
        switch (geometry)
        {
        case Geometry::SPHERICAL:
            return 2;
        default:
            return static_cast<real>(2 * M_PI);
        }
    } else {
        switch (geometry)
        {
            case Geometry::SPHERICAL:
            {
                const real x2l = get_x2face(ii, 0);
                const real x2r = get_x2face(ii, 1);
                const real dcos = std::cos(x2l) - std::cos(x2r);
                return dcos;  
            }
            default:
            {
                return dx2;
            }
        }
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real RMHD<dim>::get_x3_differential(const lint ii) const {
    if constexpr(dim == 1) {
        switch (geometry)
        {
        case Geometry::SPHERICAL:
            return static_cast<real>(2 * M_PI);
        default:
            return 1;
        }
    } else if constexpr(dim == 2) {
        switch (geometry)
        {
            case Geometry::PLANAR_CYLINDRICAL:
                return 1;
            default:
                return static_cast<real>(2 * M_PI);
        }
    } else {
        return dx3;
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
real RMHD<dim>::get_cell_volume(const lint ii, const lint jj, const lint kk) const
{
    // the volume in cartesian coordinates is only nominal
    if (geometry == Geometry::CARTESIAN) {
        return 1;
    }
    return get_x1_differential(ii) * get_x2_differential(jj) * get_x3_differential(kk);
}

template<int dim>
void RMHD<dim>::emit_troubled_cells() {
    troubled_cells.copyFromGpu();
    cons.copyFromGpu();
    prims.copyFromGpu();
    for (luint gid = 0; gid < total_zones; gid++)
    {
        if (troubled_cells[gid] != 0) {
            const luint xpg   = xactive_grid;
            const luint ypg   = yactive_grid;
            const luint kk    = helpers::get_height(gid, xpg, ypg);
            const luint jj    = helpers::get_row(gid, xpg, ypg, kk);
            const luint ii    = helpers::get_column(gid, xpg, ypg, kk);
            const lint ireal  = helpers::get_real_idx(ii, radius, xactive_grid);
            const lint jreal  = helpers::get_real_idx(jj, radius, yactive_grid); 
            const lint kreal  = helpers::get_real_idx(kk, radius, zactive_grid); 
            const real x1l    = get_x1face(ireal, 0);
            const real x1r    = get_x1face(ireal, 1);
            const real x2l    = get_x2face(jreal, 0);
            const real x2r    = get_x2face(jreal, 1);
            const real x3l    = get_x3face(kreal, 0);
            const real x3r    = get_x3face(kreal, 1);
            const real x1mean = helpers::calc_any_mean(x1l, x1r, x1_cell_spacing);
            const real x2mean = helpers::calc_any_mean(x2l, x2r, x2_cell_spacing);
            const real x3mean = helpers::calc_any_mean(x3l, x3r, x3_cell_spacing);
            const real s1 = cons[gid].momentum(1);
            const real s2 = cons[gid].momentum(2);
            const real s3 = cons[gid].momentum(3);
            const real et = (cons[gid].d + cons[gid].tau + prims[gid].p);
            const real h1 = cons[gid].bcomponent(1);
            const real h2 = cons[gid].bcomponent(2);
            const real h3 = cons[gid].bcomponent(3);
            const real s  = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);
            const real vsq = (s * s) / (et * et);
            const real mag_sq = (h1 * h1 + h2 * h2 + h3 * h3);
            const real w  = 1 / std::sqrt(1 - vsq);
            if constexpr(dim == 1) {
                printf("\nCons2Prim cannot converge\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, iter: %lu\n", 
                        cons[gid].d / w, prims[gid].p, vsq, mag_sq, x1mean, n
                );
            } else if constexpr(dim == 2) {
                printf("\nCons2Prim cannot converge\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, x2coord: %.2e, iter: %lu\n", 
                        cons[gid].d / w, prims[gid].p, vsq, mag_sq, x1mean, x2mean, n
                );
            } else {
                printf("\nCons2Prim cannot converge\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, x2coord: %.2e, x3coord: %.2e, iter: %lu\n", 
                        cons[gid].d / w, prims[gid].p, vsq, mag_sq, x1mean, x2mean, x3mean, n
                );
            }
        }
    }
}
//-----------------------------------------------------------------------------------------
//                          GET THE rm::Primitive
//-----------------------------------------------------------------------------------------
/**
 * Return the primitive
 * variables density , three-velocity, pressure
 * 
 * @param  p executation policy class  
 * @return none
 */
template<int dim>
void RMHD<dim>::cons2prim(const ExecutionPolicy<> &p)
{
    const auto gr = gamma / (gamma - 1);
    const auto* const cons_data  = cons.data();
    auto* const prim_data  = prims.data();
    auto* const edens_data = edens_guess.data(); 
    auto* const troubled_data = troubled_cells.data();
    simbi::parallel_for(p, (luint)0, total_zones, [
        prim_data,
        cons_data,
        edens_data,
        troubled_data,
        gr,
        this
    ] GPU_LAMBDA (luint gid){
        bool workLeftToDo = true;
        volatile  __shared__ bool found_failure;

        auto tid = get_threadId();
        if (tid == 0) 
            found_failure = inFailureState;
        simbi::gpu::api::synchronize();

        real invdV = 1.0;
        while (!found_failure && workLeftToDo)
        {
            if (changing_volume)
            {
                if constexpr(dim == 1) {
                    const auto ireal = helpers::get_real_idx(gid, radius, active_zones);
                    const real dV    = get_cell_volume(ireal);
                    invdV            = 1 / dV;
                } else if constexpr(dim == 2) {
                    const luint ii   = gid % nx;
                    const luint jj   = gid / nx;
                    const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
                    const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
                    const real dV    = get_cell_volume(ireal, jreal);
                    invdV = 1 / dV;
                } else {
                    const luint kk  = simbi::helpers::get_height(gid, xactive_grid, yactive_grid);
                    const luint jj  = simbi::helpers::get_row(gid, xactive_grid, yactive_grid, kk);
                    const luint ii  = simbi::helpers::get_column(gid, xactive_grid, yactive_grid, kk);
                    const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
                    const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
                    const auto kreal = helpers::get_real_idx(kk, radius, zactive_grid); 
                    const real dV    = get_cell_volume(ireal, jreal, kreal);
                    invdV = 1 / dV;
                }
            }
            
            const real d    = cons_data[gid].d * invdV;
            const real m1   = cons_data[gid].momentum(1) * invdV;
            const real m2   = cons_data[gid].momentum(2) * invdV;
            const real m3   = cons_data[gid].momentum(3) * invdV;
            const real tau  = cons_data[gid].tau * invdV;
            const real b1   = cons_data[gid].bcomponent(1) * invdV;
            const real b2   = cons_data[gid].bcomponent(2) * invdV;
            const real b3   = cons_data[gid].bcomponent(3) * invdV;
            const real dchi = cons_data[gid].chi * invdV; 
            const real s    = (m1 * b1 + m2 * b2 + m3 * b3);
            const real ssq  = s * s;
            const real msq  = (m1 * m1 + m2 * m2 + m3 * m3);
            const real bsq  = (b1 * b1 + b2 * b2 + b3 * b3);

            // Perform modified Newton Raphson based on
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930
            // so far, the convergence rate is the same, but perhaps I need a slight tweak

            // compute f(x_0)
            // f = helpers::newton_f(gamma, tau, d, S, peq);
            int iter = 0;
            real qq = edens_data[gid];
            const real tol = d * tol_scale;
            real f, g;
            do
            {
                // compute x_[k+1]
                f = helpers::newton_f_mhd(gr, tau + d, d, ssq, bsq, msq, qq);
                g = helpers::newton_g_mhd(gr, tau + d, d, ssq, bsq, msq, qq);
                qq  -= f / g;

                // compute x*_k
                // f     = helpers::newton_f(gamma, tau, d, S, qq);
                // pstar = qq - f / g;

                if (iter >= MAX_ITER || std::isnan(qq))
                {
                    troubled_data[gid] = 1;
                    dt                = INFINITY;
                    inFailureState    = true;
                    found_failure     = true;
                    break;
                }
                iter++;

            } while (std::abs(f / g) >= tol);

            const real w  = helpers::calc_rmhd_lorentz(ssq, bsq, msq, qq);
            const real pg = helpers::calc_rmhd_pg(gr, d, w, qq);
            const real v1 = (1 / qq + bsq) * (m1 + s / qq * b1);
            const real v2 = (1 / qq + bsq) * (m2 + s / qq * b2);
            const real v3 = (1 / qq + bsq) * (m3 + s / qq * b3);
            edens_data[gid] = qq;
            #if FOUR_VELOCITY
                prim_data[gid] = rm::Primitive<dim>{d/w, v1 * w, v2 * w, v3 * w, pg, b1, b2, b3, dchi / d};
            #else
                prim_data[gid] = rm::Primitive<dim>{d/w, v1, v2, v3, pg, b1, b2, b3, dchi / d};
            #endif
            workLeftToDo = false;

            if (qq < 0) {
                troubled_data[gid] = 1;
                inFailureState = true;
                found_failure  = true;
                dt = INFINITY;
            }
            simbi::gpu::api::synchronize();
        }
    });
}

/**
 * Return the primitive
 * variables density , three-velocity, pressure
 * 
 * @param  con conserved array at index
 * @param gid  current global index
 * @return none
 */
template<int dim>
RMHD<dim>::primitive_t RMHD<dim>::cons2prim(const RMHD<dim>::conserved_t &cons, const luint gid)
{
    const auto gr = gamma / (gamma - 1);
    auto* const edens_data = edens_guess.data(); 
    real invdV = 1.0;
    if (changing_volume)
    {
        if constexpr(dim == 1) {
            const auto ireal = helpers::get_real_idx(gid, radius, active_zones);
            const real dV    = get_cell_volume(ireal);
            invdV            = 1 / dV;
        } else if constexpr(dim == 2) {
            const luint ii   = gid % nx;
            const luint jj   = gid / nx;
            const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
            const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
            const real dV    = get_cell_volume(ireal, jreal);
            invdV = 1 / dV;
        } else {
            const luint kk  = simbi::helpers::get_height(gid, xactive_grid, yactive_grid);
            const luint jj  = simbi::helpers::get_row(gid, xactive_grid, yactive_grid, kk);
            const luint ii  = simbi::helpers::get_column(gid, xactive_grid, yactive_grid, kk);
            const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
            const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
            const auto kreal = helpers::get_real_idx(kk, radius, zactive_grid); 
            const real dV    = get_cell_volume(ireal, jreal, kreal);
            invdV = 1 / dV;
        }
    }
    
    const real d    = cons.d * invdV;
    const real m1   = cons.momentum(1) * invdV;
    const real m2   = cons.momentum(2) * invdV;
    const real m3   = cons.momentum(3) * invdV;
    const real tau  = cons.tau * invdV;
    const real b1   = cons.bcomponent(1) * invdV;
    const real b2   = cons.bcomponent(2) * invdV;
    const real b3   = cons.bcomponent(3) * invdV;
    const real dchi = cons.chi * invdV; 
    const real s    = (m1 * b1 + m2 * b2 + m3 * b3);
    const real ssq  = s * s;
    const real msq  = (m1 * m1 + m2 * m2 + m3 * m3);
    const real bsq  = (b1 * b1 + b2 * b2 + b3 * b3);


    // Perform modified Newton Raphson based on
    // https://www.sciencedirect.com/science/article/pii/S0893965913002930
    // so far, the convergence rate is the same, but perhaps I need a slight tweak

    // compute f(x_0)
    // f = helpers::newton_f(gamma, tau, d, S, peq);
    int iter = 0;
    real qq = edens_data[gid];
    const real tol = d * tol_scale;
    real f, g;
    do
    {
        // compute x_[k+1]
        f = helpers::newton_f_mhd(gr, tau + d, d, ssq, bsq, msq, qq);
        g = helpers::newton_g_mhd(gr, tau + d, d, ssq, bsq, msq, qq);
        qq  -= f / g;

        // compute x*_k
        // f     = helpers::newton_f(gamma, tau, d, S, qq);
        // pstar = qq - f / g;

        if (iter >= MAX_ITER || std::isnan(qq))
        {
            dt                = INFINITY;
            inFailureState    = true;
            break;
        }
        iter++;

    } while (std::abs(f / g) >= tol);

    const real w  = helpers::calc_rmhd_lorentz(ssq, bsq, msq, qq);
    const real pg = helpers::calc_rmhd_pg(gr, d, w, qq);
    const real v1 = (1 / qq + bsq) * (m1 + s / qq * b1);
    const real v2 = (1 / qq + bsq) * (m2 + s / qq * b2);
    const real v3 = (1 / qq + bsq) * (m3 + s / qq * b3);
    edens_data[gid] = qq;
    #if FOUR_VELOCITY
        return rm::Primitive<dim>{d/w, v1 * w, v2 * w, v3 * w, pg, dchi / d};
    #else
        return rm::Primitive<dim>{d/w, v1, v2, v3, pg, b1, b2, b3, dchi / d};
    #endif
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
/*
    Compute the outer wave speeds as discussed in Mignone and Bodo (2006)
*/
template<int dim>
GPU_CALLABLE_MEMBER
RMHD<dim>::eigenvals_t RMHD<dim>::calc_eigenvals(
    const RMHD<dim>::primitive_t &primsL,
    const RMHD<dim>::primitive_t &primsR,
    const luint nhat) const
{
    // Solve the quartic equation for the real roots of the wave speeds in Eq. (56)
    rm::Eigenvals<dim> lambdas;
    real lpL, lpR, lmL, lmR;

    // left side
    const real rhoL = primsL.rho;
    const real hL   = primsL.get_gas_enthalpy(gamma);
    const real cs2L = (gamma * primsL.p / (rhoL * hL));
    const auto b4L  = rm::MagFourVec<dim>(primsL); 
    const real bsqL = b4L.inner_product();
    const real bnL  = primsL.bcomponent(nhat);
    const real bn2L = bnL * bnL;
    const real vnL  = primsL.vcomponent(nhat);
    if (primsL.vsquared() < tol_scale) { // Eq.(57)
        const real fac = 1 / (rhoL * hL + bsqL);
        const real a = 1;
        const real b = - (
            bsqL + rhoL * hL * cs2L 
            + bn2L * cs2L
        ) * fac;
        const real c = cs2L * bn2L * fac;
        const real disq = std::sqrt(b * b - 4 * a * c);
        lpL = std::sqrt(0.5 * (-b + disq));
        lmL = - lpL;
    } else if (primsL.bcomponent(nhat) < tol_scale) { // Eq. (58)
        const real g2L = primsL.get_lorentz_factor_squared();
        const real vdbperp = primsL.vdotb() - primsL.vcomponent(nhat) * primsL.bcomponent(nhat);
        const real qL = bsqL - cs2L * vdbperp * vdbperp;
        const real a2 = rhoL * hL * (cs2L + g2L * (1 - cs2L)) + qL;
        const real a1 = -2 * rhoL * hL * g2L * vnL * (1 - cs2L);
        const real a0 = rhoL * hL * (-cs2L + g2L * vnL * vnL * (1 -cs2L)) - qL;
        const real disq = a1 * a1 - 4 * a2 * a0;
        lpL = 0.5 * (-a1 + std::sqrt(disq)) / a2;
        lmL = 0.5 * (-a1 - std::sqrt(disq)) / a2;
    } else { // solve the full quartic Eq. (56)
        real speeds[4];

        const real vdB    = primsL.vdotb();
        const real vdB2   = vdB * vdB;
        const real wL     = primsL.get_lorentz_factor();
        const real w2L    = wL * wL;
        const real nrg    = 1.0 / (rhoL * hL + bsqL);   
        const real vn2L   = vnL * vnL;
        const real bnred  = b4L.normal(nhat) / wL;
        const real bnred2 = bnred * bnred;  
                        
        const real invd  = cs2L * nrg;
        const real eps2  = (cs2L * rhoL * hL + bsqL) * nrg;
        const real nrgt  = w2L * rhoL * hL * (1.0 - cs2L) * nrg;

        //=========================================
        //    quartic coefficients adapted from PLUTO
        //=========================================
        const real scrh   = 2.0*(invd * vdB * bnred - eps2 * vnL);
        const real a4     = nrgt  - invd*vdB2 + eps2;
        const real fac    = 1.0 / a4;
        const real a3     = fac * (- 4.0 * vnL  * nrgt  + scrh);
        const real a2     = fac * (  6.0 * vn2L * nrgt  + invd * (vdB2 - bnred2) + eps2 * (vn2L - 1.0));
        const real a1     = fac * (- 4.0 * vnL * vn2L * nrgt  - scrh);
        const real a0     = fac * (vn2L * vn2L * nrgt  + invd * bnred2 - eps2 * vn2L);
        const auto nroots = helpers::quartic(a3, a2, a1, a0, speeds);
        
        // #if !GPU_CODE
        // if (nroots != 4) {
        //     printf("\nI'm dying on the left, %d, %.2e, %.2e\n", nroots, speeds[3], speeds[0]);
        // }
        // #endif
        lpL = speeds[3];
        lmL = speeds[0];
        
    };

    // right side
    const real rhoR = primsR.rho;
    const real hR   = primsR.get_gas_enthalpy(gamma);
    const real cs2R = (gamma * primsR.p / (rhoR * hR));
    const auto b4R  = rm::MagFourVec<dim>(primsR); 
    const real bsqR = b4R.inner_product();
    const real bnR  = primsR.bcomponent(nhat);
    const real bn2R = bnR * bnR;
    const real vnR  = primsR.vcomponent(nhat);
    if (primsR.vsquared() < minimum_real) { // Eq.(57)
        const real fac = 1 / (rhoR * hR + bsqR);
        const real a = 1;
        const real b = - (
            bsqR + rhoR * hR * cs2R 
            + bn2R * cs2R
        ) * fac;
        const real c = cs2R * bn2R * fac;
        const real disq = std::sqrt(b * b - 4 * a * c);
        lpR = std::sqrt(0.5 * (-b + disq));
        lmR = - lpR;
    } else if (primsR.bcomponent(nhat) < minimum_real) { // Eq. (58)
        const real g2R = primsR.get_lorentz_factor_squared();
        const real vdbperp = primsR.vdotb() - primsR.vcomponent(nhat) * primsR.bcomponent(nhat);
        const real qR = bsqR - cs2R * vdbperp * vdbperp;
        const real a2 = rhoR * hR * (cs2R + g2R * (1 - cs2R)) + qR;
        const real a1 = -2 * rhoR * hR * g2R * vnR * (1 - cs2R);
        const real a0 = rhoR * hR * (-cs2R + g2R * vnR * vnR * (1 -cs2R)) - qR;
        const real disq = a1 * a1 - 4 * a2 * a0;
        lpR = 0.5 * (-a1 + std::sqrt(disq)) / a2;
        lmR = 0.5 * (-a1 - std::sqrt(disq)) / a2;
    } else { // solve the full quartic Eq. (56)
        real speeds[4];

        const real vdB    = primsR.vdotb();
        const real vdB2   = vdB * vdB;
        const real wR     = primsR.get_lorentz_factor();
        const real w2R    = wR * wR;
        const real nrg    = 1.0 / (rhoR * hR + bsqR);   
        const real vn2R   = vnR * vnR;
        const real bnred  = b4R.normal(nhat) / wR;
        const real bnred2 = bnred * bnred;  
                        
        const real invd  = cs2R * nrg;
        const real eps2  = (cs2R * rhoR * hR + bsqR) * nrg;
        const real nrgt  = w2R * rhoR * hR * (1.0 - cs2R) * nrg;

        //=========================================
        //    quartic coefficients adapted from PLUTO
        //=========================================
        const real scrh   = 2.0*(invd * vdB * bnred - eps2 * vnR);
        const real a4     = nrgt  - invd*vdB2 + eps2;
        const real fac    = 1.0 / a4;
        const real a3     = fac * (- 4.0 * vnR  * nrgt  + scrh);
        const real a2     = fac * (  6.0 * vn2R * nrgt  + invd * (vdB2 - bnred2) + eps2 * (vn2R - 1.0));
        const real a1     = fac * (- 4.0 * vnR * vn2R * nrgt  - scrh);
        const real a0     = fac * (vnR * vn2R * nrgt  + invd * bnred2 - eps2 * vn2R);
        const auto nroots = helpers::quartic(a3, a2, a1, a0, speeds);
        
        lpR = speeds[3];
        lmR = speeds[0];

        // #if !GPU_CODE
        // if (nroots != 4) {
        //     printf("\nI'm dying on the right, %d, %.2e, %.2e\n", nroots, speeds[3], speeds[0]);
        // }
        // #endif
    };

    return {
        helpers::my_min(lmL, lmR),
        helpers::my_max(lpL, lpR),
        std::sqrt(cs2L),
        std::sqrt(cs2R)
    };
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template<int dim>
GPU_CALLABLE_MEMBER 
RMHD<dim>::conserved_t RMHD<dim>::prims2cons(const RMHD<dim>::primitive_t &prims) const
{
    const real rho      = prims.rho;
    const real v1       = prims.vcomponent(1);
    const real v2       = prims.vcomponent(2);
    const real v3       = prims.vcomponent(3);
    const real pg       = prims.p;
    const real b1       = prims.bcomponent(1);
    const real b2       = prims.bcomponent(2);
    const real b3       = prims.bcomponent(3);
    const real lorentz_gamma = prims.get_lorentz_factor();
    const real h             = prims.get_gas_enthalpy(gamma);
    const real vdotb         = prims.vdotb();
    const real bsq           = (b1 * b1 + b2 * b2 + b3 * b3);
    const real vsq           = (v1 * v1 + v2 * v2 + v3 * v3);

    return rm::Conserved<dim>{
         rho * lorentz_gamma, 
        (rho * h * lorentz_gamma * lorentz_gamma + bsq) * v1 - vdotb * b1,
        (rho * h * lorentz_gamma * lorentz_gamma + bsq) * v2 - vdotb * b2,
        (rho * h * lorentz_gamma * lorentz_gamma + bsq) * v3 - vdotb * b3,
         rho * h * lorentz_gamma * lorentz_gamma - pg - rho * lorentz_gamma + 0.5 * bsq + 0.5 * (vsq * bsq - vdotb * vdotb),
         b1,
         b2,
         b3,
         rho * lorentz_gamma * prims.chi
    };
};
//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditonal timestep
template<int dim>
template<TIMESTEP_TYPE dt_type>
void RMHD<dim>::adapt_dt()
{
    // singleton instance of thread pool. lazy-evaluated
    static auto &thread_pool = simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(static_cast<luint>(0), active_zones, [&](luint aid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
        const luint kk = dim < 3 ? 0 : simbi::helpers::get_height(aid, xactive_grid, yactive_grid);
        const luint jj = dim < 2 ? 0 : simbi::helpers::get_row(aid, xactive_grid, yactive_grid, kk);
        const luint ii = simbi::helpers::get_column(aid, xactive_grid, yactive_grid, kk);
        // Left/Right wave speeds
        if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
            const auto rho = prims[aid].rho;
            const auto v1  = prims[aid].vcomponent(1);
            const auto v2  = prims[aid].vcomponent(2);
            const auto v3  = prims[aid].vcomponent(3);
            const auto pre = prims[aid].p;
            const real h   = prims[aid].get_gas_enthalpy(gamma);
            const real cs  = std::sqrt(gamma * pre / (rho * h));
            v1p = std::abs(v1 + cs) / (1 + v1 * cs);
            v1m = std::abs(v1 - cs) / (1 - v1 * cs);
            if constexpr(dim > 1) {
                v2p = std::abs(v2 + cs) / (1 + v2 * cs);
                v2m = std::abs(v2 - cs) / (1 - v2 * cs);
            }
            if constexpr(dim > 2) {
                v3p = std::abs(v3 + cs) / (1 + v3 * cs);
                v3m = std::abs(v3 - cs) / (1 - v3 * cs);
            }                        
        } else {
            v1p = 1;
            v1m = 1;
            if constexpr(dim > 1) {
                v2p = 1;
                v2m = 1;
            }
            if constexpr(dim > 2) {
                v3p = 1; 
                v3m = 1; 
            }
        }

        const auto x1l = get_x1face(ii, 0);
        const auto x1r = get_x1face(ii, 1);
        const auto dx1 = x1r - x1l; 
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            if constexpr(dim == 1) {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m))
                });

            } else if constexpr(dim == 2) {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m)),
                    dx2 / (std::max(v2p, v2m))
                });
            } else {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m)),
                    dx2 / (std::max(v2p, v2m)),
                    dx3 / (std::max(v3p, v3m))
                });
            }   
            break;
        
        case simbi::Geometry::SPHERICAL:
            {
                if constexpr(dim == 1) {
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m))
                    });
                } else if constexpr(dim == 2) {
                    const auto rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::SPHERICAL);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m))
                    });
                } else {
                    const auto x2l   = get_x2face(jj, 0);
                    const auto x2r   = get_x2face(jj, 1);
                    const auto rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::SPHERICAL);
                    const real th    = static_cast<real>(0.5) * (x2r + x2l);
                    const auto rproj = rmean * std::sin(th);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m)),
                        rproj * dx3 / (std::max(v3p, v3m))
                    });
                }
                break;
            }
        default:
            {
                if constexpr(dim == 1) {
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m))
                    });
                } else if constexpr(dim == 2) {
                    switch (geometry)
                    {
                    case Geometry::AXIS_CYLINDRICAL:
                    {
                        cfl_dt = std::min({       
                            dx1 / (std::max(v1p, v1m)),
                            dx2 / (std::max(v2p, v2m))
                        });
                        break;
                    }
                    
                    default:
                    {
                        const auto rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::CYLINDRICAL);
                        cfl_dt = std::min({       
                            dx1 / (std::max(v1p, v1m)),
                            rmean * dx2 / (std::max(v2p, v2m))
                        });
                        break;
                    }
                    }
                } else {
                    const auto rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::CYLINDRICAL);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m)),
                        dx3 / (std::max(v3p, v3m))
                    });
                }
                break;
            }
        }
        pooling::update_minimum(min_dt, cfl_dt);
    });
    dt = cfl * min_dt;
};

template<int dim>
template<TIMESTEP_TYPE dt_type>
void RMHD<dim>::adapt_dt(const ExecutionPolicy<> &p)
{
    #if GPU_CODE
        if constexpr(dim == 1) {
            // LAUNCH_ASYNC((helpers::compute_dt<rm::Primitive<1>,dt_type>), p.gridSize, p.blockSize, this, prims.data(), dt_min.data());
            helpers::compute_dt<rm::Primitive<1>, dt_type><<<p.gridSize, p.blockSize>>>(this, prims.data(), dt_min.data());
        } else {
            // LAUNCH_ASYNC((helpers::compute_dt<rm::Primitive<dim>,dt_type>), p.gridSize, p.blockSize, this, prims.data(), dt_min.data(), geometry);
            helpers::compute_dt<rm::Primitive<dim>, dt_type><<<p.gridSize,p.blockSize>>>(this, prims.data(), dt_min.data(), geometry);
        }
        // LAUNCH_ASYNC((helpers::deviceReduceWarpAtomicKernel<dim>), p.gridSize, p.blockSize, this, dt_min.data(), active_zones);
        helpers::deviceReduceWarpAtomicKernel<dim><<<p.gridSize, p.blockSize>>>(this, dt_min.data(), active_zones);
        gpu::api::deviceSynch();
    #endif
}
//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
template<int dim>
GPU_CALLABLE_MEMBER
RMHD<dim>::conserved_t RMHD<dim>::prims2flux(const RMHD<dim>::primitive_t &prims, const luint nhat) const
{
    const real rho      = prims.rho;
    const real v1       = prims.vcomponent(1);
    const real v2       = prims.vcomponent(2);
    const real v3       = prims.vcomponent(3);
    const real p        = prims.total_pressure();
    const real b1       = prims.bcomponent(1);
    const real b2       = prims.bcomponent(2);
    const real b3       = prims.bcomponent(3);
    const real chi      = prims.chi;
    const real vn       = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
    const real bn       = (nhat == 1) ? b1 : (nhat == 2) ? b2 : b3;
    const real lorentz_gamma = prims.get_lorentz_factor();

    const real h     = prims.get_gas_enthalpy(gamma);
    const real bsq   = (b1 * b1 + b2 * b2 + b3 * b3);
    const real vdotb = prims.vdotb();
    const real d  = rho * lorentz_gamma;
    const real m1 = (rho * lorentz_gamma * lorentz_gamma * h + bsq) * v1 - vdotb * b1;
    const real m2 = (rho * lorentz_gamma * lorentz_gamma * h + bsq) * v2 - vdotb * b2;
    const real m3 = (rho * lorentz_gamma * lorentz_gamma * h + bsq) * v3 - vdotb * b3;
    const real mn = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;

    const auto b4 = rm::MagFourVec<dim>(prims);
    return rm::Conserved<dim>{
        d  * vn, 
        m1 * vn + helpers::kronecker(nhat, 1) * p - bn * b4.one / lorentz_gamma, 
        m2 * vn + helpers::kronecker(nhat, 2) * p - bn * b4.two / lorentz_gamma, 
        m3 * vn + helpers::kronecker(nhat, 3) * p - bn * b4.three / lorentz_gamma,  
        mn - d * vn, 
        vn * b1 - v1 * bn,
        vn * b2 - v2 * bn,
        vn * b3 - v3 * bn,
        d * vn * chi
    };
};

template<int dim>
GPU_CALLABLE_MEMBER
RMHD<dim>::conserved_t RMHD<dim>::calc_hll_flux(
    const RMHD<dim>::conserved_t &left_state, 
    const RMHD<dim>::conserved_t &right_state,
    const RMHD<dim>::conserved_t &left_flux, 
    const RMHD<dim>::conserved_t &right_flux,
    const RMHD<dim>::primitive_t &left_prims, 
    const RMHD<dim>::primitive_t &right_prims,
    const luint nhat,
    const real vface) const
{
    const rm::Eigenvals<dim> lambda = calc_eigenvals(left_prims, right_prims, nhat);
    // Grab the necessary wave speeds
    const real aL  = lambda.afL;
    const real aR  = lambda.afR;
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;
    
    rm::Conserved<dim> net_flux;
    // Compute the HLL Flux component-wise
    if (vface < aLm) {
        net_flux = left_flux - left_state * vface;
    } else if (vface > aRp) {
        net_flux = right_flux - right_state * vface;
    } else {    
        const auto f_hll = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aLm * aRp) / (aRp - aLm);
        const auto u_hll = (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);
        net_flux = f_hll - u_hll * vface;
    }

    // Upwind the scalar concentration flux
    if (net_flux.d < 0)
        net_flux.chi = right_prims.chi * net_flux.d;
    else
        net_flux.chi = left_prims.chi  * net_flux.d;

    // Compute the HLL Flux component-wise
    return net_flux;
};

template<int dim>
GPU_CALLABLE_MEMBER
RMHD<dim>::conserved_t RMHD<dim>::calc_hllc_flux(
    const RMHD<dim>::conserved_t &left_state,
    const RMHD<dim>::conserved_t &right_state,
    const RMHD<dim>::conserved_t &left_flux,
    const RMHD<dim>::conserved_t &right_flux,
    const RMHD<dim>::primitive_t &left_prims,
    const RMHD<dim>::primitive_t &right_prims,
    const luint nhat,
    const real vface) const 
{

    static auto construct_the_state = [](
            const luint nhat,
            const luint np1,
            const luint np2,
            const real d, 
            const real mnorm, 
            const real mt1, 
            const real mt2, 
            const real tau, 
            const real bnorm, 
            const real bt1, 
            const real bt2
        ) {
            rm::Conserved<dim> u;
            u.d = d;
            u.momentum(nhat) = mnorm;
            u.momentum(np1)  = mt1; 
            u.momentum(np2)  = mt2;
            u.tau = tau;
            u.bcomponent(nhat) = bnorm;
            u.bcomponent(np1)  = bt1;
            u.bcomponent(np2)  = bt2;
            return u;
    };

    const rm::Eigenvals<dim> lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL  = lambda.afL;
    const real aR  = lambda.afR;
    const real aLm = aL < 0 ? aL : 0;
    const real aRp = aR > 0 ? aR : 0;

    //---- Check Wave Speeds before wasting computations
    if (vface <= aLm) {
        return left_flux - left_state * vface;
    } else if (vface >= aRp) {
        return right_flux - right_state * vface;
    }

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux 
        = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aRp * aLm) 
            / (aRp - aLm);

    // get the perpendicular directional unit vectors
    const auto np1 = helpers::next_perm(nhat, 1);
    const auto np2 = helpers::next_perm(nhat, 2);

    // the normal component of the magnetic field is assumed to 
    // be continuos across the interace, so bnL = bnR = bnStar
    const real bnStar  = hll_state.bcomponent(nhat);
    const real bt1Star = hll_state.bcomponent(np1);
    const real bt2Star = hll_state.bcomponent(np2);

    const real uhlld   = hll_state.d;
    const real uhllm1  = hll_state.momentum(1);
    const real uhllm2  = hll_state.momentum(2);
    const real uhllm3  = hll_state.momentum(3);
    const real uhlltau = hll_state.tau;

    const real fhlld   = hll_flux.d;
    const real fhllm1  = hll_flux.momentum(1);
    const real fhllm2  = hll_flux.momentum(2);
    const real fhllm3  = hll_flux.momentum(3);
    const real fhlltau = hll_flux.tau;
    const real fhllb1  = hll_flux.bcomponent(1);
    const real fhllb2  = hll_flux.bcomponent(2);
    const real fhllb3  = hll_flux.bcomponent(3);

    const real e  = uhlltau + uhlld;
    const real s  = (nhat == 1) ? uhllm1 : (nhat == 2) ? uhllm2 : uhllm3;
    const real fe = fhlltau + fhlld;
    const real fs = (nhat == 1) ? fhllm1 : (nhat == 2) ? fhllm2 : fhllm3;
    const real fpb1 = (np1 == 1) ? fhllb1 : (np1 == 2) ? fhllb2 : fhllb3;
    const real fpb2 = (np2 == 1) ? fhllb1 : (np2 == 2) ? fhllb2 : fhllb3;  

    //------Calculate the contact wave velocity and pressure
    const real fdb2 = (bt1Star * fpb1 + bt2Star * fpb2);
    const real a = fe - fdb2;
    const real b = -(e + fs) + (bt1Star * bt1Star + bt2Star * bt2Star) + (fpb1 * fpb1 + fpb2 * fpb2);
    const real c = s - fdb2;
    const real quad  = -static_cast<real>(0.5) * (b + helpers::sgn(b) * std::sqrt(b * b - 4.0 * a * c));
    const real aStar = c * (1 / quad);
    const real vt1Star = (bt1Star * aStar - fpb1) / bnStar; // Eq. (38)
    const real vt2Star = (bt2Star * aStar - fpb2) / bnStar; // Eq. (38)
    const real invg2   = (1 - (aStar * aStar + vt1Star * vt1Star + vt2Star * vt2Star));
    const real vsdB    = (aStar * bnStar + vt1Star * bt1Star + vt2Star * bt2Star);
    const real pStar   = -aStar * (fe - bnStar * vsdB) + fs + bnStar * bnStar * invg2;

    if (vface <= aStar)
    {
        // const real pressure = left_prims.p;
        const real d        = left_state.d;
        const real m1       = left_state.momentum(1);
        const real m2       = left_state.momentum(2);
        const real m3       = left_state.momentum(3);
        const real tau      = left_state.tau;
        // const real chi      = left_state.chi;
        const real e        = tau + d;
        const real cofactor = 1 / (aL - aStar);
        const real mnorm    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;

        const real vL     = left_prims.vcomponent(nhat);
        // Left Star State in x-direction of coordinate lattice
        const real dStar      = cofactor * (aL - vL) * d;
        const real eStar      = cofactor * (aL * e - mnorm + pStar * aStar - vsdB * bnStar);
        const real mnStar     = (eStar + pStar) * aStar - vsdB * bnStar;
        const real mt1Star    = cofactor * (-bnStar * (bt1Star * invg2 + vsdB * vt1Star) + aL * left_state.momentum(np1) - left_flux.momentum(np1));
        const real mt2Star    = cofactor * (-bnStar * (bt2Star * invg2 + vsdB * vt2Star) + aL * left_state.momentum(np2) - left_flux.momentum(np2));
        const real tauStar    = eStar - dStar;
        const auto starStateL = construct_the_state(nhat, np1, np2, d, mnStar, mt1Star, mt2Star, tauStar, bnStar, bt1Star, bt2Star);
        auto hllc_flux        = left_flux + (starStateL - left_state) * aL - starStateL * vface;

        // upwind the concentration flux 
        if (hllc_flux.d < 0)
            hllc_flux.chi = right_prims.chi * hllc_flux.d;
        else
            hllc_flux.chi = left_prims.chi  * hllc_flux.d;

        return hllc_flux;
    } else {
        // const real pressure = right_prims.p;
        const real d        = right_state.d;
        const real m1       = right_state.momentum(1);
        const real m2       = right_state.momentum(2);
        const real m3       = right_state.momentum(3);
        const real tau      = right_state.tau;
        // const real chi      = right_state.chi;
        const real e        = tau + d;
        const real cofactor = 1 / (aR - aStar);
        const real mnorm    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;

        const real vR     = right_prims.vcomponent(nhat);
        // Right Star State in x-direction of coordinate lattice
        const real dStar      = cofactor * (aR - vR) * d;
        const real eStar      = cofactor * (aR * e - mnorm + pStar * aStar - vsdB * bnStar);
        const real mnStar     = (eStar + pStar) * aStar - vsdB * bnStar;
        const real mt1Star    = cofactor * (-bnStar * (bt1Star * invg2 + vsdB * vt1Star) + aR * right_state.momentum(np1) - right_flux.momentum(np1));
        const real mt2Star    = cofactor * (-bnStar * (bt2Star * invg2 + vsdB * vt2Star) + aR * right_state.momentum(np2) - right_flux.momentum(np2));
        const real tauStar    = eStar - dStar;
        const auto starStateR = construct_the_state(nhat, np1, np2, d, mnStar, mt1Star, mt2Star, tauStar, bnStar, bt1Star, bt2Star);
        auto hllc_flux        = right_flux + (starStateR - right_state) * aR - starStateR * vface;

        // upwind the concentration flux 
        if (hllc_flux.d < 0)
            hllc_flux.chi = right_prims.chi * hllc_flux.d;
        else
            hllc_flux.chi = left_prims.chi  * hllc_flux.d;

        return hllc_flux;
    }
};

template<int dim>
GPU_CALLABLE_MEMBER
RMHD<dim>::conserved_t RMHD<dim>::calc_hlld_flux(
    const RMHD<dim>::conserved_t &left_state,
    const RMHD<dim>::conserved_t &right_state,
    const RMHD<dim>::conserved_t &left_flux,
    const RMHD<dim>::conserved_t &right_flux,
    const RMHD<dim>::primitive_t &left_prims,
    const RMHD<dim>::primitive_t &right_prims,
    const luint nhat,
    const real vface,
    const luint gid) const 
{
    // rm::Conserved<dim> ua, uc;
    // const rm::Eigenvals<dim> lambda = calc_eigenvals(left_prims, right_prims, nhat);
    // const real aL  = lambda.afL;
    // const real aR  = lambda.afR;
    // const real aLm = aL < 0 ? aL : 0;
    // const real aRp = aR > 0 ? aR : 0;

    // //---- Check wave speeds before wasting computations
    // if (vface <= aLm) {
    //     return left_flux - left_state * vface;
    // } else if (vface >= aRp) {
    //     return right_flux - right_state * vface;
    // }

    //  //-------------------Calculate the HLL Intermediate State
    // const auto hll_state = 
    //     (right_state * aRp - left_state * aLm - right_flux + left_flux) / (aRp - aLm);

    // //------------------Calculate the RHLLE Flux---------------
    // const auto hll_flux 
    //     = (left_flux * aRp - right_flux * aLm + (right_state - left_state) * aRp * aLm) 
    //         / (aRp - aLm);

    // // define the magnetic field normal to the zone
    // const auto bn = hll_state.bcomponent(nhat);


    // // Eq. (12)
    // const auto rL = left_state * aLm - left_flux;
    // const auto rR = right_state * aRp - right_flux;

    // //==================================================================
    // // Helper functions to ease repetition
    // //==================================================================
    // const real qfunc = [](const rm::Conserved<dim> &r, const luint nhat, const real a, const real p) {
    //     return r.total_energy() * a + p * (1 - a * a);
    // };
    // const real gfunc =[](const luint np1, const luint np2, const rm::Conserved<dim> &r) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     } else if constexpr(dim == 2) {
    //         return (r.bcomponent(np1) * r.bcomponent(np1));
    //     } else {
    //         return (r.bcomponent(np1) * r.bcomponent(np1) + r.bcomponent(np2) * r.bcomponent(np2));
    //     }
    // };
    // const real yfunc = [](const luint np1, const luint np2, const rm::Conserved<dim> &r) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     } else if constexpr(dim == 2) {
    //         return r.bcomponent(np1) * r.momentum(np1);
    //     } else {   
    //         return r.bcomponent(np1) * r.momentum(np1) + r.bcomponent(np2) * r.momentum(np2);
    //     }
    // };
    // const real ofunc = [](const real q, const real g, const real bn, const real a) {
    //     return q - g + bn * bn * (1 - a * a);
    // };
    // const real xfunc = [](const real q, const real y, const real g, const real bn, const real a, const real p, const real et) {
    //     return bn * (q * a * bn + y) - (q + g) * (a * p + et);
    // };
    // const real vnfunc = [](const real bn, const real q, const real a, const real y, const real g, const real p, const real mn, const real x) {
    //     return (bn * (q* bn + a * y) - (q + g) * (p + mn)) / x;
    // };
    // const real vt1func = [](const real o, const real mt1, const real bt1, const real y, const real bn, const real a, const real mn, const real et, const real x) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     };
    //     return (o * mt1 + bt1 * (y + bn * (a * mn - et))) / x;
    // };
    // const real vt2func = [](const real o, const real mt2, const real bt2, const real y, const real bn, const real a, const real mn, const real et, const real x) {
    //     if constexpr(dim < 3) {
    //         return 0;
    //     };
    //     return (o * mt1 + bt2 * (y + bn * (a * mn - et))) / x;
    // };
    // const real btanfunc = [](const real rbk, const real bn, const real vn, const real a) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     };
    //     return (rbk - bn * vn) / (a - vn);
    // };

    // const real total_enthalpy(const real p, const real et, const real vdr, const real a, const real vn) {
    //     return p + (et - vdr) / (a - vn);
    // };

    // const real bkc = [](const real bkL, const real bkR, const real vaL, const real vaR, const real vnL, const real vnR, const real bn, const real vkL, const real vkR) {
    //     return (
    //           bkR * (vaR - vnR) 
    //         - bkL * (vaL - vnL)
    //         + bn  * (vkR - vkL)
    //     ) / (vaR - vaL);
    // };

    // const real vec_dot = [](const real x1, const real x2, const real x3, const real y1, const real y2, const real y3) {
    //     x1 * y1 + x2 * y2 + x3 * y3;
    // };

    // const real vec_sq = [](const real x1, const real x2, const real x3) {
    //     return x1 *x1 + x2 * x2 + x3 * x3;
    // };

    // const rm::Conserved<dim> construct_the_state = [](
    //     const luint nhat,
    //     const luint np1,
    //     const luint np2
    //     const real d, 
    //     const real vfac, 
    //     const real et, 
    //     const real p, 
    //     const real vn, 
    //     const real vdb, 
    //     const real bn, 
    //     const real bp1, 
    //     const real bp2,
    //     const real vp1,
    //     const real vp2
    // ) {
    //     rm::Conserved<dim> u;
    //     u.d = d * vfac;
    //     u.momentum(nhat) = (et + p) * vn - vdb * bn;
    //     if constexpr(dim > 1) {
    //         u.momentum(np1 > dim ? 1 : np1) = (et + p) * vp1 - vdb * bp1;
    //     }
    //     if constexpr(dim > 2) {
    //         u.momentum(np2) = (et + p) * vp2 - vdb * bp2;
    //     }
    //     u.tau = et - u.d;
    //     u.bcomponent(nhat) = bn;
    //     if constexpr(dim > 1) {
    //         u.bcomponent(np1 > dim ? 1 : np1) = bp1;
    //     }
    //     if constexpr(dim > 2) {
    //         u.bcomponent(np2) = bp2;
    //     }
    //     return u;
    // };

    // //==============================================================================
    // // initial pressure guess
    // real p0 = 0;
    // if (bn * bn / (pguess * pguess) < 0.01) {
    //     const real a = aRp - aLm;
    //     const real b = rR.total_energy() - rL.total_energy() + aRp * rL - aLm * rR;
    //     const real c = rL.momentum(nhat) * rR.total_energy() - rR.momentum(nhat) * rL.total_energy();
    //     const real quad = std::max(static_cast<real>(0), b * b - 4 * a * c);
    //     p0 = 0.5 * (-b + std::sqrt(quad)) / (aRp - aLm);
    // } else {
    //     const auto phll = cons2prim(hll_state, gid);
    //     p0 = phll.total_pressure();
    // }
    // //----------------- Jump conditions across the fast waves (section 3.1)
    // const auto np1  = helpers::next_perm(nhat, 1);
    // const auto np2  = helpers::next_perm(nhat, 2);

    // // left side
    // const auto pL   = left_prims.total_pressure();
    // const auto qL   = qfunc(rL, nhat, aLm, pL);;
    // const auto gL   = gfunc(np1, np2, rL);
    // const auto yL   = yfunc(np1, np2, rL);
    // const auto oL   = ofunc(qL, gL, bn, aLm);
    // const auto xL   = xfunc(qL, yL, gL, bn, aLm, pL, rL.total_energ());
    // // velocity components
    // const auto vnL   = vnfunc(bn, qL, aLm, yL, gL, pL, mnL, xL);
    // const auto vt1L  = vt1func(oL, rL.momentum(np1), rL.bcomponent(np1), yL, bn, aLm, rL.momentum(nhat), rL.total_energy(), xL); 
    // const auto vt2L  = vt2func(oL, rL.momentum(np2), rL.bcomponent(np2), yL, bn, aLm, rL.momentum(nhat), rL.total_energy(), xL);
    // const auto bp1L  = btanfunc(rL.bcomponent(np1), bn, vnL, vt1L, aLm); 
    // const auto bp2L  = btanfunc(rL.bcomponent(np2), bn, vnL, vt2L, aLm);
    // const auto vdrL  = vnL * rL.momentum(nhat) + vt1L * rL.momentum(np1) + vt2L * rL.momentum(np2);
    // const auto wL    = total_enthalpy(pL, rL.total_energy(), vdr, aLm, vnL);
    // const auto vdbL  = (vnL * bn + vnL1 * bp1 + vnL2 * bp2);
    // const auto vfacL = 1 / (aLm - vnL);

    // // right side
    // const auto pR   = right_prims.total_pressure();
    // const auto qR   = qfunc(rR, nhat, aRm, pR);;
    // const auto gR   = gfunc(np1, np2, rR);
    // const auto yR   = yfunc(np1, np2, rR);
    // const auto oR   = ofunc(qR, gR, bn, aRm);
    // const auto xR   = xfunc(qR, yR, gR, bn, aRm, pR, rR.total_energ());
    // // velocity components
    // const auto vnR   = vnfunc(bn, qR, aRm, yR, gR, pR, mnR, xR);
    // const auto vt1R  = vt1func(oR, rR.momentum(np1), rR.bcomponent(np1), yR, bn, aRm, rR.momentum(nhat), rR.total_energy(), xR); 
    // const auto vt2R  = vt2func(oR, rR.momentum(np2), rR.bcomponent(np2), yR, bn, aRm, rR.momentum(nhat), rR.total_energy(), xR);
    // const auto bp1R  = btanfunc(rR.bcomponent(np1), bn, vnR, vt1R, aRm); 
    // const auto bp2R  = btanfunc(rR.bcomponent(np2), bn, vnR, vt2R, aRm);
    // const auto vdrR  = vnR * rR.momentum(nhat) + vt1R * rR.momentum(np1) + vt2R * rR.momentum(np2);
    // const auto wR    = total_enthalpy(pR, rR.total_energy(), vdr, aRm, vnR);
    // const auto vdbR  = (vnR * bn + vnR1 * bp1 + vnR2 * bp2);
    // const auto vfacR = 1 / (aRm - vnR);

    // //--------------Jump conditions across the Alfven waves (section 3.2)
    // const auto etaL = - helpers::sgn(bn) * std::sqrt(wL);
    // const auto etaR =   helpers::sgn(bn) * std::sqrt(wR);
    // const auto calc_kcomp = (const int nhat, const int ehat, const rm::Conserved<dim> &r, const real p, const real a, const real eta) {
    //     return (r.momentum(nhat) + p * helpers::kronecker(ehat, nhat) + r.bcomponent(ehat) * eta) / (a * p + r.total_energy() + bn * eta);
    // }
    // const auto knL  = calc_kcomp(nhat, nhat, rL, pL, aLm, etaL);
    // const auto knR  = calc_kcomp(nhat, nhat, rR, pR, aRm, etaR);
    // const auto kt1L = calc_kcomp(nhat, np1, rL,  pL, aLm, etaL);
    // const auto kt1R = calc_kcomp(nhat, np1, rR,  pR, aRm, etaR);
    // const auto kt2L = calc_kcomp(nhat, np2, rL,  pL, aLm, etaL);
    // const auto kt2R = calc_kcomp(nhat, np2, rR,  pR, aRp, etaR);
    // // the k-normal is the Alfven wave speed
    // const auto vaL = knL;
    // const auto vaR = knR;
    // if (aLm - vaL < vface) { // return FaL
    //     ua = construct_the_state(
    //         nhat, 
    //         np1, 
    //         np2, 
    //         rL.d, 
    //         vfacL, 
    //         rL.total_energy(), 
    //         pL, 
    //         vnL, 
    //         vdbL, 
    //         bn, 
    //         bp1L, 
    //         bp2L, 
    //         vt1L, 
    //         vt2L
    //     );
    //     return left_flux + (ua - left_state) * vaL - ua * vface;
    // } else if (vaR - aRp < vface) { // return FaR
    //     ua = construct_the_state(
    //         nhat, 
    //         np1, 
    //         np2, 
    //         rR.d, 
    //         vfacR, 
    //         rR.total_energy(), 
    //         pR, 
    //         vnR, 
    //         vdbR, 
    //         bn, 
    //         bp1R, 
    //         bp2R, 
    //         vt1R, 
    //         vt2R
    //     );

    //     return right_flux + (ua - right_state) * vaR - ua * vface;
    // } else {
    //     dK  = 1 / (vaR - vaL);
    //     //---------------Jump conditions across the contact wave (section 3.3)
    //     const auto bkxn  = bn;
    //     const auto bkc1  = bkc(uaL.bcomponent(np1), uaR.bcomponent(np1), vaL, vaR, vnL, vnR, vt1L, vt1R) * dK;
    //     const auto bkc2  = bkc(uaL.bcomponent(np2), uaR.bcomponent(np2), vaL, vaR, vnL, vnR, vt2L, vt2R) * dK;
    //     const auto kdbL  = vec_dot(bkxn, bkc1, bkc2, knL, kt1L, kt2L);
    //     const auto kdbR  = vec_dot(bkxn, bkc1, bkc2, knR, kt1R, kt2R);
    //     const auto ksqL  = vec_sq(knL, kt1L, kt2L);
    //     const auto ksqR  = vec_sq(knR, kt1R, kt2R);
    //     const auto kfacL = (1 - ksqL) / (etaL - kdbL);
    //     const auto kfacR = (1 - ksqR) / (etaR - kdbR);
    //     const auto vanL  = knL  -  bn * kfacL;
    //     const auto vat1L = kt1L - bkc1 * kfacL;
    //     const auto vat2L = kt2L - bkc2 * kfacL;
    //     const auto vanR  = knR  -  bn * kfacR;
    //     const auto vat1R = kt1R - bkc1 * kfacR;
    //     const auto vat2R = kt2R - bkc2 * kfacR;
    //     const auto vakn = 0.5 * (vanL + vanR);
    //     const auto vat1 = 0.5 * (vat1L + vat1R);
    //     const auto vat2 = 0.5 * (vat2L + vat2R);
    //     const auto vdbc = vec_dot(vakn, vat1, vat2, bkxn, bkc1, bkc2);
    //     if (vakn > 0) {
    //         ua = construct_the_state(
    //             nhat, 
    //             np1, 
    //             np2, 
    //             rL.d, 
    //             vfacL, 
    //             rL.total_energy(), 
    //             pL, 
    //             vnL, 
    //             vdbL, 
    //             bn, 
    //             bp1L, 
    //             bp2L, 
    //             vt1L, 
    //             vt2L
    //         );
    //         const real etc  = (vaL * ua.total_energy() - ua.momentum(nhat) + pL * vakn - vdbc * bn) / (vaL - vakn);
    //         uc = construct_the_state(
    //             nhat, 
    //             np1, 
    //             np2, 
    //             ua.d, 
    //             (vaL - vnL) / (vaL - vakn), 
    //             etc, 
    //             pL, 
    //             vnL, 
    //             vdbc, 
    //             bn, 
    //             bkc1, 
    //             bkc2, 
    //             vat1L, 
    //             vat2L
    //         );

    //         const auto fa = left_flux + (ua - left_state) * vaL;
    //         return fa + (uc - ua) * vakn - uc * vface;
    //     } else {
    //         ua = construct_the_state(
    //             nhat, 
    //             np1, 
    //             np2, 
    //             rL.d, 
    //             vfacR, 
    //             rR.total_energy(), 
    //             pR, 
    //             vnR, 
    //             vdbR, 
    //             bn, 
    //             bp1R, 
    //             bp2R, 
    //             vt1R, 
    //             vt2R
    //         );
    //         const real etc  = (vaR * uaR.total_energy() - uaR.momentum(nhat) + pR * vakn - vdbc * bnR) / (vaR - vakn);
    //         uc = construct_the_state(
    //             nhat, 
    //             np1, 
    //             np2, 
    //             ua.d, 
    //             (vaR - vnR) / (vaR - vakn), 
    //             etc, 
    //             pR, 
    //             vnR, 
    //             vdbc, 
    //             bn, 
    //             bkc1, 
    //             bkc2, 
    //             vat1R, 
    //             vat2R
    //         );
    //         const auto fa = right_flux + (ua - right_state) * vaR;
    //         return fa + (uc - ua) * vakn - uc * vface;
    //     }
    // }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template<int dim>
void RMHD<dim>::advance(
    const ExecutionPolicy<> &p,
    const luint sx,
    const luint sy)
{
    const luint xpg = this->xactive_grid;
    const luint ypg = this->yactive_grid;
    const luint zpg = this->zactive_grid;

    const luint extent      = p.get_full_extent();
    auto* const cons_data   = cons.data();
    const auto* const prim_data   = prims.data();
    const auto* const dens_source = density_source.data();
    const auto* const mom1_source = m1_source.data();
    const auto* const mom2_source = m2_source.data();
    const auto* const mom3_source = m3_source.data();
    const auto* const mag1_source = sourceB1.data();
    const auto* const mag2_source = sourceB2.data();
    const auto* const mag3_source = sourceB3.data();
    const auto* const erg_source  = energy_source.data();
    const auto* const object_data = object_pos.data();
    const auto* const grav1_source = sourceG1.data();
    const auto* const grav2_source = sourceG2.data();
    const auto* const grav3_source = sourceG3.data();

    simbi::parallel_for(p, (luint)0, extent, [
        sx,
        sy,
        p,
        prim_data,
        cons_data,
        dens_source,
        mom1_source,
        mom2_source,
        mom3_source,
        mag1_source,
        mag2_source,
        mag3_source,
        erg_source,
        object_data,
        grav1_source,
        grav2_source,
        grav3_source,
        xpg,
        ypg,
        zpg,
        this
    ] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        auto prim_buff = shared_memory_proxy<rm::Primitive<dim>>();
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint kk  = dim < 3 ? 0 : (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::helpers::get_height(idx, xpg, ypg);
        const luint jj  = dim < 2 ? 0 : (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::helpers::get_row(idx, xpg, ypg, kk);
        const luint ii  = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::helpers::get_column(idx, xpg, ypg, kk);
        #if GPU_CODE
        if constexpr(dim == 1) {
            if (ii >= xpg) return;
        } else if constexpr(dim == 2) {
            if ((ii >= xpg) || (jj >= ypg)) return;
        } else {
            if ((ii >= xpg) || (jj >= ypg) || (kk >= zpg)) return;
        }
        #endif 

        const luint ia  = ii + radius;
        const luint ja  = dim < 2 ? 0 : jj + radius;
        const luint ka  = dim < 3 ? 0 : kk + radius;
        const luint tx  = (BuildPlatform == Platform::GPU) ? threadIdx.x : 0;
        const luint ty  = dim < 2 ? 0 : (BuildPlatform == Platform::GPU) ? threadIdx.y : 0;
        const luint tz  = dim < 3 ? 0 : (BuildPlatform == Platform::GPU) ? threadIdx.z : 0;
        const luint txa = (BuildPlatform == Platform::GPU) ? tx + radius : ia;
        const luint tya = dim < 2 ? 0 : (BuildPlatform == Platform::GPU) ? ty + radius : ja;
        const luint tza = dim < 3 ? 0 : (BuildPlatform == Platform::GPU) ? tz + radius : ka;

        rm::Conserved<dim> uxL, uxR, uyL, uyR, uzL, uzR;
        rm::Conserved<dim> fL, fR, gL, gR, hL, hR, frf, flf, grf, glf, hrf, hlf;
        rm::Primitive<dim> xprimsL, xprimsR, yprimsL, yprimsR, zprimsL, zprimsR;

        const luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            if constexpr(dim == 1) {
                luint txl = p.blockSize.x;
                // Check if the active index exceeds the active zones
                // if it does, then this thread buffer will taken on the
                // ghost index at the very end and return
                prim_buff[txa] = prim_data[ia];
                if (threadIdx.x < radius)
                {
                    if (blockIdx.x == p.gridSize.x - 1 && (ia + p.blockSize.x > nx - radius + threadIdx.x)) {
                        txl = nx - radius - ia + threadIdx.x;
                    }
                    prim_buff[txa - radius] = prim_data[ia - radius];
                    prim_buff[txa + txl]        = prim_data[ia + txl];
                }
                simbi::gpu::api::synchronize();
            } else if constexpr(dim == 2) {
                luint txl = p.blockSize.x;
                luint tyl = p.blockSize.y;
                // Load Shared memory into buffer for active zones plus ghosts
                prim_buff[tya * sx + txa * sy] = prim_data[aid];
                if (ty < radius)
                {
                    if (blockIdx.y == p.gridSize.y - 1 && (ja + p.blockSize.y > ny - radius + ty)) {
                        tyl = ny - radius - ja + ty;
                    }
                    prim_buff[(tya - radius) * sx + txa] = prim_data[(ja - radius) * nx + ia];
                    prim_buff[(tya + tyl) * sx + txa]    = prim_data[(ja + tyl) * nx + ia]; 
                }
                if (tx < radius)
                {   
                    if (blockIdx.x == p.gridSize.x - 1 && (ia + p.blockSize.x > nx - radius + tx)) {
                        txl = nx - radius - ia + tx;
                    }
                    prim_buff[tya * sx + txa - radius] =  prim_data[ja * nx + (ia - radius)];
                    prim_buff[tya * sx + txa +    txl] =  prim_data[ja * nx + (ia + txl)]; 
                }
                simbi::gpu::api::synchronize();

            } else {
                luint txl = p.blockSize.x;
                luint tyl = p.blockSize.y;
                luint tzl = p.blockSize.z;
                // Load Shared memory into buffer for active zones plus ghosts
                prim_buff[tza * sx * sy + tya * sx + txa] = prim_data[aid];
                if (tz == 0)    
                {
                    if ((blockIdx.z == p.gridSize.z - 1) && (ka + p.blockSize.z > nz - radius + tz)) {
                        tzl = nz - radius - ka + tz;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = tzl + q - 1;
                        prim_buff[(tza - q) * sx * sy + tya * sx + txa]  = prim_data[(ka - q) * nx * ny + ja * nx + ia];
                        prim_buff[(tza + re) * sx * sy + tya * sx + txa] = prim_data[(ka + re) * nx * ny + ja * nx + ia];
                    } 
                }
                if (ty == 0)    
                {
                    if ((blockIdx.y == p.gridSize.y - 1) && (ja + p.blockSize.y > ny - radius + ty)) {
                        tyl = ny - radius - ja + ty;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = tyl + q - 1;
                        prim_buff[tza * sx * sy + (tya - q) * sx + txa]  = prim_data[ka * nx * ny + (ja - q) * nx + ia];
                        prim_buff[tza * sx * sy + (tya + re) * sx + txa] = prim_data[ka * nx * ny + (ja + re) * nx + ia];
                    } 
                }
                if (tx == 0)
                {   
                    if ((blockIdx.x == p.gridSize.x - 1) && (ia + p.blockSize.x > nx - radius + tx)) {
                        txl = nx - radius - ia + tx;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = txl + q - 1;
                        prim_buff[tza * sx * sy + tya * sx + txa - q]  =  prim_data[ka * nx * ny + ja * nx + ia - q];
                        prim_buff[tza * sx * sy + tya * sx + txa + re] =  prim_data[ka * nx * ny + ja * nx + ia + re]; 
                    }
                }
                simbi::gpu::api::synchronize();
            }
        #endif

        const bool object_to_my_left  = dim < 2 ? false : object_data[kk * xpg * ypg + jj * xpg +  helpers::my_max(static_cast<lint>(ii - 1), static_cast<lint>(0))];
        const bool object_to_my_right = dim < 2 ? false : object_data[kk * xpg * ypg + jj * xpg +  helpers::my_min(ii + 1,  xpg - 1)];
        const bool object_in_front    = dim < 2 ? false : object_data[kk * xpg * ypg + helpers::my_min(jj + 1, ypg - 1) * xpg +  ii];
        const bool object_behind      = dim < 2 ? false : object_data[kk * xpg * ypg + helpers::my_max(static_cast<lint>(jj - 1), static_cast<lint>(0)) * xpg + ii];
        const bool object_above_me    = dim < 3 ? false : object_data[helpers::my_min(kk + 1, zpg - 1)  * xpg * ypg + jj * xpg +  ii];
        const bool object_below_me    = dim < 3 ? false : object_data[helpers::my_max(static_cast<lint>(kk - 1), static_cast<lint>(0)) * xpg * ypg + jj * xpg +  ii];

        const real x1l    = get_x1face(ii, 0);
        const real x1r    = get_x1face(ii, 1);
        const real vfaceL = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
        const real vfaceR = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;

        if (first_order) [[unlikely]] {
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            if constexpr(dim > 1) {
                //j+1/2
                yprimsL = prim_buff[tza * sx * sy + (tya + 0) * sx + txa];
                yprimsR = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
            }
            if constexpr(dim > 2) {
                //k+1/2
                zprimsL = prim_buff[(tza + 0) * sx * sy + tya * sx + txa];
                zprimsR = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
            }

            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                if constexpr(dim > 1) {
                    xprimsR.v2  =  xprimsL.v2;
                }
                if constexpr(dim > 2){
                    xprimsR.v3  =  xprimsL.v3;
                }
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                if constexpr(dim > 1) {
                    yprimsR.v2  = -yprimsL.v2;
                }
                if constexpr(dim > 2){
                    yprimsR.v3  =  yprimsL.v3;
                }
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                if constexpr(dim == 3) {
                    zprimsR.v2  =  zprimsL.v2;
                    zprimsR.v3  = -zprimsL.v3;
                }
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            }
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }
            // Calc HLL Flux at i+1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                if constexpr(dim == 1) {
                    frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    break;
                } else {
                    if(quirk_smoothing)
                    {
                        if (helpers::quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                            frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        } else {
                            frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        }

                        if (helpers::quirk_strong_shock(yprimsL.p, yprimsR.p)){
                            grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        } else {
                            grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        }

                        if constexpr(dim > 2) {
                            if (helpers::quirk_strong_shock(zprimsL.p, zprimsR.p)){
                                hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            } else {
                                hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            }
                        }
                        break;
                    } else {
                        frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);

                        if constexpr(dim > 2) {
                            hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                        }
                        break;
                    }
                }
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1) {
                    grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            if constexpr(dim > 1) {
                //j+1/2
                yprimsL = prim_buff[tza * sx * sy + (tya - 1) * sx + txa]; 
                yprimsR = prim_buff[tza * sx * sy + (tya + 0) * sx + txa]; 
            }
            if constexpr(dim > 2) {
                //k+1/2
                zprimsL = prim_buff[(tza - 1) * sx * sy + tya * sx + txa]; 
                zprimsR = prim_buff[(tza - 0) * sx * sy + tya * sx + txa]; 
            }

            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                if constexpr(dim > 1){
                    xprimsL.v2  =  xprimsR.v2;
                }
                if constexpr(dim > 2){
                    xprimsL.v3  =  xprimsR.v3;
                }
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                if constexpr(dim > 1) {
                    yprimsL.v2  = -yprimsR.v2;
                }
                if constexpr(dim > 2){
                    yprimsL.v3  =  yprimsR.v3;
                }
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                if constexpr(dim == 3) {
                    zprimsL.v2  =  zprimsR.v2;
                    zprimsL.v3  = -zprimsR.v3;
                }
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }
            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            } 
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            // Calc HLL Flux at i-1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                if constexpr(dim == 1) {
                    flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    break;
                } else {
                    if(quirk_smoothing)
                    {
                        if (helpers::quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                            flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        } else {
                            flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        }

                        if (helpers::quirk_strong_shock(yprimsL.p, yprimsR.p)){
                            glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        } else {
                            glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        }

                        if constexpr(dim > 2) {
                            if (helpers::quirk_strong_shock(zprimsL.p, zprimsR.p)){
                                hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            } else {
                                hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            }
                        }
                        break;
                    } else {
                        flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);

                        if constexpr(dim > 2) {
                            hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                        }
                        break;
                    }
                }
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1) {
                    glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0); 
                }
                break;
            }
        } else{
            // Coordinate X
            const rm::Primitive<dim> xleft_most  = prim_buff[tza * sx * sy + tya * sx + (txa - 2)];
            const rm::Primitive<dim> xleft_mid   = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            const rm::Primitive<dim> center      = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            const rm::Primitive<dim> xright_mid  = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            const rm::Primitive<dim> xright_most = prim_buff[tza * sx * sy + tya * sx + (txa + 2)];
            rm::Primitive<dim> yleft_most, yleft_mid, yright_mid, yright_most;
            rm::Primitive<dim> zleft_most, zleft_mid, zright_mid, zright_most;
            // Reconstructed left X rm::Primitive<dim> vector at the i+1/2 interface
            xprimsL  = center     + helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta)   * static_cast<real>(0.5); 
            xprimsR  = xright_mid - helpers::plm_gradient(xright_mid, center, xright_most, plm_theta) * static_cast<real>(0.5);

            // Coordinate Y
            if constexpr(dim > 1){
                yleft_most  = prim_buff[tza * sx * sy + (tya - 2) * sx + txa];
                yleft_mid   = prim_buff[tza * sx * sy + (tya - 1) * sx + txa];
                yright_mid  = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
                yright_most = prim_buff[tza * sx * sy + (tya + 2) * sx + txa];
                yprimsL  = center     + helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta)   * static_cast<real>(0.5);  
                yprimsR  = yright_mid - helpers::plm_gradient(yright_mid, center, yright_most, plm_theta) * static_cast<real>(0.5);
            }

            // Coordinate z
            if constexpr(dim > 2){
                zleft_most  = prim_buff[(tza - 2) * sx * sy + tya * sx + txa];
                zleft_mid   = prim_buff[(tza - 1) * sx * sy + tya * sx + txa];
                zright_mid  = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
                zright_most = prim_buff[(tza + 2) * sx * sy + tya * sx + txa];
                zprimsL  = center     + helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta)   * static_cast<real>(0.5);  
                zprimsR  = zright_mid - helpers::plm_gradient(zright_mid, center, zright_most, plm_theta) * static_cast<real>(0.5);
            }

            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                if constexpr(dim > 1) {
                    xprimsR.v2  =  xprimsL.v2;
                }
                if constexpr(dim > 2){
                    xprimsR.v3  =  xprimsL.v3;
                }
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                if constexpr(dim > 1) {
                    yprimsR.v2  = -yprimsL.v2;
                }
                if constexpr(dim > 2) {
                    yprimsR.v3  =  yprimsL.v3;
                }
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                if constexpr(dim == 3) {
                    zprimsR.v2  =  zprimsL.v2;
                    zprimsR.v3  = -zprimsL.v3;
                }
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            // Calculate the left and right states using the reconstructed PLM
            // rm::Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            }
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            switch (sim_solver)
            {
            case Solver::HLLC:
                if constexpr(dim == 1) {
                    frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                    break;
                } else {
                    if(quirk_smoothing)
                    {
                        if (helpers::quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                            frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        } else {
                            frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        }

                        if (helpers::quirk_strong_shock(yprimsL.p, yprimsR.p)){
                            grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        } else {
                            grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        }

                        if constexpr(dim > 2) {
                            if (helpers::quirk_strong_shock(zprimsL.p, zprimsR.p)){
                                hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            } else {
                                hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            }
                        }
                        break;
                    } else {
                        frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                        grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);

                        if constexpr(dim > 2) {
                            hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                        }
                        break;
                    }
                }
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1) {
                    grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            xprimsL  = xleft_mid  + helpers::plm_gradient(xleft_mid, xleft_most, center, plm_theta) * static_cast<real>(0.5); 
            xprimsR  = center     - helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta) * static_cast<real>(0.5);
            if constexpr(dim > 1) {
                yprimsL  = yleft_mid  + helpers::plm_gradient(yleft_mid, yleft_most, center, plm_theta) * static_cast<real>(0.5); 
                yprimsR  = center     - helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta) * static_cast<real>(0.5);
            }
            if constexpr(dim > 2) {
                zprimsL  = zleft_mid  + helpers::plm_gradient(zleft_mid, zleft_most, center, plm_theta) * static_cast<real>(0.5);
                zprimsR  = center     - helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta) * static_cast<real>(0.5);
            }

            
            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                if constexpr(dim > 1){
                    xprimsL.v2  =  xprimsR.v2;
                }
                if constexpr(dim > 2){
                    xprimsL.v3  =  xprimsR.v3;
                }
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                if constexpr(dim > 1) {
                    yprimsL.v2  = -yprimsR.v2;
                }
                if constexpr(dim > 2){
                    yprimsL.v3  =  yprimsR.v3;
                }
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                if constexpr(dim == 3) {
                    zprimsL.v2  =  zprimsR.v2;
                    zprimsL.v3  = -zprimsR.v3;
                }
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            // Calculate the left and right states using the reconstructed PLM rm::Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }
            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            } 
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            switch (sim_solver)
            {
            case Solver::HLLC:
                if constexpr(dim == 1) {
                    flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                    break;
                } else {
                    if(quirk_smoothing)
                    {
                        if (helpers::quirk_strong_shock(xprimsL.p, xprimsR.p) ){
                            flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        } else {
                            flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        }

                        if (helpers::quirk_strong_shock(yprimsL.p, yprimsR.p)){
                            glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        } else {
                            glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0.0);
                        }

                        if constexpr(dim > 2) {
                            if (helpers::quirk_strong_shock(zprimsL.p, zprimsR.p)){
                                hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            } else {
                                hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0.0);
                            }
                        }
                        break;
                    } else {
                        flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                        glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);

                        if constexpr(dim > 2) {
                            hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                        }
                        break;
                    }
                }
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1) {
                    glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0); 
                }
                break;
            }
        }// end else 

        //Advance depending on geometry
        const luint real_loc = kk * xpg * ypg + jj * xpg + ii;
        const real d_source  = den_source_all_zeros     ? 0.0 : dens_source[real_loc];
        const real s1_source = mom1_source_all_zeros    ? 0.0 : mom1_source[real_loc];
        const real e_source  = energy_source_all_zeros  ? 0.0 : erg_source[real_loc];
        const real b1_source = mag1_source_all_zeros  ? 0.0 : mag1_source[real_loc];
        
        const auto source_terms = [&]{
            const real s2_source = mom2_source_all_zeros ? 0.0 : mom2_source[real_loc];
            const real s3_source = mom3_source_all_zeros ? 0.0 : mom3_source[real_loc];
            const real b2_source = mag2_source_all_zeros ? 0.0 : mag2_source[real_loc];
            const real b3_source = mag3_source_all_zeros ? 0.0 : mag3_source[real_loc];
            return rm::Conserved<dim>{d_source, s1_source, s2_source, s3_source, e_source, b1_source, b2_source, b3_source} * time_constant;
        }();

        // Gravity
        const auto gs1_source = zero_gravity1 ? 0 : grav1_source[real_loc] * cons_data[aid].d;
        const auto tid = tza * sx * sy + tya * sx + txa;
        const auto gravity = [&]{
            const auto gs2_source = zero_gravity2 ? 0 : grav2_source[real_loc] * cons_data[aid].d;
            const auto gs3_source = zero_gravity3 ? 0 : grav3_source[real_loc] * cons_data[aid].d;
            const auto ge_source  = gs1_source * prim_buff[tid].v1 + gs2_source * prim_buff[tid].v2 + gs3_source * prim_buff[tid].v3;
            return rm::Conserved<dim>{0, gs1_source, gs2_source, gs3_source, ge_source, 0, 0, 0};
        }();

        if constexpr(dim == 1) {
            switch(geometry)
            {
                case simbi::Geometry::CARTESIAN:
                {
                    // printf("flf: %.2e, frf: %.2e, cons: %.2e\n", flf.d, frf.d, cons_data[ia].d);
                    cons_data[ia] -= ((frf - flf) * invdx1 - source_terms - gravity) * dt * step;
                    break;
                }
                default:
                {
                    const real rlf    = x1l + vfaceL * step * dt; 
                    const real rrf    = x1r + vfaceR * step * dt;
                    const real rmean  = helpers::get_cell_centroid(rrf, rlf, geometry);
                    const real sR     = 4.0 * M_PI * rrf * rrf; 
                    const real sL     = 4.0 * M_PI * rlf * rlf; 
                    const real dV     = 4.0 * M_PI * rmean * rmean * (rrf - rlf);    
                    const real factor = (mesh_motion) ? dV : 1;         
                    const real pc     = prim_buff[txa].total_pressure();
                    const real invdV  = 1 / dV;
                    const auto geom_sources = rm::Conserved<1>{0.0, pc * (sR - sL) * invdV, 0.0, 0.0};
                    
                    cons_data[ia] -= ( (frf * sR - flf * sL) * invdV - geom_sources - source_terms - gravity) * step * dt * factor;
                    break;
                }
            } // end switch
        } else if constexpr(dim == 2) {
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                {
                    cons_data[aid] -= ( (frf - flf) * invdx1 + (grf - glf) * invdx2 - source_terms - gravity) * step * dt;
                    break;
                }
                
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl           = x1l + vfaceL * step * dt; 
                        const real rr           = x1r + vfaceR * step * dt;
                        const real rmean        = helpers::get_cell_centroid(rr, rl, geometry);
                        const real tl           = helpers::my_max<real>(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                        const real tr           = helpers::my_min<real>(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dcos         = std::cos(tl) - std::cos(tr);
                        const real dV           = 2.0 * M_PI * (1.0 / 3.0) * (rr * rr * rr - rl * rl * rl) * dcos;
                        const real invdV        = 1.0 / dV;
                        const real s1R          = 2.0 * M_PI * rr * rr * dcos; 
                        const real s1L          = 2.0 * M_PI * rl * rl * dcos; 
                        const real s2R          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tr);
                        const real s2L          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tl);
                        const real factor       = (mesh_motion) ? dV : 1;  

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].get_v1();
                        const real vc   = prim_buff[tid].get_v2();
                        const real pc   = prim_buff[tid].total_pressure();
                        const real hc   = prim_buff[tid].get_gas_enthalpy(gamma);
                        const auto b4c  = rm::MagFourVec<dim>(prim_buff[tid]);
                        const real gam2 = 1/(1 - (uc * uc + vc * vc));

                        const rm::Conserved<2> geom_source  = {
                            0, 
                            (rhoc * hc * gam2 * vc * vc - b4c.two * b4c.two) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * hc * gam2 * uc * vc - b4c.one * b4c.two) / rmean + pc * (s2R - s2L) * invdV, 
                            0,
                            0,
                            0
                        };

                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step * factor;
                        break;
                    }
                case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt; 
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = helpers::get_cell_centroid(rr, rl, simbi::Geometry::PLANAR_CYLINDRICAL);
                        // const real tl           = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2 , x2min);
                        // const real tr           = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dV        = rmean * (rr - rl) * dx2;
                        const real invdV        = 1.0 / dV;
                        const real s1R          = rr * dx2; 
                        const real s1L          = rl * dx2; 
                        const real s2R          = (rr - rl); 
                        const real s2L          = (rr - rl); 

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].get_v1();
                        const real vc   = prim_buff[tid].get_v2();
                        const real pc   = prim_buff[tid].total_pressure();
                        const auto b4c  = rm::MagFourVec<dim>(prim_buff[tid]);
                        
                        const real hc   = prim_buff[tid].get_gas_enthalpy(gamma);
                        const real gam2 = 1/(1 - (uc * uc + vc * vc));
                        const rm::Conserved<2> geom_source  = {
                            0, 
                            (rhoc * hc * gam2 * vc * vc - b4c.two * b4c.two) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * hc * gam2 * uc * vc - b4c.one * b4c.two) / rmean, 
                            0,
                            0,
                            0
                        };
                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms 
                            - gravity
                        ) * dt * step;
                        break;
                    }
                default:
                    {
                        const real rl           = x1l + vfaceL * step * dt; 
                        const real rr           = x1r + vfaceR * step * dt;
                        const real rmean        = helpers::get_cell_centroid(rr, rl, simbi::Geometry::AXIS_CYLINDRICAL);
                        const real dV           = rmean * (rr - rl) * dx2;
                        const real invdV        = 1.0 / dV;
                        const real s1R          = rr * dx2; 
                        const real s1L          = rl * dx2; 
                        const real s2R          = rmean * (rr - rl); 
                        const real s2L          = rmean * (rr - rl);  

                        // Grab central primitives
                        const real pc   = prim_buff[tid].total_pressure();
                        const auto b4c  = rm::MagFourVec<dim>(prim_buff[tid]);
                        const auto geom_source  = rm::Conserved<2>{
                            0, 
                            (- b4c.two * b4c.two) / rmean + pc * (s1R - s1L) * invdV, 
                            (+ b4c.one * b4c.two) / rmean, 
                            0,
                            0,
                            0
                        };
                        
                        cons_data[aid] -= ( 
                              (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step;
                        break;
                    }
            } // end switch
        } else {
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons_data[aid] -= ( (frf  - flf ) * invdx1 + (grf - glf) * invdx2 + (hrf - hlf) * invdx3 - source_terms - gravity) * dt * step;
                        break;
                    }
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl     = x1l + vfaceL * step * dt; 
                        const real rr     = x1r + vfaceR * step * dt;
                        const real tl     = get_x2face(jj, 0);
                        const real tr     = get_x2face(jj, 1); 
                        const real ql     = get_x3face(kk, 0);
                        const real qr     = get_x3face(kk, 1); 
                        const real rmean  = helpers::get_cell_centroid(rr, rl, simbi::Geometry::SPHERICAL);
                        const real s1R    = rr * rr; 
                        const real s1L    = rl * rl; 
                        const real s2R    = std::sin(tr);
                        const real s2L    = std::sin(tl);
                        const real thmean = static_cast<real>(0.5) * (tl + tr);
                        const real sint   = std::sin(thmean);
                        const real dV1    = rmean * rmean * (rr - rl);             
                        const real dV2    = rmean * sint  * (tr - tl); 
                        const real dV3    = rmean * sint  * (qr - ql); 
                        const real cot    = std::cos(thmean) / sint;

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].get_v1();
                        const real vc   = prim_buff[tid].get_v2();
                        const real wc   = prim_buff[tid].get_v3();
                        const real pc   = prim_buff[tid].total_pressure();
                        const auto b4c  = rm::MagFourVec<dim>(prim_buff[tid]);

                        const real hc   = prim_buff[tid].get_gas_enthalpy(gamma);
                        const real gam2 = 1/(1 - (uc * uc + vc * vc + wc * wc));

                        const auto geom_source  = rm::Conserved<3>{
                            0, 
                            (rhoc * hc * gam2 * (vc * vc + wc * wc) - b4c.two * b4c.two - b4c.three * b4c.three) / rmean + pc * (s1R - s1L) / dV1,
                            (rhoc * hc * gam2 * (wc * wc * cot - uc * vc) - b4c.three * b4c.three * cot + b4c.one * b4c.two) / rmean + pc * (s2R - s2L)/dV2 , 
                            - (rhoc * hc * gam2 * wc * (uc + vc * cot) - b4c.three * b4c.one - b4c.three * b4c.two * cot ) / rmean, 
                            0,
                            0,
                            0,
                            0
                        };
                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) / dV1 
                            + (grf * s2R - glf * s2L) / dV2 
                            + (hrf - hlf) / dV3 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step;
                        break;
                    }
                default:
                    {
                        const real rl     = x1l + vfaceL * step * dt; 
                        const real rr     = x1r + vfaceR * step * dt;
                        const real ql     = get_x2face(jj, 0);
                        const real qr     = get_x2face(jj, 1); 
                        const real zl     = get_x3face(kk, 0);
                        const real zr     = get_x3face(kk, 1); 
                        const real rmean  = helpers::get_cell_centroid(rr, rl, simbi::Geometry::CYLINDRICAL);
                        const real s1R    = rr * (zr - zl) * (qr - ql); 
                        const real s1L    = rl * (zr - zl) * (qr - ql); 
                        const real s2R    = (rr - rl) * (zr - rl);
                        const real s2L    = (rr - rl) * (zr - rl);
                        // const real s3L          = rmean * (rr - rl) * (tr - tl);
                        // const real s3R          = s3L;
                        // const real thmean       = static_cast<real>(0.5) * (tl + tr);
                        const real dV    = rmean * (rr - rl) * (zr - zl) * (qr - ql);
                        const real invdV = 1/ dV;

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].get_v1();
                        const real vc   = prim_buff[tid].get_v2();
                        const real wc   = prim_buff[tid].get_v3();
                        const real pc   = prim_buff[tid].total_pressure();
                        const auto b4c  = rm::MagFourVec<dim>(prim_buff[tid]);

                        const real hc   = prim_buff[tid].get_gas_enthalpy(gamma);
                        const real gam2 = 1/(1 - (uc * uc + vc * vc + wc * wc));

                        const auto geom_source  = rm::Conserved<3>{
                            0, 
                            (rhoc * hc * gam2 * (vc * vc) - b4c.two * b4c.two - b4c.three * b4c.three) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * hc * gam2 * uc * vc - b4c.one * b4c.two) / rmean , 
                            0, 
                            0,
                            0,
                            0,
                            0
                        };
                        cons_data[aid] -= ( (frf * s1R - flf * s1L) * invdV + (grf * s2R - glf * s2L) * invdV + (hrf - hlf) * invdV - geom_source - source_terms) * dt * step;
                        break;
                    }
            } // end switch
        }
    });
}
// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template<int dim>
void RMHD<dim>::simulate(
    std::function<real(real)> const &a,
    std::function<real(real)> const &adot,
    std::optional<RMHD<dim>::function_t> const &d_outer,
    std::optional<RMHD<dim>::function_t> const &s1_outer,
    std::optional<RMHD<dim>::function_t> const &s2_outer,
    std::optional<RMHD<dim>::function_t> const &s3_outer,
    std::optional<RMHD<dim>::function_t> const &e_outer)
{   
    helpers::anyDisplayProps();
    // set the primtive functionals
    this->dens_outer = d_outer.value_or(nullptr);
    this->mom1_outer = s1_outer.value_or(nullptr);
    this->mom2_outer = s2_outer.value_or(nullptr);
    this->mom3_outer = s3_outer.value_or(nullptr);
    this->enrg_outer = e_outer.value_or(nullptr);

    if constexpr(dim == 1) {
        this->all_outer_bounds = (
            d_outer.has_value()
            && s1_outer.has_value() 
            && e_outer.has_value()
         );
    } else if constexpr(dim == 2) {
        this->all_outer_bounds = (
            d_outer.has_value()
            && s1_outer.has_value() 
            && s2_outer.has_value() 
            && e_outer.has_value()
        );
    } else {
        this->all_outer_bounds = (
            d_outer.has_value()
            && s1_outer.has_value() 
            && s2_outer.has_value() 
            && s3_outer.has_value() 
            && e_outer.has_value()
        );
    }

    // Stuff for moving mesh 
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->changing_volume = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (mesh_motion && all_outer_bounds) {
        if constexpr(dim == 1) {
            outer_zones.resize(first_order ? 1 : 2);
            const real dV  = get_cell_volume(active_zones - 1);
            outer_zones[0] = conserved_t{
                dens_outer(x1max), 
                mom1_outer(x1max), 
                mom2_outer(x1max),
                mom3_outer(x1max),
                enrg_outer(x1max),
                mag1_outer(x1max),
                mag2_outer(x1max),
                mag3_outer(x1max)} * dV;
            outer_zones.copyToGpu();
        } else if constexpr(dim == 2) {
            outer_zones.resize(ny);
            for (luint jj = 0; jj < ny; jj++) {
                const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid);
                const real dV    = get_cell_volume(xactive_grid - 1, jreal);
                outer_zones[jj]  = conserved_t{
                    dens_outer(x1max, x2[jreal]), 
                    mom1_outer(x1max, x2[jreal]), 
                    mom2_outer(x1max, x2[jreal]),
                    mom3_outer(x1max, x2[jreal]), 
                    enrg_outer(x1max, x2[jreal]),
                    mag1_outer(x1max, x2[jreal]),
                    mag2_outer(x1max, x2[jreal]),
                    mag3_outer(x1max, x2[jreal])} * dV;
            }
            outer_zones.copyToGpu();
        } else {
            outer_zones.resize(ny * nz);
            for (luint kk = 0; kk < nz; kk++)
            {       
                const auto kreal = helpers::get_real_idx(kk, radius, zactive_grid);    
                for (luint jj = 0; jj < ny; jj++) {
                    const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid);
                    const real dV    = get_cell_volume(xactive_grid - 1, jreal, kreal);
                    outer_zones[kk * ny + jj]  = conserved_t{
                        dens_outer(x1max, x2[jreal], x3[kreal]), 
                        mom1_outer(x1max, x2[jreal], x3[kreal]), 
                        mom2_outer(x1max, x2[jreal], x3[kreal]),
                        mom3_outer(x1max, x2[jreal], x3[kreal]), 
                        enrg_outer(x1max, x2[jreal], x3[kreal]),
                        mag1_outer(x1max, x2[jreal], x3[kreal]), 
                        mag2_outer(x1max, x2[jreal], x3[kreal]),
                        mag3_outer(x1max, x2[jreal], x3[kreal])} * dV;
                }
            }
            outer_zones.copyToGpu();
        }
    }

    if (x2max == 0.5 * M_PI){
        this->half_sphere = true;
    }

    inflow_zones.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        this->inflow_zones[i] = rm::Conserved<dim>{
            boundary_sources[i][0], 
            boundary_sources[i][1], 
            boundary_sources[i][2], 
            boundary_sources[i][3], 
            boundary_sources[i][4],
            boundary_sources[i][5],
            boundary_sources[i][6],
            boundary_sources[i][7]
        };
    }

    // Write some info about the setup for writeup later
    setup.x1max = x1[xactive_grid - 1];
    setup.x1min = x1[0];
    setup.x1    = x1;
    if constexpr(dim > 1) {
        setup.x2max = x2[yactive_grid - 1];
        setup.x2min = x2[0];
        setup.x2    = x2;
    }
    if constexpr(dim > 2) {
        setup.x3max = x3[zactive_grid - 1];
        setup.x3min = x3[0];
        setup.x3    = x3;
    }
    setup.nx                  = nx;
    setup.ny                  = ny;
    setup.nz                  = nz;
    setup.xactive_zones       = xactive_grid;
    setup.yactive_zones       = yactive_grid;
    setup.zactive_zones       = zactive_grid;
    setup.x1_cell_spacing      = cell2str.at(x1_cell_spacing);
    setup.x2_cell_spacing      = cell2str.at(x2_cell_spacing);
    setup.x3_cell_spacing      = cell2str.at(x3_cell_spacing);
    setup.ad_gamma            = gamma;
    setup.first_order         = first_order;
    setup.coord_system        = coord_system;
    setup.using_fourvelocity  = (VelocityType == Velocity::FourVelocity);
    setup.regime              = "srmhd";
    setup.mesh_motion         = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions          = dim;

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(active_zones);
    edens_guess.resize(total_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++)
    {
        const auto d    = state[0][i];
        const auto m1   = state[1][i];
        const auto m2   = state[2][i];
        const auto m3   = state[3][i];
        const auto tau  = state[4][i];
        const auto b1   = state[5][i];
        const auto b2   = state[6][i];
        const auto b3   = state[7][i];
        const auto dchi = state[8][i];
        edens_guess[i]  = state[9][i];
        cons[i] = rm::Conserved<dim>{d, m1, m2, m3, tau, b1, b2, b3, dchi};
    }
    cons.copyToGpu();
    prims.copyToGpu();
    edens_guess.copyToGpu();
    dt_min.copyToGpu();
    density_source.copyToGpu();
    m1_source.copyToGpu();
    if constexpr(dim > 1) {
        m2_source.copyToGpu();
    }
    if constexpr(dim > 2) {
        m3_source.copyToGpu();
    }
    if constexpr(dim > 1) {
        object_pos.copyToGpu();
    }
    energy_source.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();
    troubled_cells.copyToGpu();
    sourceG1.copyToGpu();
    if constexpr(dim > 1) {
        sourceG2.copyToGpu();
    }
    if constexpr(dim > 2) {
        sourceG3.copyToGpu();
    }
    sourceB1.copyToGpu();
    if constexpr(dim > 1) {
        sourceB2.copyToGpu();
    }
    if constexpr(dim > 2) {
        sourceB3.copyToGpu();
    }

    // Setup the system
    const luint xblockdim    = xactive_grid > gpu_block_dimx ? gpu_block_dimx : xactive_grid;
    const luint yblockdim    = yactive_grid > gpu_block_dimy ? gpu_block_dimy : yactive_grid;
    const luint zblockdim    = zactive_grid > gpu_block_dimz ? gpu_block_dimz : zactive_grid;
    this->radius             = (first_order) ? 1 : 2;
    this->step               = (first_order) ? 1 : static_cast<real>(0.5);
    const luint xstride      = (BuildPlatform == Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint ystride      = (dim < 3) ? 1 : (BuildPlatform == Platform::GPU) ? yblockdim + 2 * radius: ny;
    const auto  xblockspace  =  xblockdim + 2 * radius;
    const auto  yblockspace  = (dim < 2) ? 1 : yblockdim + 2 * radius;
    const auto  zblockspace  = (dim < 3) ? 1 : zblockdim + 2 * radius;
    const luint shBlockSpace = xblockspace * yblockspace * zblockspace;
    const luint shBlockBytes = shBlockSpace * sizeof(rm::Primitive<dim>);
    const auto fullP         = simbi::ExecutionPolicy({nx, ny, nz}, {xblockdim, yblockdim, zblockdim});
    const auto activeP       = simbi::ExecutionPolicy({xactive_grid, yactive_grid, zactive_grid}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    
    if constexpr(BuildPlatform == Platform::GPU){
        writeln("Requested shared memory: {} bytes", shBlockBytes);
    }
    
    cons2prim(fullP);
    if constexpr(BuildPlatform == Platform::GPU) {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(activeP);
    } else {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }

    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) { 
        rm::write2file<dim>(*this, setup, data_directory, t, 0, chkpt_interval, checkpoint_zones);
        if constexpr(dim == 1) {
            helpers::config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
        } else if constexpr(dim == 2) {
            helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
        } else {
            helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
        }
    }

    this->n = 0;
    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&] {
        if (inFailureState){
            return;
        }
        advance(activeP, xstride, ystride);
        cons2prim(fullP);
        if constexpr(dim == 1) {
            helpers::config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
        } else if constexpr(dim == 2) {
            helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
        } else {
            helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
        }

        if constexpr(BuildPlatform == Platform::GPU) {
            adapt_dt(activeP);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
        if (mesh_motion){
            // update x1 endpoints  
            const real vmin = (geometry == simbi::Geometry::SPHERICAL) ? x1min * hubble_param : hubble_param;
            const real vmax = (geometry == simbi::Geometry::SPHERICAL) ? x1max * hubble_param : hubble_param;
            x1max += step * dt * vmax;
            x1min += step * dt * vmin;
            hubble_param = adot(t) / a(t);
        }
        // std::cout << "pause" << "\n";
        // std::cin.get();
    });

    if (inFailureState){
        emit_troubled_cells();
    }
};
