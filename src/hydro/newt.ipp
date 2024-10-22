#include "util/device_api.hpp"     // for syncrohonize, devSynch, ...
#include "util/logger.hpp"         // for logger
#include "util/parallel_for.hpp"   // for parallel_for
#include "util/printb.hpp"         // for writeln
#include <cmath>                   // for max, min

using namespace simbi;
using namespace simbi::util;
using namespace simbi::helpers;

// Default Constructor
template <int dim>
Newtonian<dim>::Newtonian() = default;

// Overloaded Constructor
template <int dim>
Newtonian<dim>::Newtonian(
    std::vector<std::vector<real>>& state,
    const InitialConditions& init_conditions
)
    : HydroBase(state, init_conditions)
{
}

// Destructor
template <int dim>
Newtonian<dim>::~Newtonian() = default;

// Helpers
template <int dim>
DUAL constexpr real
Newtonian<dim>::get_x1face(const lint ii, const int side) const
{
    switch (x1_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x1l = my_max<real>(x1min + (ii - 0.5) * dx1, x1min);
                if (side == 0) {
                    return x1l;
                }
                return my_min<real>(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
            }
        default:
            {
                const real x1l = my_max<real>(
                    x1min * std::pow(10.0, (ii - 0.5) * dlogx1),
                    x1min
                );
                if (side == 0) {
                    return x1l;
                }
                return my_min<real>(
                    x1l * std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0)),
                    x1max
                );
            }
    }
}

template <int dim>
DUAL constexpr real
Newtonian<dim>::get_x2face(const lint ii, const int side) const
{
    switch (x2_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x2l = my_max<real>(x2min + (ii - 0.5) * dx2, x2min);
                if (side == 0) {
                    return x2l;
                }
                return my_min<real>(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
            }
        default:
            {
                const real x2l = my_max<real>(
                    x2min * std::pow(10.0, (ii - 0.5) * dlogx2),
                    x2min
                );
                if (side == 0) {
                    return x2l;
                }
                return my_min<real>(
                    x2l * std::pow(10.0, dlogx2 * (ii == 0 ? 0.5 : 1.0)),
                    x2max
                );
            }
    }
}

template <int dim>
DUAL constexpr real
Newtonian<dim>::get_x3face(const lint ii, const int side) const
{
    switch (x3_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x3l = my_max<real>(x3min + (ii - 0.5) * dx3, x3min);
                if (side == 0) {
                    return x3l;
                }
                return my_min<real>(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
            }
        default:
            {
                const real x3l = my_max<real>(
                    x3min * std::pow(10.0, (ii - 0.5) * dlogx3),
                    x3min
                );
                if (side == 0) {
                    return x3l;
                }
                return my_min<real>(
                    x3l * std::pow(10.0, dlogx3 * (ii == 0 ? 0.5 : 1.0)),
                    x3max
                );
            }
    }
}

template <int dim>
DUAL constexpr real Newtonian<dim>::get_x1_differential(const lint ii) const
{
    const real x1l   = get_x1face(ii, 0);
    const real x1r   = get_x1face(ii, 1);
    const real xmean = get_cell_centroid(x1r, x1l, geometry);
    switch (geometry) {
        case Geometry::SPHERICAL:
            return xmean * xmean * (x1r - x1l);
        default:
            return xmean * (x1r - x1l);
    }
}

template <int dim>
DUAL constexpr real Newtonian<dim>::get_x2_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return 2.0;
            default:
                return (2.0 * M_PI);
        }
    }
    else {
        switch (geometry) {
            case Geometry::SPHERICAL:
                {
                    const real x2l  = get_x2face(ii, 0);
                    const real x2r  = get_x2face(ii, 1);
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

template <int dim>
DUAL constexpr real Newtonian<dim>::get_x3_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return (2.0 * M_PI);
            default:
                return 1.0;
        }
    }
    else if constexpr (dim == 2) {
        switch (geometry) {
            case Geometry::PLANAR_CYLINDRICAL:
                return 1.0;
            default:
                return (2.0 * M_PI);
        }
    }
    else {
        return dx3;
    }
}

template <int dim>
DUAL real Newtonian<dim>::get_cell_volume(
    const lint ii,
    const lint jj,
    const lint kk
) const
{
    if (geometry == Geometry::CARTESIAN) {
        return 1.0;
    }
    return get_x1_differential(ii) * get_x2_differential(jj) *
           get_x3_differential(kk);
}

template <int dim>
void Newtonian<dim>::emit_troubled_cells() const
{
    for (luint gid = 0; gid < total_zones; gid++) {
        if (troubled_cells[gid] != 0) {
            const luint kk    = get_height(gid, nx, ny);
            const luint jj    = get_row(gid, nx, ny, kk);
            const luint ii    = get_column(gid, nx, ny, kk);
            const lint ireal  = get_real_idx(ii, radius, xag);
            const lint jreal  = get_real_idx(jj, radius, yag);
            const lint kreal  = get_real_idx(kk, radius, zag);
            const real x1l    = get_x1face(ireal, 0);
            const real x1r    = get_x1face(ireal, 1);
            const real x2l    = get_x2face(jreal, 0);
            const real x2r    = get_x2face(jreal, 1);
            const real x3l    = get_x3face(kreal, 0);
            const real x3r    = get_x3face(kreal, 1);
            const real x1mean = calc_any_mean(x1l, x1r, x1_cell_spacing);
            const real x2mean = calc_any_mean(x2l, x2r, x2_cell_spacing);
            const real x3mean = calc_any_mean(x3l, x3r, x3_cell_spacing);
            const real rho    = cons[gid].dens();
            const real v1     = cons[gid].momentum(1) / rho;
            const real v2     = (dim < 2) ? cons[gid].momentum(2) / rho : 0.0;
            const real v3     = (dim < 3) ? cons[gid].momentum(3) / rho : 0.0;
            const real vsq    = v1 * v1 + v2 * v2 + v3 * v3;
            if constexpr (dim == 1) {
                fprintf(
                    stderr,
                    "\nPrimitives in bad  state\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, iter: %" PRIu64 "\n",
                    cons[gid].dens(),
                    prims[gid].p(),
                    vsq,
                    x1mean,
                    global_iter
                );
            }
            else if constexpr (dim == 2) {
                fprintf(
                    stderr,
                    "\nPrimitives in bad  state\n"
                    "Density: %.2e, "
                    "Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, iter: "
                    "%" PRIu64 "\n",
                    cons[gid].dens(),
                    prims[gid].p(),
                    vsq,
                    x1mean,
                    x2mean,
                    global_iter
                );
            }
            else {
                fprintf(
                    stderr,
                    "\nPrimitives in bad  state\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, "
                    "x3coord: %.2e, iter: %" PRIu64 "\n",
                    cons[gid].dens(),
                    prims[gid].p(),
                    vsq,
                    x1mean,
                    x2mean,
                    x3mean,
                    global_iter
                );
            }
        }
    }
}

//-----------------------------------------------------------------------------------------
//                          Get The Primitive
//-----------------------------------------------------------------------------------------
/**
 * Return the primitive
 * variables density , three-velocity, pressure
 *
 * @param  p execution policy class
 * @return none
 */
template <int dim>
void Newtonian<dim>::cons2prim()
{
    const auto* const ccons = cons.data();
    simbi::parallel_for(fullP, [ccons, this] DEV(const luint gid) {
        real invdV = 1.0;
        if (homolog) {
            if constexpr (dim == 1) {
                const auto ireal = get_real_idx(gid, radius, active_zones);
                const real dV    = get_cell_volume(ireal);
                invdV            = 1.0 / dV;
            }
            else if constexpr (dim == 2) {
                const luint ii   = gid % nx;
                const luint jj   = gid / nx;
                const auto ireal = get_real_idx(ii, radius, xag);
                const auto jreal = get_real_idx(jj, radius, yag);
                const real dV    = get_cell_volume(ireal, jreal);
                invdV            = 1.0 / dV;
            }
            else {
                const luint kk   = get_height(gid, xag, yag);
                const luint jj   = get_row(gid, xag, yag, kk);
                const luint ii   = get_column(gid, xag, yag, kk);
                const auto ireal = get_real_idx(ii, radius, xag);
                const auto jreal = get_real_idx(jj, radius, yag);
                const auto kreal = get_real_idx(kk, radius, zag);
                const real dV    = get_cell_volume(ireal, jreal, kreal);
                invdV            = 1.0 / dV;
            }
        }
        const real rho     = ccons[gid].dens() * invdV;
        const real v1      = (ccons[gid].momentum(1) / rho) * invdV;
        const real v2      = (ccons[gid].momentum(2) / rho) * invdV;
        const real v3      = (ccons[gid].momentum(3) / rho) * invdV;
        const real rho_chi = ccons[gid].chi() * invdV;
        const real pre =
            (gamma - 1.0) *
            (ccons[gid].nrg() - 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3));
        if constexpr (dim == 1) {
            prims[gid] = {rho, v1, pre, rho_chi / rho};
        }
        else if constexpr (dim == 2) {
            prims[gid] = {rho, v1, v2, pre, rho_chi / rho};
        }
        else {
            prims[gid] = {rho, v1, v2, v3, pre, rho_chi / rho};
        }

        if (pre < 0 || !std::isfinite(pre)) {
            troubled_cells[gid] = 1;
            inFailureState      = true;
            dt                  = INFINITY;
        }
    });
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template <int dim>
DUAL Newtonian<dim>::eigenvals_t Newtonian<dim>::calc_eigenvals(
    const Newtonian<dim>::primitive_t& primsL,
    const Newtonian<dim>::primitive_t& primsR,
    const luint nhat
) const
{
    const real rhoL = primsL.rho();
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p();

    const real rhoR = primsR.rho();
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.p();

    const real csR = std::sqrt(gamma * pR / rhoR);
    const real csL = std::sqrt(gamma * pL / rhoL);
    switch (sim_solver) {
        case Solver::HLLC:
            {
                // const real cbar   = 0.5 * (csL + csR);
                // const real rhoBar = 0.5 * (rhoL + rhoR);
                // const real pStar =
                //     0.5 * (pL + pR) + 0.5 * (vL - vR) * cbar * rhoBar;

                // Steps to Compute HLLC as described in Toro et al. 2019
                const real num = csL + csR - (gamma - 1.0) * 0.5 * (vR - vL);
                const real denom =
                    csL * std::pow(pL, -hllc_z) + csR * std::pow(pR, -hllc_z);
                const real p_term = num / denom;
                const real pStar  = std::pow(p_term, (1.0 / hllc_z));

                const real qL = (pStar <= pL)
                                    ? 1.0
                                    : std::sqrt(
                                          1.0 + ((gamma + 1.0) / (2.0 * gamma)
                                                ) * (pStar / pL - 1.0)
                                      );

                const real qR = (pStar <= pR)
                                    ? 1.0
                                    : std::sqrt(
                                          1.0 + ((gamma + 1.0) / (2.0 * gamma)
                                                ) * (pStar / pR - 1.0)
                                      );

                const real aL = vL - qL * csL;
                const real aR = vR + qR * csR;

                const real aStar =
                    ((pR - pL + rhoL * vL * (aL - vL) - rhoR * vR * (aR - vR)) /
                     (rhoL * (aL - vL) - rhoR * (aR - vR)));

                if constexpr (dim == 1) {
                    return {aL, aR, aStar, pStar};
                }
                else {
                    return {aL, aR, csL, csR, aStar, pStar};
                }
            }

        default:
            {
                const real aR =
                    my_max<real>(my_max<real>(vL + csL, vR + csR), 0.0);
                const real aL =
                    my_min<real>(my_min<real>(vL - csL, vR - csR), 0.0);
                return {aL, aR};
            }
    }
};

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditional timestep
template <int dim>
void Newtonian<dim>::adapt_dt()
{
    // singleton instance of thread pool. lazy-evaluated
    static auto& thread_pool =
        simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(total_zones, [&](luint gid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
        const luint kk    = axid<dim, BlkAx::K>(gid, nx, ny);
        const luint jj    = axid<dim, BlkAx::J>(gid, nx, ny, kk);
        const luint ii    = axid<dim, BlkAx::I>(gid, nx, ny, kk);
        const luint ireal = get_real_idx(ii, radius, xag);
        // Left/Right wave speeds
        const real rho = prims[gid].rho();
        const real v1  = prims[gid].vcomponent(1);
        const real v2  = prims[gid].vcomponent(2);
        const real v3  = prims[gid].vcomponent(3);
        const real pre = prims[gid].p();
        const real cs  = std::sqrt(gamma * pre / rho);

        v1m = std::abs(v1 - cs);
        v1p = std::abs(v1 + cs);
        if constexpr (dim > 1) {
            v2m = std::abs(v2 - cs);
            v2p = std::abs(v2 + cs);
        }
        if constexpr (dim > 2) {
            v3m = std::abs(v3 - cs);
            v3p = std::abs(v3 + cs);
        }

        const real x1l = get_x1face(ireal, 0);
        const real x1r = get_x1face(ireal, 1);
        const real dx1 = x1r - x1l;
        switch (geometry) {
            case simbi::Geometry::CARTESIAN:
                if constexpr (dim == 1) {
                    cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
                }
                else if constexpr (dim == 2) {
                    cfl_dt = std::min(
                        {dx1 / (std::max(v1p, v1m)), dx2 / (std::max(v2p, v2m))}
                    );
                }
                else {
                    cfl_dt = std::min(
                        {dx1 / (std::max(v1p, v1m)),
                         dx2 / (std::max(v2p, v2m)),
                         dx3 / (std::max(v3p, v3m))}
                    );
                }
                break;

            case simbi::Geometry::SPHERICAL:
                {
                    if constexpr (dim == 1) {
                        cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
                    }
                    else if constexpr (dim == 2) {
                        const real rmean = get_cell_centroid(
                            x1r,
                            x1l,
                            simbi::Geometry::SPHERICAL
                        );
                        cfl_dt = std::min(
                            {dx1 / (std::max(v1p, v1m)),
                             rmean * dx2 / (std::max(v2p, v2m))}
                        );
                    }
                    else {
                        const real x2l   = get_x2face(jj, 0);
                        const real x2r   = get_x2face(jj, 1);
                        const real rmean = get_cell_centroid(
                            x1r,
                            x1l,
                            simbi::Geometry::SPHERICAL
                        );
                        const real th    = 0.5 * (x2r + x2l);
                        const real rproj = rmean * std::sin(th);
                        cfl_dt           = std::min(
                            {dx1 / (std::max(v1p, v1m)),
                                       rmean * dx2 / (std::max(v2p, v2m)),
                                       rproj * dx3 / (std::max(v3p, v3m))}
                        );
                    }
                    break;
                }
            default:
                {
                    if constexpr (dim == 1) {
                        cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
                    }
                    else if constexpr (dim == 2) {
                        switch (geometry) {
                            case Geometry::AXIS_CYLINDRICAL:
                                {
                                    cfl_dt = std::min(
                                        {dx1 / (std::max(v1p, v1m)),
                                         dx2 / (std::max(v2p, v2m))}
                                    );
                                    break;
                                }

                            default:
                                {
                                    const real rmean = get_cell_centroid(
                                        x1r,
                                        x1l,
                                        simbi::Geometry::CYLINDRICAL
                                    );
                                    cfl_dt = std::min(
                                        {dx1 / (std::max(v1p, v1m)),
                                         rmean * dx2 / (std::max(v2p, v2m))}
                                    );
                                    break;
                                }
                        }
                    }
                    else {
                        const real rmean = get_cell_centroid(
                            x1r,
                            x1l,
                            simbi::Geometry::CYLINDRICAL
                        );
                        cfl_dt = std::min(
                            {dx1 / (std::max(v1p, v1m)),
                             rmean * dx2 / (std::max(v2p, v2m)),
                             dx3 / (std::max(v3p, v3m))}
                        );
                    }
                    break;
                }
        }
        pooling::update_minimum(min_dt, cfl_dt);
    });
    dt = cfl * min_dt;
};

template <int dim>
void Newtonian<dim>::adapt_dt(const ExecutionPolicy<>& p)
{
#if GPU_CODE
    if constexpr (dim == 1) {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data());
        compute_dt<primitive_t>
            <<<p.gridSize, p.blockSize>>>(this, prims.data(), dt_min.data());
    }
    else {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data(),
        // geometry);
        compute_dt<primitive_t><<<p.gridSize, p.blockSize>>>(
            this,
            prims.data(),
            dt_min.data(),
            geometry
        );
    }
    // LAUNCH_ASYNC((deviceReduceWarpAtomicKernel<dim>), p.gridSize,
    // p.blockSize, this, dt_min.data(), active_zones);
    deviceReduceWarpAtomicKernel<dim>
        <<<p.gridSize, p.blockSize>>>(this, dt_min.data(), total_zones);
    gpu::api::deviceSynch();
#endif
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::calc_hlle_flux(
    const Newtonian<dim>::primitive_t& prL,
    const Newtonian<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL;
    const real aR     = lambda.aR;
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, nhat);
    const auto fR     = prR.to_flux(gamma, nhat);

    auto net_flux = [&] {
        // Compute the HLL Flux component-wise
        if (vface <= aL) {
            return fL - uL * vface;
        }
        else if (vface >= aR) {
            return fR - uR * vface;
        }
        else {
            const auto f_hll =
                (fL * aR - fR * aL + (uR - uL) * aR * aL) / (aR - aL);
            const auto u_hll = (uR * aR - uL * aL - fR + fL) / (aR - aL);
            return f_hll - u_hll * vface;
        }
    }();

    // Upwind the scalar concentration
    if (net_flux.dens() < 0.0) {
        net_flux.chi() = prR.chi() * net_flux.dens();
    }
    else {
        net_flux.chi() = prL.chi() * net_flux.dens();
    }

    return net_flux;
};

template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::calc_hllc_flux(
    const Newtonian<dim>::primitive_t& prL,
    const Newtonian<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL;
    const real aR     = lambda.aR;
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, nhat);
    const auto fR     = prR.to_flux(gamma, nhat);

    // Quick checks before moving on with rest of computation
    if (vface <= aL) {
        return fL - uL * vface;
    }
    else if (vface >= aR) {
        return fR - uR * vface;
    }

    if constexpr (dim == 1) {
        const real aStar = lambda.aStar;
        const real pStar = lambda.pStar;
        if (vface <= aStar) {
            real pressure = prL.p();
            real v        = prL.v1();
            real rho      = uL.dens();
            real m        = uL.m1();
            real energy   = uL.nrg();
            real cofac    = 1.0 / (aL - aStar);

            real rhoStar = cofac * (aL - v) * rho;
            real mstar   = cofac * (m * (aL - v) - pressure + pStar);
            real eStar =
                cofac * (energy * (aL - v) + pStar * aStar - pressure * v);

            auto star_state = conserved_t{rhoStar, mstar, eStar};

            // Compute the intermediate left flux
            return fL + (star_state - uL) * aL - star_state * vface;
        }
        else {
            real pressure = prR.p();
            real v        = prR.v1();
            real rho      = uR.dens();
            real m        = uR.m1();
            real energy   = uR.nrg();
            real cofac    = 1.0 / (aR - aStar);

            real rhoStar = cofac * (aR - v) * rho;
            real mstar   = cofac * (m * (aR - v) - pressure + pStar);
            real eStar =
                cofac * (energy * (aR - v) + pStar * aStar - pressure * v);

            auto star_state = conserved_t{rhoStar, mstar, eStar};

            // Compute the intermediate right flux
            return fR + (star_state - uR) * aR - star_state * vface;
        }
    }
    else {
        const real cL    = lambda.csL;
        const real cR    = lambda.csR;
        const real aStar = lambda.aStar;
        const real pStar = lambda.pStar;
        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020:
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        constexpr real ma_lim = 0.10;

        // --------------Compute the L Star State----------
        real pressure = prL.p();
        real rho      = uL.dens();
        real m1       = uL.momentum(1);
        real m2       = uL.momentum(2);
        real m3       = uL.momentum(3);
        real edens    = uL.nrg();
        real cofactor = 1.0 / (aL - aStar);

        const real vL = prL.vcomponent(nhat);
        const real vR = prR.vcomponent(nhat);

        // Left Star State in x-direction of coordinate lattice
        real rhostar = cofactor * (aL - vL) * rho;
        real m1star  = cofactor * (m1 * (aL - vL) +
                                  kronecker(nhat, 1) * (-pressure + pStar));
        real m2star  = cofactor * (m2 * (aL - vL) +
                                  kronecker(nhat, 2) * (-pressure + pStar));
        real m3star  = cofactor * (m3 * (aL - vL) +
                                  kronecker(nhat, 3) * (-pressure + pStar));
        real estar =
            cofactor * (edens * (aL - vL) + pStar * aStar - pressure * vL);
        const auto starStateL = [=] {
            if constexpr (dim == 2) {
                return conserved_t{rhostar, m1star, m2star, estar};
            }
            else {
                return conserved_t{rhostar, m1star, m2star, m3star, estar};
            }
        }();

        pressure = prR.p();
        rho      = uR.dens();
        m1       = uR.momentum(1);
        m2       = uR.momentum(2);
        m3       = uR.momentum(3);
        edens    = uR.nrg();
        cofactor = 1.0 / (aR - aStar);

        rhostar = cofactor * (aR - vR) * rho;
        m1star  = cofactor *
                 (m1 * (aR - vR) + kronecker(nhat, 1) * (-pressure + pStar));
        m2star = cofactor *
                 (m2 * (aR - vR) + kronecker(nhat, 2) * (-pressure + pStar));
        m3star = cofactor *
                 (m3 * (aR - vR) + kronecker(nhat, 3) * (-pressure + pStar));
        estar = cofactor * (edens * (aR - vR) + pStar * aStar - pressure * vR);
        const auto starStateR = [=] {
            if constexpr (dim == 2) {
                return conserved_t{rhostar, m1star, m2star, estar};
            }
            else {
                return conserved_t{rhostar, m1star, m2star, m3star, estar};
            }
        }();

        const real ma_local = my_max(std::abs(vL / cL), std::abs(vR / cR));
        const real phi =
            std::sin(my_min<real>(1.0, ma_local / ma_lim) * M_PI * 0.5);
        const real aL_lm          = phi * aL;
        const real aR_lm          = phi * aR;
        const auto face_starState = (aStar <= 0) ? starStateR : starStateL;
        auto net_flux             = (fL + fR) * 0.5 +
                        ((starStateL - uL) * aL_lm +
                         (starStateL - starStateR) * std::abs(aStar) +
                         (starStateR - uR) * aR_lm) *
                            0.5 -
                        face_starState * vface;

        // upwind the concentration
        if (net_flux.dens() < 0.0) {
            net_flux.chi() = prR.chi() * net_flux.dens();
        }
        else {
            net_flux.chi() = prL.chi() * net_flux.dens();
        }

        return net_flux;
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void Newtonian<dim>::advance()
{
    const auto prim_dat = prims.data();
    simbi::parallel_for(activeP, [prim_dat, this] DEV(const luint idx) {
        conserved_t fri[2], gri[2], hri[2];
        primitive_t pL, pLL, pR, pRR;

        // primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        // const auto prb = sm_or_identity(prim_dat);
        const auto prb = sm_proxy<primitive_t>(prim_dat);

        const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
        const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
        const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

        if constexpr (global::on_gpu) {
            if constexpr (dim == 1) {
                if (ii >= xag) {
                    return;
                }
            }
            else if constexpr (dim == 2) {
                if ((ii >= xag) || (jj >= yag)) {
                    return;
                }
            }
            else {
                if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
                    return;
                }
            }
        }
        const luint ia  = ii + radius;
        const luint ja  = dim < 2 ? 0 : jj + radius;
        const luint ka  = dim < 3 ? 0 : kk + radius;
        const luint tx  = (global::on_sm) ? threadIdx.x : 0;
        const luint ty  = dim < 2 ? 0 : (global::on_sm) ? threadIdx.y : 0;
        const luint tz  = dim < 3 ? 0 : (global::on_sm) ? threadIdx.z : 0;
        const luint txa = (global::on_sm) ? tx + radius : ia;
        const luint tya = dim < 2 ? 0 : (global::on_sm) ? ty + radius : ja;
        const luint tza = dim < 3 ? 0 : (global::on_sm) ? tz + radius : ka;
        const luint aid = idx3(ia, ja, ka, nx, ny, nz);

        if constexpr (global::on_sm) {
            load_shared_buffer<dim>(
                activeP,
                prb,
                prim_dat,
                nx,
                ny,
                nz,
                sx,
                sy,
                tx,
                ty,
                tz,
                txa,
                tya,
                tza,
                ia,
                ja,
                ka,
                radius
            );
        }

        const real x1l    = get_x1face(ii, 0);
        const real x1r    = get_x1face(ii, 1);
        const real vfaceL = (homolog) ? x1l * hubble_param : hubble_param;
        const real vfaceR = (homolog) ? x1r * hubble_param : hubble_param;

        const auto il = get_real_idx(ii - 1, 0, xag);
        const auto ir = get_real_idx(ii + 1, 0, xag);
        const auto jl = get_real_idx(jj - 1, 0, yag);
        const auto jr = get_real_idx(jj + 1, 0, yag);
        const auto kl = get_real_idx(kk - 1, 0, zag);
        const auto kr = get_real_idx(kk + 1, 0, zag);

        // object to left or right? (x1-direction)
        const bool object_x[2] = {
          ib_check<dim>(object_pos, il, jj, kk, xag, yag, 1),
          ib_check<dim>(object_pos, ir, jj, kk, xag, yag, 1)
        };

        // object in front or behind? (x2-direction)
        const bool object_y[2] = {
          ib_check<dim>(object_pos, ii, jl, kk, xag, yag, 2),
          ib_check<dim>(object_pos, ii, jr, kk, xag, yag, 2)
        };

        // object above or below? (x3-direction)
        const bool object_z[2] = {
          ib_check<dim>(object_pos, ii, jj, kl, xag, yag, 3),
          ib_check<dim>(object_pos, ii, jj, kr, xag, yag, 3)
        };

        // Calc Rimeann Flux at all interfaces
        for (int q = 0; q < 2; q++) {
            // fluxes in i direction
            pL = prb[idx3(txa + q - 1, tya, tza, sx, sy, 0)];
            pR = prb[idx3(txa + q + 0, tya, tza, sx, sy, 0)];

            if (!use_pcm) {
                pLL = prb[idx3(txa + q - 2, tya, tza, sx, sy, 0)];
                pRR = prb[idx3(txa + q + 1, tya, tza, sx, sy, 0)];

                pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
            }
            ib_modify<dim>(pR, pL, object_x[q], 1);
            fri[q] = (this->*riemann_solve)(pL, pR, 1, 0);
            if constexpr (dim > 1) {
                // fluxes in j direction
                pL = prb[idx3(txa, tya + q - 1, tza, sx, sy, 0)];
                pR = prb[idx3(txa, tya + q + 0, tza, sx, sy, 0)];

                if (!use_pcm) {
                    pLL = prb[idx3(txa, tya + q - 2, tza, sx, sy, 0)];
                    pRR = prb[idx3(txa, tya + q + 1, tza, sx, sy, 0)];

                    pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                    pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
                }
                ib_modify<dim>(pR, pL, object_y[q], 2);
                gri[q] = (this->*riemann_solve)(pL, pR, 2, 0);

                if constexpr (dim > 2) {
                    // fluxes in k direction
                    pL = prb[idx3(txa, tya, tza + q - 1, sx, sy, 0)];
                    pR = prb[idx3(txa, tya, tza + q + 0, sx, sy, 0)];

                    if (!use_pcm) {
                        pLL = prb[idx3(txa, tya, tza + q - 2, sx, sy, 0)];
                        pRR = prb[idx3(txa, tya, tza + q + 1, sx, sy, 0)];

                        pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                        pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
                    }
                    ib_modify<dim>(pR, pL, object_z[q], 3);
                    hri[q] = (this->*riemann_solve)(pL, pR, 3, 0);
                }
            }
        }

        // TODO: Implement functional source terms
        auto source_terms = conserved_t{};
        // Gravity
        auto gravity = conserved_t{};

        const auto tid = tza * sx * sy + tya * sx + txa;

        if constexpr (dim == 1) {
            switch (geometry) {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons[ia] -= ((fri[RF] - fri[LF]) * invdx1 -
                                     source_terms - gravity) *
                                    dt * step;
                        break;
                    }
                default:
                    {
                        const real rlf = x1l + vfaceL * step * dt;
                        const real rrf = x1r + vfaceR * step * dt;
                        const real rmean =
                            get_cell_centroid(rrf, rlf, geometry);
                        const real sR = 4.0 * M_PI * rrf * rrf;
                        const real sL = 4.0 * M_PI * rlf * rlf;
                        const real dV =
                            4.0 * M_PI * rmean * rmean * (rrf - rlf);
                        const real factor = (mesh_motion) ? dV : 1;
                        const real pc     = prb[txa].p();
                        const real invdV  = 1.0 / dV;
                        const auto geom_sources =
                            conserved_t{0.0, pc * (sR - sL) * invdV, 0.0};
                        cons[ia] -= ((fri[RF] * sR - fri[LF] * sL) * invdV -
                                     geom_sources - source_terms - gravity) *
                                    step * dt * factor;
                        break;
                    }
            }   // end switch
        }
        else if constexpr (dim == 2) {
            switch (geometry) {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons[aid] -= ((fri[RF] - fri[LF]) * invdx1 +
                                      (gri[RF] - gri[LF]) * invdx2 -
                                      source_terms - gravity) *
                                     step * dt;
                        break;
                    }

                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = get_cell_centroid(rr, rl, geometry);
                        const real tl =
                            my_max<real>(x2min + (jj - 0.5) * dx2, x2min);
                        const real tr = my_min<real>(
                            tl + dx2 * (jj == 0 ? 0.5 : 1.0),
                            x2max
                        );
                        const real dcos = std::cos(tl) - std::cos(tr);
                        const real dV   = 2.0 * M_PI * (1.0 / 3.0) *
                                        (rr * rr * rr - rl * rl * rl) * dcos;
                        const real invdV = 1.0 / dV;
                        const real s1R   = 2.0 * M_PI * rr * rr * dcos;
                        const real s1L   = 2.0 * M_PI * rl * rl * dcos;
                        const real s2R   = 2.0 * M_PI * 0.5 *
                                         (rr * rr - rl * rl) * std::sin(tr);
                        const real s2L = 2.0 * M_PI * 0.5 *
                                         (rr * rr - rl * rl) * std::sin(tl);
                        const real factor = (mesh_motion) ? dV : 1;

                        // Grab central primitives
                        const real rhoc = prb[tid].rho();
                        const real uc   = prb[tid].v1();
                        const real vc   = prb[tid].v2();
                        const real pc   = prb[tid].p();

                        const conserved_t geom_source = {
                          0.0,
                          (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV,
                          -(rhoc * uc * vc) / rmean + pc * (s2R - s2L) * invdV,
                          0.0
                        };

                        cons[aid] -= ((fri[RF] * s1R - fri[LF] * s1L) * invdV +
                                      (gri[RF] * s2R - gri[LF] * s2L) * invdV -
                                      geom_source - source_terms - gravity) *
                                     dt * step * factor;
                        break;
                    }
                case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = get_cell_centroid(
                            rr,
                            rl,
                            simbi::Geometry::PLANAR_CYLINDRICAL
                        );
                        // const real tl           = my_max(x2min +
                        // (jj - 0.5) * dx2 , x2min); const real tr =
                        // my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0),
                        // x2max);
                        const real dV    = rmean * (rr - rl) * dx2;
                        const real invdV = 1.0 / dV;
                        const real s1R   = rr * dx2;
                        const real s1L   = rl * dx2;
                        const real s2R   = (rr - rl);
                        const real s2L   = (rr - rl);

                        // Grab central primitives
                        const real rhoc = prb[tid].rho();
                        const real uc   = prb[tid].v1();
                        const real vc   = prb[tid].v2();
                        const real pc   = prb[tid].p();

                        const conserved_t geom_source = {
                          0.0,
                          (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV,
                          -(rhoc * uc * vc) / rmean,
                          0.0
                        };
                        cons[aid] -= ((fri[RF] * s1R - fri[LF] * s1L) * invdV +
                                      (gri[RF] * s2R - gri[LF] * s2L) * invdV -
                                      geom_source - source_terms - gravity) *
                                     dt * step;
                        break;
                    }
                default:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = get_cell_centroid(
                            rl,
                            rr,
                            simbi::Geometry::AXIS_CYLINDRICAL
                        );
                        const real dV    = rmean * (rr - rl) * dx2;
                        const real invdV = 1.0 / dV;
                        const real s1R   = rr * dx2;
                        const real s1L   = rl * dx2;
                        const real s2R   = rmean * (rr - rl);
                        const real s2L   = rmean * (rr - rl);

                        // Grab central primitives
                        const real pc          = prb[tid].p();
                        const auto geom_source = conserved_t{
                          0.0,
                          pc * (s1R - s1L) * invdV,
                          0.0,
                          0.0
                        };
                        cons[aid] -= ((fri[RF] * s1R - fri[LF] * s1L) * invdV +
                                      (gri[RF] * s2R - gri[LF] * s2L) * invdV -
                                      geom_source - source_terms - gravity) *
                                     dt * step;
                        break;
                    }
            }   // end switch
        }
        else {
            switch (geometry) {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons[aid] -= ((fri[RF] - fri[LF]) * invdx1 +
                                      (gri[RF] - gri[LF]) * invdx2 +
                                      (hri[RF] - hri[LF]) * invdx3 -
                                      source_terms - gravity) *
                                     dt * step;
                        break;
                    }
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real tl    = get_x2face(jj, 0);
                        const real tr    = get_x2face(jj, 1);
                        const real ql    = get_x3face(kk, 0);
                        const real qr    = get_x3face(kk, 1);
                        const real rmean = get_cell_centroid(
                            rr,
                            rl,
                            simbi::Geometry::SPHERICAL
                        );
                        const real s1R    = rr * rr;
                        const real s1L    = rl * rl;
                        const real s2R    = std::sin(tr);
                        const real s2L    = std::sin(tl);
                        const real thmean = 0.5 * (tl + tr);
                        const real sint   = std::sin(thmean);
                        const real dV1    = rmean * rmean * (rr - rl);
                        const real dV2    = rmean * sint * (tr - tl);
                        const real dV3    = rmean * sint * (qr - ql);
                        const real cot    = std::cos(thmean) / sint;

                        // Grab central primitives
                        const real rhoc = prb[tid].rho();
                        const real uc   = prb[tid].v1();
                        const real vc   = prb[tid].v2();
                        const real wc   = prb[tid].v3();
                        const real pc   = prb[tid].p();

                        const auto geom_source = conserved_t{
                          0.0,
                          (rhoc * (vc * vc + wc * wc)) / rmean +
                              pc * (s1R - s1L) / dV1,
                          rhoc * (wc * wc * cot - uc * vc) / rmean +
                              pc * (s2R - s2L) / dV2,
                          -rhoc * wc * (uc + vc * cot) / rmean,
                          0.0
                        };
                        cons[aid] -= ((fri[RF] * s1R - fri[LF] * s1L) / dV1 +
                                      (gri[RF] * s2R - gri[LF] * s2L) / dV2 +
                                      (hri[RF] - hri[LF]) / dV3 - geom_source -
                                      source_terms - gravity) *
                                     dt * step;
                        break;
                    }
                default:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real ql    = get_x2face(jj, 0);
                        const real qr    = get_x2face(jj, 1);
                        const real zl    = get_x3face(kk, 0);
                        const real zr    = get_x3face(kk, 1);
                        const real rmean = get_cell_centroid(
                            rr,
                            rl,
                            simbi::Geometry::CYLINDRICAL
                        );
                        const real s1R = rr * (zr - zl) * (qr - ql);
                        const real s1L = rl * (zr - zl) * (qr - ql);
                        const real s2R = (rr - rl) * (zr - zl);
                        const real s2L = (rr - rl) * (zr - zl);
                        const real s3L = rmean * (rr - rl) * (zr - zl);
                        const real s3R = s3L;
                        // const real thmean = 0.5 * (tl + tr);
                        const real dV =
                            rmean * (rr - rl) * (zr - zl) * (qr - ql);
                        const real invdV = 1.0 / dV;

                        // Grab central primitives
                        const real rhoc = prb[tid].rho();
                        const real uc   = prb[tid].v1();
                        const real vc   = prb[tid].v2();
                        // const real wc   = prb[tid].v3;
                        const real pc = prb[tid].p();

                        const auto geom_source = conserved_t{
                          0.0,
                          (rhoc * (vc * vc)) / rmean + pc * (s1R - s1L) * invdV,
                          -(rhoc * uc * vc) / rmean,
                          0.0,
                          0.0
                        };
                        cons[aid] -= ((fri[RF] * s1R - fri[LF] * s1L) * invdV +
                                      (gri[RF] * s2R - gri[LF] * s2L) * invdV +
                                      (hri[RF] * s3R - hri[LF] * s3L) * invdV -
                                      geom_source - source_terms) *
                                     dt * step;
                        break;
                    }
            }   // end switch
        }
    });
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
template <int dim>
void Newtonian<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot,
    std::optional<Newtonian<dim>::function_t> const& d_outer,
    std::optional<Newtonian<dim>::function_t> const& m1_outer,
    std::optional<Newtonian<dim>::function_t> const& m2_outer,
    std::optional<Newtonian<dim>::function_t> const& m3_outer,
    std::optional<Newtonian<dim>::function_t> const& e_outer
)
{
    anyDisplayProps();
    // set the primitive functionals
    this->dens_outer = d_outer.value_or(nullptr);
    this->mom1_outer = m1_outer.value_or(nullptr);
    this->mom2_outer = m2_outer.value_or(nullptr);
    this->mom3_outer = m3_outer.value_or(nullptr);
    this->enrg_outer = e_outer.value_or(nullptr);

    if constexpr (dim == 1) {
        this->all_outer_bounds =
            (d_outer.has_value() && m1_outer.has_value() && enrg_outer);
    }
    else if constexpr (dim == 2) {
        this->all_outer_bounds =
            (d_outer.has_value() && m1_outer.has_value() &&
             m2_outer.has_value() && e_outer.has_value());
    }
    else {
        this->all_outer_bounds =
            (d_outer.has_value() && m1_outer.has_value() &&
             m2_outer.has_value() && m3_outer.has_value() &&
             e_outer.has_value());
    }

    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (mesh_motion && all_outer_bounds) {
        if constexpr (dim == 1) {
            outer_zones.resize(spatial_order == "pcm" ? 1 : 2);
            const real dV = get_cell_volume(active_zones - 1);
            outer_zones[0] =
                conserved_t{
                  dens_outer(x1max),
                  mom1_outer(x1max),
                  enrg_outer(x1max)
                } *
                dV;
            outer_zones.copyToGpu();
        }
        else if constexpr (dim == 2) {
            outer_zones.resize(ny);
            for (luint jj = 0; jj < ny; jj++) {
                const auto jreal = get_real_idx(jj, radius, yag);
                const real dV    = get_cell_volume(nxv - 1, jreal);
                outer_zones[jj] =
                    conserved_t{
                      dens_outer(x1max, x2[jreal]),
                      mom1_outer(x1max, x2[jreal]),
                      mom2_outer(x1max, x2[jreal]),
                      enrg_outer(x1max, x2[jreal])
                    } *
                    dV;
            }
            outer_zones.copyToGpu();
        }
        else {
            outer_zones.resize(ny * nz);
            for (luint kk = 0; kk < nz; kk++) {
                const auto kreal = get_real_idx(kk, radius, zag);
                for (luint jj = 0; jj < ny; jj++) {
                    const auto jreal = get_real_idx(jj, radius, yag);
                    const real dV    = get_cell_volume(nxv - 1, jreal, kreal);
                    outer_zones[kk * ny + jj] =
                        conserved_t{
                          dens_outer(x1max, x2[jreal], x3[kreal]),
                          mom1_outer(x1max, x2[jreal], x3[kreal]),
                          mom2_outer(x1max, x2[jreal], x3[kreal]),
                          mom3_outer(x1max, x2[jreal], x3[kreal]),
                          enrg_outer(x1max, x2[jreal], x3[kreal])
                        } *
                        dV;
                }
            }
            outer_zones.copyToGpu();
        }
    }

    if (x2max == 0.5 * M_PI) {
        this->half_sphere = true;
    }

    inflow_zones.resize(dim * 2);
    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
        if constexpr (dim == 1) {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2]
            };
        }
        else if constexpr (dim == 2) {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2],
              boundary_sources[i][3]
            };
        }
        else {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2],
              boundary_sources[i][3],
              boundary_sources[i][4]
            };
        }
    }

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(total_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        const real rho = state[0][i];
        const real m1  = state[1][i];
        const real m2  = [&] {
            if constexpr (dim < 2) {
                return static_cast<real>(0.0);
            }
            return state[2][i];
        }();
        const real m3 = [&] {
            if constexpr (dim < 3) {
                return static_cast<real>(0.0);
            }
            return state[3][i];
        }();
        const real e = [&] {
            if constexpr (dim == 1) {
                return state[2][i];
            }
            else if constexpr (dim == 2) {
                return state[3][i];
            }
            else {
                return state[4][i];
            }
        }();

        const real rho_chi = [&] {
            if constexpr (dim == 1) {
                return state[3][i];
            }
            else if constexpr (dim == 2) {
                return state[4][i];
            }
            else {
                return state[5][i];
            }
        }();
        if constexpr (dim == 1) {
            cons[i] = {rho, m1, e, rho_chi};
        }
        else if constexpr (dim == 2) {
            cons[i] = {rho, m1, m2, e, rho_chi};
        }
        else {
            cons[i] = {rho, m1, m2, m3, e, rho_chi};
        }
    }
    // Deallocate duplicate memory and setup the system
    set_output_params(dim, "euler");
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    print_shared_mem();
    set_the_riemann_solver();

    cons2prim();
    if constexpr (global::on_gpu) {
        adapt_dt(fullP);
    }
    else {
        adapt_dt();
    }

    // Using a sigmoid decay function to represent when the source terms turn
    // off.
    time_constant = sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) {
        write_to_file(*this);
        if constexpr (dim == 1) {
            config_ghosts1D(
                fullP,
                cons.data(),
                nx,
                spatial_order == "pcm",
                bcs.data(),
                outer_zones.data(),
                inflow_zones.data()
            );
        }
        else if constexpr (dim == 2) {
            config_ghosts2D(
                fullP,
                cons.data(),
                nx,
                ny,
                spatial_order == "pcm",
                geometry,
                bcs.data(),
                outer_zones.data(),
                inflow_zones.data(),
                half_sphere
            );
        }
        else {
            config_ghosts3D(
                fullP,
                cons.data(),
                nx,
                ny,
                nz,
                spatial_order == "pcm",
                bcs.data(),
                inflow_zones.data(),
                half_sphere,
                geometry
            );
        }
    }

    // Simulate :)
    try {
        simbi::detail::logger::with_logger(*this, tend, [&] {
            advance();
            cons2prim();
            if constexpr (dim == 1) {
                config_ghosts1D(
                    fullP,
                    cons.data(),
                    nx,
                    spatial_order == "pcm",
                    bcs.data(),
                    outer_zones.data(),
                    inflow_zones.data()
                );
            }
            else if constexpr (dim == 2) {
                config_ghosts2D(
                    fullP,
                    cons.data(),
                    nx,
                    ny,
                    spatial_order == "pcm",
                    geometry,
                    bcs.data(),
                    outer_zones.data(),
                    inflow_zones.data(),
                    half_sphere
                );
            }
            else {
                config_ghosts3D(
                    fullP,
                    cons.data(),
                    nx,
                    ny,
                    nz,
                    spatial_order == "pcm",
                    bcs.data(),
                    inflow_zones.data(),
                    half_sphere,
                    geometry
                );
            }

            if constexpr (global::on_gpu) {
                adapt_dt(fullP);
            }
            else {
                adapt_dt();
            }
            time_constant =
                sigmoid(t, engine_duration, step * dt, constant_sources);
            t += step * dt;
            if (mesh_motion) {
                // update x1 endpoints
                const real vmin =
                    (homolog) ? x1min * hubble_param : hubble_param;
                const real vmax =
                    (homolog) ? x1max * hubble_param : hubble_param;
                x1max += step * dt * vmax;
                x1min += step * dt * vmin;
                hubble_param = adot(t) / a(t);
            }
        });
    }
    catch (const SimulationFailureException& e) {
        std::cerr << std::string(80, '=') << "\n";
        std::cerr << e.what() << '\n';
        std::cerr << std::string(80, '=') << "\n";
        troubled_cells.copyFromGpu();
        cons.copyFromGpu();
        prims.copyFromGpu();
        hasCrashed = true;
        write_to_file(*this);
        emit_troubled_cells();
    }
};
