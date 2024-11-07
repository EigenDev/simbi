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
            const auto cell   = this->cell_factors(ireal, jreal, kreal);
            const real x1mean = cell.x1mean;
            const real x2mean = cell.x2mean;
            const real x3mean = cell.x3mean;
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
    simbi::parallel_for(fullP, total_zones, [ccons, this] DEV(const luint gid) {
        real invdV = 1.0;
        if (homolog) {
            if constexpr (dim == 1) {
                const auto ireal = get_real_idx(gid, radius, active_zones);
                const auto cell  = this->cell_factors(ireal);
                const real dV    = cell.dV;
                invdV            = 1.0 / dV;
            }
            else if constexpr (dim == 2) {
                const luint ii   = gid % nx;
                const luint jj   = gid / nx;
                const auto ireal = get_real_idx(ii, radius, xag);
                const auto jreal = get_real_idx(jj, radius, yag);
                const auto cell  = this->cell_factors(ireal, jreal);
                const real dV    = cell.dV;
                invdV            = 1.0 / dV;
            }
            else {
                const luint kk   = get_height(gid, xag, yag);
                const luint jj   = get_row(gid, xag, yag, kk);
                const luint ii   = get_column(gid, xag, yag, kk);
                const auto ireal = get_real_idx(ii, radius, xag);
                const auto jreal = get_real_idx(jj, radius, yag);
                const auto kreal = get_real_idx(kk, radius, zag);
                const auto cell  = this->cell_factors(ireal, jreal, kreal);
                const real dV    = cell.dV;
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
        case Solver::HLLC: {
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
                                      1.0 + ((gamma + 1.0) / (2.0 * gamma)) *
                                                (pStar / pL - 1.0)
                                  );

            const real qR = (pStar <= pR)
                                ? 1.0
                                : std::sqrt(
                                      1.0 + ((gamma + 1.0) / (2.0 * gamma)) *
                                                (pStar / pR - 1.0)
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

        default: {
            const real aR = my_max3<real>(vL + csL, vR + csR, 0.0);
            const real aL = my_min3<real>(vL - csL, vR - csR, 0.0);
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
        const luint jreal = get_real_idx(jj, radius, yag);
        const luint kreal = get_real_idx(kk, radius, zag);
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
        const auto cell = this->cell_factors(ireal, jreal, kreal);
        const real x1l  = cell.x1L();
        const real x1r  = cell.x1R();
        const real dx1  = x1r - x1l;
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

            case simbi::Geometry::SPHERICAL: {
                if constexpr (dim == 1) {
                    cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
                }
                else if constexpr (dim == 2) {
                    const real rmean = cell.x1mean;
                    cfl_dt           = std::min(
                        {dx1 / (std::max(v1p, v1m)),
                                   rmean * dx2 / (std::max(v2p, v2m))}
                    );
                }
                else {
                    const real x2l   = cell.x2L();
                    const real x2r   = cell.x2R();
                    const real rmean = cell.x1mean;
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
            default: {
                if constexpr (dim == 1) {
                    cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
                }
                else if constexpr (dim == 2) {
                    switch (geometry) {
                        case Geometry::AXIS_CYLINDRICAL: {
                            cfl_dt = std::min(
                                {dx1 / (std::max(v1p, v1m)),
                                 dx2 / (std::max(v2p, v2m))}
                            );
                            break;
                        }

                        default: {
                            const real rmean = cell.x1mean;
                            cfl_dt           = std::min(
                                {dx1 / (std::max(v1p, v1m)),
                                           rmean * dx2 / (std::max(v2p, v2m))}
                            );
                            break;
                        }
                    }
                }
                else {
                    const real rmean = cell.x1mean;
                    cfl_dt           = std::min(
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
    const real aL     = lambda.aL();
    const real aR     = lambda.aR();
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
    const real aL     = lambda.aL();
    const real aR     = lambda.aR();
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
        const real aStar = lambda.aStar();
        const real pStar = lambda.pStar();
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
        const real cL    = lambda.csL();
        const real cR    = lambda.csR();
        const real aStar = lambda.aStar();
        const real pStar = lambda.pStar();
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
//                                           SOURCE TERMS
//===================================================================================================================
template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::hydro_sources(const auto& cell
) const
{
    if (null_sources) {
        return conserved_t{};
    }
    const auto x1c = cell.x1mean;
    const auto x2c = cell.x2mean;
    const auto x3c = cell.x3mean;

    conserved_t res;
    if constexpr (dim == 1) {
        for (int q = 0; q < conserved_t::nmem; q++) {
            res[q] = hsources[q](x1c, t);
        }
    }
    else if constexpr (dim == 2) {
        for (int q = 0; q < conserved_t::nmem; q++) {
            res[q] = hsources[q](x1c, x2c, t);
        }
    }
    else {
        for (int q = 0; q < conserved_t::nmem; q++) {
            res[q] = hsources[q](x1c, x2c, x3c, t);
        }
    }

    return res;
}

template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::gravity_sources(
    const Newtonian<dim>::primitive_t& prims,
    const auto& cell
) const
{
    if (null_gravity) {
        return conserved_t{};
    }
    const auto x1c = cell.x1mean;

    conserved_t res;
    // gravity only changes the momentum and energy
    if constexpr (dim > 1) {
        const auto x2c = cell.x2mean;
        if constexpr (dim > 2) {
            const auto x3c = cell.x3mean;
            for (int q = 1; q < dimensions + 1; q++) {
                res[q] = gsources[q](x1c, x2c, x3c, t);
            }
            res[dimensions + 1] = gsources[1](x1c, x2c, x3c, t) * prims[1] +
                                  gsources[2](x1c, x2c, x3c, t) * prims[2] +
                                  gsources[3](x1c, x2c, x3c, t) * prims[3];
        }
        else {
            for (int q = 1; q < dimensions + 1; q++) {
                res[q] = gsources[q](x1c, x2c, t);
            }
            res[dimensions + 1] = gsources[1](x1c, x2c, t) * prims[1] +
                                  gsources[2](x1c, x2c, t) * prims[2];
        }
    }
    else {
        for (int q = 1; q < dimensions + 1; q++) {
            res[q] = gsources[q](x1c, t);
        }
        res[dimensions + 1] = gsources[1](x1c, t) * prims[1];
    }

    return res;
}

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
        const auto prb = sm_or_identity(prim_dat);

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
        const luint tid = idx3(txa, tya, tza, sx, sy, sz);

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

        const auto cell   = this->cell_factors(ii, jj, kk);
        const real vfs[2] = {cell.v1fL(), cell.v1fR()};

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
            pL = prb[idx3(txa + q - 1, tya, tza, sx, sy, sz)];
            pR = prb[idx3(txa + q + 0, tya, tza, sx, sy, sz)];

            if (!use_pcm) {
                pLL = prb[idx3(txa + q - 2, tya, tza, sx, sy, sz)];
                pRR = prb[idx3(txa + q + 1, tya, tza, sx, sy, sz)];

                pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
            }
            ib_modify<dim>(pR, pL, object_x[q], 1);
            fri[q] = (this->*riemann_solve)(pL, pR, 1, vfs[q]);

            if constexpr (dim > 1) {
                // fluxes in j direction
                pL = prb[idx3(txa, tya + q - 1, tza, sx, sy, sz)];
                pR = prb[idx3(txa, tya + q + 0, tza, sx, sy, sz)];

                if (!use_pcm) {
                    pLL = prb[idx3(txa, tya + q - 2, tza, sx, sy, sz)];
                    pRR = prb[idx3(txa, tya + q + 1, tza, sx, sy, sz)];

                    pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                    pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
                }
                ib_modify<dim>(pR, pL, object_y[q], 2);
                gri[q] = (this->*riemann_solve)(pL, pR, 2, 0);

                if constexpr (dim > 2) {
                    // fluxes in k direction
                    pL = prb[idx3(txa, tya, tza + q - 1, sx, sy, sz)];
                    pR = prb[idx3(txa, tya, tza + q + 0, sx, sy, sz)];

                    if (!use_pcm) {
                        pLL = prb[idx3(txa, tya, tza + q - 2, sx, sy, sz)];
                        pRR = prb[idx3(txa, tya, tza + q + 1, sx, sy, sz)];

                        pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                        pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
                    }
                    ib_modify<dim>(pR, pL, object_z[q], 3);
                    hri[q] = (this->*riemann_solve)(pL, pR, 3, 0);
                }
            }
        }

        // TODO: implement functional source and gravity
        const auto source_terms = hydro_sources(cell);
        // Gravity
        const auto gravity = gravity_sources(prb[tid], cell);

        // geometric source terms
        const auto geom_source = cell.geom_sources(prb[tid]);

        if constexpr (dim == 1) {
            cons[aid] -=
                ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1() -
                 source_terms - gravity - geom_source) *
                dt * step;
        }
        else if constexpr (dim == 2) {
            cons[aid] -=
                ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1() +
                 (gri[RF] * cell.a2R() - gri[LF] * cell.a2L()) * cell.idV2() -
                 source_terms - gravity - geom_source) *
                dt * step;
        }
        else {
            cons[aid] -=
                ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1() +
                 (gri[RF] * cell.a2R() - gri[LF] * cell.a2L()) * cell.idV2() +
                 (hri[RF] * cell.a3R() - hri[LF] * cell.a3L()) * cell.idV3() -
                 source_terms - gravity - geom_source) *
                dt * step;
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
    const std::vector<std::optional<Newtonian<dim>::function_t>>& bsources,
    const std::vector<std::optional<Newtonian<dim>::function_t>>& hsources,
    const std::vector<std::optional<Newtonian<dim>::function_t>>& gsources
)
{
    anyDisplayProps();
    // set the boundary, hydro, and gravity sources terms respectively
    for (auto&& q : bsources) {
        this->bsources.push_back(q.value_or(nullptr));
    }
    for (auto&& q : hsources) {
        this->hsources.push_back(q.value_or(nullptr));
    }
    for (auto&& q : gsources) {
        this->gsources.push_back(q.value_or(nullptr));
    }

    // check if ~all~ boundary sources have been set.
    // if the user forgot one, the code will run with
    // and outflow outer boundary condition
    this->all_outer_bounds = std::all_of(
        this->bsources.begin(),
        this->bsources.end(),
        [](const auto& q) { return q != nullptr; }
    );

    this->null_gravity = std::all_of(
        this->gsources.begin(),
        this->gsources.end(),
        [](const auto& q) { return q == nullptr; }
    );

    this->null_sources = std::all_of(
        this->hsources.begin(),
        this->hsources.end(),
        [](const auto& q) { return q == nullptr; }
    );
    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (x2max == 0.5 * M_PI) {
        this->half_sphere = true;
    }

    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
    }

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(total_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        for (int q = 0; q < conserved_t::nmem; q++) {
            cons[i][q] = state[q][i];
        }
    }
    // Deallocate duplicate memory and setup the system
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    print_shared_mem();
    init_riemann_solver();
    this->set_mesh_funcs();

    config_ghosts(this);
    cons2prim();
    if constexpr (global::on_gpu) {
        adapt_dt(fullP);
    }
    else {
        adapt_dt();
    }

    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) {
        write_to_file(*this);
    }

    // Simulate :)
    try {
        simbi::detail::logger::with_logger(*this, tend, [&] {
            advance();
            config_ghosts(this);
            cons2prim();

            if constexpr (global::on_gpu) {
                adapt_dt(fullP);
            }
            else {
                adapt_dt();
            }

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
