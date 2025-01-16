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

//-----------------------------------------------------------------------------------------
//                          Get The Primitive
//-----------------------------------------------------------------------------------------
template <int dim>
void Newtonian<dim>::cons2prim()
{
    shared_atomic_bool local_failure;
    auto to_primitive = [gamma = this->gamma,
                         loc   = &local_failure] DEV(const auto& cons
                        ) -> Maybe<primitive_t> {
        const real rho = cons.dens();
        const real v1  = (cons.momentum(1) / rho);
        const real v2  = (cons.momentum(2) / rho);
        const real v3  = (cons.momentum(3) / rho);
        const real chi = cons.chi();
        const real pre =
            (gamma - 1.0) *
            (cons.nrg() - 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3));

        if (pre < 0 || !std::isfinite(pre)) {
            // store the invalid state
            loc->store(true);
            return simbi::Nothing;
        }

        if constexpr (dim == 1) {
            return primitive_t{rho, v1, pre, chi / rho};
        }
        else if constexpr (dim == 2) {
            return primitive_t{rho, v1, v2, pre, chi / rho};
        }
        else {
            return primitive_t{rho, v1, v2, v3, pre, chi / rho};
        }
    };

    prims = cons.transform_parallel(fullPolicy, to_primitive);

    if (local_failure.load()) {
        inFailureState.store(true);
    }
}

/**
 * Return the primitive
 * variables density , three-velocity, pressure
 *
 * @param  p execution policy class
 * @return none
 */
// template <int dim>
// void Newtonian<dim>::cons2prim()
// {
// const auto* const ccons = cons.data();
// simbi::parallel_for(
//     fullPolicy,
//     total_zones,
//     [ccons, this] DEV(const luint gid) {
//         real invdV = 1.0;
//         if (homolog) {
//             if constexpr (dim == 1) {
//                 const auto ireal = get_real_idx(gid, radius,
//                 active_zones); const auto cell  =
//                 this->cell_geometry(ireal); const real dV    = cell.dV;
//                 invdV            = 1.0 / dV;
//             }
//             else if constexpr (dim == 2) {
//                 const luint ii   = gid % nx;
//                 const luint jj   = gid / nx;
//                 const auto ireal = get_real_idx(ii, radius, xag);
//                 const auto jreal = get_real_idx(jj, radius, yag);
//                 const auto cell  = this->cell_geometry(ireal, jreal);
//                 const real dV    = cell.dV;
//                 invdV            = 1.0 / dV;
//             }
//             else {
//                 const luint kk   = get_height(gid, xag, yag);
//                 const luint jj   = get_row(gid, xag, yag, kk);
//                 const luint ii   = get_column(gid, xag, yag, kk);
//                 const auto ireal = get_real_idx(ii, radius, xag);
//                 const auto jreal = get_real_idx(jj, radius, yag);
//                 const auto kreal = get_real_idx(kk, radius, zag);
//                 const auto cell  = this->cell_geometry(ireal, jreal,
//                 kreal); const real dV    = cell.dV; invdV = 1.0 / dV;
//             }
//         }
//         const real rho     = ccons[gid].dens() * invdV;
//         const real v1      = (ccons[gid].momentum(1) / rho) * invdV;
//         const real v2      = (ccons[gid].momentum(2) / rho) * invdV;
//         const real v3      = (ccons[gid].momentum(3) / rho) * invdV;
//         const real rho_chi = ccons[gid].chi() * invdV;
//         const real pre =
//             (gamma - 1.0) *
//             (ccons[gid].nrg() - 0.5 * rho * (v1 * v1 + v2 * v2 + v3 *
//             v3));
//         if constexpr (dim == 1) {
//             prims[gid] = {rho, v1, pre, rho_chi / rho};
//         }
//         else if constexpr (dim == 2) {
//             prims[gid] = {rho, v1, v2, pre, rho_chi / rho};
//         }
//         else {
//             prims[gid] = {rho, v1, v2, v3, pre, rho_chi / rho};
//         }

//         if (pre < 0 || !std::isfinite(pre)) {
//             troubled_cells[gid] = 1;
//             inFailureState.store(true);
//             dt = INFINITY;
//         }
//     }
// );
// }

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template <int dim>
DUAL Newtonian<dim>::eigenvals_t Newtonian<dim>::calc_eigenvals(
    const auto& primsL,
    const auto& primsR,
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
    auto calc_wave_speeds = [gamma = this->gamma](const Maybe<primitive_t>& prim
                            ) -> WaveSpeeds {
        const real cs = std::sqrt(gamma * prim->p() / prim->rho());
        const real v1 = prim->vcomponent(1);
        const real v2 = prim->vcomponent(2);
        const real v3 = prim->vcomponent(3);

        return WaveSpeeds{
          .v1p = std::abs(v1 + cs),
          .v1m = std::abs(v1 - cs),
          .v2p = std::abs(v2 + cs),
          .v2m = std::abs(v2 - cs),
          .v3p = std::abs(v3 + cs),
          .v3m = std::abs(v3 - cs),
        };
    };
    auto calc_local_dt =
        [this](const WaveSpeeds& speeds, const auto& cell) -> real {
        switch (geometry) {
            case Geometry::CARTESIAN:
                if constexpr (dim == 1) {
                    return (cell.x1R() - cell.x1L()) /
                           (std::max(speeds.v1p, speeds.v1m));
                }
                else if constexpr (dim == 2) {
                    return std::min(
                        (cell.x1R() - cell.x1L()) /
                            (std::max(speeds.v1p, speeds.v1m)),
                        (cell.x2R() - cell.x2L()) /
                            (std::max(speeds.v2p, speeds.v2m))
                    );
                }
                else {
                    return std::min(
                        {(cell.x1R() - cell.x1L()) /
                             (std::max(speeds.v1p, speeds.v1m)),
                         (cell.x2R() - cell.x2L()) /
                             (std::max(speeds.v2p, speeds.v3m)),
                         (cell.x3R() - cell.x3L()) /
                             (std::max(speeds.v3p, speeds.v3m))}
                    );
                }
            case Geometry::SPHERICAL: {
                if constexpr (dim == 1) {
                    return (cell.x1R() - cell.x1L()) /
                           (std::max(speeds.v1p, speeds.v1m));
                }
                else if constexpr (dim == 2) {
                    const real rmean = cell.x1mean;
                    return std::min(
                        {(cell.x1R() - cell.x1L()) /
                             (std::max(speeds.v1p, speeds.v1m)),
                         rmean * (cell.x2R() - cell.x2L()) /
                             (std::max(speeds.v2p, speeds.v2m))}
                    );
                }
                else {
                    const real rmean = cell.x1mean;
                    const real th    = 0.5 * (cell.x2R() + cell.x2L());
                    const real rproj = rmean * std::sin(th);
                    return std::min(
                        {(cell.x1R() - cell.x1L()) /
                             (std::max(speeds.v1p, speeds.v1m)),
                         rmean * (cell.x2R() - cell.x2L()) /
                             (std::max(speeds.v2p, speeds.v2m)),
                         rproj * (cell.x3R() - cell.x3L()) /
                             (std::max(speeds.v3p, speeds.v3m))}
                    );
                }
            }
            default:
                if constexpr (dim == 1) {
                    return (cell.x1R() - cell.x1L()) /
                           (std::max(speeds.v1p, speeds.v1m));
                }
                else if constexpr (dim == 2) {
                    switch (geometry) {
                        case Geometry::AXIS_CYLINDRICAL: {
                            return std::min(
                                (cell.x1R() - cell.x1L()) /
                                    (std::max(speeds.v1p, speeds.v1m)),
                                (cell.x2R() - cell.x2L()) /
                                    (std::max(speeds.v2p, speeds.v2m))
                            );
                        }

                        default: {
                            const real rmean = cell.x1mean;
                            return std::min(
                                {(cell.x1R() - cell.x1L()) /
                                     (std::max(speeds.v1p, speeds.v1m)),
                                 rmean * (cell.x2R() - cell.x2L()) /
                                     (std::max(speeds.v2p, speeds.v2m))}
                            );
                        }
                    }
                }
                else {
                    const real rmean = cell.x1mean;
                    return std::min(
                        {(cell.x1R() - cell.x1L()) /
                             (std::max(speeds.v1p, speeds.v1m)),
                         rmean * (cell.x2R() - cell.x2L()) /
                             (std::max(speeds.v2p, speeds.v2m)),
                         (cell.x3R() - cell.x3L()) /
                             (std::max(speeds.v3p, speeds.v3m))}
                    );
                }
        }
    };

    dt = prims
             .transform_parallel(
                 fullPolicy,
                 [this, calc_wave_speeds, calc_local_dt](
                     const Maybe<primitive_t>& prim,
                     const size_t gid
                 ) -> real {
                     // get indices, speeds, and cell parameters
                     const auto [ii, jj, kk] = get_indices(gid, nx, ny);
                     const auto speeds       = calc_wave_speeds(prim);
                     const auto cell         = this->cell_geometry(ii, jj, kk);
                     return calc_local_dt(speeds, cell);
                 }
             )
             .reduce(
                 fullPolicy,
                 INFINITY,
                 [](real a, real b) { return std::min(a, b); }
             ) *
         cfl;

    // std::atomic<real> min_dt = INFINITY;
    // pooling::getThreadPool().parallel_for(total_zones, [&](luint gid) {
    //     real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
    //     const luint kk    = axid<dim, BlkAx::K>(gid, nx, ny);
    //     const luint jj    = axid<dim, BlkAx::J>(gid, nx, ny, kk);
    //     const luint ii    = axid<dim, BlkAx::I>(gid, nx, ny, kk);
    //     const luint ireal = get_real_idx(ii, radius, xag);
    //     const luint jreal = get_real_idx(jj, radius, yag);
    //     const luint kreal = get_real_idx(kk, radius, zag);
    //     // Left/Right wave speeds
    //     const real rho = prims[gid]->rho();
    //     const real v1  = prims[gid]->vcomponent(1);
    //     const real v2  = prims[gid]->vcomponent(2);
    //     const real v3  = prims[gid]->vcomponent(3);
    //     const real pre = prims[gid]->p();
    //     const real cs  = std::sqrt(gamma * pre / rho);

    //     v1m = std::abs(v1 - cs);
    //     v1p = std::abs(v1 + cs);
    //     if constexpr (dim > 1) {
    //         v2m = std::abs(v2 - cs);
    //         v2p = std::abs(v2 + cs);
    //     }
    //     if constexpr (dim > 2) {
    //         v3m = std::abs(v3 - cs);
    //         v3p = std::abs(v3 + cs);
    //     }
    //     const auto cell = this->cell_geometry(ireal, jreal, kreal);
    //     const real x1l  = cell.x1L();
    //     const real x1r  = cell.x1R();
    //     const real dx1  = x1r - x1l;
    //     switch (geometry) {
    //         case simbi::Geometry::CARTESIAN:
    //             if constexpr (dim == 1) {
    //                 cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
    //             }
    //             else if constexpr (dim == 2) {
    //                 cfl_dt = std::min(
    //                     {dx1 / (std::max(v1p, v1m)), dx2 / (std::max(v2p,
    //                     v2m))}
    //                 );
    //             }
    //             else {
    //                 cfl_dt = std::min(
    //                     {dx1 / (std::max(v1p, v1m)),
    //                      dx2 / (std::max(v2p, v2m)),
    //                      dx3 / (std::max(v3p, v3m))}
    //                 );
    //             }
    //             break;

    //         case simbi::Geometry::SPHERICAL: {
    //             if constexpr (dim == 1) {
    //                 cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
    //             }
    //             else if constexpr (dim == 2) {
    //                 const real rmean = cell.x1mean;
    //                 cfl_dt           = std::min(
    //                     {dx1 / (std::max(v1p, v1m)),
    //                                rmean * dx2 / (std::max(v2p, v2m))}
    //                 );
    //             }
    //             else {
    //                 const real x2l   = cell.x2L();
    //                 const real x2r   = cell.x2R();
    //                 const real rmean = cell.x1mean;
    //                 const real th    = 0.5 * (x2r + x2l);
    //                 const real rproj = rmean * std::sin(th);
    //                 cfl_dt           = std::min(
    //                     {dx1 / (std::max(v1p, v1m)),
    //                                rmean * dx2 / (std::max(v2p, v2m)),
    //                                rproj * dx3 / (std::max(v3p, v3m))}
    //                 );
    //             }
    //             break;
    //         }
    //         default: {
    //             if constexpr (dim == 1) {
    //                 cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
    //             }
    //             else if constexpr (dim == 2) {
    //                 switch (geometry) {
    //                     case Geometry::AXIS_CYLINDRICAL: {
    //                         cfl_dt = std::min(
    //                             {dx1 / (std::max(v1p, v1m)),
    //                              dx2 / (std::max(v2p, v2m))}
    //                         );
    //                         break;
    //                     }

    //                     default: {
    //                         const real rmean = cell.x1mean;
    //                         cfl_dt           = std::min(
    //                             {dx1 / (std::max(v1p, v1m)),
    //                                        rmean * dx2 / (std::max(v2p,
    //                                        v2m))}
    //                         );
    //                         break;
    //                     }
    //                 }
    //             }
    //             else {
    //                 const real rmean = cell.x1mean;
    //                 cfl_dt           = std::min(
    //                     {dx1 / (std::max(v1p, v1m)),
    //                                rmean * dx2 / (std::max(v2p, v2m)),
    //                                dx3 / (std::max(v3p, v3m))}
    //                 );
    //             }
    //             break;
    //         }
    //     }
    //     pooling::update_minimum(min_dt, cfl_dt);
    // });
    // dt = cfl * min_dt;
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
    const auto& prL,
    const auto& prR,
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
    const auto& prL,
    const auto& prR,
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
        hydro_source(x1c, t, res);
    }
    else if constexpr (dim == 2) {
        hydro_source(x1c, x2c, t, res);
    }
    else {
        hydro_source(x1c, x2c, x3c, t, res);
    }

    return res;
}

template <int dim>
DUAL Newtonian<dim>::conserved_t
Newtonian<dim>::gravity_sources(const auto& maybe_prims, const auto& cell) const
{
    const auto prims = maybe_prims.value();
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
            gravity_source(x1c, x2c, x3c, t, res);
            res[dimensions + 1] =
                res[1] * prims[1] + res[2] * prims[2] + res[3] * prims[3];
        }
        else {
            gravity_source(x1c, x2c, t, res);
            res[dimensions + 1] = res[1] * prims[1] + res[2] * prims[2];
        }
    }
    else {
        gravity_source(x1c, t, res);
        res[dimensions + 1] = res[1] * prims[1];
    }

    return res;
}

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void Newtonian<dim>::update_mesh_motion(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot
)
{
    if (!mesh_motion) {
        return;
    }

    auto update = [this](real x, real h) {
        return x + step * dt * (homolog ? x * h : h);
    };

    hubble_param = adot(t) / a(t);
    x1max        = update(x1max, hubble_param);
    x1min        = update(x1min, hubble_param);
}

template <int dim>
void Newtonian<dim>::advance()
{
    auto update_conserved = [this] DEV(
                                const auto& fri,
                                const auto& gri,
                                const auto& hri,
                                const auto& source_terms,
                                const auto& gravity,
                                const auto& geom_source,
                                const auto& cell
                            ) -> conserved_t {
        conserved_t res;
        for (int q = 1; q > -1; q--) {
            // q = 0 is L, q = 1 is R
            const auto sign = (q == 1) ? 1 : -1;
            res -= fri[q] * cell.idV1() * cell.area(0 + q) * sign;
            if constexpr (dim > 1) {
                res -= gri[q] * cell.idV2() * cell.area(2 + q) * sign;
                if constexpr (dim > 2) {
                    res -= hri[q] * cell.idV3() * cell.area(4 + q) * sign;
                }
            }
        }

        res += source_terms;
        res += gravity;
        res += geom_source;

        return res * step * dt;
    };

    auto calc_flux = [this, update_conserved] DEV(const auto& stencil) {
        conserved_t fri[2], gri[2], hri[2];

        // Calculate fluxes using stencil
        for (int q = 0; q < 2; q++) {
            // X-direction flux
            const auto& pL = stencil.at(q - 1, 0, 0);
            const auto& pR = stencil.at(q - 0, 0, 0);
            if (!use_pcm) {
                const auto& pLL = stencil.at(q - 2, 0, 0);
                const auto& pRR = stencil.at(q + 1, 0, 0);
                // compute the reconstructed states
                const auto pLr =
                    pL + plm_gradient(*pL, *pLL, *pR, plm_theta) * 0.5;
                const auto pRr =
                    pR - plm_gradient(*pR, *pL, *pRR, plm_theta) * 0.5;
                fri[q] = (this->*riemann_solve)(pLr, pRr, 1, 0);
            }
            else {
                fri[q] = (this->*riemann_solve)(pL, pR, 1, 0);
            }

            if constexpr (dim > 1) {
                // Y-direction flux
                const auto& pL_y = stencil.at(0, q - 1, 0);
                const auto& pR_y = stencil.at(0, q - 0, 0);
                if (!use_pcm) {
                    const auto& pLL_y = stencil.at(0, q - 2, 0);
                    const auto& pRR_y = stencil.at(0, q + 1, 0);
                    const auto pLr_y =
                        pL_y +
                        plm_gradient(*pL_y, *pLL_y, *pR_y, plm_theta) * 0.5;
                    const auto pRr_y =
                        pR_y -
                        plm_gradient(*pR_y, *pL_y, *pRR_y, plm_theta) * 0.5;
                    gri[q] = (this->*riemann_solve)(pLr_y, pRr_y, 2, 0);
                }
                else {
                    gri[q] = (this->*riemann_solve)(pL_y, pR_y, 2, 0);
                }

                if constexpr (dim > 2) {
                    // Z-direction flux
                    const auto& pL_z = stencil.at(0, 0, q - 1);
                    const auto& pR_z = stencil.at(0, 0, q - 0);
                    if (!use_pcm) {
                        const auto& pLL_z = stencil.at(0, 0, q - 2);
                        const auto& pRR_z = stencil.at(0, 0, q + 1);
                        const auto pLr_z =
                            pL_z +
                            plm_gradient(*pL_z, *pLL_z, *pR_z, plm_theta) * 0.5;
                        const auto pRr_z =
                            pR_z -
                            plm_gradient(*pR_z, *pL_z, *pRR_z, plm_theta) * 0.5;
                        hri[q] = (this->*riemann_solve)(pLr_z, pRr_z, 3, 0);
                    }
                    else {
                        hri[q] = (this->*riemann_solve)(pL_z, pR_z, 3, 0);
                    }
                }
            }
        }

        // Calculate sources
        const auto [ii, jj, kk] = stencil.indices();
        const auto cell         = this->cell_geometry(ii, jj, kk);
        const auto source_terms = hydro_sources(cell);
        const auto gravity      = gravity_sources(stencil.center(), cell);
        const auto geom_source  = cell.geom_sources(stencil.center());

        // Return updated conserved values
        return update_conserved(
            fri,
            gri,
            hri,
            source_terms,
            gravity,
            geom_source,
            cell
        );
    };

    // Transform using stencil operations
    cons.template transform_stencil_with<Maybe<primitive_t>>(
        activePolicy,
        prims,
        radius,
        calc_flux
    );
}

// template <int dim>
// void Newtonian<dim>::advance()
// {
//     const auto prim_dat = prims.data();
//     simbi::parallel_for(activePolicy, [prim_dat, this] DEV(const luint idx) {
//         conserved_t fri[2], gri[2], hri[2];
//         primitive_t pL, pLL, pR, pRR;

//         // primitive buffer that returns dynamic shared array
//         // if working with shared memory on GPU, identity otherwise
//         const auto prb = sm_or_identity(prim_dat);

//         const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
//         const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
//         const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

//         if constexpr (global::on_gpu) {
//             if constexpr (dim == 1) {
//                 if (ii >= xag) {
//                     return;
//                 }
//             }
//             else if constexpr (dim == 2) {
//                 if ((ii >= xag) || (jj >= yag)) {
//                     return;
//                 }
//             }
//             else {
//                 if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
//                     return;
//                 }
//             }
//         }
//         const luint ia  = ii + radius;
//         const luint ja  = dim < 2 ? 0 : jj + radius;
//         const luint ka  = dim < 3 ? 0 : kk + radius;
//         const luint tx  = (global::on_sm) ? threadIdx.x : 0;
//         const luint ty  = dim < 2 ? 0 : (global::on_sm) ? threadIdx.y : 0;
//         const luint tz  = dim < 3 ? 0 : (global::on_sm) ? threadIdx.z : 0;
//         const luint txa = (global::on_sm) ? tx + radius : ia;
//         const luint tya = dim < 2 ? 0 : (global::on_sm) ? ty + radius : ja;
//         const luint tza = dim < 3 ? 0 : (global::on_sm) ? tz + radius : ka;
//         const luint aid = idx3(ia, ja, ka, nx, ny, nz);
//         const luint tid = idx3(txa, tya, tza, sx, sy, sz);

//         if constexpr (global::on_sm) {
//             load_shared_buffer<dim>(
//                 activePolicy,
//                 prb,
//                 prim_dat,
//                 nx,
//                 ny,
//                 nz,
//                 sx,
//                 sy,
//                 tx,
//                 ty,
//                 tz,
//                 txa,
//                 tya,
//                 tza,
//                 ia,
//                 ja,
//                 ka,
//                 radius
//             );
//         }

//         const auto cell   = this->cell_geometry(ii, jj, kk);
//         const real vfs[2] = {
//           cell.template velocity<Side::X1L>(),
//           cell.template velocity<Side::X1R>()
//         };

//         const auto il = get_real_idx(ii - 1, 0, xag);
//         const auto ir = get_real_idx(ii + 1, 0, xag);
//         const auto jl = get_real_idx(jj - 1, 0, yag);
//         const auto jr = get_real_idx(jj + 1, 0, yag);
//         const auto kl = get_real_idx(kk - 1, 0, zag);
//         const auto kr = get_real_idx(kk + 1, 0, zag);

//         // object to left or right? (x1-direction)
//         const bool object_x[2] = {
//           ib_check<dim>(object_pos, il, jj, kk, xag, yag, 1),
//           ib_check<dim>(object_pos, ir, jj, kk, xag, yag, 1)
//         };

//         // object in front or behind? (x2-direction)
//         const bool object_y[2] = {
//           ib_check<dim>(object_pos, ii, jl, kk, xag, yag, 2),
//           ib_check<dim>(object_pos, ii, jr, kk, xag, yag, 2)
//         };

//         // object above or below? (x3-direction)
//         const bool object_z[2] = {
//           ib_check<dim>(object_pos, ii, jj, kl, xag, yag, 3),
//           ib_check<dim>(object_pos, ii, jj, kr, xag, yag, 3)
//         };

//         // Calc Rimeann Flux at all interfaces
//         for (int q = 0; q < 2; q++) {
//             // fluxes in i direction
//             pL = prb[idx3(txa + q - 1, tya, tza, sx, sy, sz)];
//             pR = prb[idx3(txa + q + 0, tya, tza, sx, sy, sz)];

//             if (!use_pcm) {
//                 pLL = prb[idx3(txa + q - 2, tya, tza, sx, sy, sz)];
//                 pRR = prb[idx3(txa + q + 1, tya, tza, sx, sy, sz)];

//                 pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
//                 pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
//             }
//             ib_modify<dim>(pR, pL, object_x[q], 1);
//             fri[q] = (this->*riemann_solve)(pL, pR, 1, vfs[q]);

//             if constexpr (dim > 1) {
//                 // fluxes in j direction
//                 pL = prb[idx3(txa, tya + q - 1, tza, sx, sy, sz)];
//                 pR = prb[idx3(txa, tya + q + 0, tza, sx, sy, sz)];

//                 if (!use_pcm) {
//                     pLL = prb[idx3(txa, tya + q - 2, tza, sx, sy, sz)];
//                     pRR = prb[idx3(txa, tya + q + 1, tza, sx, sy, sz)];

//                     pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
//                     pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
//                 }
//                 ib_modify<dim>(pR, pL, object_y[q], 2);
//                 gri[q] = (this->*riemann_solve)(pL, pR, 2, 0);

//                 if constexpr (dim > 2) {
//                     // fluxes in k direction
//                     pL = prb[idx3(txa, tya, tza + q - 1, sx, sy, sz)];
//                     pR = prb[idx3(txa, tya, tza + q + 0, sx, sy, sz)];

//                     if (!use_pcm) {
//                         pLL = prb[idx3(txa, tya, tza + q - 2, sx, sy, sz)];
//                         pRR = prb[idx3(txa, tya, tza + q + 1, sx, sy, sz)];

//                         pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
//                         pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
//                     }
//                     ib_modify<dim>(pR, pL, object_z[q], 3);
//                     hri[q] = (this->*riemann_solve)(pL, pR, 3, 0);
//                 }
//             }
//         }

//         // TODO: implement functional source and gravity
//         const auto source_terms = hydro_sources(cell);
//         // Gravity
//         const auto gravity = gravity_sources(prb[tid], cell);

//         // geometric source terms
//         const auto geom_source = cell.geom_sources(prb[tid]);

//         if constexpr (dim == 1) {
//             cons[aid] -=
//                 ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1()
//                 -
//                  source_terms - gravity - geom_source) *
//                 dt * step;
//         }
//         else if constexpr (dim == 2) {
//             cons[aid] -=
//                 ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1()
//                 +
//                  (gri[RF] * cell.a2R() - gri[LF] * cell.a2L()) * cell.idV2()
//                  - source_terms - gravity - geom_source) *
//                 dt * step;
//         }
//         else {
//             cons[aid] -=
//                 ((fri[RF] * cell.a1R() - fri[LF] * cell.a1L()) * cell.idV1()
//                 +
//                  (gri[RF] * cell.a2R() - gri[LF] * cell.a2L()) * cell.idV2()
//                  + (hri[RF] * cell.a3R() - hri[LF] * cell.a3L()) *
//                  cell.idV3() - source_terms - gravity - geom_source) *
//                 dt * step;
//         }
//     });
// }

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
template <int dim>
void Newtonian<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot
)
{
    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
    }
    load_functions();
    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    if constexpr (global::on_gpu) {
        dt_min.resize(total_zones);
    }

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
    init_riemann_solver();
    // this->set_mesh_funcs();
    config_ghosts(this);
    cons2prim();
    if constexpr (global::on_gpu) {
        adapt_dt(fullPolicy);
    }
    else {
        adapt_dt();
    }

    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&] {
        advance();
        config_ghosts(this);
        cons2prim();
        adapt_dt();

        // std::cout << "dt: " << dt << "\n";

        t += step * dt;
        update_mesh_motion(a, adot);
    });
};
