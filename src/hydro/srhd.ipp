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
SRHD<dim>::SRHD() = default;

// Overloaded Constructor
template <int dim>
SRHD<dim>::SRHD(
    std::vector<std::vector<real>>& state,
    const InitialConditions& init_conditions
)
    : HydroBase(state, init_conditions)
{
}

// Destructor
template <int dim>
SRHD<dim>::~SRHD() = default;

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
void SRHD<dim>::cons2prim()
{
    const auto* const cons_data = cons.data();
    simbi::parallel_for(fullP, total_zones, [cons_data, this] DEV(luint gid) {
        bool workLeftToDo = true;
        volatile __shared__ bool found_failure;

        auto tid = get_threadId();
        if (tid == 0) {
            found_failure = inFailureState;
        }
        simbi::gpu::api::synchronize();

        real invdV = 1.0;
        while (!found_failure && workLeftToDo) {
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

            const real d    = cons_data[gid].dens() * invdV;
            const real s1   = cons_data[gid].momentum(1) * invdV;
            const real s2   = cons_data[gid].momentum(2) * invdV;
            const real s3   = cons_data[gid].momentum(3) * invdV;
            const real tau  = cons_data[gid].nrg() * invdV;
            const real dchi = cons_data[gid].chi() * invdV;
            const real s    = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);

            // Perform modified Newton Raphson based on
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930
            // so far, the convergence rate is the same, but perhaps I need
            // a slight tweak

            // compute f(x_0)
            // f = newton_f(gamma, tau, d, S, peq);
            int iter       = 0;
            real peq       = pressure_guess[gid];
            const real tol = d * global::epsilon;
            real dp;
            do {
                // compute x_[k+1]
                auto [f, g] = newton_fg(gamma, tau, d, s, peq);
                dp          = f / g;
                peq -= dp;

                // compute x*_k
                // f     = newton_f(gamma, tau, d, S, peq);
                // pstar = peq - f / g;

                if (iter >= global::MAX_ITER || !std::isfinite(peq)) {
                    troubled_cells[gid] = 1;
                    dt                  = INFINITY;
                    inFailureState      = true;
                    found_failure       = true;
                    break;
                }
                iter++;

            } while (std::abs(dp) >= tol);

            const real inv_et   = 1.0 / (tau + d + peq);
            real v1             = s1 * inv_et;
            pressure_guess[gid] = peq;
            if constexpr (dim == 1) {
                const real w = 1.0 / std::sqrt(1.0 - v1 * v1);
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                }
                prims[gid] = primitive_t{d / w, v1, peq, dchi / d};
            }
            else if constexpr (dim == 2) {
                real v2      = s2 * inv_et;
                const real w = 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2));
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                    v2 *= w;
                }
                prims[gid] = primitive_t{d / w, v1, v2, peq, dchi / d};
            }
            else {
                real v2        = s2 * inv_et;
                real v3        = s3 * inv_et;
                const real vsq = v1 * v1 + v2 * v2 + v3 * v3;
                const real w   = 1.0 / std::sqrt(1.0 - vsq);
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                    v2 *= w;
                    v3 *= w;
                }
                prims[gid] = primitive_t{d / w, v1, v2, v3, peq, dchi / d};
            }
            workLeftToDo = false;

            if (peq < 0) {
                troubled_cells[gid] = 1;
                inFailureState      = true;
                found_failure       = true;
                dt                  = INFINITY;
            }
            simbi::gpu::api::synchronize();
        }
    });
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template <int dim>
DUAL SRHD<dim>::eigenvals_t SRHD<dim>::calc_eigenvals(
    const SRHD<dim>::primitive_t& primsL,
    const SRHD<dim>::primitive_t& primsR,
    const luint nhat
) const
{
    // Separate the left and right Primitive
    const real rhoL = primsL.rho();
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p();
    const real hL   = primsL.enthalpy(gamma);

    const real rhoR = primsR.rho();
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.p();
    const real hR   = primsR.enthalpy(gamma);

    const real csR = std::sqrt(gamma * pR / (hR * rhoR));
    const real csL = std::sqrt(gamma * pL / (hL * rhoL));

    switch (comp_wave_speed) {
        //-----------Calculate wave speeds based on Schneider et al. 1993
        case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93: {
            const real vbar = 0.5 * (vL + vR);
            const real cbar = 0.5 * (csL + csR);
            const real bl   = (vbar - cbar) / (1.0 - cbar * vbar);
            const real br   = (vbar + cbar) / (1.0 + cbar * vbar);
            const real aL   = my_min<real>(bl, (vL - csL) / (1.0 - vL * csL));
            const real aR   = my_max<real>(br, (vR + csR) / (1.0 + vR * csR));

            return {aL, aR, csL, csR};
        }
        //-----------Calculate wave speeds based on Mignone & Bodo 2005
        case simbi::WaveSpeeds::MIGNONE_AND_BODO_05: {
            // Get Wave Speeds based on Mignone & Bodo Eqs. (21.0 - 23)
            const real lorentzL = 1.0 / std::sqrt(1.0 - (vL * vL));
            const real lorentzR = 1.0 / std::sqrt(1.0 - (vR * vR));
            const real sL =
                csL * csL / (lorentzL * lorentzL * (1.0 - csL * csL));
            const real sR =
                csR * csR / (lorentzR * lorentzR * (1.0 - csR * csR));
            // Define temporaries to save computational cycles
            const real qfL   = 1.0 / (1.0 + sL);
            const real qfR   = 1.0 / (1.0 + sR);
            const real sqrtR = std::sqrt(sR * (1.0 - vR * vR + sR));
            const real sqrtL = std::sqrt(sL * (1.0 - vL * vL + sL));

            const real lamLm = (vL - sqrtL) * qfL;
            const real lamRm = (vR - sqrtR) * qfR;
            const real lamLp = (vL + sqrtL) * qfL;
            const real lamRp = (vR + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

            return {aL, aR, csL, csR};
        }
        //-----------Calculate wave speeds based on Huber & Kissmann 2021
        case simbi::WaveSpeeds::HUBER_AND_KISSMANN_2021: {
            const real gammaL = 1.0 / std::sqrt(1.0 - (vL * vL));
            const real gammaR = 1.0 / std::sqrt(1.0 - (vR * vR));
            const real uL     = gammaL * vL;
            const real uR     = gammaR * vR;
            const real sL     = csL * csL / (1.0 - csL * csL);
            const real sR     = csR * csR / (1.0 - csR * csR);
            const real sqrtR = std::sqrt(sR * (gammaR * gammaR - uR * uR + sR));
            const real sqrtL = std::sqrt(sL * (gammaL * gammaL - uL * uL + sL));
            const real qfL   = 1.0 / (gammaL * gammaL + sL);
            const real qfR   = 1.0 / (gammaR * gammaR + sR);

            const real lamLm = (gammaL * uL - sqrtL) * qfL;
            const real lamRm = (gammaR * uR - sqrtR) * qfR;
            const real lamLp = (gammaL * uL + sqrtL) * qfL;
            const real lamRp = (gammaR * uR + sqrtR) * qfR;

            const real aL = lamLm < lamRm ? lamLm : lamRm;
            const real aR = lamLp > lamRp ? lamLp : lamRp;

            return {aL, aR, csL, csR};
        }
        default:   // NAIVE wave speeds
        {
            const real aLm = (vL - csL) / (1.0 - vL * csL);
            const real aLp = (vL + csL) / (1.0 + vL * csL);
            const real aRm = (vR - csR) / (1.0 - vR * csR);
            const real aRp = (vR + csR) / (1.0 + vR * csR);

            const real aL = my_min(aLm, aRm);
            const real aR = my_max(aLp, aRp);
            return {aL, aR, csL, csR};
        }
    }
};

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditional time step
template <int dim>
template <TIMESTEP_TYPE dt_type>
void SRHD<dim>::adapt_dt()
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
        if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
            const real rho = prims[gid].rho();
            const real v1  = prims[gid].vcomponent(1);
            const real v2  = prims[gid].vcomponent(2);
            const real v3  = prims[gid].vcomponent(3);
            const real pre = prims[gid].p();
            const real h   = prims[gid].enthalpy(gamma);
            const real cs  = std::sqrt(gamma * pre / (rho * h));
            v1p            = std::abs(v1 + cs) / (1.0 + v1 * cs);
            v1m            = std::abs(v1 - cs) / (1.0 - v1 * cs);
            if constexpr (dim > 1) {
                v2p = std::abs(v2 + cs) / (1.0 + v2 * cs);
                v2m = std::abs(v2 - cs) / (1.0 - v2 * cs);
            }
            if constexpr (dim > 2) {
                v3p = std::abs(v3 + cs) / (1.0 + v3 * cs);
                v3m = std::abs(v3 - cs) / (1.0 - v3 * cs);
            }
        }
        else {
            v1p = 1.0;
            v1m = 1.0;
            if constexpr (dim > 1) {
                v2p = 1.0;
                v2m = 1.0;
            }
            if constexpr (dim > 2) {
                v3p = 1.0;
                v3m = 1.0;
            }
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
template <TIMESTEP_TYPE dt_type>
void SRHD<dim>::adapt_dt(const ExecutionPolicy<>& p)
{
#if GPU_CODE
    if constexpr (dim == 1) {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data());
        compute_dt<primitive_t, dt_type>
            <<<p.gridSize, p.blockSize>>>(this, prims.data(), dt_min.data());
    }
    else {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data(),
        // geometry);
        compute_dt<primitive_t, dt_type><<<p.gridSize, p.blockSize>>>(
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
#endif
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

template <int dim>
DUAL SRHD<dim>::conserved_t SRHD<dim>::calc_hlle_flux(
    const SRHD<dim>::primitive_t& prL,
    const SRHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, nhat);
    const auto fR     = prR.to_flux(gamma, nhat);
    const auto lambda = calc_eigenvals(prL, prR, nhat);

    // Grab the necessary wave speeds
    const real aL  = lambda.aL();
    const real aR  = lambda.aR();
    const real aLm = aL < 0.0 ? aL : 0.0;
    const real aRp = aR > 0.0 ? aR : 0.0;

    auto net_flux = [&] {
        // Compute the HLL Flux component-wise
        if (vface <= aLm) {
            return fL - uL * vface;
        }
        else if (vface >= aRp) {
            return fR - uR * vface;
        }
        else {
            const auto f_hll =
                (fL * aRp - fR * aLm + (uR - uL) * aLm * aRp) / (aRp - aLm);
            const auto u_hll = (uR * aRp - uL * aLm - fR + fL) / (aRp - aLm);
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

    // Compute the HLL Flux component-wise
    return net_flux;
};

template <int dim>
DUAL SRHD<dim>::conserved_t SRHD<dim>::calc_hllc_flux(
    const SRHD<dim>::primitive_t& prL,
    const SRHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, nhat);
    const auto fR     = prR.to_flux(gamma, nhat);
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL();
    const real aR     = lambda.aR();
    const real aLm    = aL < 0 ? aL : 0;
    const real aRp    = aR > 0 ? aR : 0;

    //---- Check Wave Speeds before wasting computations
    if (vface <= aLm) {
        return fL - uL * vface;
    }
    else if (vface >= aRp) {
        return fR - uR * vface;
    }

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = (uR * aRp - uL * aLm - fR + fL) / (aRp - aLm);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux =
        (fL * aRp - fR * aLm + (uR - uL) * aRp * aLm) / (aRp - aLm);

    if (quirk_smoothing) {
        if (quirk_strong_shock(prL.p(), prR.p())) {
            return hll_flux;
        }
    }
    const real uhlld   = hll_state.dens();
    const real uhlls1  = hll_state.momentum(1);
    const real uhlls2  = hll_state.momentum(2);
    const real uhlls3  = hll_state.momentum(3);
    const real uhlltau = hll_state.nrg();
    const real fhlld   = hll_flux.dens();
    const real fhlls1  = hll_flux.momentum(1);
    const real fhlls2  = hll_flux.momentum(2);
    const real fhlls3  = hll_flux.momentum(3);
    const real fhlltau = hll_flux.nrg();
    const real e       = uhlltau + uhlld;
    const real s       = (nhat == 1) ? uhlls1 : (nhat == 2) ? uhlls2 : uhlls3;
    const real fe      = fhlltau + fhlld;
    const real fs      = (nhat == 1) ? fhlls1 : (nhat == 2) ? fhlls2 : fhlls3;

    //------Calculate the contact wave velocity and pressure
    const real a     = fe;
    const real b     = -(e + fs);
    const real c     = s;
    const real quad  = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
    const real aStar = c * (1.0 / quad);
    const real pStar = -aStar * fe + fs;

    if constexpr (dim == 1) {
        if (vface <= aStar) {
            const real v        = prL.get_v1();
            const real pressure = prL.p();
            const real d        = uL.dens();
            const real s        = uL.m1();
            const real tau      = uL.nrg();
            const real e        = tau + d;
            const real cofactor = 1.0 / (aLm - aStar);

            //--------------Compute the L Star State----------
            // Left Star State in x-direction of coordinate lattice
            const real dStar = cofactor * (aLm - v) * d;
            const real sStar = cofactor * (s * (aLm - v) - pressure + pStar);
            const real eStar =
                cofactor * (e * (aLm - v) + pStar * aStar - pressure * v);
            const real tauStar = eStar - dStar;

            const auto star_stateL = conserved_t{dStar, sStar, tauStar};

            //---------Compute the L Star Flux
            // Compute the HLL Flux component-wise
            auto hllc_flux = fL + (star_stateL - uL) * aLm;
            return hllc_flux - star_stateL * vface;
        }
        else {
            const real v        = prR.get_v1();
            const real pressure = prR.p();
            const real d        = uR.dens();
            const real s        = uR.m1();
            const real tau      = uR.nrg();
            const real e        = tau + d;
            const real cofactor = 1.0 / (aRp - aStar);

            //--------------Compute the R Star State----------
            // Left Star State in x-direction of coordinate lattice
            const real dStar = cofactor * (aRp - v) * d;
            const real sStar = cofactor * (s * (aRp - v) - pressure + pStar);
            const real eStar =
                cofactor * (e * (aRp - v) + pStar * aStar - pressure * v);
            const real tauStar = eStar - dStar;

            const auto star_stateR = conserved_t{dStar, sStar, tauStar};

            //---------Compute the R Star Flux
            auto hllc_flux = fR + (star_stateR - uR) * aRp;
            return hllc_flux - star_stateR * vface;
        }
    }
    else {
        switch (comp_hllc_type) {
            case HLLCTYPE::FLEISCHMANN: {
                // Apply the low-Mach HLLC fix found in Fleischmann et al
                // 2020:
                // https://www.sciencedirect.com/science/article/pii/S0021999120305362
                const real csL        = lambda.csL();
                const real csR        = lambda.csR();
                constexpr real ma_lim = 5.0;

                // --------------Compute the L Star State----------
                real pressure = prL.p();
                real d        = uL.dens();
                real s1       = uL.momentum(1);
                real s2       = uL.momentum(2);
                real s3       = uL.momentum(3);
                real tau      = uL.nrg();
                real e        = tau + d;
                real cofactor = 1.0 / (aL - aStar);

                const real vL = prL.vcomponent(nhat);
                const real vR = prR.vcomponent(nhat);
                // Left Star State in x-direction of coordinate lattice
                real dStar = cofactor * (aL - vL) * d;
                real s1star =
                    cofactor *
                    (s1 * (aL - vL) + kronecker(nhat, 1) * (-pressure + pStar));
                real s2star =
                    cofactor *
                    (s2 * (aL - vL) + kronecker(nhat, 2) * (-pressure + pStar));
                real s3star =
                    cofactor *
                    (s3 * (aL - vL) + kronecker(nhat, 3) * (-pressure + pStar));
                real eStar =
                    cofactor * (e * (aL - vL) + pStar * aStar - pressure * vL);
                real tauStar          = eStar - dStar;
                const auto starStateL = [&] {
                    if constexpr (dim == 2) {
                        return conserved_t{dStar, s1star, s2star, tauStar};
                    }
                    else {
                        return conserved_t{
                          dStar,
                          s1star,
                          s2star,
                          s3star,
                          tauStar
                        };
                    }
                }();

                pressure = prR.p();
                d        = uR.dens();
                s1       = uR.momentum(1);
                s2       = uR.momentum(2);
                s3       = uR.momentum(3);
                tau      = uR.nrg();
                e        = tau + d;
                cofactor = 1.0 / (aR - aStar);

                dStar  = cofactor * (aR - vR) * d;
                s1star = cofactor * (s1 * (aR - vR) +
                                     kronecker(nhat, 1) * (-pressure + pStar));
                s2star = cofactor * (s2 * (aR - vR) +
                                     kronecker(nhat, 2) * (-pressure + pStar));
                s3star = cofactor * (s3 * (aR - vR) +
                                     kronecker(nhat, 3) * (-pressure + pStar));
                eStar =
                    cofactor * (e * (aR - vR) + pStar * aStar - pressure * vR);
                tauStar               = eStar - dStar;
                const auto starStateR = [&] {
                    if constexpr (dim == 2) {
                        return conserved_t{dStar, s1star, s2star, tauStar};
                    }
                    else {
                        return conserved_t{
                          dStar,
                          s1star,
                          s2star,
                          s3star,
                          tauStar
                        };
                    }
                }();
                const real ma_left =
                    vL / csL * std::sqrt((1.0 - csL * csL) / (1.0 - vL * vL));
                const real ma_right =
                    vR / csR * std::sqrt((1.0 - csR * csR) / (1.0 - vR * vR));
                const real ma_local =
                    my_max(std::abs(ma_left), std::abs(ma_right));
                const real phi =
                    std::sin(my_min<real>(1, ma_local / ma_lim) * M_PI * 0.5);
                const real aL_lm = phi == 0 ? aL : phi * aL;
                const real aR_lm = phi == 0 ? aR : phi * aR;

                const auto face_starState =
                    (aStar <= 0) ? starStateR : starStateL;
                auto net_flux = (fL + fR) * 0.5 +
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

            default: {
                if (vface <= aStar) {
                    const real pressure = prL.p();
                    const real d        = uL.dens();
                    const real s1       = uL.momentum(1);
                    const real s2       = uL.momentum(2);
                    const real s3       = uL.momentum(3);
                    const real tau      = uL.nrg();
                    const real chi      = uL.chi();
                    const real e        = tau + d;
                    const real cofactor = 1.0 / (aL - aStar);

                    const real vL = prL.vcomponent(nhat);
                    // Left Star State in x-direction of coordinate lattice
                    const real dStar   = cofactor * (aL - vL) * d;
                    const real chistar = cofactor * (aL - vL) * chi;
                    const real s1star =
                        cofactor * (s1 * (aL - vL) +
                                    kronecker(nhat, 1) * (-pressure + pStar));
                    const real s2star =
                        cofactor * (s2 * (aL - vL) +
                                    kronecker(nhat, 2) * (-pressure + pStar));
                    const real s3star =
                        cofactor * (s3 * (aL - vL) +
                                    kronecker(nhat, 3) * (-pressure + pStar));
                    const real eStar =
                        cofactor *
                        (e * (aL - vL) + pStar * aStar - pressure * vL);
                    const real tauStar    = eStar - dStar;
                    const auto starStateL = [=] {
                        if constexpr (dim == 2) {
                            return conserved_t{
                              dStar,
                              s1star,
                              s2star,
                              tauStar,
                              chistar
                            };
                        }
                        else {
                            return conserved_t{
                              dStar,
                              s1star,
                              s2star,
                              s3star,
                              tauStar,
                              chistar
                            };
                        }
                    }();

                    auto hllc_flux =
                        fL + (starStateL - uL) * aL - starStateL * vface;

                    // upwind the concentration
                    if (hllc_flux.dens() < 0.0) {
                        hllc_flux.chi() = prR.chi() * hllc_flux.dens();
                    }
                    else {
                        hllc_flux.chi() = prL.chi() * hllc_flux.dens();
                    }

                    return hllc_flux;
                }
                else {
                    const real pressure = prR.p();
                    const real d        = uR.dens();
                    const real s1       = uR.momentum(1);
                    const real s2       = uR.momentum(2);
                    const real s3       = uR.momentum(3);
                    const real tau      = uR.nrg();
                    const real chi      = uR.chi();
                    const real e        = tau + d;
                    const real cofactor = 1.0 / (aR - aStar);

                    const real vR      = prR.vcomponent(nhat);
                    const real dStar   = cofactor * (aR - vR) * d;
                    const real chistar = cofactor * (aR - vR) * chi;
                    const real s1star =
                        cofactor * (s1 * (aR - vR) +
                                    kronecker(nhat, 1) * (-pressure + pStar));
                    const real s2star =
                        cofactor * (s2 * (aR - vR) +
                                    kronecker(nhat, 2) * (-pressure + pStar));
                    const real s3star =
                        cofactor * (s3 * (aR - vR) +
                                    kronecker(nhat, 3) * (-pressure + pStar));
                    const real eStar =
                        cofactor *
                        (e * (aR - vR) + pStar * aStar - pressure * vR);
                    const real tauStar    = eStar - dStar;
                    const auto starStateR = [=] {
                        if constexpr (dim == 2) {
                            return conserved_t{
                              dStar,
                              s1star,
                              s2star,
                              tauStar,
                              chistar
                            };
                        }
                        else {
                            return conserved_t{
                              dStar,
                              s1star,
                              s2star,
                              s3star,
                              tauStar,
                              chistar
                            };
                        }
                    }();

                    auto hllc_flux =
                        fR + (starStateR - uR) * aR - starStateR * vface;

                    // upwind the concentration
                    if (hllc_flux.dens() < 0.0) {
                        hllc_flux.chi() = prR.chi() * hllc_flux.dens();
                    }
                    else {
                        hllc_flux.chi() = prL.chi() * hllc_flux.dens();
                    }

                    return hllc_flux;
                }
            }
        }   // end switch
    }
};

//===================================================================================================================
//                                           SOURCE TERMS
//===================================================================================================================
template <int dim>
DUAL SRHD<dim>::conserved_t SRHD<dim>::hydro_sources(const auto& cell) const
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
DUAL SRHD<dim>::conserved_t SRHD<dim>::gravity_sources(
    const SRHD<dim>::primitive_t& prims,
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
void SRHD<dim>::advance()
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

        // // Calc Rimeann Flux at all interfaces
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

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void SRHD<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot,
    const std::vector<std::optional<SRHD<dim>::function_t>>& bsources,
    const std::vector<std::optional<SRHD<dim>::function_t>>& hsources,
    const std::vector<std::optional<SRHD<dim>::function_t>>& gsources
)
{
    // set the boundary, hydro, and gracity sources terms
    // respectively
    for (auto&& q : bsources) {
        this->bsources.push_back(q.value_or(nullptr));
    }
    for (auto&& q : hsources) {
        this->hsources.push_back(q.value_or(nullptr));
    }
    for (auto&& q : gsources) {
        this->gsources.push_back(q.value_or(nullptr));
    }
    check_sources();

    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
    }

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(total_zones);
    pressure_guess.resize(total_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        for (int q = 0; q < conserved_t::nmem; q++) {
            cons[i][q] = state[q][i];
        }
        const auto d = cons[i][0];
        const real s = std::sqrt(
            cons[i][1] * cons[i][1] + cons[i][2] * cons[i][2] +
            cons[i][3] * cons[i][3]
        );
        const auto e      = cons[i][dim + 1];
        pressure_guess[i] = std::abs(s - d - e);
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
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(fullP);
    }
    else {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }

    // Simulate :)
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
            const real vmin = (homolog) ? x1min * hubble_param : hubble_param;
            const real vmax = (homolog) ? x1max * hubble_param : hubble_param;
            x1max += step * dt * vmax;
            x1min += step * dt * vmin;
            hubble_param = adot(t) / a(t);
        }
    });
};
