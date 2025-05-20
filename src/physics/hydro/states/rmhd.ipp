#include "build_options.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/atomic_bool.hpp"   // for shared_atomic_bool
#include "core/types/utility/enums.hpp"
#include "geometry/vector_calculus.hpp"   // for curl_component
#include "io/exceptions.hpp"
#include "physics/hydro/schemes/ct/emf_field.hpp"   // for EMField
#include "physics/hydro/types/generic_structs.hpp"
#include "util/tools/helpers.hpp"
#include <cmath>   // for max, min
#include <limits>

using namespace simbi;
using namespace simbi::util;
using namespace simbi::helpers;
using namespace simbi::vector_calculus;

// Default Constructor
template <int dim>
RMHD<dim>::RMHD() = default;

// Overloaded Constructor
template <int dim>
RMHD<dim>::RMHD(
    std::vector<std::vector<real>>& state,
    InitialConditions& init_conditions
)
    : HydroBase<RMHD<dim>, dim, Regime::RMHD>(state, init_conditions),
      bstag1(std::move(init_conditions.bfield[0])),
      bstag2(std::move(init_conditions.bfield[1])),
      bstag3(std::move(init_conditions.bfield[2]))
{
}

// Destructor
template <int dim>
RMHD<dim>::~RMHD() = default;

//-----------------------------------------------------------------------------------------
//                          Get The Primitive
//-----------------------------------------------------------------------------------------
template <int dim>
void RMHD<dim>::cons2prim_impl()
{
    atomic::simbi_atomic<bool> local_failure{false};
    this->prims_.transform(
        [gamma = this->gamma,
         loc   = local_failure.get(
         )] DEV(auto& prim, const auto& c) -> Maybe<primitive_t> {
            const real d      = c.dens();
            const auto mom    = c.momentum();
            const real tau    = c.nrg();
            const auto bfield = c.bfield();
            const real dchi   = c.chi();

            //==================================================================
            // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
            //==================================================================

            //======= rescale the variables Eqs. (22) - (25)
            const real invd   = 1.0 / d;
            const real isqrtd = std::sqrt(invd);
            const real q      = tau * invd;
            const auto rvec   = mom * invd;
            const auto rsq    = vecops::dot(rvec, rvec);
            const real rmag   = std::sqrt(rsq);
            const auto hvec   = bfield * isqrtd;
            const auto beesq  = vecops::dot(hvec, hvec) + global::epsilon;
            const auto rdb    = vecops::dot(rvec, hvec);
            const real rdbsq  = rdb * rdb;
            // r-parallel Eq. (25.a)
            const auto rparr = rdb / beesq * hvec;
            // r-perpendicular, Eq. (25.b)
            const auto rperp = rvec - rparr;
            const auto rpsq  = vecops::dot(rperp, rperp);

            // We use the false position method to solve for the roots
            real mu_lower = 0.0;
            real mu_upper = find_mu_plus(beesq, rdbsq, rmag);
            // Evaluate the master function (Eq. 44) at the roots
            real f_lower =
                kkc_fmu44(mu_lower, rmag, rpsq, beesq, rdbsq, q, d, gamma);
            real f_upper =
                kkc_fmu44(mu_upper, rmag, rpsq, beesq, rdbsq, q, d, gamma);
            size_type iter = 0;
            real mu, ff;
            do {
                mu = (mu_lower * f_upper - mu_upper * f_lower) /
                     (f_upper - f_lower);
                ff = kkc_fmu44(mu, rmag, rpsq, beesq, rdbsq, q, d, gamma);
                if (ff * f_upper < 0.0) {
                    mu_lower = mu;
                    f_lower  = ff;
                }
                else {
                    // use Illinois algorithm to avoid stagnation
                    f_lower  = 0.5 * f_lower;
                    mu_upper = mu;
                    f_upper  = ff;
                }
                if (iter >= global::MAX_ITER || !std::isfinite(ff)) {
                    loc->store(true);
                    return simbi::None([iter]() -> const ErrorCode {
                        if (iter >= global::MAX_ITER) {
                            return ErrorCode::MAX_ITER;
                        }
                        else {
                            return ErrorCode::NON_FINITE_ROOT;
                        }
                    }());
                }
                iter++;
            } while (std::abs(mu_lower - mu_upper) > global::epsilon &&
                     std::abs(ff) > global::epsilon);

            if (!std::isfinite(mu)) {
                loc->store(true);
                return simbi::None();
            }

            // Ok, we have the roots. Now we can compute the primitive
            // variables Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = rsq * x * x + mu * x * (1.0 + x) * rdbsq;

            // Equation (39)
            const real qbar =
                q - 0.5 * (beesq + mu * mu * x * x * beesq * rpsq);

            // Equation (32)
            const real vsq  = mu * mu * rbar_sq;
            const real gbsq = vsq / (1.0 - vsq);
            const real w    = std::sqrt(1.0 + gbsq);

            // Equation (41)
            const real rhohat = d / w;

            // Equation (42)
            const real eps = w * (qbar - mu * rbar_sq) + gbsq / (1.0 + w);
            // zero-temperature limit for gamma-law EoS
            constexpr auto pfloor = 1.0e-3;
            const real epshat = my_max(eps, pfloor / (rhohat * (gamma - 1.0)));

            // Equation (43)
            const real pg = (gamma - 1.0) * rhohat * epshat;

            if (!std::isfinite(pg) || pg < 0.0) {
                loc->store(true);
                return simbi::None(
                    ErrorCode::NEGATIVE_PRESSURE |
                    ErrorCode::NON_FINITE_PRESSURE
                );
            }

            // velocities Eq. (68)
            auto vel = mu * x * (rvec + hvec * rdb * mu);
            // if (vel.norm() > 1.0) {
            //     loc->store(true);
            //     return simbi::None(ErrorCode::SUPERLUMINAL_VELOCITY);
            // }
            if constexpr (global::using_four_velocity) {
                vel *= w;
            }

            return primitive_t{rhohat, vel, pg, dchi / d, bfield};
        },
        this->full_policy(),
        this->cons_
    );

    if (local_failure.load()) {
        this->set_failure_state(true);
    }
}

/**
 * Return the primitive
 * variables density , three-velocity, pressure
 *
 * @param con conserved array at index
 * @param gid  current global index
 * @return none
 */
template <int dim>
DEV simbi::Maybe<typename RMHD<dim>::primitive_t>
RMHD<dim>::cons2prim_single(const auto& cons) const
{
    const real d      = cons.dens();
    const auto mom    = cons.momentum();
    const real tau    = cons.nrg();
    const auto bfield = cons.bfield();
    const real dchi   = cons.chi();

    //==================================================================
    // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
    //==================================================================

    //======= rescale the variables Eqs. (22) - (25)
    const real invd   = 1.0 / d;
    const real isqrtd = std::sqrt(invd);
    const real q      = tau * invd;
    const auto rvec   = mom * invd;
    const auto rsq    = vecops::dot(rvec, rvec);
    const real rmag   = std::sqrt(rsq);
    const auto hvec   = bfield * isqrtd;
    const auto beesq  = vecops::dot(hvec, hvec) + global::epsilon;
    const auto rdb    = vecops::dot(rvec, hvec);
    const real rdbsq  = rdb * rdb;
    // r-parallel Eq. (25.a)
    const auto rparr = rdb / beesq * hvec;
    // r-perpendicular, Eq. (25.b)
    const auto rperp = rvec - rparr;
    const auto rpsq  = vecops::dot(rperp, rperp);

    // We use the false position method to solve for the roots
    real mu_lower  = 0.0;
    real mu_upper  = find_mu_plus(beesq, rdbsq, rmag);
    real f_lower   = kkc_fmu44(mu_lower, rmag, rpsq, beesq, rdbsq, q, d, gamma);
    real f_upper   = kkc_fmu44(mu_upper, rmag, rpsq, beesq, rdbsq, q, d, gamma);
    size_type iter = 0.0;
    real mu, ff;
    do {
        mu = (mu_lower * f_upper - mu_upper * f_lower) / (f_upper - f_lower);
        ff = kkc_fmu44(mu, rmag, rpsq, beesq, rdbsq, q, d, gamma);
        if (ff * f_upper < 0.0) {
            mu_lower = mu_upper;
            f_lower  = f_upper;
            mu_upper = mu;
            f_upper  = ff;
        }
        else {
            // use Illinois algorithm to avoid stagnation
            f_lower  = 0.5 * f_lower;
            mu_upper = mu;
            f_upper  = ff;
        }
        if (iter >= global::MAX_ITER || !std::isfinite(ff)) {
            return simbi::None([iter]() -> const ErrorCode {
                if (iter >= global::MAX_ITER) {
                    return ErrorCode::MAX_ITER;
                }
                else {
                    return ErrorCode::NON_FINITE_ROOT;
                }
            }());
        }
        iter++;
    } while (std::abs(mu_lower - mu_upper) > global::epsilon &&
             std::abs(ff) > global::epsilon);

    if (!std::isfinite(mu)) {
        return simbi::None();
    }

    // Ok, we have the roots. Now we can compute the primitive
    // variables Equation (26)
    const real x = 1.0 / (1.0 + mu * beesq);

    // Equation (38)
    const real rbar_sq = rsq * x * x + mu * x * (1.0 + x) * rdbsq;

    // Equation (39)
    const real qbar = q - 0.5 * (beesq + mu * mu * x * x * beesq * rpsq);

    // Equation (32)
    const real vsq  = mu * mu * rbar_sq;
    const real gbsq = vsq / (1.0 - vsq);
    const real w    = std::sqrt(1.0 + gbsq);

    // Equation (41)
    const real rhohat = d / w;

    // Equation (42)
    const real eps = w * (qbar - mu * rbar_sq) + gbsq / (1.0 + w);
    // zero-temperature limit for gamma-law EoS
    constexpr auto pfloor = 1.0e-3;
    const real epshat     = my_max(eps, pfloor / (rhohat * (gamma - 1.0)));

    // Equation (43)
    const real pg = (gamma - 1.0) * rhohat * epshat;

    if (!std::isfinite(pg) || pg < 0.0) {
        return simbi::None(
            ErrorCode::NEGATIVE_PRESSURE | ErrorCode::NON_FINITE_PRESSURE
        );
    }

    // velocities Eq. (68)
    auto vel = mu * x * (rvec + hvec * rdb * mu);
    if (vel.norm() > 1.0) {
        return simbi::None(ErrorCode::SUPERLUMINAL_VELOCITY);
    }
    if constexpr (global::using_four_velocity) {
        vel *= w;
    }

    return primitive_t{rhohat, vel, pg, dchi / d, bfield};
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
/*
    Compute the outer wave speeds as discussed in Mignone and Bodo (2006)
*/

template <int dim>
DUAL auto
RMHD<dim>::calc_max_wave_speeds(const auto& prims, const luint nhat) const
{
    /*
    evaluate the full quartic if the simplifying conditions are not met.
    Polynomial coefficients were calculated using sympy in the following
    way:

    In [1]: import sympy
    In [2]: rho = sympy.Symbol('rho')
    In [3]: h = sympy.Symbol('h')
    In [4]: vx = sympy.Symbol('vx')
    In [5]: b0 = sympy.Symbol('b0')
    In [6]: bx = sympy.Symbol('bx')
    In [7]: cs = sympy.Symbol('cs')
    In [8]: x = sympy.Symbol('x')
    In [9]: b2 = sympy.Symbol('b2')
    In [10]: g = sympy.Symbol('g')
    In [11]: p = sympy.Poly(sympy.expand(rho * h * (1 - cs**2) * (g * (x -
    vx))**4 - (1 - x**2)*((b2 + rho * h * cs**2) * (g*(x-vx))**2 - cs**2 *
    (bx - x * b0)**2)), domain='ZZ[rho, h, cs, g,
    ...: vx, b2, bx, b0]') # --------------- Eq. (56)
    net_flux.set_mdz_vars(MignoneDelZannaVariables{
    })

    In [12]: p.coeffs()
    Out[12]:
    [-b0**2*cs**2 + b2*g**2 - cs**2*g**4*h*rho + cs**2*g**2*h*rho +
    g**4*h*rho, 2*b0*bx*cs**2 - 2*b2*g**2*vx + 4*cs**2*g**4*h*rho*vx -
    2*cs**2*g**2*h*rho*vx
    - 4*g**4*h*rho*vx,
    b0**2*cs**2 + b2*g**2*vx**2 - b2*g**2 - bx**2*cs**2 -
    6*cs**2*g**4*h*rho*vx**2 + cs**2*g**2*h*rho*vx**2 - cs**2*g**2*h*rho +
    6*g**4*h*rho*vx**2,
    -2*b0*bx*cs**2 + 2*b2*g**2*vx + 4*cs**2*g**4*h*rho*vx**3
    + 2*cs**2*g**2*h*rho*vx - 4*g**4*h*rho*vx**3,
    -b2*g**2*vx**2 + bx**2*cs**2 - cs**2*g**4*h*rho*vx**4 -
    cs**2*g**2*h*rho*vx**2 + g**4*h*rho*vx**4]
    */
    const real rho   = prims.rho();
    const real h     = prims.enthalpy(gamma);
    const real cs2   = (gamma * prims.press() / (rho * h));
    const auto bmu   = prims.calc_magnetic_four_vector();
    const real bmusq = bmu.inner_product(bmu);
    const real bn    = prims.bcomponent(nhat);
    const real bn2   = bn * bn;
    const real vn    = prims.vcomponent(nhat);
    if (prims.vsquared() < global::epsilon) {   // Eq.(57)
        const real fac     = 1.0 / (rho * h + bmusq);
        const real a       = 1.0;
        const real b       = -(bmusq + rho * h * cs2 + bn2 * cs2) * fac;
        const real c       = cs2 * bn2 * fac;
        const real disq    = std::sqrt(b * b - 4.0 * a * c);
        const auto lambdaR = std::sqrt(0.5 * (-b + disq));
        const auto lambdaL = -lambdaR;
        return std::make_tuple(lambdaL, lambdaR);
    }
    else if (bn2 < global::epsilon) {   // Eq. (58)
        const real g2      = prims.lorentz_factor_squared();
        const real vdbperp = prims.vdotb() - vn * bn;
        const real q       = bmusq - cs2 * vdbperp * vdbperp;
        const real a2      = rho * h * (cs2 + g2 * (1.0 - cs2)) + q;
        const real a1      = -2.0 * rho * h * g2 * vn * (1.0 - cs2);
        const real a0      = rho * h * (-cs2 + g2 * vn * vn * (1.0 - cs2)) - q;
        const real disq    = a1 * a1 - 4.0 * a2 * a0;
        const auto lambdaR = 0.5 * (-a1 + std::sqrt(disq)) / a2;
        const auto lambdaL = 0.5 * (-a1 - std::sqrt(disq)) / a2;
        return std::make_tuple(lambdaL, lambdaR);
    }
    else {   // solve the full quartic Eq. (56)
        // initialize quartic speed array
        real speeds[4] = {0.0, 0.0, 0.0, 0.0};

        const real bmu0 = bmu[0];
        const real bmun = bmu.spatial_dot(unit_vectors::get<dim>(nhat));
        const real w    = prims.lorentz_factor();
        const real w2   = w * w;
        const real vn2  = vn * vn;

        const real a4 =
            (-bmu0 * bmu0 * cs2 + bmusq * w2 - cs2 * w2 * w2 * h * rho +
             cs2 * w2 * h * rho + w2 * w2 * h * rho);
        const real fac = 1.0 / a4;

        const real a3 = fac * (2.0 * bmu0 * bmun * cs2 - 2.0 * bmusq * w2 * vn +
                               4.0 * cs2 * w2 * w2 * h * rho * vn -
                               2.0 * cs2 * w2 * h * rho * vn -
                               4.0 * w2 * w2 * h * rho * vn);
        const real a2 =
            fac * (bmu0 * bmu0 * cs2 + bmusq * w2 * vn2 - bmusq * w2 -
                   bmun * bmun * cs2 - 6.0 * cs2 * w2 * w2 * h * rho * vn2 +
                   cs2 * w2 * h * rho * vn2 - cs2 * w2 * h * rho +
                   6.0 * w2 * w2 * h * rho * vn2);

        const real a1 =
            fac * (-2.0 * bmu0 * bmun * cs2 + 2.0 * bmusq * w2 * vn +
                   4.0 * cs2 * w2 * w2 * h * rho * vn * vn2 +
                   2.0 * cs2 * w2 * h * rho * vn -
                   4.0 * w2 * w2 * h * rho * vn * vn2);

        const real a0 =
            fac * (-bmusq * w2 * vn2 + bmun * bmun * cs2 -
                   cs2 * w2 * w2 * h * rho * vn2 * vn2 -
                   cs2 * w2 * h * rho * vn2 + w2 * w2 * h * rho * vn2 * vn2);

        const auto nroots = solve_quartic(a3, a2, a1, a0, speeds);

        // if there are no roots, return null
        if (nroots == 0) {
            return std::make_tuple(0.0, 0.0);
        }
        return std::make_tuple(speeds[0], speeds[nroots - 1]);

        if constexpr (global::debug_mode) {
            if (nroots != 4) {
                printf(
                    "\n number of quartic roots less than 4, nroots: %d."
                    "fastest wave"
                    ": % .2e, slowest_wave"
                    ": % .2e\n ",
                    nroots,
                    speeds[3],
                    speeds[0]
                );
            }
            else {
                printf(
                    "slowest wave: %.2e, fastest wave: %.2e\n",
                    speeds[0],
                    speeds[3]
                );
            }
        }
    }
}

template <int dim>
DUAL RMHD<dim>::eigenvals_t RMHD<dim>::calc_eigenvals(
    const auto& primsL,
    const auto& primsR,
    const luint nhat
) const
{
    // left side
    const auto [lmL, lpL] = calc_max_wave_speeds(primsL, nhat);
    // right_side
    const auto [lmR, lpR] = calc_max_wave_speeds(primsR, nhat);

    const auto aR = my_max(lpL, lpR);
    const auto aL = my_min(lmL, lmR);

    return {aL, aR};
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::calc_hlle_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const real bface
) const
{
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR     = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    // Grab the fastest wave speeds
    const auto aL  = lambda.afL();
    const auto aR  = lambda.afR();
    const auto aLm = aL < 0.0 ? aL : 0.0;
    const auto aRp = aR > 0.0 ? aR : 0.0;

    const auto stationary = aLm == aRp;

    auto net_flux = [&] {
        // Compute the HLL Flux component-wise
        if (stationary) {
            return (fL + fR) * 0.5 - (uR + uL) * 0.5 * vface;
        }
        else if (vface <= aLm) {
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

    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        const auto nj = next_perm(nhat, 1);
        const auto nk = next_perm(nhat, 2);
        if (vface <= aLm) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = 1.0,
                  .aR   = 0.0,
                  .dL   = 0.0,
                  .dR   = 0.0,
                  .vjL  = prL.vcomponent(nj),
                  .vjR  = 0.0,
                  .vkL  = prL.vcomponent(nk),
                  .vkR  = 0.0,
                }
            );
        }
        else if (vface >= aRp) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = 0.0,
                  .aR   = 1.0,
                  .dL   = 0.0,
                  .dR   = 0.0,
                  .vjL  = 0.0,
                  .vjR  = prR.vcomponent(nj),
                  .vkL  = 0.0,
                  .vkR  = prR.vcomponent(nk),
                }
            );
        }
        else {
            // set the wave coefficients
            const auto afac = 1.0 / (aRp - aLm);
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = +aRp * afac,
                  .aR   = -aLm * afac,
                  .dL   = -aRp * aLm * afac,
                  .dR   = -aRp * aLm * afac,
                  .vjL  = prL.vcomponent(nj),
                  .vjR  = prR.vcomponent(nj),
                  .vkL  = prL.vcomponent(nk),
                  .vkR  = prR.vcomponent(nk),
                }
            );
        }
    }
    else {
        net_flux.calc_electric_field(unit_vectors::get<dim>(nhat), nhat);
    }
    return net_flux;
};

template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::calc_hllc_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const real bface
) const
{
    real aS;
    [[maybe_unused]] real chiL, chiR;
    bool null_normal_field = false;
    const auto uL          = prL.to_conserved(gamma);
    const auto uR          = prR.to_conserved(gamma);
    const auto fL          = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR          = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

    const auto lambda     = calc_eigenvals(prL, prR, nhat);
    const real aL         = lambda.afL();
    const real aR         = lambda.afR();
    const real aLm        = aL < 0.0 ? aL : 0.0;
    const real aRp        = aR > 0.0 ? aR : 0.0;
    const auto stationary = aLm == aRp;
    auto net_flux         = [&]() {
        //---- Check Wave Speeds before wasting computations
        if (stationary) {
            return (fL + fR) * 0.5 - (uR + uL) * 0.5 * vface;
        }
        else if (vface <= aLm) {
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

        if (this->quirk_smoothing() &&
            quirk_strong_shock(prL.press(), prR.press())) {
            return hll_flux - hll_state * vface;
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);
        // the normal component of the magnetic field is assumed to
        // be continuous across the interface, so bnL = bnR = bn
        // const auto bn = hll_state.bcomponent(nhat);
        const auto bn = bface;
        // const real bn  = hll_state.bcomponent(nhat);
        const real bp1 = hll_state.bcomponent(np1);
        const real bp2 = hll_state.bcomponent(np2);

        // check if normal magnetic field is approaching zero
        null_normal_field = goes_to_zero(bn);

        const real uhlld = hll_state.dens();
        const real uhllm = hll_state.mcomponent(nhat);
        const real uhlle = hll_state.nrg() + uhlld;

        const real fhlld = hll_flux.dens();
        const real fhllm = hll_flux.mcomponent(nhat);
        const real fhlle = hll_flux.nrg() + fhlld;
        const real fpb1  = hll_flux.bcomponent(np1);
        const real fpb2  = hll_flux.bcomponent(np2);

        // //------Calculate the contact wave velocity and pressure
        real a, b, c;
        if (null_normal_field) {
            a = fhlle;
            b = -(fhllm + uhlle);
            c = uhllm;
        }
        else {
            const real fdb   = bp1 * fpb1 + bp2 * fpb2;
            const real bpsq  = bp1 * bp1 + bp2 * bp2;
            const real fbpsq = fpb1 * fpb1 + fpb2 * fpb2;
            a                = fhlle - fdb;
            b                = -(fhllm + uhlle) + bpsq + fbpsq;
            c                = uhllm - fdb;
        }

        const real quad = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
        aS              = c / quad;

        // set the chi values if using MdZ21 CT
        if constexpr (comp_ct_type == CTTYPE::MdZ) {
            chiL = -(prL.vcomponent(nhat) - aS) / (aLm - aS);
            chiR = -(prR.vcomponent(nhat) - aS) / (aRp - aS);
        }

        const auto on_left = vface < aS;
        const auto u       = on_left ? uL : uR;
        const auto f       = on_left ? fL : fR;
        const auto pr      = on_left ? prL : prR;
        const auto ws      = on_left ? aLm : aRp;

        const real d     = u.dens();
        const real mnorm = u.mcomponent(nhat);
        const real ump1  = u.mcomponent(np1);
        const real ump2  = u.mcomponent(np2);
        const real fmp1  = f.mcomponent(np1);
        const real fmp2  = f.mcomponent(np2);
        const real et    = u.nrg() + d;
        const real cfac  = 1.0 / (ws - aS);

        const real v  = pr.vcomponent(nhat);
        const real vs = cfac * (ws - v);
        const real ds = vs * d;
        // star state
        conserved_t us;
        if (null_normal_field) {
            const real pS       = -aS * fhlle + fhllm;
            const real es       = cfac * (ws * et - mnorm + pS * aS);
            const real mn       = (es + pS) * aS;
            const real mp1      = vs * ump1;
            const real mp2      = vs * ump2;
            us.dens()           = ds;
            us.mcomponent(nhat) = mn;
            us.mcomponent(np1)  = mp1;
            us.mcomponent(np2)  = mp2;
            us.nrg()            = es - ds;
            us.bcomponent(nhat) = bn;
            us.bcomponent(np1)  = vs * pr.bcomponent(np1);
            us.bcomponent(np2)  = vs * pr.bcomponent(np2);
        }
        else {
            const real vp1   = (bp1 * aS - fpb1) / bn;
            const real vp2   = (bp2 * aS - fpb2) / bn;
            const real invg2 = (1.0 - (aS * aS + vp1 * vp1 + vp2 * vp2));
            const real vsdB  = (aS * bn + vp1 * bp1 + vp2 * bp2);
            const real pS = -aS * (fhlle - bn * vsdB) + fhllm + bn * bn * invg2;
            const real es = cfac * (ws * et - mnorm + pS * aS - vsdB * bn);
            const real mn = (es + pS) * aS - vsdB * bn;
            const real mp1 =
                cfac * (-bn * (bp1 * invg2 + vsdB * vp1) + ws * ump1 - fmp1);
            const real mp2 =
                cfac * (-bn * (bp2 * invg2 + vsdB * vp2) + ws * ump2 - fmp2);

            us.dens()           = ds;
            us.mcomponent(nhat) = mn;
            us.mcomponent(np1)  = mp1;
            us.mcomponent(np2)  = mp2;
            us.nrg()            = es - ds;
            us.bcomponent(nhat) = bn;
            us.bcomponent(np1)  = bp1;
            us.bcomponent(np2)  = bp2;
        }

        //------Return the HLLC flux
        return f + (us - u) * ws - us * vface;
    }();
    // upwind the concentration
    if (net_flux.dens() < 0.0) {
        net_flux.chi() = prR.chi() * net_flux.dens();
    }
    else {
        net_flux.chi() = prL.chi() * net_flux.dens();
    }

    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        const auto nj = next_perm(nhat, 1);
        const auto nk = next_perm(nhat, 2);
        if (vface <= aLm) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = 1.0,
                  .aR   = 0.0,
                  .dL   = 0.0,
                  .dR   = 0.0,
                  .vjL  = prL.vcomponent(nj),
                  .vjR  = 0.0,
                  .vkL  = prL.vcomponent(nk),
                  .vkR  = 0.0,
                }
            );
        }
        else if (vface >= aRp) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = 0.0,
                  .aR   = 1.0,
                  .dL   = 0.0,
                  .dR   = 0.0,
                  .vjL  = 0.0,
                  .vjR  = prR.vcomponent(nj),
                  .vkL  = 0.0,
                  .vkR  = prR.vcomponent(nk)
                }
            );
        }
        else {
            // if Bn is zero, then the HLLC solver admits a jummp in the
            // transverse magnetic field components across the middle wave.
            // If not, HLLC has the same flux and diffusion coefficients as
            // the HLL solver.
            if (null_normal_field) {
                constexpr auto half = static_cast<real>(0.5);
                const auto aaS      = std::abs(aS);
                net_flux.set_mdz_vars(
                    MignoneDelZannaVariables{
                      .lamL = aLm,
                      .lamR = aRp,
                      .aL   = half,
                      .aR   = half,
                      .dL   = half * (aaS - std::abs(aLm)) * chiL + half * aaS,
                      .dR   = half * (aaS - std::abs(aRp)) * chiR + half * aaS,
                      .vjL  = prL.vcomponent(nj),
                      .vjR  = prR.vcomponent(nj),
                      .vkL  = prL.vcomponent(nk),
                      .vkR  = prR.vcomponent(nk)
                    }
                );
            }
            else {
                const auto afac = 1.0 / (aRp - aLm);
                net_flux.set_mdz_vars(
                    MignoneDelZannaVariables{
                      .lamL = aLm,
                      .lamR = aRp,
                      .aL   = +aRp * afac,
                      .aR   = -aLm * afac,
                      .dL   = -aRp * aLm * afac,
                      .dR   = -aRp * aLm * afac,
                      .vjL  = prL.vcomponent(nj),
                      .vjR  = prR.vcomponent(nj),
                      .vkL  = prL.vcomponent(nk),
                      .vkR  = prR.vcomponent(nk)
                    }
                );
            }
        }
    }
    else {
        net_flux.calc_electric_field(unit_vectors::get<dim>(nhat));
    }
    return net_flux;
};

template <int dim>
DUAL real RMHD<dim>::hlld_vdiff(
    const real p,
    const RMHD<dim>::conserved_t r[2],
    const real lam[2],
    const real bn,
    const luint nhat,
    auto& praL,
    auto& praR,
    auto& prC
) const

{
    real eta[2], enthalpy[2];
    real kv[2][3], bv[2][3], vv[2][3];
    const auto sgnBn = sgn(bn) + global::epsilon;

    const auto np1 = next_perm(nhat, 1);
    const auto np2 = next_perm(nhat, 2);

    // compute Alfven terms
    for (int ii = 0; ii < 2; ii++) {
        const auto aS   = lam[ii];
        const auto rS   = r[ii];
        const auto rmn  = rS.mcomponent(nhat);
        const auto rmp1 = rS.mcomponent(np1);
        const auto rmp2 = rS.mcomponent(np2);
        const auto rbn  = rS.bcomponent(nhat);
        const auto rbp1 = rS.bcomponent(np1);
        const auto rbp2 = rS.bcomponent(np2);
        const auto ret  = rS.total_energy();

        // Eqs (26) - (30)
        const real a  = rmn - aS * ret + p * (1.0 - aS * aS);
        const real g  = rbp1 * rbp1 + rbp2 * rbp2;
        const real ag = (a + g);
        const real c  = rbp1 * rmp1 + rbp2 * rmp2;
        const real q  = -ag + bn * bn * (1.0 - aS * aS);
        const real x  = bn * (a * aS * bn + c) - ag * (aS * p + ret);

        // Eqs (23) - (25)
        const real term = (c + bn * (aS * rmn - ret));
        const real vn   = (bn * (a * bn + aS * c) - ag * (p + rmn)) / x;
        const real vp1  = (q * rmp1 + rbp1 * term) / x;
        const real vp2  = (q * rmp2 + rbp2 * term) / x;

        // Equation (21)
        const real var1 = 1.0 / (aS - vn);
        const real bp1  = (rbp1 - bn * vp1) * var1;
        const real bp2  = (rbp2 - bn * vp2) * var1;

        // Equation (31)
        const real rdv = (vn * rmn + vp1 * rmp1 + vp2 * rmp2);
        const real wt  = p + (ret - rdv) * var1;

        enthalpy[ii] = wt;

        // Equation (35) & (43)
        eta[ii]         = (ii == 0 ? -1.0 : 1.0) * sgnBn * std::sqrt(wt);
        const auto etaS = eta[ii];
        const real var2 = 1.0 / (aS * p + ret + bn * etaS);
        const real kn   = (rmn + p + rbn * etaS) * var2;
        const real kp1  = (rmp1 + rbp1 * etaS) * var2;
        const real kp2  = (rmp2 + rbp2 * etaS) * var2;

        vv[ii][0] = vn;
        vv[ii][1] = vp1;
        vv[ii][2] = vp2;

        // the normal component of the k-vector is the Alfven speed
        kv[ii][0] = kn;
        kv[ii][1] = kp1;
        kv[ii][2] = kp2;

        bv[ii][0] = bn;
        bv[ii][1] = bp1;
        bv[ii][2] = bp2;
    }

    // Load left and right vars
    const auto kL   = kv[LF];
    const auto kR   = kv[RF];
    const auto bL   = bv[LF];
    const auto bR   = bv[RF];
    const auto vL   = vv[LF];
    const auto vR   = vv[RF];
    const auto etaL = eta[LF];
    const auto etaR = eta[RF];

    // Compute contact terms
    // Equation (45)
    const auto dkn  = (kR[0] - kL[0]) + global::epsilon;
    const auto var3 = 1.0 / dkn;
    const auto bcn  = bn;
    const auto bcp1 = ((bR[1] * (kR[0] - vR[0]) + bn * vR[1]) -
                       (bL[1] * (kL[0] - vL[0]) + bn * vL[1])) *
                      var3;
    const auto bcp2 = ((bR[2] * (kR[0] - vR[0]) + bn * vR[2]) -
                       (bL[2] * (kL[0] - vL[0]) + bn * vL[2])) *
                      var3;
    // Left side Eq.(49)
    const auto kcnL  = kL[0];
    const auto kcp1L = kL[1];
    const auto kcp2L = kL[2];
    const auto ksqL  = kcnL * kcnL + kcp1L * kcp1L + kcp2L * kcp2L;
    const auto kdbL  = kcnL * bcn + kcp1L * bcp1 + kcp2L * bcp2;
    const auto regL  = (1.0 - ksqL) / (etaL - kdbL);
    const auto yL    = regL * var3;

    // Left side Eq.(47)
    const auto vncL  = kcnL - bcn * regL;
    const auto vpc1L = kcp1L - bcp1 * regL;
    const auto vpc2L = kcp2L - bcp2 * regL;

    // Right side Eq. (49)
    const auto kcnR  = kR[0];
    const auto kcp1R = kR[1];
    const auto kcp2R = kR[2];
    const auto ksqR  = kcnR * kcnR + kcp1R * kcp1R + kcp2R * kcp2R;
    const auto kdbR  = kcnR * bcn + kcp1R * bcp1 + kcp2R * bcp2;
    const auto regR  = (1.0 - ksqR) / (etaR - kdbR);
    const auto yR    = regR * var3;

    // Right side Eq. (47)
    const auto vncR  = kcnR - bcn * regR;
    const auto vpc1R = kcp1R - bcp1 * regR;
    const auto vpc2R = kcp2R - bcp2 * regR;

    // Equation (48)
    const auto f = [=] {
        if (goes_to_zero(dkn)) {
            return 0.0;
        }
        else if (goes_to_zero(bn)) {
            return dkn;
        }
        else {
            return dkn * (1.0 - bn * (yR - yL));
        }
    }();

    // check if solution is physically consistent, Eq. (54)
    auto eqn54ok = (vncL - kL[0]) > -global::epsilon;
    eqn54ok &= (kR[0] - vncR) > -global::epsilon;
    eqn54ok &= (lam[0] - vL[0]) < 0.0;
    eqn54ok &= (lam[1] - vR[0]) > 0.0;
    eqn54ok &= (enthalpy[1] - p) > 0.0;
    eqn54ok &= (enthalpy[0] - p) > 0.0;
    eqn54ok &= (kL[0] - lam[0]) > -global::epsilon;
    eqn54ok &= (lam[1] - kR[0]) > -global::epsilon;

    if (!eqn54ok) {
        return std::numeric_limits<real>::infinity();
    }

    // Fill in the Alfven (L / R) and Contact Prims
    praL.vcomponent(nhat) = vL[0];
    praL.vcomponent(np1)  = vL[1];
    praL.vcomponent(np2)  = vL[2];
    praL.bcomponent(nhat) = bL[0];
    praL.bcomponent(np1)  = bL[1];
    praL.bcomponent(np2)  = bL[2];
    praL.alfven()         = kL[0];

    praR.vcomponent(nhat) = vR[0];
    praR.vcomponent(np1)  = vR[1];
    praR.vcomponent(np2)  = vR[2];
    praR.bcomponent(nhat) = bR[0];
    praR.bcomponent(np1)  = bR[1];
    praR.bcomponent(np2)  = bR[2];
    praR.alfven()         = kR[0];

    prC.vcomponent(nhat) = 0.5 * (vncR + vncL);
    prC.vcomponent(np1)  = 0.5 * (vpc1R + vpc1L);
    prC.vcomponent(np2)  = 0.5 * (vpc2R + vpc2L);
    prC.bcomponent(nhat) = bcn;
    prC.bcomponent(np1)  = bcp1;
    prC.bcomponent(np2)  = bcp2;

    return f;
};

template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::calc_hlld_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const real bface
) const
{
    [[maybe_unused]] real chiL, chiR, laL, laR, veeL, veeR, veeS;
    const auto uL = prL.to_conserved(gamma);
    const auto uR = prR.to_conserved(gamma);
    const auto fL = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

    const auto lambda     = calc_eigenvals(prL, prR, nhat);
    const real aL         = lambda.afL();
    const real aR         = lambda.afR();
    const real aLm        = aL < 0.0 ? aL : 0.0;
    const real aRp        = aR > 0.0 ? aR : 0.0;
    const auto stationary = aLm == aRp;

    auto net_flux = [&]() {
        //---- Check Wave Speeds before wasting computations
        if (stationary) {
            return (fL + fR) * 0.5 - (uR + uL) * 0.5 * vface;
        }
        else if (vface <= aLm) {
            return fL - uL * vface;
        }
        else if (vface >= aRp) {
            return fR - uR * vface;
        }

        const real afac = 1.0 / (aRp - aLm);

        //-------------------Calculate the HLL Intermediate State
        const auto hll_state = (uR * aRp - uL * aLm - fR + fL) * afac;

        //------------------Calculate the RHLLE Flux---------------
        const auto hll_flux =
            (fL * aRp - fR * aLm + (uR - uL) * aRp * aLm) * afac;

        if (this->quirk_smoothing()) {
            if (quirk_strong_shock(prL.press(), prR.press())) {
                return hll_flux - hll_state * vface;
            }
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);
        // the normal component of the magnetic field is assumed to
        // be continuous across the interface, so bnL = bnR = bn
        const auto bn = bface;
        // const real bn = hll_state.bcomponent(nhat);

        // Eq. (12)
        const conserved_t r[2] = {uL * aLm - fL, uR * aRp - fR};
        const real lam[2]      = {aLm, aRp};

        //------------------------------------
        // Iteratively solve for the pressure
        //------------------------------------
        //------------ initial pressure guess
        const auto maybe_prim = cons2prim_single(hll_state);
        if (!maybe_prim.has_value()) {
            return hll_flux - hll_state * vface;
        }
        auto p0 = maybe_prim.value().total_pressure();

        // params to smoothen secant method if HLLD fails
        constexpr real feps          = global::epsilon;
        constexpr real peps          = global::epsilon;
        constexpr real prat_lim      = 0.01;    // pressure ratio limit
        constexpr real pguess_offset = 1.e-6;   // pressure guess offset
        constexpr int num_tries      = 15;      // secant tries before giving up
        bool hlld_success            = true;

        // L / R Alfven prims and Contact prims
        primitive_t prAL, prAR, prC;
        const auto p = [&] {
            if (bn * bn / p0 < prat_lim) {   // Eq.(53)
                // in this limit, the pressure is found through Eq. (55)
                const real et_hll  = hll_state.total_energy();
                const real fet_hll = hll_flux.total_energy();
                const real mn_hll  = hll_state.mcomponent(nhat);
                const real fmn_hll = hll_flux.mcomponent(nhat);

                const real b    = et_hll - fmn_hll;
                const real c    = fet_hll * mn_hll - et_hll * fmn_hll;
                const real quad = my_max(0.0, b * b - 4.0 * c);
                p0              = 0.5 * (-b + std::sqrt(quad));
            }

            auto f0 = hlld_vdiff(p0, r, lam, bn, nhat, prAL, prAR, prC);
            if (std::abs(f0) < feps) {
                return p0;
            }

            const real ptol = p0 * peps;
            real p1         = p0 * (1.0 + pguess_offset);
            auto iter       = 0;
            real dp;
            // Use the secant method to solve for the pressure
            do {
                auto f_lower =
                    hlld_vdiff(p1, r, lam, bn, nhat, prAL, prAR, prC);

                dp = (p1 - p0) / (f_lower - f0) * f_lower;
                p0 = p1;
                f0 = f_lower;
                p1 -= dp;

                if (iter++ > num_tries || !std::isfinite(f_lower)) {
                    hlld_success = false;
                    break;
                }

            } while (std::abs(dp) > ptol || std::abs(f0) > feps);

            return p1;
        }();
        // speed of the contact wave
        const auto vnc = prC.vcomponent(nhat);

        // Alfven speeds
        laL = prAL.alfven();
        laR = prAR.alfven();

        if constexpr (comp_ct_type == CTTYPE::MdZ) {
            // Eq. (46)
            auto degenerate = std::abs(laR - laL) < 1.e-9 * std::abs(aRp - aLm);
            const auto xL   = (prAL.vcomponent(nhat) - vnc) * (aLm - vnc) /
                            ((laL - aLm) * (laL + aLm - 2.0 * vnc));
            const auto xR = (prAR.vcomponent(nhat) - vnc) * (aRp - vnc) /
                            ((laR - aRp) * (laR + aRp - 2.0 * vnc));
            chiL = (laL - aLm) * xL;
            chiR = (laR - aRp) * xR;
            veeL = (laL + aLm) / (std::abs(laL) + std::abs(aLm));
            veeR = (laR + aRp) / (std::abs(laR) + std::abs(aRp));
            veeS = [=] {
                if (degenerate) {
                    return 0.0;
                }
                return (laR + laL) / (std::abs(laR) + std::abs(laL));
                ;
            }();
        }

        if (!hlld_success) {
            return hll_flux - hll_state * vface;
        }

        // do compound inequalities in two steps
        const auto on_left =
            (safe_less_than(vface, vnc) && safe_greater_than(vface, laL)) ||
            (safe_less_than(vface, laL) && safe_greater_than(vface, aLm));
        const auto at_contact =
            (safe_less_than(vface, laR) && safe_greater_than(vface, vnc)) ||
            (safe_less_than(vface, vnc) && safe_greater_than(vface, laL));

        const auto uc = on_left ? uL : uR;
        const auto pa = on_left ? prAL : prAR;
        const auto fc = on_left ? fL : fR;
        const auto rc = on_left ? r[LF] : r[RF];
        const auto lc = on_left ? aLm : aRp;
        const auto la = on_left ? laL : laR;

        // compute intermediate state across fast waves (Section 3.1)
        // === Fast / Slow Waves ===
        const real vna  = pa.vcomponent(nhat);
        const real vp1  = pa.vcomponent(np1);
        const real vp2  = pa.vcomponent(np2);
        const real bp1  = pa.bcomponent(np1);
        const real bp2  = pa.bcomponent(np2);
        const real vdba = pa.vdotb();

        const real fac = 1.0 / (lc - vna);
        const real da  = rc.dens() * fac;
        const real ea  = (rc.total_energy() + p * vna - vdba * bn) * fac;
        const real mn  = (ea + p) * vna - vdba * bn;
        const real mp1 = (ea + p) * vp1 - vdba * bp1;
        const real mp2 = (ea + p) * vp2 - vdba * bp2;

        conserved_t ua;
        ua.dens()           = da;
        ua.mcomponent(nhat) = mn;
        ua.mcomponent(np1)  = mp1;
        ua.mcomponent(np2)  = mp2;
        ua.nrg()            = ea - da;
        ua.bcomponent(nhat) = bn;
        ua.bcomponent(np1)  = bp1;
        ua.bcomponent(np2)  = bp2;

        const auto fa = fc + (ua - uc) * lc;

        if (!at_contact) {
            return fa - ua * vface;
        }

        // === Contact Wave ===
        // compute jump conditions across alfven waves (Section 3.3)
        const real vdbC = prC.vdotb();
        const real bnC  = prC.bcomponent(nhat);
        const real bp1C = prC.bcomponent(np1);
        const real bp2C = prC.bcomponent(np2);
        const real vp1C = prC.vcomponent(np1);
        const real vp2C = prC.vcomponent(np2);
        const real fac2 = 1.0 / (la - vnc);
        const real dc   = da * (la - vna) * fac2;
        const real ec   = (ea * la - mn + p * vnc - vdbC * bn) * fac2;
        const real mnc  = (ec + p) * vnc - vdbC * bn;
        const real mpc1 = (ec + p) * vp1C - vdbC * bp1C;
        const real mpc2 = (ec + p) * vp2C - vdbC * bp2C;

        conserved_t ut;
        ut.dens()           = dc;
        ut.mcomponent(nhat) = mnc;
        ut.mcomponent(np1)  = mpc1;
        ut.mcomponent(np2)  = mpc2;
        ut.nrg()            = ec - dc;
        ut.bcomponent(nhat) = bnC;
        ut.bcomponent(np1)  = bp1C;
        ut.bcomponent(np2)  = bp2C;

        return fa + (ut - ua) * la - ut * vface;
    }();

    // upwind the concentration
    if (net_flux.dens() < 0.0) {
        net_flux.chi() = prR.chi() * net_flux.dens();
    }
    else {
        net_flux.chi() = prL.chi() * net_flux.dens();
    }

    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        const auto nj       = next_perm(nhat, 1);
        const auto nk       = next_perm(nhat, 2);
        constexpr auto half = static_cast<real>(0.5);
        if (vface <= aLm) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL  = aLm,
                  .lamR  = aRp,
                  .aL    = 1.0,
                  .aR    = 0.0,
                  .dL    = 0.0,
                  .dR    = 0.0,
                  .vjL   = prL.vcomponent(nj),
                  .vjR   = 0.0,
                  .vkL   = prL.vcomponent(nk),
                  .vkR   = 0.0,
                  .vnorm = prL.vcomponent(nhat)
                }
            );
        }
        else if (vface >= aRp) {
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL  = aLm,
                  .lamR  = aRp,
                  .aL    = 0.0,
                  .aR    = 1.0,
                  .dL    = 0.0,
                  .dR    = 0.0,
                  .vjL   = 0.0,
                  .vjR   = prR.vcomponent(nj),
                  .vkL   = 0.0,
                  .vkR   = prR.vcomponent(nk),
                  .vnorm = prR.vcomponent(nhat)
                }
            );
        }
        else {
            // if Bn is zero, then the HLLC solver admits a jummp in the
            // transverse magnetic field components across the middle wave.
            // If not, HLLC has the same flux and diffusion coefficients as
            // the HLL solver.
            net_flux.set_mdz_vars(
                MignoneDelZannaVariables{
                  .lamL = aLm,
                  .lamR = aRp,
                  .aL   = half * (1.0 + veeS),
                  .aR   = half * (1.0 - veeS),
                  .dL   = half * (veeL - veeS) * chiL +
                        half * (std::abs(laL) - veeS * laL),
                  .dR = half * (veeR - veeS) * chiR +
                        half * (std::abs(laR) - veeS * laR),
                  .vjL   = prL.vcomponent(nj),
                  .vjR   = prR.vcomponent(nj),
                  .vkL   = prL.vcomponent(nk),
                  .vkR   = prR.vcomponent(nk),
                  .vnorm = 0.5 * (prL.vcomponent(nhat) + prR.vcomponent(nhat))
                }
            );
        }
    }
    else {
        net_flux.calc_electric_field(unit_vectors::get<dim>(nhat));
    }
    return net_flux;
};

template <int dim>
void RMHD<dim>::sync_flux_boundaries()
{
    // sync only in perpendicular directions
    this->conserved_boundary_manager().template sync_boundaries<flux_tag>(
        this->full_xvertex_policy(),
        fri,
        fri.contract({1, 1, 0}),
        this->bcs(),
        this->mesh()
    );
    this->conserved_boundary_manager().template sync_boundaries<flux_tag>(
        this->full_yvertex_policy(),
        gri,
        gri.contract({1, 0, 1}),
        this->bcs(),
        this->mesh()
    );
    this->conserved_boundary_manager().template sync_boundaries<flux_tag>(
        this->full_zvertex_policy(),
        hri,
        hri.contract({0, 1, 1}),
        this->bcs(),
        this->mesh()
    );
}

template <int dim>
void RMHD<dim>::sync_magnetic_boundaries()
{
    bfield_man_.sync_boundaries(
        this->full_xvertex_policy(),
        bstag1,
        bstag1.contract({1, 1, 0}),
        this->bcs(),
        this->mesh()
    );
    bfield_man_.sync_boundaries(
        this->full_yvertex_policy(),
        bstag2,
        bstag2.contract({1, 0, 1}),
        this->bcs(),
        this->mesh()
    );
    bfield_man_.sync_boundaries(
        this->full_zvertex_policy(),
        bstag3,
        bstag3.contract({0, 1, 1}),
        this->bcs(),
        this->mesh()
    );

    // check if the magnetic field is divergence free
    // for (size_type kk = 0; kk < this->mesh().grid().active_gridsize(2); kk++)
    // {
    //     for (size_type jj = 0; jj < this->mesh().grid().active_gridsize(1);
    //          jj++) {
    //         for (size_type ii = 0; ii <
    //         this->mesh().grid().active_gridsize(0);
    //              ii++) {
    //             const auto cell =
    //                 this->mesh().get_cell_from_indices(ii, jj, kk);
    //             const auto b1L = bstag1.at(ii + 0, jj + 1, kk + 1);
    //             const auto b1R = bstag1.at(ii + 1, jj + 1, kk + 1);
    //             const auto b2L = bstag2.at(ii + 1, jj + 0, kk + 1);
    //             const auto b2R = bstag2.at(ii + 1, jj + 1, kk + 1);
    //             const auto b3L = bstag3.at(ii + 1, jj + 1, kk + 0);
    //             const auto b3R = bstag3.at(ii + 1, jj + 1, kk + 1);
    //             const auto divB =
    //                 divergence(cell, b1L, b1R, b2L, b2R, b3L, b3R);

    //             if (!goes_to_zero(divB)) {
    //                 printf(
    //                     "b3L: %f, b3R: %f, b3L(ghost): %f, b3R(ghost) %f\n",
    //                     b3L,
    //                     b3R,
    //                     bstag3.at(ii, jj, kk + 0),
    //                     bstag3.at(ii, jj, kk + 1)
    //                 );
    //                 std::cout << "[" << this->current_iter() << "] "
    //                           << "Divergence of B is not zero at (" << ii
    //                           << ", " << jj << ", " << kk << ") " << divB
    //                           << std::endl;
    //                 std::cin.get();
    //             }
    //         }
    //     }
    // }
}

template <int dim>
void RMHD<dim>::riemann_fluxes()
{
    fri.contract({1, 1, 0}).stencil_transform(
        [this] DEV(auto& fr, const auto& prim, const auto& bface) {
            // fluxes in i direction, centered at i-1/2
            const auto& pL = prim.at(-1, 0, 0);
            const auto& pR = prim.at(+0, 0, 0);

            if (!this->using_pcm()) {
                const auto& pLL = prim.at(-2, 0, 0);
                const auto& pRR = prim.at(+1, 0, 0);

                const auto pLr =
                    pL + plm_gradient(pL, pLL, pR, this->plm_theta()) * 0.5;
                const auto pRr =
                    pR - plm_gradient(pR, pL, pRR, this->plm_theta()) * 0.5;
                return (this->*riemann_solve)(pLr, pRr, 1, 0.0, bface.value());
            }
            else {
                return (this->*riemann_solve)(pL, pR, 1, 0.0, bface.value());
            }
        },
        this->xvertex_policy(),
        this->prims_.contract(this->halo_radius()),
        bstag1.contract({1, 1, 0})
    );

    gri.contract({1, 0, 1}).stencil_transform(
        [this] DEV(auto& gr, const auto& prim, const auto& bface) {
            // fluxes in j direction, centered at j-1/2
            const auto& pL = prim.at(0, -1, 0);
            const auto& pR = prim.at(0, +0, 0);

            if (!this->using_pcm()) {
                const auto& pLL = prim.at(0, -2, 0);
                const auto& pRR = prim.at(0, +1, 0);

                const auto pLr =
                    pL + plm_gradient(pL, pLL, pR, this->plm_theta()) * 0.5;
                const auto pRr =
                    pR - plm_gradient(pR, pL, pRR, this->plm_theta()) * 0.5;
                return (this->*riemann_solve)(pLr, pRr, 2, 0.0, bface.value());
            }
            else {
                return (this->*riemann_solve)(pL, pR, 2, 0.0, bface.value());
            }
        },
        this->yvertex_policy(),
        this->prims_.contract(this->halo_radius()),
        bstag2.contract({1, 0, 1})
    );

    hri.contract({0, 1, 1}).stencil_transform(
        [this] DEV(auto& hr, const auto& prim, const auto& bface) {
            // fluxes in k direction, centered at k-1/2
            const auto& pL = prim.at(0, 0, -1);
            const auto& pR = prim.at(0, 0, +0);

            if (!this->using_pcm()) {
                const auto& pLL = prim.at(0, 0, -2);
                const auto& pRR = prim.at(0, 0, +1);

                const auto pLr =
                    pL + plm_gradient(pL, pLL, pR, this->plm_theta()) * 0.5;
                const auto pRr =
                    pR - plm_gradient(pR, pL, pRR, this->plm_theta()) * 0.5;
                return (this->*riemann_solve)(pLr, pRr, 3, 0.0, bface.value());
            }
            else {
                return (this->*riemann_solve)(pL, pR, 3, 0.0, bface.value());
            }
        },
        this->zvertex_policy(),
        this->prims_.contract(this->halo_radius()),
        bstag3.contract({0, 1, 1})
    );
}

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
template <int nhat>
void RMHD<dim>::update_magnetic_component(const ExecutionPolicy<>& policy)
{
    auto& b_stag = (nhat == 1) ? bstag1 : (nhat == 2) ? bstag2 : bstag3;
    constexpr auto inner_region = [&]() {
        if constexpr (nhat == 1) {
            return collapsable<dim>{1, 1, 0};
        }
        else if constexpr (nhat == 2) {
            return collapsable<dim>{1, 0, 1};
        }
        else {
            return collapsable<dim>{0, 1, 1};
        }
    }();

    // Create contracted views for EMF calculation
    // all staggered fields are center at the {i,j,k}-1/2 location
    auto magnetic_update = [this] DEV(auto& b_view, const auto& prim) -> real {
        const auto [ii, jj, kk] = b_view.position();
        const auto cell = this->mesh().get_cell_from_indices(ii, jj, kk);
        ct::EMField<ct_scheme_t> efield;

        // Configure edges based on component
        efield.compute_edge_components<next_perm(nhat, 1), next_perm(nhat, 2)>(
            fri,
            gri,
            hri,
            bstag1_old,
            bstag2_old,
            bstag3_old,
            prim
        );

        // if (nhat == 3) {
        //     const auto res =
        //         b_view.value() -
        //         this->time_step() * curl_component<nhat>(cell, efield);
        //     if (ii == 0 && jj == 0 && kk == 0) {
        //         std::cout << "iter: " << this->current_iter() << std::endl;
        //         std::cout << "B3 at k-1/2: " << b_view.value() << " -> " <<
        //         res
        //                   << std::endl;
        //     }
        //     else if (ii == 0 && jj == 0 && kk == 1) {
        //         std::cout << "iter: " << this->current_iter() << std::endl;
        //         std::cout << "B3 at k+1/2: " << b_view.value() << " -> " <<
        //         res
        //                   << std::endl;
        //     }
        // }
        return b_view.value() -
               this->time_step() * curl_component<nhat>(cell, efield);
    };

    // Transform using stencil view
    b_stag.contract(inner_region)
        .stencil_transform(
            magnetic_update,
            policy,
            this->prims_.contract(this->halo_radius())
        );
}

template <int dim>
void RMHD<dim>::advance_magnetic_fields()
{
    // for the magnetic field components, one needs
    // to be careful on what fluxes to consider and
    // for each plane. For example, for B1, the electric
    // field components in question will be E2 and E3
    // or, if going by cyclic permutation, EJ = E2, EK = E3
    // which require the IK planar fluxes for EJ aqnd IJ
    // planar fluxes for EK
    update_magnetic_component<1>(this->xvertex_policy());
    update_magnetic_component<2>(this->yvertex_policy());
    update_magnetic_component<3>(this->zvertex_policy());
}

template <int dim>
void RMHD<dim>::advance_conserved()
{
    auto dcons = [this] DEV(
                     const auto& fr,
                     const auto& gr,
                     const auto& hr,
                     const auto& source_terms,
                     const auto& gravity,
                     const auto& geometrical_sources,
                     const auto& cell
                 ) -> conserved_t {
        conserved_t res;
        for (int q = 1; q > -1; q--) {
            // q = 0 is L, q = 1 is R
            const auto sign = (q == 1) ? 1 : -1;
            res -= fr.at(q, 0, 0) * cell.inverse_volume(0) * cell.area(0 + q) *
                   sign;
            if constexpr (dim > 1) {
                res -= gr.at(0, q, 0) * cell.inverse_volume(1) *
                       cell.area(2 + q) * sign;
                if constexpr (dim > 2) {
                    res -= hr.at(0, 0, q) * cell.inverse_volume(2) *
                           cell.area(4 + q) * sign;
                }
            }
        }

        res += source_terms;
        res += gravity;
        res += geometrical_sources;

        return res * this->time_step();
    };

    auto update_conserved = [this, dcons] DEV(
                                auto& con,
                                const auto& prim,
                                const auto& fr,
                                const auto& gr,
                                const auto& hr,
                                const auto& b1,
                                const auto& b2,
                                const auto& b3
                            ) {
        const auto [ii, jj, kk] = con.position();
        // mesh factors
        const auto cell = this->mesh().get_cell_from_indices(ii, jj, kk);

        // compute the change in conserved variables
        const auto dc = dcons(
            fr,
            gr,
            hr,
            this->hydro_sources(prim.value(), cell),
            this->gravity_sources(prim.value(), cell),
            cell.geometrical_sources(prim.value(), gamma),
            cell
        );

        // update the mean magnetic field, separate from
        // the conserved gas variables. Only up to second order accurate
        // averaing -1/2 with +1/2 face for each component
        con.value().bcomponent(1) = 0.5 * (b1.at(0, 0, 0) + b1.at(1, 0, 0));
        con.value().bcomponent(2) = 0.5 * (b2.at(0, 0, 0) + b2.at(0, 1, 0));
        con.value().bcomponent(3) = 0.5 * (b3.at(0, 0, 0) + b3.at(0, 0, 1));

        // update conserved (gas) variables
        return con.value().increment_gas_terms(dc);
    };

    // Transform using stencil operations
    this->cons_.contract(this->halo_radius())
        .stencil_transform(
            update_conserved,
            this->interior_policy(),
            this->prims_.contract(this->halo_radius()),
            fri.contract({1, 1, 0}),
            gri.contract({1, 0, 1}),
            hri.contract({0, 1, 1}),
            bstag1.contract({1, 1, 0}),
            bstag2.contract({1, 0, 1}),
            bstag3.contract({0, 1, 1})
        );
}

template <int dim>
void RMHD<dim>::advance_impl()
{
    riemann_fluxes();
    sync_flux_boundaries();
    advance_magnetic_fields();
    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        sync_magnetic_boundaries();
    }
    advance_conserved();
    this->apply_boundary_conditions();

    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        // copy_from does a deep copy, but on device if need be,
        // so we need not do any syncs
        bstag1_old.copy_from(bstag1, this->full_xvertex_policy());
        bstag2_old.copy_from(bstag2, this->full_yvertex_policy());
        bstag3_old.copy_from(bstag3, this->full_zvertex_policy());
    }
}

//===================================================================================================================
//                                            INITIALIZE SIMULATE
//===================================================================================================================
template <int dim>
void RMHD<dim>::init_simulation()
{
    init_riemann_solver();
    this->apply_boundary_conditions();
    const auto& xP = this->full_xvertex_policy();
    const auto& yP = this->full_yvertex_policy();
    const auto& zP = this->full_zvertex_policy();
    // allocate space for Riemann fluxes
    fri.resize(xP.get_active_extent())
        .reshape(
            {this->active_nz() + 2,
             this->active_ny() + 2,
             this->active_nx() + 1}
        );
    gri.resize(yP.get_active_extent())
        .reshape(
            {this->active_nz() + 2,
             this->active_ny() + 1,
             this->active_nx() + 2}
        );
    hri.resize(zP.get_active_extent())
        .reshape(
            {this->active_nz() + 1,
             this->active_ny() + 2,
             this->active_nx() + 2}
        );

    bstag1.reshape(
        {this->active_nz() + 2, this->active_ny() + 2, this->active_nx() + 1}
    );
    bstag2.reshape(
        {this->active_nz() + 2, this->active_ny() + 1, this->active_nx() + 2}
    );
    bstag3.reshape(
        {this->active_nz() + 1, this->active_ny() + 2, this->active_nx() + 2}
    );

    // will need the current bfields if using the
    // CT scheme of Mignone & Del Zanna (2021)
    if constexpr (comp_ct_type == CTTYPE::MdZ) {
        bstag1_old = bstag1;
        bstag2_old = bstag2;
        bstag3_old = bstag3;
    }

    sync_all_to_device();
};
