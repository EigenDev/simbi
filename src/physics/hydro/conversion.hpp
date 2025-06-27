#ifndef SIMBI_PHYSICS_CONVERSION_HPP
#define SIMBI_PHYSICS_CONVERSION_HPP

#include "compute/functional/monad/maybe.hpp"   // for Maybe, None
#include "config.hpp"                           // for global::epsilon
#include "core/base/concepts.hpp"               // for is_hydro_conserved_c
#include "core/types/alias.hpp"                 // for real, std::uint64_t
#include "physics/eos/ideal.hpp"                // for ideal_gas_eos_t
#include "physics/hydro/physics.hpp"            // for pressure_from_conserved
#include "system/io/exceptions.hpp"             // for ErrorCode
#include <cmath>                                // for abs, isfinite, sqrt

namespace simbi::hydro::newtonian {
    using namespace simbi::eos;

    template <
        concepts::is_hydro_conserved_c conserved_t,
        typename EoS = ideal_gas_eos_t<conserved_t::regime>>
    DEV auto to_primitive(const conserved_t& cons, real gamma)
        -> Maybe<typename conserved_t::counterpart_t>
    {
        using primitive_t = typename conserved_t::counterpart_t;
        primitive_t prim;
        prim.rho = cons.den;
        prim.vel = cons.mom / cons.den;
        prim.pre = pressure_from_conserved<EoS>(cons, gamma);

        if (prim.pre < 0.0 || !std::isfinite(prim.pre)) {
            return None(
                ErrorCode::NEGATIVE_PRESSURE | ErrorCode::NON_FINITE_PRESSURE
            );
        }
        return prim;
    }
}   // namespace simbi::hydro::newtonian

namespace simbi::hydro::srhd {
    using namespace simbi::build::types;
    using namespace simbi::eos;

    template <
        concepts::is_hydro_conserved_c conserved_t,
        typename EoS = ideal_gas_eos_t<conserved_t::regime>>
    DEV constexpr auto to_primitive(const conserved_t& cons, real gamma)
        -> Maybe<typename conserved_t::counterpart_t>
    {
        using primitive_t = typename conserved_t::counterpart_t;
        const auto& d     = cons.den;
        const auto& svec  = cons.mom;
        const auto& tau   = cons.nrg;
        const auto& dchi  = cons.chi;
        const auto smag   = svec.norm();

        // Perform modified Newton Raphson based on
        // https://www.sciencedirect.com/science/article/pii/S0893965913002930
        // so far, the convergence rate is the same, but perhaps I need
        // a slight tweak
        std::uint64_t iter = 0;
        real peq           = std::abs(smag - d - tau);
        const real tol     = d * global::epsilon;
        real dp;
        do {
            // compute x_[k+1]
            auto [f, g] = newton_fg(gamma, tau, d, smag, peq);
            dp          = f / g;
            peq -= dp;

            if (iter >= constants::max_iterations || !std::isfinite(peq)) {
                return simbi::None(
                    ErrorCode::MAX_ITER | ErrorCode::NON_FINITE_ROOT
                );
            }
            iter++;

        } while (std::abs(dp) >= tol);

        if (peq < 0) {
            return simbi::None(ErrorCode::NEGATIVE_PRESSURE);
        }

        const auto inv_et   = 1.0 / (tau + d + peq);
        const auto velocity = svec * inv_et;
        const auto w = 1.0 / std::sqrt(1.0 - vecops::dot(velocity, velocity));

        return primitive_t{
          d / w,
          velocity * (global::using_four_velocity ? w : 1.0),
          peq,
          dchi / d
        };
    }

}   // namespace simbi::hydro::srhd

namespace simbi::hydro::rmhd {
    using namespace simbi::eos;
    using namespace simbi::concepts;

    template <
        is_mhd_conserved_c conserved_t,
        typename EoS = ideal_gas_eos_t<conserved_t::regime>>
    DEV constexpr auto to_primitive(const conserved_t& cons, real gamma)
        -> Maybe<typename conserved_t::counterpart_t>
    {
        using primitive_t = typename conserved_t::counterpart_t;
        const auto d      = cons.den;
        const auto mom    = cons.mom;
        const auto tau    = cons.nrg;
        const auto bfield = cons.mag;
        const auto dchi   = cons.chi;

        //==================================================================
        // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
        //==================================================================

        //======= rescale the variables Eqs. (22) - (25)
        const auto invd   = 1.0 / d;
        const auto isqrtd = std::sqrt(invd);
        const auto q      = tau * invd;
        const auto rvec   = mom * invd;
        const auto rsq    = vecops::dot(rvec, rvec);
        const auto rmag   = std::sqrt(rsq);
        const auto hvec   = bfield * isqrtd;
        const auto beesq  = vecops::dot(hvec, hvec) + global::epsilon;
        const auto rdb    = vecops::dot(rvec, hvec);
        const auto rdbsq  = rdb * rdb;
        // r-parallel Eq. (25.a)
        const auto rparr = rdb / beesq * hvec;
        // r-perpendicular, Eq. (25.b)
        const auto rperp = rvec - rparr;
        const auto rpsq  = vecops::dot(rperp, rperp);

        // We use the false position method to solve for the roots
        auto mu_lower = 0.0;
        auto mu_upper = find_mu_plus(beesq, rdbsq, rmag);
        auto f_lower =
            kkc_fmu44(mu_lower, rmag, rpsq, beesq, rdbsq, q, d, gamma);
        auto f_upper =
            kkc_fmu44(mu_upper, rmag, rpsq, beesq, rdbsq, q, d, gamma);

        std::uint64_t iter = 0.0;
        real mu, ff;
        do {
            mu =
                (mu_lower * f_upper - mu_upper * f_lower) / (f_upper - f_lower);
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
            if (iter >= constants::max_iterations || !std::isfinite(ff)) {
                return simbi::None([iter]() -> ErrorCode {
                    if (iter >= constants::max_iterations) {
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
        const auto x = 1.0 / (1.0 + mu * beesq);

        // Equation (38)
        const auto rbar_sq = rsq * x * x + mu * x * (1.0 + x) * rdbsq;

        // Equation (39)
        const auto qbar = q - 0.5 * (beesq + mu * mu * x * x * beesq * rpsq);

        // Equation (32)
        const auto vsq  = mu * mu * rbar_sq;
        const auto gbsq = vsq / (1.0 - vsq);
        const auto w    = std::sqrt(1.0 + gbsq);

        // Equation (41)
        const auto rhohat = d / w;

        // Equation (42)
        const auto eps = w * (qbar - mu * rbar_sq) + gbsq / (1.0 + w);
        // zero-temperature limit for gamma-law EoS
        constexpr auto pfloor = 1.0e-3;
        const auto epshat     = my_max(eps, pfloor / (rhohat * (gamma - 1.0)));

        // Equation (43)
        const auto pg = (gamma - 1.0) * rhohat * epshat;

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

        return primitive_t{rhohat, vel, pg, bfield, dchi / d};
    }
}   // namespace simbi::hydro::rmhd
#endif
