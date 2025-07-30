#ifndef SIMBI_PHYSICS_WAVE_SPEEDS_HPP
#define SIMBI_PHYSICS_WAVE_SPEEDS_HPP

#include "base/concepts.hpp"
#include "config.hpp"               // for DEV, real, global
#include "contact_properties.hpp"   // for wave_speeds_t
#include "containers/vector.hpp"    // for unit_vector_t
#include "physics/hydro/physics.hpp"   // for is_hydro_primitive_c, is_mhd_primitive_c, is_rmhd_c, is_srhd_c
#include "utility/enums.hpp"     // for WaveSpeedEstimate
#include "utility/helpers.hpp"   // for solve_quartic,

#include <algorithm>     // for std::min, std::max
#include <cmath>         // for std::sqrt, std::pow
#include <cstdio>        // for printf
#include <tuple>         // for std::tuple_size, std::tuple_element
#include <type_traits>   // for std::integral_constant, std::is_same_v

namespace simbi::hydro {
    struct wave_speeds_t;
}   // namespace simbi::hydro

// structured bindings for wave_speeds_t
namespace std {
    template <>
    struct tuple_size<simbi::hydro::wave_speeds_t>
        : std::integral_constant<size_t, 2> {
    };

    template <>
    struct tuple_element<0, simbi::hydro::wave_speeds_t> {
        using type = simbi::real;
    };

    template <>
    struct tuple_element<1, simbi::hydro::wave_speeds_t> {
        using type = simbi::real;
    };
}   // namespace std

namespace simbi::hydro {
    struct wave_speeds_t {
        real left, right;
        DEV constexpr auto min() const { return left; }
        DEV constexpr auto max() const { return right; }
        // structured bindings support
        template <std::size_t Index>
        std::tuple_element_t<Index, wave_speeds_t>& get()
        {
            if constexpr (Index == 0) {
                return left;
            }
            else if constexpr (Index == 1) {
                return right;
            }
            else {
                // this will never be reached due to tuple_size specialization,
                // but it's needed to satisfy the compiler
                static_assert(Index < 2, "Index out of bounds");
                return left;   // unreachable, but needed for compilation
            }
        }

        template <std::size_t Index>
        const std::tuple_element_t<Index, wave_speeds_t>& get() const
        {
            if constexpr (Index == 0) {
                return left;
            }
            else if constexpr (Index == 1) {
                return right;
            }
            else {
                // this will never be reached due to tuple_size specialization,
                // but it's needed to satisfy the compiler
                static_assert(Index < 2, "Index out of bounds");
                return left;   // unreachable, but needed for compilation
            }
        }
    };
}   // namespace simbi::hydro

namespace simbi::hydro::newtonian {
    struct wave_properties_t {
        wave_speeds_t speeds;
        contact_properties_t contact;
    };

    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t wave_speeds(
        const primitive_t& prim,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto cs = std::sqrt(gamma * prim.pre / prim.rho);
        const auto vn = vecops::dot(prim.vel, nhat);
        return {vn - cs, vn + cs};
    }

    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t extremal_speeds(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto left_waves  = wave_speeds(primL, nhat, gamma);
        const auto right_waves = wave_speeds(primR, nhat, gamma);
        return {
          std::min({left_waves.min(), right_waves.min(), 0.0}),
          std::max({left_waves.max(), right_waves.max(), 0.0})
        };
    }

    template <is_hydro_primitive_c primitive_t>
    DEV wave_properties_t wave_properties(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto rhoL = primL.rho;
        const auto rhoR = primR.rho;
        const auto pL   = primL.pre;
        const auto pR   = primR.pre;
        const auto csL  = std::sqrt(gamma * pL / rhoL);
        const auto csR  = std::sqrt(gamma * pR / rhoR);
        const auto vL   = vecops::dot(primL.vel, nhat);
        const auto vR   = vecops::dot(primR.vel, nhat);

        real pStar, qL, qR;
        // ---- Standard adiabatic case ----
        const auto rho_bar = 0.5 * (rhoL + rhoR);
        const auto c_bar   = 0.5 * (csL + csR);
        const real pvrs = 0.5 * (pL + pR) - 0.5 * (vR - vL) * rho_bar * c_bar;
        const real pmin = std::min(pL, pR);
        const real pmax = std::max(pL, pR);

        // Section 9.5.2 of Toro's book suggests a user-defined
        // threshold for the PVRS case
        constexpr auto q_user = 2.0;
        if (pmax / pmin <= q_user && pmin <= pvrs && pvrs <= pmax) {
            // PVRS case
            pStar = pvrs;
        }
        else if (pvrs <= pmin) {
            // two-rarefaction case
            const real gamma_factor = (gamma - 1.0) / (2.0 * gamma);
            const real pL_pow       = std::pow(pL, gamma_factor);
            const real pR_pow       = std::pow(pR, gamma_factor);
            pStar                   = std::pow(
                (csL + csR - 0.5 * (gamma - 1.0) * (vR - vL)) /
                    (csL / pL_pow + csR / pR_pow),
                1.0 / gamma_factor
            );
        }
        else {
            // two-shock case
            const real alpha_L = 2.0 / ((gamma + 1) * rhoL);
            const real alpha_R = 2.0 / ((gamma + 1) * rhoR);
            const real beta_L  = (gamma - 1) / (gamma + 1) * pL;
            const real beta_R  = (gamma - 1) / (gamma + 1) * pR;
            const real p0      = std::max(0.0, pvrs);
            const real gL      = std::sqrt(alpha_L / (p0 + beta_L));
            const real gR      = std::sqrt(alpha_R / (p0 + beta_R));
            pStar              = (gL * pL + gR * pR - (vR - vL)) / (gL + gR);
        }

        // compute q factors for adiabatic case
        if (pStar <= pL) {
            qL = 1.0;
        }
        else {
            qL = std::sqrt(
                1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pL - 1.0)
            );
        }
        if (pStar <= pR) {
            qR = 1.0;
        }
        else {
            qR = std::sqrt(
                1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pR - 1.0)
            );
        }

        // signal speeds
        const real aL = vL - csL * qL;
        const real aR = vR + csR * qR;

        // middle wave speed (contact discontinuity)
        const real aStar =
            (pR - pL + rhoL * vL * (aL - vL) - rhoR * vR * (aR - vR)) /
            (rhoL * (aL - vL) - rhoR * (aR - vR));

        return {{aL, aR}, {aStar, pStar}};
    }

}   // namespace simbi::hydro::newtonian

namespace simbi::hydro::srhd {
    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t wave_speeds(
        const primitive_t& prim,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto vn = vecops::dot(prim.vel, nhat);
        const auto cs = sound_speed(prim, gamma);
        switch (comp_wave_speed) {
            case WaveSpeedEstimate::MIGNONE_AND_BODO_05: {
                // get wave speeds based on Mignone & Bodo Eqs. (21.0 - 23)
                const auto w = 1.0 / std::sqrt(1.0 - (vn * vn));
                const auto s = cs * cs / (w * w * (1.0 - cs * cs));
                // define temporaries to save computational cycles
                const real qf  = 1.0 / (1.0 + s);
                const real fac = std::sqrt(s * (1.0 - vn * vn + s));
                return {(vn - fac) * qf, (vn + fac) * qf};
            }
            default:   // Davis wave speeds
            {
                return {
                  (vn - cs) / (1.0 - cs * vn),
                  (vn + cs) / (1.0 + cs * vn)
                };
            }
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t extremal_speeds(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto left_waves  = wave_speeds(primL, nhat, gamma);
        const auto right_waves = wave_speeds(primR, nhat, gamma);
        return {
          std::min({left_waves.min(), right_waves.min(), 0.0}),
          std::max({left_waves.max(), right_waves.max(), 0.0})
        };
    }

}   // namespace simbi::hydro::srhd

namespace simbi::hydro::rmhd {
    using namespace simbi::helpers;
    template <is_mhd_primitive_c primitive_t>
    DEV wave_speeds_t wave_speeds(
        const primitive_t& prim,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
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
        const auto rho   = prim.rho;
        const auto w     = lorentz_factor(prim);
        const auto h     = enthalpy(prim, gamma);
        const auto cssq  = sound_speed_squared(prim, gamma);
        const auto bmu   = magnetic_four_vector(prim);
        const auto bmusq = bmu.inner_product(bmu);
        const auto bn    = vecops::dot(prim.mag, nhat);
        const auto bnsq  = bn * bn;
        const auto vn    = vecops::dot(prim.vel, nhat);
        const auto vsq   = vecops::dot(prim.vel, prim.vel);
        if (vsq < global::epsilon) {   // Eq.(57)
            const auto fac     = 1.0 / (rho * h + bmusq);
            const auto a       = 1.0;
            const auto b       = -(bmusq + rho * h * cssq + bnsq * cssq) * fac;
            const auto c       = cssq * bnsq * fac;
            const auto disq    = std::sqrt(b * b - 4.0 * a * c);
            const auto lambdaR = std::sqrt(0.5 * (-b + disq));
            const auto lambdaL = -lambdaR;
            return {lambdaL, lambdaR};
        }
        else if (bnsq < global::epsilon) {   // Eq. (58)
            const real g2      = w * w;
            const real vdbperp = vecops::dot(prim.vel, prim.mag) - vn * bn;
            const real q       = bmusq - cssq * vdbperp * vdbperp;
            const real a2      = rho * h * (cssq + g2 * (1.0 - cssq)) + q;
            const real a1      = -2.0 * rho * h * g2 * vn * (1.0 - cssq);
            const real a0 = rho * h * (-cssq + g2 * vn * vn * (1.0 - cssq)) - q;
            const real disq    = a1 * a1 - 4.0 * a2 * a0;
            const auto lambdaR = 0.5 * (-a1 + std::sqrt(disq)) / a2;
            const auto lambdaL = 0.5 * (-a1 - std::sqrt(disq)) / a2;
            return {lambdaL, lambdaR};
        }
        else {   // solve the full quartic Eq. (56)
            // initialize quartic speed array
            real speeds[4] = {0.0, 0.0, 0.0, 0.0};

            const auto bmu0 = bmu[0];
            const auto bmun = bmu.spatial_dot(nhat);
            const auto w2   = w * w;
            const auto vn2  = vn * vn;

            const auto a4 =
                (-bmu0 * bmu0 * cssq + bmusq * w2 - cssq * w2 * w2 * h * rho +
                 cssq * w2 * h * rho + w2 * w2 * h * rho);
            const auto fac = 1.0 / a4;

            const auto a3 =
                fac *
                (2.0 * bmu0 * bmun * cssq - 2.0 * bmusq * w2 * vn +
                 4.0 * cssq * w2 * w2 * h * rho * vn -
                 2.0 * cssq * w2 * h * rho * vn - 4.0 * w2 * w2 * h * rho * vn);
            const auto a2 =
                fac *
                (bmu0 * bmu0 * cssq + bmusq * w2 * vn2 - bmusq * w2 -
                 bmun * bmun * cssq - 6.0 * cssq * w2 * w2 * h * rho * vn2 +
                 cssq * w2 * h * rho * vn2 - cssq * w2 * h * rho +
                 6.0 * w2 * w2 * h * rho * vn2);

            const auto a1 =
                fac * (-2.0 * bmu0 * bmun * cssq + 2.0 * bmusq * w2 * vn +
                       4.0 * cssq * w2 * w2 * h * rho * vn * vn2 +
                       2.0 * cssq * w2 * h * rho * vn -
                       4.0 * w2 * w2 * h * rho * vn * vn2);

            const auto a0 = fac * (-bmusq * w2 * vn2 + bmun * bmun * cssq -
                                   cssq * w2 * w2 * h * rho * vn2 * vn2 -
                                   cssq * w2 * h * rho * vn2 +
                                   w2 * w2 * h * rho * vn2 * vn2);

            const auto nroots = solve_quartic(a3, a2, a1, a0, speeds);

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

            // if there are no roots, return null vector
            if (nroots == 0) {
                return {0.0, 0.0};
            }
            return {speeds[0], speeds[nroots - 1]};
        }
    }

    template <is_mhd_primitive_c primitive_t>
    DEV wave_speeds_t extremal_speeds(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        const auto left_waves  = wave_speeds(primL, nhat, gamma);
        const auto right_waves = wave_speeds(primR, nhat, gamma);
        return {
          std::min({left_waves.min(), right_waves.min(), 0.0}),
          std::max({left_waves.max(), right_waves.max(), 0.0})
        };
    }
}   // namespace simbi::hydro::rmhd

namespace simbi::hydro {
    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t wave_speeds(
        const primitive_t& prim,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        if constexpr (is_rmhd_c<primitive_t>) {
            return rmhd::wave_speeds(prim, nhat, gamma);
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            return srhd::wave_speeds(prim, nhat, gamma);
        }
        else {   // newtonian hydro
            return newtonian::wave_speeds(prim, nhat, gamma);
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV wave_speeds_t extremal_speeds(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        if constexpr (is_rmhd_c<primitive_t>) {
            return rmhd::extremal_speeds(primL, primR, nhat, gamma);
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            return srhd::extremal_speeds(primL, primR, nhat, gamma);
        }
        else {   // newtonian hydro
            return newtonian::extremal_speeds(primL, primR, nhat, gamma);
        }
    }
}   // namespace simbi::hydro

#endif   // SIMBI_PHYSICS_WAVE_SPEEDS_HPP
