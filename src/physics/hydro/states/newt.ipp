#include "core/managers/boundary_manager.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/utility/atomic_bool.hpp"   // for shared_atomic_bool
#include "io/exceptions.hpp"
#include "physics/hydro/schemes/viscosity/viscous.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include "util/tools/device_api.hpp"
#include "util/tools/helpers.hpp"
#include <cmath>   // for max, min

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
    InitialConditions& init_conditions
)
    : HydroBase<Newtonian<dim>, dim, Regime::NEWTONIAN>(state, init_conditions),
      isothermal_(init_conditions.isothermal),
      shakura_sunyaev_alpha_(init_conditions.shakura_sunyaev_alpha),
      sound_speed_squared_(init_conditions.sound_speed_squared)
{
    this->context_.gamma               = gamma;
    this->context_.is_isothermal       = goes_to_zero(gamma - 1.0);
    this->context_.alpha_ss            = shakura_sunyaev_alpha_;
    this->context_.ambient_sound_speed = std::sqrt(sound_speed_squared_);
}

// Destructor
template <int dim>
Newtonian<dim>::~Newtonian() = default;

//-----------------------------------------------------------------------------------------
//                          Get The Primitive
//-----------------------------------------------------------------------------------------
template <int dim>
void Newtonian<dim>::cons2prim_impl()
{
    atomic::simbi_atomic<bool> local_failure{false};
    this->prims_.transform(
        [gamma = this->gamma, loc = local_failure.get(), iso = isothermal_] DEV(
            auto& /* prim */,
            const auto& cons_var
        ) -> Maybe<primitive_t> {
            const auto& rho = cons_var.dens();
            const auto vel  = cons_var.momentum() / rho;
            const auto chi  = cons_var.chi() / rho;
            const auto pre  = cons_var.pressure(gamma, iso);

            if (pre < 0 || !std::isfinite(pre)) {
                // store the invalid state
                loc->store(true);
                return simbi::None(
                    ErrorCode::NEGATIVE_PRESSURE |
                    ErrorCode::NON_FINITE_PRESSURE
                );
            }

            return primitive_t{rho, vel, pre, chi};
        },
        this->full_policy(),
        this->cons_
    );

    if (local_failure.load()) {
        this->set_failure_state(true);
    }
}

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
    const real pL   = primsL.press();

    const real rhoR = primsR.rho();
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.press();

    const auto csL = std::sqrt(gamma * pL / rhoL);
    const auto csR = std::sqrt(gamma * pR / rhoR);

    switch (this->solver_type()) {
        case Solver::HLLC: {
            if (isothermal_) {

                // For isothermal: cs should be constant, but use local values
                // for robustness
                const real cs_avg = 0.5 * (csL + csR);

                // Isothermal p* calculation using Riemann invariants
                // u* = 0.5 * [(uL + uR) + cs * ln(rhoL/rhoR)]
                const real u_star =
                    0.5 * (vL + vR + cs_avg * std::log(rhoL / rhoR));

                // For isothermal: solve for rho* using Riemann invariants
                // From left state: u* + cs*ln(rho*) = uL + cs*ln(rhoL)
                // From right state: u* - cs*ln(rho*) = uR - cs*ln(rhoR)
                const real ln_rho_star_from_L =
                    (vL - u_star) / cs_avg + std::log(rhoL);
                const real ln_rho_star_from_R =
                    (u_star - vR) / cs_avg + std::log(rhoR);
                const real ln_rho_star =
                    0.5 * (ln_rho_star_from_L + ln_rho_star_from_R);
                const real rho_star = std::exp(ln_rho_star);

                // p* = rho* * cs^2 for isothermal
                const real pStar = rho_star * cs_avg * cs_avg;

                // Wave speeds for isothermal case
                real qL, qR;

                if (pStar <= pL) {
                    // Rarefaction wave on left
                    qL = 1.0;
                }
                else {
                    // Shock wave on left - isothermal Rankine-Hugoniot
                    // For isothermal shocks: rho2/rho1 = p2/p1 (since cs =
                    // constant)
                    qL = std::sqrt(pStar / pL);
                }

                if (pStar <= pR) {
                    // Rarefaction wave on right
                    qR = 1.0;
                }
                else {
                    // Shock wave on right - isothermal Rankine-Hugoniot
                    qR = std::sqrt(pStar / pR);
                }

                // Signal speeds
                const real aL = vL - csL * qL;
                const real aR = vR + csR * qR;

                // Contact wave speed (same formula works for isothermal)
                const real aStar =
                    (pR - pL + rhoL * vL * (aL - vL) - rhoR * vR * (aR - vR)) /
                    (rhoL * (aL - vL) - rhoR * (aR - vR));

                return {aL, aR, aStar, pStar};
            }
            else {
                real pStar, qL, qR;
                // ---- Standard adiabatic case ----
                const auto rho_bar = 0.5 * (rhoL + rhoR);
                const auto c_bar   = 0.5 * (csL + csR);
                const real pvrs =
                    0.5 * (pL + pR) - 0.5 * (vR - vL) * rho_bar * c_bar;
                const real pmin = my_min(pL, pR);
                const real pmax = my_max(pL, pR);

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
                    const real p0      = my_max(0.0, pvrs);
                    const real gL      = std::sqrt(alpha_L / (p0 + beta_L));
                    const real gR      = std::sqrt(alpha_R / (p0 + beta_R));
                    pStar = (gL * pL + gR * pR - (vR - vL)) / (gL + gR);
                }

                // Compute q factors for adiabatic case
                if (pStar <= pL) {
                    qL = 1.0;
                }
                else {
                    qL = std::sqrt(
                        1.0 +
                        ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pL - 1.0)
                    );
                }
                if (pStar <= pR) {
                    qR = 1.0;
                }
                else {
                    qR = std::sqrt(
                        1.0 +
                        ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pR - 1.0)
                    );
                }

                // Signal speeds
                const real aL = vL - csL * qL;
                const real aR = vR + csR * qR;

                // Middle wave speed (contact discontinuity)
                const real aStar =
                    (pR - pL + rhoL * vL * (aL - vL) - rhoR * vR * (aR - vR)) /
                    (rhoL * (aL - vL) - rhoR * (aR - vR));

                return {aL, aR, aStar, pStar};
            }
        }

        default: {
            const real aR = my_max3<real>(vL + csL, vR + csR, 0.0);
            const real aL = my_min3<real>(vL - csL, vR - csR, 0.0);
            return {aL, aR};
        }
    }
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::calc_hlle_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const auto& viscL,
    const auto& viscR
) const
{
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL();
    const real aR     = lambda.aR();
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    auto fL           = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    auto fR           = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));
    if (!goes_to_zero(this->viscosity())) {
        fL -= viscL;   // add the viscous stress to momentum flux
        fR -= viscR;   // add the viscous stress to momentum flux
    }

    auto net_flux = [&] {
        // Compute the HLL Flux component-wise
        if (vface <= aL) {
            return fL - uL * vface;
        }
        else if (vface >= aR) {
            return fR - uR * vface;
        }
        else {
            auto f_hll = (fL * aR - fR * aL + (uR - uL) * aR * aL) / (aR - aL);
            auto u_hll = (uR * aR - uL * aL - fR + fL) / (aR - aL);
            return f_hll - u_hll * vface;
        }
    }();

    // // Upwind the scalar concentration
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
    const real vface,
    const auto& viscL,
    const auto& viscR
) const
{
    if constexpr (dim > 1) {
        if (this->quirk_smoothing() &&
            quirk_strong_shock(prL.press(), prR.press())) {
            return calc_hlle_flux(prL, prR, nhat, vface, viscL, viscR);
        }
    }

    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL();
    const real aR     = lambda.aR();
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    auto fL           = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    auto fR           = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));
    if (!goes_to_zero(this->viscosity())) {
        fL -= viscL;   // subtract the viscous stress from momentum flux tensor
        fR -= viscR;   // subtract the viscous stress from momentum flux tensor
    }

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
            real pressure = prL.press();
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
            real pressure = prR.press();
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
        const real aStar = lambda.aStar();
        const real pStar = lambda.pStar();

        // --------------Compute the L Star State----------
        real pressure = prL.press();
        real rho      = uL.dens();
        auto mom      = uL.momentum();
        real nrg      = uL.nrg();
        real fac      = 1.0 / (aL - aStar);

        const real vL   = prL.vcomponent(nhat);
        const real vR   = prR.vcomponent(nhat);
        const auto ehat = unit_vectors::get<dim>(nhat);

        // Left Star State in x-direction of coordinate lattice
        real rhostar = fac * (aL - vL) * rho;
        auto mstar   = fac * (mom * (aL - vL) + ehat * (pStar - pressure));
        real estar = fac * (nrg * (aL - vL) + (pStar * aStar - pressure * vL));
        const auto starStateL = conserved_t{rhostar, mstar, estar};

        pressure = prR.press();
        rho      = uR.dens();
        mom      = uR.momentum();
        nrg      = uR.nrg();
        fac      = 1.0 / (aR - aStar);

        rhostar = fac * (aR - vR) * rho;
        mstar   = fac * (mom * (aR - vR) + ehat * (-pressure + pStar));
        estar   = fac * (nrg * (aR - vR) + (pStar * aStar - pressure * vR));
        const auto starStateR = conserved_t{rhostar, mstar, estar};

        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020:
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        const real phi = EntropyDetector::compute_adaptive_phi(
            prL,
            prR,
            unit_vectors::get<dim>(nhat),
            gamma,
            // flag for Fleischmann et al. (2020) low-Mach fix
            this->fleischmann_limiter()
        );
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

template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::calc_slau_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const auto& viscL,
    const auto& viscR
) const
{
    // I had a lot of fun checking out
    // A sequel to AUSM, Part II: AUSM+-up for all speeds by Meng-Sing Liou
    // and this works great for low-Mach flows and isothermal flows
    // https://www.sciencedirect.com/science/article/pii/S0021999105004274
    // and then I learned about Simple Low Dissipation AUSM (SLAU)
    // by Shima & Kitamura (2009-2011) adn wanted to implement it
    // slau parameters
    constexpr real beta  = 1.0 / 8.0;   // pressure diffusion parameter
    constexpr real alpha = 2.0;         // mach number scaling

    // get primitive variables
    const real rhoL = prL.rho();
    const real rhoR = prR.rho();
    const real pL   = prL.press();
    const real pR   = prR.press();
    const real vL   = prL.vcomponent(nhat);
    const real vR   = prR.vcomponent(nhat);

    // compute sound speeds - gamma handles isothermal case
    const real csL = std::sqrt(gamma * pL / rhoL);
    const real csR = std::sqrt(gamma * pR / rhoR);

    // interface sound speed and density
    const real cs_half  = 0.5 * (csL + csR);
    const real rho_half = 0.5 * (rhoL + rhoR);

    // mach numbers
    const real ML = vL / csL;
    const real MR = vR / csR;

    // slau mach number functions
    auto M_plus_slau = [](real M) -> real {
        if (M >= 1.0) {
            return M;
        }
        else if (M <= -1.0) {
            return 0.0;
        }
        else {
            return 0.25 * (M + 1.0) * (M + 1.0);
        }
    };

    auto M_minus_slau = [](real M) -> real {
        if (M >= 1.0) {
            return 0.0;
        }
        else if (M <= -1.0) {
            return M;
        }
        else {
            return -0.25 * (M - 1.0) * (M - 1.0);
        }
    };

    // slau pressure functions
    auto P_plus_slau = [](real M) -> real {
        if (M >= 1.0) {
            return 1.0;
        }
        else if (M <= -1.0) {
            return 0.0;
        }
        else {
            return 0.25 * (M + 1.0) * (M + 1.0) * (2.0 - M);
        }
    };

    auto P_minus_slau = [](real M) -> real {
        if (M >= 1.0) {
            return 0.0;
        }
        else if (M <= -1.0) {
            return 1.0;
        }
        else {
            return 0.25 * (M - 1.0) * (M - 1.0) * (2.0 + M);
        }
    };

    // split mach numbers and pressures
    const real M_plus_L  = M_plus_slau(ML);
    const real M_minus_R = M_minus_slau(MR);
    const real P_plus_L  = P_plus_slau(ML);
    const real P_minus_R = P_minus_slau(MR);

    // slau interface mach number with pressure weighting
    const real M_ausm = M_plus_L + M_minus_R;

    // pressure-weighted interface velocity for better contact preservation
    const real u_tilde = (pL * vR + pR * vL) / (pL + pR);
    const real M_tilde = u_tilde / cs_half;

    // slau modification parameter
    const real chi = my_min(1.0, my_max(std::abs(ML), std::abs(MR)));

    // slau interface mach number
    const real M_half = chi * M_ausm + (1.0 - chi) * M_tilde;

    // interface pressure with slau enhancement
    const real p_ausm = P_plus_L * pL + P_minus_R * pR;

    // pressure diffusion term for contact preservation
    const real p_diffusion = -beta * rho_half * cs_half *
                             my_max(std::abs(ML), std::abs(MR)) * (pR - pL);

    // velocity diffusion term
    const real vel_diffusion =
        -alpha * cs_half * rho_half * P_plus_L * P_minus_R * (vR - vL);

    const real p_half = p_ausm + p_diffusion;

    // mass flux
    const real mass_flux = cs_half * rho_half * M_half;

    // get conservative variables and base fluxes
    const auto uL = prL.to_conserved(gamma);
    const auto uR = prR.to_conserved(gamma);
    auto fL       = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    auto fR       = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

    // apply viscous corrections if needed
    if (!goes_to_zero(this->viscosity())) {
        fL -= viscL;
        fR -= viscR;
    }

    // construct slau flux
    conserved_t slau_flux;

    // density flux
    slau_flux.dens() = mass_flux;

    // momentum flux with pressure weighting for better contact preservation
    if (mass_flux >= 0.0) {
        // upwind from left but with slau modifications
        slau_flux.mcomponent(nhat) = mass_flux * vL + p_half + vel_diffusion;

        // transverse momentum components
        if constexpr (dim > 1) {
            for (luint i = 1; i <= dim; ++i) {
                if (i != nhat) {
                    const real u_trans_L = prL.vcomponent(i);
                    const real u_trans_R = prR.vcomponent(i);
                    // pressure-weighted transverse velocity
                    const real u_trans_interface =
                        (pL * u_trans_R + pR * u_trans_L) / (pL + pR);
                    slau_flux.mcomponent(i) =
                        mass_flux *
                        (chi * u_trans_L + (1.0 - chi) * u_trans_interface);
                }
            }
        }

        // energy flux
        if (!isothermal_) {
            const real HL = (uL.nrg() + pL) / rhoL;   // specific enthalpy
            const real HR = (uR.nrg() + pR) / rhoR;
            // pressure-weighted enthalpy for better contact preservation
            const real H_interface = (pL * HR + pR * HL) / (pL + pR);
            slau_flux.nrg() =
                mass_flux * (chi * HL + (1.0 - chi) * H_interface);
        }
        else {
            // for isothermal, energy is not evolved
            slau_flux.nrg() = 0.0;
        }
    }
    else {
        // upwind from right but with slau modifications
        slau_flux.mcomponent(nhat) = mass_flux * vR + p_half + vel_diffusion;

        // transverse momentum components
        if constexpr (dim > 1) {
            for (luint i = 1; i <= dim; ++i) {
                if (i != nhat) {
                    const real u_trans_L = prL.vcomponent(i);
                    const real u_trans_R = prR.vcomponent(i);
                    // pressure-weighted transverse velocity
                    const real u_trans_interface =
                        (pL * u_trans_R + pR * u_trans_L) / (pL + pR);
                    slau_flux.mcomponent(i) =
                        mass_flux *
                        (chi * u_trans_R + (1.0 - chi) * u_trans_interface);
                }
            }
        }

        // energy flux
        if (!isothermal_) {
            const real HL = (uL.nrg() + pL) / rhoL;   // specific enthalpy
            const real HR = (uR.nrg() + pR) / rhoR;
            // pressure-weighted enthalpy for better contact preservation
            const real H_interface = (pL * HR + pR * HL) / (pL + pR);
            slau_flux.nrg() =
                mass_flux * (chi * HR + (1.0 - chi) * H_interface);
        }
        else {
            // for isothermal, energy is not evolved
            slau_flux.nrg() = 0.0;
        }
    }

    // handle scalar concentration with slau pressure weighting
    const real chi_interface = (pL * prR.chi() + pR * prL.chi()) / (pL + pR);
    if (mass_flux >= 0.0) {
        slau_flux.chi() =
            mass_flux * (chi * prL.chi() + (1.0 - chi) * chi_interface);
    }
    else {
        slau_flux.chi() =
            mass_flux * (chi * prR.chi() + (1.0 - chi) * chi_interface);
    }

    // apply face velocity correction
    const auto avg_state = (mass_flux >= 0.0) ? uL : uR;
    return slau_flux - avg_state * vface;
}

template <int dim>
DUAL Newtonian<dim>::conserved_t Newtonian<dim>::calc_ausm_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface,
    const auto& viscL,
    const auto& viscR
) const
{
    constexpr real alpha = 3.0 / 16.0;   // Low-Mach enhancement parameter
    constexpr real Kp    = 0.75;         // Pressure diffusion coefficient
    constexpr real sigma = 1.0;   // Scaling parameter for pressure diffusion

    // get primitive variables
    const real rhoL = prL.rho();
    const real rhoR = prR.rho();
    const real pL   = prL.press();
    const real pR   = prR.press();
    const real vL   = prL.vcomponent(nhat);
    const real vR   = prR.vcomponent(nhat);

    // compute sound speeds
    const real csL = std::sqrt(gamma * pL / rhoL);
    const real csR = std::sqrt(gamma * pR / rhoR);

    // interface sound speed (Roe mean for simplicity)
    const real cs_half = (std::sqrt(rhoL) * csL + std::sqrt(rhoR) * csR) /
                         (std::sqrt(rhoL) + std::sqrt(rhoR));

    // Mach numbers
    const real ML = vL / csL;
    const real MR = vR / csR;

    // AUSM+-up Mach number splitting functions
    auto M_plus_up = [](real M) -> real {
        if (std::abs(M) >= 1.0) {
            return 0.5 * (M + std::abs(M));
        }
        else {
            const real M2   = M * M;
            const real beta = alpha * M * (M2 - 1.0) * (M2 - 1.0);
            return 0.25 * (M + 1.0) * (M + 1.0) + beta;
        }
    };

    auto M_minus_up = [](real M) -> real {
        if (std::abs(M) >= 1.0) {
            return 0.5 * (M - std::abs(M));
        }
        else {
            const real M2   = M * M;
            const real beta = alpha * M * (M2 - 1.0) * (M2 - 1.0);
            return -0.25 * (M - 1.0) * (M - 1.0) - beta;
        }
    };

    // AUSM+-up  pressure splitting functions
    auto P_plus_up = [](real M) -> real {
        if (std::abs(M) >= 1.0) {
            return 0.5 * (1.0 + sgn(M));
        }
        else {
            const real M2          = M * M;
            const real base        = 0.25 * (M + 1.0) * (M + 1.0) * (2.0 - M);
            const real enhancement = alpha * M * (M2 - 1.0) * (M2 - 1.0);
            return base + enhancement;
        }
    };

    auto P_minus_up = [](real M) -> real {
        if (std::abs(M) >= 1.0) {
            return 0.5 * (1.0 - sgn(M));
        }
        else {
            const real M2          = M * M;
            const real base        = 0.25 * (M - 1.0) * (M - 1.0) * (2.0 + M);
            const real enhancement = alpha * M * (M2 - 1.0) * (M2 - 1.0);
            return base - enhancement;
        }
    };

    // split Mach numbers and pressures using AUSM+-up functions
    const real M_plus_L  = M_plus_up(ML);
    const real M_minus_R = M_minus_up(MR);
    const real P_plus_L  = P_plus_up(ML);
    const real P_minus_R = P_minus_up(MR);

    // interface Mach number
    const real M_half = M_plus_L + M_minus_R;

    // AUSM+-up pressure with diffusion term
    const real p_base      = P_plus_L * pL + P_minus_R * pR;
    const real p_diffusion = -Kp * P_plus_L * P_minus_R * (rhoL + rhoR) *
                             cs_half * (vR - vL) * sigma;
    const real p_half = p_base + p_diffusion;

    // mass flux
    const real mass_flux =
        cs_half * ((M_half > 0.0) ? M_half * rhoL : M_half * rhoR);

    // get conservative variables and base fluxes
    const auto uL = prL.to_conserved(gamma);
    const auto uR = prR.to_conserved(gamma);
    auto fL       = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    auto fR       = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

    // apply viscous corrections if needed
    if (!goes_to_zero(this->viscosity())) {
        fL -= viscL;
        fR -= viscR;
    }

    // construct AUSM+-up flux
    conserved_t ausm_flux;

    if (mass_flux >= 0.0) {
        // upwind from left
        ausm_flux.dens() = mass_flux;

        // momentum fluxes: convective part + pressure part
        ausm_flux.mcomponent(nhat) = mass_flux * vL + p_half;

        // transverse momentum components
        if constexpr (dim > 1) {
            for (luint i = 1; i <= dim; ++i) {
                if (i != nhat) {
                    ausm_flux.mcomponent(i) = mass_flux * prL.vcomponent(i);
                }
            }
        }

        // energy flux
        if (!isothermal_) {
            const real HL   = (uL.nrg() + pL) / rhoL;   // specific enthalpy
            ausm_flux.nrg() = mass_flux * HL;
        }
        else {
            // for isothermal, energy is not evolved
            ausm_flux.nrg() = 0.0;
        }
    }
    else {
        // upwind from right
        ausm_flux.dens() = mass_flux;

        // momentum fluxes: convective part + pressure part
        ausm_flux.mcomponent(nhat) = mass_flux * vR + p_half;

        // transverse momentum components
        if constexpr (dim > 1) {
            for (luint i = 1; i <= dim; ++i) {
                if (i != nhat) {
                    ausm_flux.mcomponent(i) = mass_flux * prR.vcomponent(i);
                }
            }
        }

        // energy flux
        if (!isothermal_) {
            const real HR   = (uR.nrg() + pR) / rhoR;   // specific enthalpy
            ausm_flux.nrg() = mass_flux * HR;
        }
        else {
            // for isothermal, energy is not evolved
            ausm_flux.nrg() = 0.0;
        }
    }

    // handle scalar concentration
    if (ausm_flux.dens() < 0.0) {
        ausm_flux.chi() = prR.chi() * ausm_flux.dens();
    }
    else {
        ausm_flux.chi() = prL.chi() * ausm_flux.dens();
    }

    // Aapply face velocity correction
    const auto avg_state = (mass_flux >= 0.0) ? uL : uR;
    return ausm_flux - avg_state * vface;
}

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void Newtonian<dim>::advance_impl()
{
    auto dcons = [this] DEV(
                     const auto& fri,
                     const auto& gri,
                     const auto& hri,
                     const auto& source_terms,
                     const auto& gravity,
                     const auto& geometrical_sources,
                     const auto& cell
                 ) -> conserved_t {
        conserved_t res;
        for (int q = 1; q > -1; q--) {
            // q = 0 is L, q = 1 is R
            const auto sign = (q == 1) ? 1 : -1;
            res -= fri[q] * cell.inverse_volume() * cell.area(0 + q) * sign;
            if constexpr (dim > 1) {
                res -= gri[q] * cell.inverse_volume() * cell.area(2 + q) * sign;
                if constexpr (dim > 2) {
                    res -= hri[q] * cell.inverse_volume() * cell.area(4 + q) *
                           sign;
                }
            }
        }
        res += source_terms;
        res += gravity;
        res += geometrical_sources;

        return res * this->time_step();
    };

    auto calc_flux = [this, dcons] DEV(auto& con, const auto& prim) {
        conserved_t fri[2], gri[2], hri[2];
        primitive_t pLx, pRx, pLy, pRy, pLz, pRz;

        const auto [ii, jj, kk] = con.position();
        const auto cell = this->mesh().get_cell_from_indices(ii, jj, kk);

        // Calculate fluxes using prim
        for (int q = 0; q < 2; q++) {
            // X-direction flux
            // const auto& pL = prim.at(q - 1, 0, 0);
            // const auto& pR = prim.at(q - 0, 0, 0);
            pLx = prim.at(q - 1, 0, 0);
            pRx = prim.at(q - 0, 0, 0);

            auto vface = cell.velocity(q);

            if (!this->using_pcm()) {
                const auto& pLL = prim.at(q - 2, 0, 0);
                const auto& pRR = prim.at(q + 1, 0, 0);
                // compute the reconstructed states
                pLx += plm_gradient(pLx, pLL, pRx, this->plm_theta()) * 0.5;
                pRx -= plm_gradient(pRx, pLx, pRR, this->plm_theta()) * 0.5;
            }

            if constexpr (dim > 1) {
                // Y-direction flux
                pLy = prim.at(0, q - 1, 0);
                pRy = prim.at(0, q - 0, 0);
                if (!this->using_pcm()) {
                    const auto& pLLy = prim.at(0, q - 2, 0);
                    const auto& pRRy = prim.at(0, q + 1, 0);

                    pLy +=
                        plm_gradient(pLy, pLLy, pRy, this->plm_theta()) * 0.5;
                    pRy -=
                        plm_gradient(pRy, pLy, pRRy, this->plm_theta()) * 0.5;
                }
            }

            if constexpr (dim > 2) {
                // Z-direction flux
                // const auto& pL_z = prim.at(0, 0, q - 1);
                // const auto& pR_z = prim.at(0, 0, q - 0);
                pLz = prim.at(0, 0, q - 1);
                pRz = prim.at(0, 0, q - 0);
                if (!this->using_pcm()) {
                    const auto& pLLz = prim.at(0, 0, q - 2);
                    const auto& pRRz = prim.at(0, 0, q + 1);

                    pLz +=
                        plm_gradient(pLz, pLLz, pRz, this->plm_theta()) * 0.5;
                    pRz -=
                        plm_gradient(pRz, pLz, pRRz, this->plm_theta()) * 0.5;
                }
            }

            const auto [viscxL, viscxR] = visc::viscous_flux<
                1>(pLx, pRx, pLy, pRy, pLz, pRz, cell, this->viscosity());

            fri[q] = (this->*riemann_solve)(pLx, pRx, 1, vface, viscxL, viscxR);

            if constexpr (dim > 1) {
                vface = cell.velocity(q + 2);

                const auto [viscyL, viscyR] = visc::viscous_flux<
                    2>(pLx, pRx, pLy, pRy, pLz, pRz, cell, this->viscosity());
                gri[q] =
                    (this->*riemann_solve)(pLy, pRy, 2, vface, viscyL, viscyR);
            }
            if constexpr (dim > 2) {
                vface = cell.velocity(q + 4);

                const auto [visczL, visczR] = visc::viscous_flux<
                    3>(pLx, pRx, pLy, pRy, pLz, pRz, cell, this->viscosity());
                hri[q] =
                    (this->*riemann_solve)(pLz, pRz, 3, vface, visczL, visczR);
            }
        }

        auto delta_con = dcons(
            fri,
            gri,
            hri,
            this->hydro_sources(prim.value(), cell),
            this->gravity_sources(prim.value(), cell),
            cell.geometrical_sources(prim.value(), gamma),
            cell
        );

        // if immersed boundaries are present, include their effects
        // on the fluid
        if (this->has_immersed_bodies()) {
            delta_con += this->ib_sources(
                prim.value(),
                cell,
                std::make_tuple(ii, jj, kk)
            );
        }

        // Return updated conserved values (if isothermal, energy is not
        // updated)
        return con.value().increment_gas_terms(delta_con, isothermal_);
    };

    // Transform using stencil operations
    this->cons_.contract(this->halo_radius())
        .stencil_transform(
            calc_flux,
            this->interior_policy(),
            this->prims_.contract(this->halo_radius())
        );

    this->apply_boundary_conditions();
}

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
template <int dim>
void Newtonian<dim>::init_simulation()
{
    init_riemann_solver();
    this->sync_to_device();
    this->apply_boundary_conditions();
    // use parent's sync to device method
    // this->sync_to_device();
};
