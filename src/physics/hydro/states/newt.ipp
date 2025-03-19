#include "core/managers/boundary_manager.hpp"
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
      sound_speed_squared_(init_conditions.sound_speed_squared)
{
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
    shared_atomic_bool local_failure;
    this->prims_.transform(
        [gamma      = this->gamma,
         loc        = &local_failure,
         isothermal = isothermal_,
         cs2 = sound_speed_squared_] DEV(auto& prim, const auto& cons_var)
            -> Maybe<primitive_t> {
            const auto& rho = cons_var.dens();
            const auto vel  = cons_var.momentum() / rho;
            const auto& chi = cons_var.chi() / rho;
            const auto pre  = cons_var.pressure(gamma, isothermal, cs2);

            if (pre < 0 || !std::isfinite(pre)) {
                // store the invalid state
                loc->store(true);
                return simbi::Nothing;
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

    const real csR = std::sqrt(gamma * pR / rhoR);
    const real csL = std::sqrt(gamma * pL / rhoL);
    switch (this->solver_type()) {
        case Solver::HLLC: {
            // Pressure-based wave speed estimates (Batten et al. 1997)
            const real pvrs = 0.5 * (pL + pR) - 0.5 * (vR - vL) *
                                                    (rhoL + rhoR) * 0.5 *
                                                    (csL + csR);
            const real pmin = my_min(pL, pR);
            const real pmax = my_max(pL, pR);

            real pStar;
            if (pmax / pmin <= 2.0 && pmin <= pvrs && pvrs <= pmax) {
                // PVRS estimate if pressure ratio is small
                pStar = pvrs;
            }
            else {
                if (isothermal_) {
                    // Isothermal case - simplified wave speed estimate
                    // Using isothermal sound speed directly
                    pStar = (rhoL * csL * pR + rhoR * csR * pL -
                             rhoL * rhoR * csL * csR * (vR - vL)) /
                            (rhoL * csL + rhoR * csR);
                }
                else {
                    // Two-rarefaction approximation for adiabatic case
                    const real gamma_factor = (gamma - 1.0) / (2.0 * gamma);
                    const real pL_pow       = std::pow(pL, gamma_factor);
                    const real pR_pow       = std::pow(pR, gamma_factor);

                    pStar = std::pow(
                        (csL + csR - 0.5 * (gamma - 1.0) * (vR - vL)) /
                            (csL / pL_pow + csR / pR_pow),
                        2.0 * gamma / (gamma - 1.0)
                    );
                }
            }

            // Compute wave speeds using pressure estimate
            real qL, qR;
            if (pStar <= pL) {
                // Rarefaction wave on left side
                qL = 1.0;
            }
            else {
                // Shock wave on left side - use Rankine-Hugoniot relation
                qL = std::sqrt(
                    1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pL - 1.0)
                );
            }

            if (pStar <= pR) {
                // Rarefaction wave on right side
                qR = 1.0;
            }
            else {
                // Shock wave on right side - use Rankine-Hugoniot relation
                qR = std::sqrt(
                    1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (pStar / pR - 1.0)
                );
            }

            // Signal speeds - left and right waves
            const real aL = vL - csL * qL;   // Left wave speed
            const real aR = vR + csR * qR;   // Right wave speed

            // Middle wave speed (contact discontinuity)
            // Using a more robust formula that correctly handles vacuum
            // conditions
            const real aStar =
                (pR - pL + rhoL * vL * (aL - vL) - rhoR * vR * (aR - vR)) /
                (rhoL * (aL - vL) - rhoR * (aR - vR));

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
    const auto fL     = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR     = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

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
    const auto fL     = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR     = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));

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
        const real cL    = lambda.csL();
        const real cR    = lambda.csR();
        const real aStar = lambda.aStar();
        const real pStar = lambda.pStar();
        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020:
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        constexpr real ma_lim = 0.10;

        // --------------Compute the L Star State----------
        real pressure = prL.press();
        real rho      = uL.dens();
        real m1       = uL.mcomponent(1);
        real m2       = uL.mcomponent(2);
        real m3       = uL.mcomponent(3);
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

        pressure = prR.press();
        rho      = uR.dens();
        m1       = uR.mcomponent(1);
        m2       = uR.mcomponent(2);
        m3       = uR.mcomponent(3);
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
            res -= fri[q] * cell.inverse_volume(0) * cell.area(0 + q) * sign;
            if constexpr (dim > 1) {
                res -=
                    gri[q] * cell.inverse_volume(1) * cell.area(2 + q) * sign;
                if constexpr (dim > 2) {
                    res -= hri[q] * cell.inverse_volume(2) * cell.area(4 + q) *
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

        const auto [ii, jj, kk] = con.position();
        const auto cell = this->mesh().get_cell_from_indices(ii, jj, kk);

        // Calculate fluxes using prim
        for (int q = 0; q < 2; q++) {
            // X-direction flux
            const auto& pL = prim.at(q - 1, 0, 0);
            const auto& pR = prim.at(q - 0, 0, 0);

            auto vface = cell.velocity(q);

            if (!this->using_pcm()) {
                const auto& pLL = prim.at(q - 2, 0, 0);
                const auto& pRR = prim.at(q + 1, 0, 0);
                // compute the reconstructed states
                const auto pLr =
                    pL + plm_gradient(pL, pLL, pR, this->plm_theta()) * 0.5;
                const auto pRr =
                    pR - plm_gradient(pR, pL, pRR, this->plm_theta()) * 0.5;
                fri[q] = (this->*riemann_solve)(pLr, pRr, 1, vface);
            }
            else {
                fri[q] = (this->*riemann_solve)(pL, pR, 1, vface);
            }

            if constexpr (dim > 1) {
                vface = cell.velocity(q + 2);
                // Y-direction flux
                const auto& pL_y = prim.at(0, q - 1, 0);
                const auto& pR_y = prim.at(0, q - 0, 0);
                if (!this->using_pcm()) {
                    const auto& pLL_y = prim.at(0, q - 2, 0);
                    const auto& pRR_y = prim.at(0, q + 1, 0);
                    const auto pLr_y =
                        pL_y +
                        plm_gradient(pL_y, pLL_y, pR_y, this->plm_theta()) *
                            0.5;
                    const auto pRr_y =
                        pR_y -
                        plm_gradient(pR_y, pL_y, pRR_y, this->plm_theta()) *
                            0.5;
                    gri[q] = (this->*riemann_solve)(pLr_y, pRr_y, 2, vface);
                }
                else {
                    gri[q] = (this->*riemann_solve)(pL_y, pR_y, 2, vface);
                }

                if constexpr (dim > 2) {
                    vface = cell.velocity(q + 4);
                    // Z-direction flux
                    const auto& pL_z = prim.at(0, 0, q - 1);
                    const auto& pR_z = prim.at(0, 0, q - 0);
                    if (!this->using_pcm()) {
                        const auto& pLL_z = prim.at(0, 0, q - 2);
                        const auto& pRR_z = prim.at(0, 0, q + 1);
                        const auto pLr_z =
                            pL_z +
                            plm_gradient(pL_z, pLL_z, pR_z, this->plm_theta()) *
                                0.5;
                        const auto pRr_z =
                            pR_z -
                            plm_gradient(pR_z, pL_z, pRR_z, this->plm_theta()) *
                                0.5;
                        hri[q] = (this->*riemann_solve)(pLr_z, pRr_z, 3, vface);
                    }
                    else {
                        hri[q] = (this->*riemann_solve)(pL_z, pR_z, 3, vface);
                    }
                }
            }
        }

        const auto delta_con = dcons(
            fri,
            gri,
            hri,
            this->hydro_sources(cell),
            this->gravity_sources(prim.value(), cell),
            cell.geometrical_sources(prim.value(), gamma),
            cell
        );
        // Return updated conserved values
        return con.value() + delta_con;
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
    this->apply_boundary_conditions();
    // use parent's sync to device method
    this->sync_to_device();
};
