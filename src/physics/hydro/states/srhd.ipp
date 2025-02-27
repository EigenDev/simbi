#include "core/managers/boundary_manager.hpp"   // for BoundaryManager
#include <cmath>                                // for max, min

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
    InitialConditions& init_conditions
)
    : HydroBase<SRHD<dim>, dim, Regime::SRHD>(state, init_conditions)
{
}

// Destructor
template <int dim>
SRHD<dim>::~SRHD() = default;

//-----------------------------------------------------------------------------------------
//                          Get The Primitive
//-----------------------------------------------------------------------------------------
template <int dim>
void SRHD<dim>::cons2prim_impl()
{
    shared_atomic_bool local_failure;
    this->prims_.transform(
        [gamma = this->gamma,
         loc   = &local_failure] DEV(auto& prim, const auto& cvar, auto& pguess)
            -> Maybe<primitive_t> {
            const auto& d    = cvar.dens();
            const auto& svec = cvar.momentum();
            const auto& tau  = cvar.nrg();
            const auto& dchi = cvar.chi();
            const auto smag  = svec.norm();

            // Perform modified Newton Raphson based on
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930
            // so far, the convergence rate is the same, but perhaps I need
            // a slight tweak
            int iter       = 0;
            real peq       = pguess;
            const real tol = d * global::epsilon;
            real dp;
            do {
                // compute x_[k+1]
                auto [f, g] = newton_fg(gamma, tau, d, smag, peq);
                dp          = f / g;
                peq -= dp;

                if (iter >= global::MAX_ITER || !std::isfinite(peq)) {
                    loc->store(true);
                    return simbi::Nothing;
                }
                iter++;

            } while (std::abs(dp) >= tol);

            if (peq < 0) {
                loc->store(true);
                return simbi::Nothing;
            }

            const auto inv_et   = 1.0 / (tau + d + peq);
            const auto velocity = svec * inv_et;
            const auto w        = 1.0 / std::sqrt(1.0 - velocity.dot(velocity));
            pguess              = peq;
            return primitive_t{
              d / w,
              velocity * (global::using_four_velocity ? w : 1.0),
              peq,
              dchi / d
            };
        },
        this->full_policy(),
        this->cons_,
        pressure_guesses_
    );
    if (local_failure.load()) {
        this->set_failure_state(true);
    }
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template <int dim>
DUAL SRHD<dim>::eigenvals_t SRHD<dim>::calc_eigenvals(
    const auto& primsL,
    const auto& primsR,
    const luint nhat
) const
{
    // Separate the left and right Primitive
    const real rhoL = primsL.rho();
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.press();
    const real hL   = primsL.enthalpy(gamma);

    const real rhoR = primsR.rho();
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.press();
    const real hR   = primsR.enthalpy(gamma);

    const real csR = std::sqrt(gamma * pR / (hR * rhoR));
    const real csL = std::sqrt(gamma * pL / (hL * rhoL));

    switch (comp_wave_speed) {
        //-----------Calculate wave speeds based on Schneider et al. 1993
        case simbi::WaveSpeedEstimate::SCHNEIDER_ET_AL_93: {
            const real vbar = 0.5 * (vL + vR);
            const real cbar = 0.5 * (csL + csR);
            const real bl   = (vbar - cbar) / (1.0 - cbar * vbar);
            const real br   = (vbar + cbar) / (1.0 + cbar * vbar);
            const real aL   = my_min<real>(bl, (vL - csL) / (1.0 - vL * csL));
            const real aR   = my_max<real>(br, (vR + csR) / (1.0 + vR * csR));

            return {aL, aR, csL, csR};
        }
        //-----------Calculate wave speeds based on Mignone & Bodo 2005
        case simbi::WaveSpeedEstimate::MIGNONE_AND_BODO_05: {
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
        case simbi::WaveSpeedEstimate::HUBER_AND_KISSMANN_2021: {
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

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

template <int dim>
DUAL SRHD<dim>::conserved_t SRHD<dim>::calc_hlle_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR     = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));
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
DUAL auto SRHD<dim>::calc_star_state(
    const auto& uL,
    const auto& uR,
    const auto& fL,
    const auto& fR,
    const real aL,
    const real aR,
    const luint nhat
) const -> std::pair<real, real>
{
    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = (uR * aR - uL * aL - fR + fL) / (aR - aL);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux = (fL * aR - fR * aL + (uR - uL) * aR * aL) / (aR - aL);

    const auto& uhlld   = hll_state.dens();
    const auto& uhlls   = hll_state.momentum();
    const auto& uhlltau = hll_state.nrg();
    const auto& fhlld   = hll_flux.dens();
    const auto& fhlls   = hll_flux.momentum();
    const auto& fhlltau = hll_flux.nrg();
    const auto e        = uhlltau + uhlld;
    const auto snorm    = uhlls.dot(unit_vectors::get<dim>(nhat));
    const auto fe       = fhlltau + fhlld;
    const auto fsnorm   = fhlls.dot(unit_vectors::get<dim>(nhat));

    //------Calculate the contact wave velocity and pressure
    const auto a     = fe;
    const auto b     = -(e + fsnorm);
    const auto c     = snorm;
    const auto quad  = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
    const auto aStar = c * (1.0 / quad);
    const auto pStar = -aStar * fe + fsnorm;
    return {aStar, pStar};
}

template <int dim>
DUAL SRHD<dim>::conserved_t SRHD<dim>::calc_hllc_flux(
    const auto& prL,
    const auto& prR,
    const luint nhat,
    const real vface
) const
{
    if constexpr (dim > 1) {
        if (this->quirk_smoothing()) {
            if (quirk_strong_shock(prL.press(), prR.press())) {
                return calc_hlle_flux(prL, prR, nhat, vface);
            }
        }
    }
    // Initial setup
    const auto uL     = prL.to_conserved(gamma);
    const auto uR     = prR.to_conserved(gamma);
    const auto fL     = prL.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto fR     = prR.to_flux(gamma, unit_vectors::get<dim>(nhat));
    const auto lambda = calc_eigenvals(prL, prR, nhat);

    // Wave speeds
    const auto aL  = lambda.aL();
    const auto aR  = lambda.aR();
    const auto aLm = std::min(aL, 0.0);
    const auto aRp = std::max(aR, 0.0);

    // Quick returns for outer waves
    if (vface <= aLm) {
        return fL - uL * vface;
    }
    if (vface >= aRp) {
        return fR - uR * vface;
    }

    // Calculate intermediate state
    const auto [aStar, pStar] = calc_star_state(uL, uR, fL, fR, aLm, aRp, nhat);

    // Apply HLLC solver
    if (vface <= aStar) {
        auto star_state = compute_star_state(prL, uL, aL, aStar, pStar, nhat);
        return apply_hllc(star_state, fL, uL, aLm, vface, prL, prR);
    }
    else {
        auto star_state = compute_star_state(prR, uR, aR, aStar, pStar, nhat);
        return apply_hllc(star_state, fR, uR, aRp, vface, prL, prR);
    }
}

//===================================================================================================================
//                                           SOURCE TERMS
//===================================================================================================================
// template <int dim>
// DUAL SRHD<dim>::conserved_t SRHD<dim>::hydro_sources(const auto& cell) const
// {
//     if (null_sources) {
//         return conserved_t{};
//     }
//     const auto x1c = cell.centroid_coordinate(0);
//     const auto x2c = cell.centroid_coordinate(1);
//     const auto x3c = cell.centroid_coordinate(2);

//     conserved_t res;
//     if constexpr (dim == 1) {
//         hydro_source(x1c, t, res);
//     }
//     else if constexpr (dim == 2) {
//         hydro_source(x1c, x2c, t, res);
//     }
//     else {
//         hydro_source(x1c, x2c, x3c, t, res);
//     }

//     return res;
// }

// template <int dim>
// DUAL SRHD<dim>::conserved_t
// SRHD<dim>::this->gravity_sources(const auto& prims, const auto& cell) const
// {
//     if (null_gravity) {
//         return conserved_t{};
//     }
//     const auto x1c = cell.centroid_coordinate(0);

//     conserved_t res;
//     // gravity only changes the momentum and energy
//     if constexpr (dim > 1) {
//         const auto x2c = cell.centroid_coordinate(1);
//         if constexpr (dim > 2) {
//             const auto x3c = cell.centroid_coordinate(2);
//             gravity_source(x1c, x2c, x3c, t, res);
//             res[dimensions + 1] =
//                 res[1] * prims[1] + res[2] * prims[2] + res[3] * prims[3];
//         }
//         else {
//             gravity_source(x1c, x2c, t, res);
//             res[dimensions + 1] = res[1] * prims[1] + res[2] * prims[2];
//         }
//     }
//     else {
//         gravity_source(x1c, t, res);
//         res[dimensions + 1] = res[1] * prims[1];
//     }

//     return res;
// }

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void SRHD<dim>::advance_impl()
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
            auto vface = cell.velocity(q);
            // X-direction flux
            const auto& pL = prim.at(q - 1, 0, 0);
            const auto& pR = prim.at(q - 0, 0, 0);

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

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void SRHD<dim>::init_simulation()
{
    // load_functions();
    init_riemann_solver();
    this->apply_boundary_conditions();
    pressure_guesses_.resize(this->cons_.size())
        .reshape(this->get_shape(this->full_policy()));
    pressure_guesses_.transform(
        [](auto& p, const auto& cons) {
            const auto d = cons.dens();
            const auto s = cons.momentum().norm();
            const auto e = cons.nrg();
            return std::abs(s - d - e);
        },
        this->full_policy(),
        this->cons_
    );
    sync_all_to_device();
};
