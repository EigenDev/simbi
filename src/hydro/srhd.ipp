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
    InitialConditions& init_conditions
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
template <int dim>
void SRHD<dim>::cons2prim()
{
    shared_atomic_bool local_failure;
    prims.transform(
        [gamma = this->gamma,
         loc   = &local_failure] DEV(auto& prim, const auto& c, auto& pguess)
            -> Maybe<primitive_t> {
            const real d    = c.dens();
            const real s1   = c.momentum(1);
            const real s2   = c.momentum(2);
            const real s3   = c.momentum(3);
            const real tau  = c.nrg();
            const real dchi = c.chi();
            const real s    = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);

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
                auto [f, g] = newton_fg(gamma, tau, d, s, peq);
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

            const real inv_et = 1.0 / (tau + d + peq);
            real v1           = s1 * inv_et;
            pguess            = peq;
            if constexpr (dim == 1) {
                const real w = 1.0 / std::sqrt(1.0 - v1 * v1);
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                }
                return primitive_t{d / w, v1, peq, dchi / d};
            }
            else if constexpr (dim == 2) {
                real v2      = s2 * inv_et;
                const real w = 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2));
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                    v2 *= w;
                }
                return primitive_t{d / w, v1, v2, peq, dchi / d};
            }
            else {
                real v2 = s2 * inv_et;
                real v3 = s3 * inv_et;
                const real w =
                    1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
                if constexpr (global::VelocityType ==
                              global::Velocity::FourVelocity) {
                    v1 *= w;
                    v2 *= w;
                    v3 *= w;
                }
                return primitive_t{d / w, v1, v2, v3, peq, dchi / d};
            }
        },
        fullPolicy,
        cons,
        pressure_guess
    );

    if (local_failure.load()) {
        inFailureState.store(true);
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

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
template <int dim>
template <TIMESTEP_TYPE dt_type>
void SRHD<dim>::adapt_dt()
{
    auto calc_wave_speeds = [gamma =
                                 this->gamma] DEV(const Maybe<primitive_t>& prim
                            ) -> WaveSpeeds {
        if constexpr (dt_type == TIMESTEP_TYPE::MINIMUM) {
            return WaveSpeeds{
              .v1p = 1.0,
              .v1m = 1.0,
              .v2p = 1.0,
              .v2m = 1.0,
              .v3p = 1.0,
              .v3m = 1.0,
            };
        }
        const real h  = prim->enthalpy(gamma);
        const real cs = std::sqrt(gamma * prim->p() / (prim->rho() * h));
        const real v1 = prim->vcomponent(1);
        const real v2 = prim->vcomponent(2);
        const real v3 = prim->vcomponent(3);

        return WaveSpeeds{
          .v1p = std::abs((v1 + cs) / (1.0 + v1 * cs)),
          .v1m = std::abs((v1 - cs) / (1.0 - v1 * cs)),
          .v2p = std::abs((v2 + cs) / (1.0 + v2 * cs)),
          .v2m = std::abs((v2 - cs) / (1.0 - v2 * cs)),
          .v3p = std::abs((v3 + cs) / (1.0 + v3 * cs)),
          .v3m = std::abs((v3 - cs) / (1.0 - v3 * cs)),
        };
    };
    auto calc_local_dt =
        [this] DEV(const WaveSpeeds& speeds, const auto& cell) -> real {
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

    dt = prims.reduce(
             static_cast<real>(INFINITY),
             [calc_wave_speeds,
              calc_local_dt,
              this](const auto& acc, const auto& prim, const luint gid) {
                 const auto [ii, jj, kk] = get_indices(gid, nx, ny);
                 const auto speeds       = calc_wave_speeds(prim);
                 const auto cell         = this->cell_geometry(ii, jj, kk);
                 const auto local_dt     = calc_local_dt(speeds, cell);
                 return std::min(acc, local_dt);
             },
             fullPolicy
         ) *
         cfl;
}

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
    const auto& prL,
    const auto& prR,
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
DUAL SRHD<dim>::conserved_t
SRHD<dim>::gravity_sources(const auto& prims, const auto& cell) const
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
void SRHD<dim>::advance()
{
    auto update_conserved = [this] DEV(
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
        res += geometrical_sources;

        return res * step * dt;
    };

    auto calc_flux = [this, update_conserved] DEV(auto& con, const auto& prim) {
        conserved_t fri[2], gri[2], hri[2];

        // Calculate fluxes using stencil
        for (int q = 0; q < 2; q++) {
            // X-direction flux
            const auto& pL = prim.at(q - 1, 0, 0);
            const auto& pR = prim.at(q - 0, 0, 0);
            if (!use_pcm) {
                const auto& pLL = prim.at(q - 2, 0, 0);
                const auto& pRR = prim.at(q + 1, 0, 0);
                // compute the reconstructed states
                const auto pLr =
                    pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                const auto pRr =
                    pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
                fri[q] = (this->*riemann_solve)(pLr, pRr, 1, 0);
            }
            else {
                fri[q] = (this->*riemann_solve)(pL, pR, 1, 0);
            }

            if constexpr (dim > 1) {
                // Y-direction flux
                const auto& pL_y = prim.at(0, q - 1, 0);
                const auto& pR_y = prim.at(0, q - 0, 0);
                if (!use_pcm) {
                    const auto& pLL_y = prim.at(0, q - 2, 0);
                    const auto& pRR_y = prim.at(0, q + 1, 0);
                    const auto pLr_y =
                        pL_y + plm_gradient(pL_y, pLL_y, pR_y, plm_theta) * 0.5;
                    const auto pRr_y =
                        pR_y - plm_gradient(pR_y, pL_y, pRR_y, plm_theta) * 0.5;
                    gri[q] = (this->*riemann_solve)(pLr_y, pRr_y, 2, 0);
                }
                else {
                    gri[q] = (this->*riemann_solve)(pL_y, pR_y, 2, 0);
                }

                if constexpr (dim > 2) {
                    // Z-direction flux
                    const auto& pL_z = prim.at(0, 0, q - 1);
                    const auto& pR_z = prim.at(0, 0, q - 0);
                    if (!use_pcm) {
                        const auto& pLL_z = prim.at(0, 0, q - 2);
                        const auto& pRR_z = prim.at(0, 0, q + 1);
                        const auto pLr_z =
                            pL_z +
                            plm_gradient(pL_z, pLL_z, pR_z, plm_theta) * 0.5;
                        const auto pRr_z =
                            pR_z -
                            plm_gradient(pR_z, pL_z, pRR_z, plm_theta) * 0.5;
                        hri[q] = (this->*riemann_solve)(pLr_z, pRr_z, 3, 0);
                    }
                    else {
                        hri[q] = (this->*riemann_solve)(pL_z, pR_z, 3, 0);
                    }
                }
            }
        }

        // Calculate sources
        const auto [ii, jj, kk]        = con.position();
        const auto cell                = this->cell_geometry(ii, jj, kk);
        const auto source_terms        = hydro_sources(cell);
        const auto gravity             = gravity_sources(prim.value(), cell);
        const auto geometrical_sources = cell.geometrical_sources(prim.value());

        // Return updated conserved values
        return con.value() + update_conserved(
                                 fri,
                                 gri,
                                 hri,
                                 source_terms,
                                 gravity,
                                 geometrical_sources,
                                 cell
                             );
    };

    // Transform using stencil operations
    cons.contract(2)
        .stencil_transform(calc_flux, activePolicy, prims.contract(2));
}

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void SRHD<dim>::simulate(
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

    cons.resize(total_zones).reshape({nz, ny, nx});
    prims.resize(total_zones).reshape({nz, ny, nx});
    // dt_min.reshape({nz, ny, nx});
    pressure_guess.resize(total_zones).reshape({nz, ny, nx});

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
    init_riemann_solver();

    // create boundary manager
    boundary_manager<conserved_t, dim> bman;
    bman.sync_boundaries(fullPolicy, cons, cons.contract(2), bcs);
    cons2prim();
    adapt_dt<TIMESTEP_TYPE::MINIMUM>();

    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&] {
        advance();
        bman.sync_boundaries(fullPolicy, cons, cons.contract(2), bcs);
        cons2prim();
        adapt_dt();

        t += step * dt;
        update_mesh_motion(a, adot);
    });
};
