#include "core/types/idx_sequence.hpp"   // for for_sequence, make_index_sequence
#include "io/console/logger.hpp"         // for logger
#include "util/parallel/parallel_for.hpp"   // for parallel_for
#include "util/tools/device_api.hpp"        // for syncrohonize, devSynch, ...
#include <cmath>                            // for max, min

using namespace simbi;
using namespace simbi::util;
using namespace simbi::helpers;

// Default Constructor
template <int dim>
RMHD<dim>::RMHD() = default;

// Overloaded Constructor
template <int dim>
RMHD<dim>::RMHD(
    std::vector<std::vector<real>>& state,
    InitialConditions& init_conditions
)
    : HydroBase(state, init_conditions)
{
}

// Destructor
template <int dim>
RMHD<dim>::~RMHD()
{
    if (hsource_handle) {
        dlclose(hsource_handle);
    }

    if (gsource_handle) {
        dlclose(gsource_handle);
    }

    if (bsource_handle) {
        dlclose(bsource_handle);
    }

    // Show the cursor
};

/**
 * @brief           implement Gardiner & Stone 2005
 * https://ui.adsabs.harvard.edu/abs/2005JCoPh.205..509G/abstract
 * @param[in/out/in,out]ew west efield:
 * @param[in/out/in,out]ee east efield:
 * @param[in/out/in,out]es south efield:
 * @param[in/out/in,out]en north efield:
 * @param[in/out/in,out]bstagp1 staggered magnetic field 1:
 * @param[in/out/in,out]bstagp2 staggered magnetic field 2:
 * @param[in/out/in,out]prims primitive variables:
 * @param[in/out/in,out]ii:
 * @param[in/out/in,out]jj:
 * @param[in/out/in,out]kk:
 * @param[in/out/in,out]ia:
 * @param[in/out/in,out]ja:
 * @param[in/out/in,out]ka:
 * @param[in/out/in,out]nhat normal direction:
 * @return          HD
 * @retval
 */
template <int dim>
template <Plane P, Corner C>
DUAL real RMHD<dim>::calc_edge_emf(
    const auto& fw,
    const auto& fe,
    const auto& fs,
    const auto& fn,
    const auto* prims,
    const luint ii,
    const luint jj,
    const luint kk,
    const luint ia,
    const luint ja,
    const luint ka,
    const luint nhat,
    const real bw,
    const real be,
    const real bs,
    const real bn
) const
{
    const real ew = fw.ecomponent(nhat);
    const real ee = fe.ecomponent(nhat);
    const real es = fs.ecomponent(nhat);
    const real en = fn.ecomponent(nhat);

    // cell-centered indices for mean values
    const auto swidx = cidx<P, C, Dir::SW>(ia, ja, ka, sx, sy, sz);
    const auto seidx = cidx<P, C, Dir::SE>(ia, ja, ka, sx, sy, sz);
    const auto nwidx = cidx<P, C, Dir::NW>(ia, ja, ka, sx, sy, sz);
    const auto neidx = cidx<P, C, Dir::NE>(ia, ja, ka, sx, sy, sz);

    // get surrounding primitives
    const auto swp = prims[swidx];
    const auto sep = prims[seidx];
    const auto nwp = prims[nwidx];
    const auto nep = prims[neidx];

    // get mean e-fields
    const real esw        = swp->ecomponent(nhat);
    const real ese        = sep->ecomponent(nhat);
    const real enw        = nwp->ecomponent(nhat);
    const real ene        = nep->ecomponent(nhat);
    const real one_eighth = static_cast<real>(0.125);
    const real eavg       = static_cast<real>(0.25) * (ew + ee + es + en);

    // Decides at compile time which method to use
    switch (comp_ct_type) {
        case CTTYPE::ZERO:   // Eq. (40)
        {
            return (
                static_cast<real>(0.5) * (es + en + ew + ee) -
                static_cast<real>(0.25) * (esw + enw + ese + ene)
            );
        }
        case CTTYPE::CONTACT:   // Eq. (51)
        {
            // j + 1/4
            const real de_dqjL = [&] {
                if (fw.dens() > 0.0) {
                    return static_cast<real>(2.0) * (es - esw);
                }
                else if (fw.dens() < 0.0) {
                    return static_cast<real>(2.0) * (en - enw);
                }
                return es - esw + en - enw;
            }();

            // j + 3/4
            const real de_dqjR = [&] {
                if (fe.dens() > 0.0) {
                    return static_cast<real>(2.0) * (ese - es);
                }
                else if (fe.dens() < 0.0) {
                    return static_cast<real>(2.0) * (ene - en);
                }
                return ese - es + ene - en;
            }();

            // k + 1/4
            const real de_dqkL = [&] {
                if (fs.dens() > 0.0) {
                    return static_cast<real>(2.0) * (ew - esw);
                }
                else if (fs.dens() < 0.0) {
                    return static_cast<real>(2.0) * (ee - ese);
                }
                return ew - esw + ee - ese;
            }();

            // k + 3/4
            const real de_dqkR = [&] {
                if (fn.dens() > 0.0) {
                    return static_cast<real>(2.0) * (enw - ew);
                }
                else if (fn.dens() < 0.0) {
                    return static_cast<real>(2.0) * (ene - ee);
                }
                return enw - ew + ene - ee;
            }();

            return (
                eavg + one_eighth * (de_dqjL - de_dqjR + de_dqkL - de_dqkR)
            );
        }
        case CTTYPE::MdZ: {
            // d-coefficients from MdZ (2021), Eqns. (34 & 35)a
            const auto dw = static_cast<real>(0.5) * (fs.dL + fn.dL);
            const auto de = static_cast<real>(0.5) * (fs.dR + fn.dR);
            const auto ds = static_cast<real>(0.5) * (fw.dL + fe.dL);
            const auto dn = static_cast<real>(0.5) * (fw.dR + fe.dR);

            // a-coefficients from MdZ (2021), Eqns. (34 & 35)b
            const auto aw = static_cast<real>(0.5) * (fs.aL + fn.aL);
            const auto ae = static_cast<real>(0.5) * (fs.aR + fn.aR);
            const auto as = static_cast<real>(0.5) * (fw.aL + fe.aL);
            const auto an = static_cast<real>(0.5) * (fw.aR + fe.aR);

            // average velocity coefficients, just after Eq. (27)
            const auto lPv = my_max(fn.lamR, fs.lamR);
            const auto lMv = my_min(fn.lamL, fs.lamL);
            const auto lPh = my_max(fe.lamR, fw.lamR);
            const auto lMh = my_min(fe.lamL, fw.lamL);

            // compute transverse velocities according to Eq. (29)
            const auto nj  = nhat % 2 == 0 ? 1 : 2;
            const auto nk  = nhat % 2 == 0 ? 2 : 1;
            const auto aph = lPh / (lPh - lMh);
            const auto amh = lMh / (lPh - lMh);
            const auto apv = lPv / (lPv - lMv);
            const auto amv = lMv / (lPv - lMv);
            const auto vw  = aph * fw.vLtrans(nj) - amh * fw.vRtrans(nj);
            const auto ve  = aph * fe.vLtrans(nj) - amh * fe.vRtrans(nj);
            const auto vn  = apv * fn.vLtrans(nk) - amv * fn.vRtrans(nk);
            const auto vs  = apv * fs.vLtrans(nk) - amv * fs.vRtrans(nk);

            const auto sign = nhat == 2 ? -1.0 : 1.0;
            const auto f_we = +(aw * vw * bw) + (ae * ve * be);
            const auto f_ns = +(an * vn * bn) + (as * vs * bs);
            // dissipation terms
            const auto phi_we = +(dw * bw) - (de * be);
            const auto phi_ns = +(dn * bn) - (ds * bs);

            return sign * ((f_ns - phi_ns) - (f_we + phi_we));
        }
        default:   // ALPHA, Eq. (49)
        {
            return 0.0;
            // constexpr real alpha = 0.1;
            // // compute permutation indices
            // const auto np1 = (P == Plane::JK) ? 2 : 1;
            // const auto np2 = (P == Plane::IJ) ? 2 : 3;

            // // face-center magnetic field indices
            // const auto [nx1, ny1, nz1] = [&] {
            //     if constexpr (P == Plane::JK) {
            //         return std::make_tuple(xag + 2, nyv, zag + 2);   //
            //         B2
            //     }
            //     return std::make_tuple(nxv, yag + 2, zag + 2);   // B1
            // }();
            // const auto sidx = cidx<P, C, Dir::S>(ii, jj, kk, nx1, ny1,
            // nz1); const auto nidx = cidx<P, C, Dir::N>(ii, jj, kk, nx1,
            // ny1, nz1);

            // const auto [nx2, ny2, nz2] = [&] {
            //     if constexpr (P == Plane::IJ) {   // B2
            //         return std::make_tuple(xag + 2, nyv, zag + 2);
            //     }
            //     return std::make_tuple(xag + 2, yag + 2, nzv);   // B3
            // }();
            // const auto eidx = cidx<P, C, Dir::E>(ii, jj, kk, nx2, ny2,
            // nz2); const auto widx = cidx<P, C, Dir::W>(ii, jj, kk, nx2,
            // ny2, nz2);

            // // perpendicular mean field 1
            // const auto bp1sw = swp.bcomponent(np1);
            // const auto bp1nw = nwp.bcomponent(np1);
            // const auto bp1se = sep.bcomponent(np1);
            // const auto bp1ne = nep.bcomponent(np1);

            // // perpendicular mean field 2
            // const auto bp2sw = swp.bcomponent(np2);
            // const auto bp2nw = nwp.bcomponent(np2);
            // const auto bp2se = sep.bcomponent(np2);
            // const auto bp2ne = nep.bcomponent(np2);

            // // perpendicular staggered field 1
            // const auto bp1s = bstagp1[sidx];
            // const auto bp1n = bstagp1[nidx];
            // // perpendicular staggered field 2
            // const auto bp2e = bstagp2[eidx];
            // const auto bp2w = bstagp2[widx];

            // const real de_dq2L = (ew - esw + ee - ese) +
            //                      alpha * (bp2e - bp2se - bp2w + bp2sw);
            // const real de_dq2R = (enw - ew + ene - ee) +
            //                      alpha * (bp2ne - bp2e - bp2nw + bp2w);
            // const real de_dq1L = (es - esw + en - enw) +
            //                      alpha * (bp1s - bp1sw - bp1n + bp1nw);
            // const real de_dq1R = (ese - es + ene - en) +
            //                      alpha * (bp1se - bp1s - bp1ne + bp1n);

            // return (
            //     eavg + one_eighth * (de_dq2L - de_dq2R + de_dq1L -
            //     de_dq1R)
            // );
        }
    }
};

template <int dim>
DUAL real RMHD<dim>::curl_e(
    const luint nhat,
    const real ej[4],
    const real ek[4],
    const auto& cell,
    const int side
) const
{
    switch (geometry) {
        case Geometry::CARTESIAN: {
            if (nhat == 1) {
                if (side == 0) {
                    return cell.idx2() * (ek[IJ::NW] - ek[IJ::SW]) -
                           cell.idx3() * (ej[IK::NW] - ej[IK::SW]);
                }
                return cell.idx2() * (ek[IJ::NE] - ek[IJ::SE]) -
                       cell.idx3() * (ej[IK::NE] - ej[IK::SE]);
            }
            else if (nhat == 2) {
                if (side == 0) {
                    return cell.idx3() * (ek[JK::NW] - ek[JK::SW]) -
                           cell.idx1() * (ej[IJ::SE] - ej[IJ::SW]);
                }
                return cell.idx3() * (ek[JK::NE] - ek[JK::SE]) -
                       cell.idx1() * (ej[IJ::NE] - ej[IJ::NW]);
            }
            else {
                if (side == 0) {
                    return cell.idx1() * (ek[IK::SE] - ek[IK::SW]) -
                           cell.idx2() * (ej[JK::SE] - ej[JK::SW]);
                }
                return cell.idx1() * (ek[IK::NE] - ek[IK::NW]) -
                       cell.idx2() * (ej[JK::NE] - ej[JK::NW]);
            }
        }
        case Geometry::SPHERICAL: {
            if (nhat == 1) {
                // compute the curl in the radial direction
                const real tr  = cell.x2R();
                const real tl  = cell.x2L();
                const real dth = tr - tl;
                const real dph = cell.x3R() - cell.x3L();
                if (side == 0) {
                    return ((ek[IJ::NW] * std::sin(tr) -
                             ek[IJ::SW] * std::sin(tl)) /
                                dth -
                            (ej[IK::NW] - ej[IK::SW]) / dph) /
                           (cell.x1L() * std::sin(cell.x2mean));
                }
                return ((ek[IJ::NE] * std::sin(tr) - ek[IJ::SE] * std::sin(tl)
                        ) / dth -
                        (ej[IK::NE] - ej[IK::SE]) / dph) /
                       (cell.x1R() * std::sin(cell.x2mean));
            }
            else if (nhat == 2) {
                // compute the curl in the theta-hat direction
                const real dr  = cell.x1R() - cell.x1L();
                const real dph = cell.x3R() - cell.x3L();
                if (side == 0) {
                    if (cell.at_pole(cell.x2L())) {
                        return (-(ej[IJ::SE] * cell.x1R() -
                                  ej[IJ::SW] * cell.x1L()) /
                                dr) /
                               cell.x1mean;
                    }
                    return (-(ej[IJ::SE] * cell.x1R() - ej[IJ::SW] * cell.x1L()
                            ) / dr +
                            (ek[JK::NW] - ek[JK::SW]) /
                                (std::sin(cell.x2L()) * dph)) /
                           cell.x1mean;
                }
                if (cell.at_pole(cell.x2R())) {
                    return (-(ej[IJ::NE] * cell.x1R() - ej[IJ::NW] * cell.x1L()
                            ) /
                            dr) /
                           cell.x1mean;
                }
                return (-(ej[IJ::NE] * cell.x1R() - ej[IJ::NW] * cell.x1L()) /
                            dr +
                        (ek[JK::NE] - ek[JK::SE]) / (std::sin(cell.x2R()) * dph)
                       ) /
                       cell.x1mean;
            }
            else {
                // compute the curl in the phi-hat direction
                const real dr  = cell.x1R() - cell.x1L();
                const real dth = cell.x2R() - cell.x2L();
                if (side == 0) {
                    return ((ek[IK::SE] * cell.x1R() - ek[IK::SW] * cell.x1L()
                            ) / dr -
                            (ej[JK::SE] - ej[JK::SW]) / dth) /
                           cell.x1mean;
                }
                return ((ek[IK::NE] * cell.x1R() - ek[IK::NW] * cell.x1L()) /
                            dr -
                        (ej[JK::NE] - ej[JK::NW]) / dth) /
                       cell.x1mean;
            }
        }
        default:   // cylindrical
            if (nhat == 1) {
                // curl in the radial direction
                const real dph = cell.x2R() - cell.x2L();
                const real dz  = cell.x3R() - cell.x3L();
                if (side == 0) {
                    return (ek[IJ::NW] - ek[IJ::SW]) / (dph * cell.x1mean) -
                           (ej[IK::NW] - ej[IK::SW]) / dz;
                }
                return (ek[IJ::NE] - ek[IJ::SE]) / (dph * cell.x1mean) -
                       (ej[IK::NE] - ej[IK::SE]) / dz;
            }
            else if (nhat == 2) {
                // curl in the phi-hat direction
                const real dr = cell.x1R() - cell.x1L();
                const real dz = cell.x3R() - cell.x3L();
                if (side == 0) {
                    return (ek[JK::NW] - ek[JK::SW]) / dz -
                           (ej[IJ::SE] - ej[IJ::SW]) / dr;
                }
                return (ek[JK::NE] - ek[JK::SE]) / dz -
                       (ej[IJ::NE] - ej[IJ::NW]) / dr;
            }
            else {
                // curl in the z-hat direction
                const real dr  = cell.x1R() - cell.x1L();
                const real dph = cell.x2R() - cell.x2L();
                if (side == 0) {
                    return ((ek[IK::SE] * cell.x1R() - ek[IK::SW] * cell.x1L()
                            ) / dr -
                            (ej[JK::SE] - ej[JK::SW]) / dph) /
                           cell.x1mean;
                }
                return ((ek[IK::NE] * cell.x1R() - ek[IK::NW] * cell.x1L()) /
                            dr -
                        (ej[JK::NE] - ej[JK::NW]) / dph) /
                       cell.x1mean;
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
void RMHD<dim>::cons2prim()
{
    // functional-style implementation
    shared_atomic_bool local_failure;
    prims.transform(
        [gamma = this->gamma,
         loc   = &local_failure] DEV(auto& prim, const auto& c)
            -> Maybe<primitive_t> {
            const real d    = c.dens();
            const real m1   = c.mcomponent(1);
            const real m2   = c.mcomponent(2);
            const real m3   = c.mcomponent(3);
            const real tau  = c.nrg();
            const real b1   = c.bcomponent(1);
            const real b2   = c.bcomponent(2);
            const real b3   = c.bcomponent(3);
            const real dchi = c.chi();

            //==================================================================
            // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
            //==================================================================

            //======= rescale the variables Eqs. (22) - (25)
            const real invd   = 1.0 / d;
            const real isqrtd = std::sqrt(invd);
            const real q      = tau * invd;
            const real r1     = m1 * invd;
            const real r2     = m2 * invd;
            const real r3     = m3 * invd;
            const real rsq    = r1 * r1 + r2 * r2 + r3 * r3;
            const real rmag   = std::sqrt(rsq);
            const real h1     = b1 * isqrtd;
            const real h2     = b2 * isqrtd;
            const real h3     = b3 * isqrtd;
            const real beesq  = h1 * h1 + h2 * h2 + h3 * h3 + global::epsilon;
            const real rdb    = r1 * h1 + r2 * h2 + r3 * h3;
            const real rdbsq  = rdb * rdb;
            // r-parallel Eq. (25.a)
            const real rp1 = (rdb / beesq) * h1;
            const real rp2 = (rdb / beesq) * h2;
            const real rp3 = (rdb / beesq) * h3;
            // r-perpendicular, Eq. (25.b)
            const real rq1 = r1 - rp1;
            const real rq2 = r2 - rp2;
            const real rq3 = r3 - rp3;

            const real rparr = std::sqrt(rp1 * rp1 + rp2 * rp2 + rp3 * rp3);
            const real rq    = std::sqrt(rq1 * rq1 + rq2 * rq2 + rq3 * rq3);

            // We use the false position method to solve for the roots
            real y1 = 0.0;
            real y2 = 1.0;
            real f1 = kkc_fmu49(y1, beesq, rdbsq, rmag);
            real f2 = kkc_fmu49(y2, beesq, rdbsq, rmag);

            bool good_guesses = false;
            // if good guesses, use them
            if (std::abs(f1) + std::abs(f2) < 2.0 * global::epsilon) {
                good_guesses = true;
            }

            int iter = 0.0;
            // compute yy in case the initial bracket is not good
            real yy = static_cast<real>(0.5) * (y1 + y2);
            real f;
            if (!good_guesses) {
                do {
                    yy = (y1 * f2 - y2 * f1) / (f2 - f1);
                    f  = kkc_fmu49(yy, beesq, rdbsq, rmag);
                    if (f * f2 < 0.0) {
                        y1 = y2;
                        f1 = f2;
                        y2 = yy;
                        f2 = f;
                    }
                    else {
                        // use Illinois algorithm to avoid stagnation
                        f1 = 0.5 * f1;
                        y2 = yy;
                        f2 = f;
                    }

                    if (iter >= global::MAX_ITER || !std::isfinite(f)) {
                        loc->store(true);
                        return simbi::Nothing;
                    }
                    iter++;
                } while (std::abs(y1 - y2) > global::epsilon &&
                         std::abs(f) > global::epsilon);
            }

            // We found good brackets. Now we can solve for the roots
            y1 = 0.0;
            y2 = yy;

            // Evaluate the master function (Eq. 44) at the roots
            f1   = kkc_fmu44(y1, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
            f2   = kkc_fmu44(y2, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
            iter = 0.0;
            do {
                yy = (y1 * f2 - y2 * f1) / (f2 - f1);
                f  = kkc_fmu44(yy, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
                if (f * f2 < 0.0) {
                    y1 = y2;
                    f1 = f2;
                    y2 = yy;
                    f2 = f;
                }
                else {
                    // use Illinois algorithm to avoid stagnation
                    f1 = 0.5 * f1;
                    y2 = yy;
                    f2 = f;
                }
                if (iter >= global::MAX_ITER || !std::isfinite(f)) {
                    loc->store(true);
                    return simbi::Nothing;
                }
                iter++;
            } while (std::abs(y1 - y2) > global::epsilon &&
                     std::abs(f) > global::epsilon);

            if (!std::isfinite(yy)) {
                loc->store(true);
                return simbi::Nothing;
            }

            // Ok, we have the roots. Now we can compute the primitive
            // variables Equation (26)
            const real x = 1.0 / (1.0 + yy * beesq);

            // Equation (38)
            const real rbar_sq = rsq * x * x + yy * x * (1.0 + x) * rdbsq;

            // Equation (39)
            const real qbar =
                q - 0.5 * (beesq + yy * yy * x * x * beesq * rq * rq);

            // Equation (32) inverted and squared
            const real vsq  = yy * yy * rbar_sq;
            const real gbsq = vsq / std::abs(1.0 - vsq);
            const real w    = std::sqrt(1.0 + gbsq);

            // Equation (41)
            const real rhohat = d / w;

            // Equation (42)
            const real epshat = w * (qbar - yy * rbar_sq) + gbsq / (1.0 + w);

            // Equation (43)
            const real pg = (gamma - 1.0) * rhohat * epshat;

            // velocities Eq. (68)
            real v1 = yy * x * (r1 + h1 * rdb * yy);
            real v2 = yy * x * (r2 + h2 * rdb * yy);
            real v3 = yy * x * (r3 + h3 * rdb * yy);

            if constexpr (global::VelocityType ==
                          global::Velocity::FourVelocity) {
                v1 *= w;
                v2 *= w;
                v3 *= w;
            }

            return primitive_t{rhohat, v1, v2, v3, pg, b1, b2, b3, dchi / d};
        },
        fullPolicy,
        cons
    );

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
// void RMHD<dim>::cons2prim()
// {
//     simbi::parallel_for(fullPolicy, total_zones, [this] DEV(luint gid) {
//         bool workLeftToDo = true;

//         shared_atomic_bool found_failure;
//         if constexpr (global::on_gpu) {
//             auto tid = get_threadId();
//             if (tid == 0) {
//                 found_failure.store(inFailureState.load());
//             }
//             simbi::gpu::api::synchronize();
//         }
//         else {
//             found_failure.store(inFailureState.load());
//         }
//         // shared_atomic_bool found_failure(inFailureState.load());
//         // simbi::gpu::api::synchronize();

//         real invdV = 1.0;
//         while (!found_failure && workLeftToDo) {
//             if (homolog) {
//                 const luint kk   = get_height(gid, xag, yag);
//                 const luint jj   = get_row(gid, xag, yag, kk);
//                 const luint ii   = get_column(gid, xag, yag, kk);
//                 const auto ireal = get_real_idx(ii, radius, xag);
//                 const auto jreal = get_real_idx(jj, radius, yag);
//                 const auto kreal = get_real_idx(kk, radius, zag);
//                 const auto cell  = this->cell_geometry(ireal, jreal, kreal);
//                 const real dV    = cell.dV;
//                 invdV            = 1.0 / dV;
//             }
//             const real d    = cons[gid].dens() * invdV;
//             const real m1   = cons[gid].mcomponent(1) * invdV;
//             const real m2   = cons[gid].mcomponent(2) * invdV;
//             const real m3   = cons[gid].mcomponent(3) * invdV;
//             const real tau  = cons[gid].nrg() * invdV;
//             const real b1   = cons[gid].bcomponent(1) * invdV;
//             const real b2   = cons[gid].bcomponent(2) * invdV;
//             const real b3   = cons[gid].bcomponent(3) * invdV;
//             const real dchi = cons[gid].chi() * invdV;

//             //==================================================================
//             // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
//             //==================================================================

//             //======= rescale the variables Eqs. (22) - (25)
//             const real invd   = 1.0 / d;
//             const real isqrtd = std::sqrt(invd);
//             const real q      = tau * invd;
//             const real r1     = m1 * invd;
//             const real r2     = m2 * invd;
//             const real r3     = m3 * invd;
//             const real rsq    = r1 * r1 + r2 * r2 + r3 * r3;
//             const real rmag   = std::sqrt(rsq);
//             const real h1     = b1 * isqrtd;
//             const real h2     = b2 * isqrtd;
//             const real h3     = b3 * isqrtd;
//             const real beesq  = h1 * h1 + h2 * h2 + h3 * h3 +
//             global::epsilon; const real rdb    = r1 * h1 + r2 * h2 + r3 * h3;
//             const real rdbsq  = rdb * rdb;
//             // r-parallel Eq. (25.a)
//             const real rp1 = (rdb / beesq) * h1;
//             const real rp2 = (rdb / beesq) * h2;
//             const real rp3 = (rdb / beesq) * h3;
//             // r-perpendicular, Eq. (25.b)
//             const real rq1 = r1 - rp1;
//             const real rq2 = r2 - rp2;
//             const real rq3 = r3 - rp3;

//             const real rparr = std::sqrt(rp1 * rp1 + rp2 * rp2 + rp3 * rp3);
//             const real rq    = std::sqrt(rq1 * rq1 + rq2 * rq2 + rq3 * rq3);

//             // We use the false position method to solve for the roots
//             real y1 = 0.0;
//             real y2 = 1.0;
//             real f1 = kkc_fmu49(y1, beesq, rdbsq, rmag);
//             real f2 = kkc_fmu49(y2, beesq, rdbsq, rmag);

//             bool good_guesses = false;
//             // if good guesses, use them
//             if (std::abs(f1) + std::abs(f2) < 2.0 * global::epsilon) {
//                 good_guesses = true;
//             }

//             int iter = 0.0;
//             // compute yy in case the initial bracket is not good
//             real yy = static_cast<real>(0.5) * (y1 + y2);
//             real f;
//             if (!good_guesses) {
//                 do {
//                     yy = (y1 * f2 - y2 * f1) / (f2 - f1);
//                     f  = kkc_fmu49(yy, beesq, rdbsq, rmag);
//                     if (f * f2 < 0.0) {
//                         y1 = y2;
//                         f1 = f2;
//                         y2 = yy;
//                         f2 = f;
//                     }
//                     else {
//                         // use Illinois algorithm to avoid stagnation
//                         f1 = 0.5 * f1;
//                         y2 = yy;
//                         f2 = f;
//                     }

//                     if (iter >= global::MAX_ITER || !std::isfinite(f)) {
//                         troubled_cells[gid] = 1;
//                         dt                  = INFINITY;
//                         inFailureState.store(true);
//                         found_failure.store(true);
//                         break;
//                     }
//                     iter++;
//                 } while (std::abs(y1 - y2) > global::epsilon &&
//                          std::abs(f) > global::epsilon);
//             }

//             // We found good brackets. Now we can solve for the roots
//             y1 = 0.0;
//             y2 = yy;

//             // Evaluate the master function (Eq. 44) at the roots
//             f1   = kkc_fmu44(y1, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
//             f2   = kkc_fmu44(y2, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
//             iter = 0.0;
//             do {
//                 yy = (y1 * f2 - y2 * f1) / (f2 - f1);
//                 f  = kkc_fmu44(yy, rmag, rparr, rq, beesq, rdbsq, q, d,
//                 gamma); if (f * f2 < 0.0) {
//                     y1 = y2;
//                     f1 = f2;
//                     y2 = yy;
//                     f2 = f;
//                 }
//                 else {
//                     // use Illinois algorithm to avoid stagnation
//                     f1 = 0.5 * f1;
//                     y2 = yy;
//                     f2 = f;
//                 }
//                 if (iter >= global::MAX_ITER || !std::isfinite(f)) {
//                     troubled_cells[gid] = 1;
//                     dt                  = INFINITY;
//                     inFailureState.store(true);
//                     found_failure.store(true);
//                     break;
//                 }
//                 iter++;
//             } while (std::abs(y1 - y2) > global::epsilon &&
//                      std::abs(f) > global::epsilon);

//             // Ok, we have the roots. Now we can compute the primitive
//             // variables Equation (26)
//             const real x = 1.0 / (1.0 + yy * beesq);

//             // Equation (38)
//             const real rbar_sq = rsq * x * x + yy * x * (1.0 + x) * rdbsq;

//             // Equation (39)
//             const real qbar =
//                 q - 0.5 * (beesq + yy * yy * x * x * beesq * rq * rq);

//             // Equation (32) inverted and squared
//             const real vsq  = yy * yy * rbar_sq;
//             const real gbsq = vsq / std::abs(1.0 - vsq);
//             const real w    = std::sqrt(1.0 + gbsq);

//             // Equation (41)
//             const real rhohat = d / w;

//             // Equation (42)
//             const real epshat = w * (qbar - yy * rbar_sq) + gbsq / (1.0 + w);

//             // Equation (43)
//             const real pg = (gamma - 1.0) * rhohat * epshat;

//             // velocities Eq. (68)
//             real v1 = yy * x * (r1 + h1 * rdb * yy);
//             real v2 = yy * x * (r2 + h2 * rdb * yy);
//             real v3 = yy * x * (r3 + h3 * rdb * yy);

//             if constexpr (global::VelocityType ==
//                           global::Velocity::FourVelocity) {
//                 v1 *= w;
//                 v2 *= w;
//                 v3 *= w;
//             }

//             prims[gid] =
//                 primitive_t{rhohat, v1, v2, v3, pg, b1, b2, b3, dchi / d};

//             workLeftToDo = false;

//             if (!std::isfinite(yy)) {
//                 troubled_cells[gid] = 1;
//                 inFailureState.store(true);
//                 found_failure.store(true);
//                 dt = INFINITY;
//             }
//             simbi::gpu::api::synchronize();
//         }
//     });
// }

/**
 * Return the primitive
 * variables density , three-velocity, pressure
 *
 * @param con conserved array at index
 * @param gid  current global index
 * @return none
 */
template <int dim>
DEV auto RMHD<dim>::cons2prim(const auto& cons) const
{
    const real d    = cons.dens();
    const real m1   = cons.mcomponent(1);
    const real m2   = cons.mcomponent(2);
    const real m3   = cons.mcomponent(3);
    const real tau  = cons.nrg();
    const real b1   = cons.bcomponent(1);
    const real b2   = cons.bcomponent(2);
    const real b3   = cons.bcomponent(3);
    const real dchi = cons.chi();

    //==================================================================
    // ATTEMPT TO RECOVER PRIMITIVES USING KASTAUN ET AL. 2021
    //==================================================================

    //======= rescale the variables Eqs. (22) - (25)
    const real invd   = 1.0 / d;
    const real isqrtd = std::sqrt(invd);
    const real q      = tau * invd;
    const real r1     = m1 * invd;
    const real r2     = m2 * invd;
    const real r3     = m3 * invd;
    const real rsq    = r1 * r1 + r2 * r2 + r3 * r3;
    const real rmag   = std::sqrt(rsq);
    const real h1     = b1 * isqrtd;
    const real h2     = b2 * isqrtd;
    const real h3     = b3 * isqrtd;
    const real beesq  = h1 * h1 + h2 * h2 + h3 * h3;
    const real rdb    = r1 * h1 + r2 * h2 + r3 * h3;
    const real rdbsq  = rdb * rdb;
    // r-parallel
    const real rp1 = (rdb / beesq) * h1;
    const real rp2 = (rdb / beesq) * h2;
    const real rp3 = (rdb / beesq) * h3;
    // r-perpendicular
    const real rq1 = r1 - rp1;
    const real rq2 = r2 - rp2;
    const real rq3 = r3 - rp3;

    const real rparr = std::sqrt(rp1 * rp1 + rp2 * rp2 + rp3 * rp3);
    const real rq    = std::sqrt(rq1 * rq1 + rq2 * rq2 + rq3 * rq3);

    // We use the false position method to solve for the roots
    real y1 = 0.0;
    real y2 = 1.0;
    real f1 = kkc_fmu49(y1, beesq, rdbsq, rmag);
    real f2 = kkc_fmu49(y2, beesq, rdbsq, rmag);

    bool good_guesses = false;
    // if good guesses, use them
    if (std::abs(f1) + std::abs(f2) < 2.0 * global::epsilon) {
        good_guesses = true;
    }

    int iter = 0.0;
    // compute yy in case the initial bracket is good
    real yy = 0.5 * (y1 + y2);
    real f;
    if (!good_guesses) {
        do {
            yy = (y1 * f2 - y2 * f1) / (f2 - f1);
            f  = kkc_fmu49(yy, beesq, rdbsq, rmag);
            if (f * f2 < 0.0) {
                y1 = y2;
                f1 = f2;
                y2 = yy;
                f2 = f;
            }
            else {
                // use Illinois algorithm to avoid stagnation
                f1 = 0.5 * f1;
                y2 = yy;
                f2 = f;
            }

            if (iter >= global::MAX_ITER || !std::isfinite(f)) {
                break;
            }
            iter++;
        } while (std::abs(y1 - y2) > global::epsilon &&
                 std::abs(f) > global::epsilon);
    }

    // We found good brackets. Now we can solve for the roots
    y1 = 0.0;
    y2 = yy;

    // Evaluate the master function (Eq. 44) at the roots
    f1   = kkc_fmu44(y1, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
    f2   = kkc_fmu44(y2, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
    iter = 0.0;
    do {
        yy = (y1 * f2 - y2 * f1) / (f2 - f1);
        f  = kkc_fmu44(yy, rmag, rparr, rq, beesq, rdbsq, q, d, gamma);
        if (f * f2 < 0.0) {
            y1 = y2;
            f1 = f2;
            y2 = yy;
            f2 = f;
        }
        else {
            // use Illinois algorithm to avoid stagnation
            f1 = 0.5 * f1;
            y2 = yy;
            f2 = f;
        }
        if (iter >= global::MAX_ITER || !std::isfinite(f)) {
            break;
        }
        iter++;
    } while (std::abs(y1 - y2) > global::epsilon &&
             std::abs(f) > global::epsilon);

    // Ok, we have the roots. Now we can compute the primitive variables
    // Equation (26)
    const real x = 1.0 / (1.0 + yy * beesq);

    // Equation (38)
    const real rbar_sq = rsq * x * x + yy * x * (1.0 + x) * rdbsq;

    // Equation (39)
    const real qbar = q - 0.5 * (beesq + yy * yy * x * x * beesq * rq * rq);

    // Equation (32) inverted and squared
    const real vsq  = yy * yy * rbar_sq;
    const real gbsq = vsq / std::abs(1.0 - vsq);
    const real w    = std::sqrt(1.0 + gbsq);

    // Equation (41)
    const real rhohat = d / w;

    // Equation (42)
    const real epshat = w * (qbar - yy * rbar_sq) + gbsq / (1.0 + w);

    // Equation (43)
    const real pg = (gamma - 1.0) * rhohat * epshat;

    // velocities Eq. (68)
    real v1 = yy * x * (r1 + h1 * rdb * yy);
    real v2 = yy * x * (r2 + h2 * rdb * yy);
    real v3 = yy * x * (r3 + h3 * rdb * yy);

    if constexpr (global::VelocityType == global::Velocity::FourVelocity) {
        v1 *= w;
        v2 *= w;
        v3 *= w;
    }
    return primitive_t{d / w, v1, v2, v3, pg, b1, b2, b3, dchi / d};
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
    const real rho = prims.rho();
    const real h   = prims.enthalpy(gamma);
    const real cs2 = (gamma * prims.press() / (rho * h));
    const auto bmu =
        prims.bfield().as_fourvec(prims.velocity(), prims.lorentz_factor());
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
    auto [lmL, lpL] = calc_max_wave_speeds(primsL, nhat);
    // right_side
    auto [lmR, lpR] = calc_max_wave_speeds(primsR, nhat);

    const auto aR = my_max(lpL, lpR);
    const auto aL = my_min(lmL, lmR);

    return {aL, aR};
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::prims2cons(const auto& prims) const
{
    const real rho   = prims.rho();
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real pg    = prims.press();
    const real b1    = prims.bcomponent(1);
    const real b2    = prims.bcomponent(2);
    const real b3    = prims.bcomponent(3);
    const real lf    = prims.lorentz_factor();
    const real h     = prims.enthalpy(gamma);
    const real vdotb = prims.vdotb();
    const real bsq   = prims.bsquared();
    const real vsq   = prims.vsquared();
    const real d     = rho * lf;
    const real ed    = d * h * lf;

    return {
      d,
      (ed + bsq) * v1 - vdotb * b1,
      (ed + bsq) * v2 - vdotb * b2,
      (ed + bsq) * v3 - vdotb * b3,
      ed - pg - d + static_cast<real>(0.5) * (bsq + vsq * bsq - vdotb * vdotb),
      b1,
      b2,
      b3,
      d * prims.chi()
    };
};

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditional timestep
template <int dim>
template <TIMESTEP_TYPE dt_type>
void RMHD<dim>::adapt_dt()
{
    auto calc_wave_speeds = [this] DEV(const auto& prims) {
        if constexpr (dt_type == TIMESTEP_TYPE::MINIMUM) {
            return WaveSpeeds{
              .v1p = 1.0,
              .v1m = 1.0,
              .v2p = 1.0,
              .v2m = 1.0,
              .v3p = 1.0,
              .v3m = 1.0
            };
        }

        const auto [v1m, v1p] = this->calc_max_wave_speeds(prims.value(), 1);
        const auto [v2m, v2p] = this->calc_max_wave_speeds(prims.value(), 2);
        const auto [v3m, v3p] = this->calc_max_wave_speeds(prims.value(), 3);
        return WaveSpeeds{
          .v1p = v1p,
          .v1m = v1m,
          .v2p = v2p,
          .v2m = v2m,
          .v3p = v3p,
          .v3m = v3m
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

    // std::atomic<real> min_dt = INFINITY;
    // pooling::getThreadPool().parallel_for(active_zones, [&](luint gid) {
    //     real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
    //     const luint kk  = axid<dim, BlkAx::K>(gid, xag, yag);
    //     const luint jj  = axid<dim, BlkAx::J>(gid, xag, yag, kk);
    //     const luint ii  = axid<dim, BlkAx::I>(gid, xag, yag, kk);
    //     const luint ia  = ii + radius;
    //     const luint ja  = jj + radius;
    //     const luint ka  = kk + radius;
    //     const luint aid = idx3(ia, ja, ka, nx, ny, nz);
    //     // Left/Right wave speeds
    //     if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
    //         std::tie(v1m, v1p) = calc_max_wave_speeds(*prims[aid], 1);
    //         v1p                = std::abs(v1p);
    //         v1m                = std::abs(v1m);
    //         std::tie(v2m, v2p) = calc_max_wave_speeds(*prims[aid], 2);
    //         v2p                = std::abs(v2p);
    //         v2m                = std::abs(v2m);
    //         std::tie(v3m, v3p) = calc_max_wave_speeds(*prims[aid], 3);
    //         v3p                = std::abs(v3p);
    //         v3m                = std::abs(v3m);
    //     }
    //     else {
    //         v1p = 1.0;
    //         v1m = 1.0;
    //         v2p = 1.0;
    //         v2m = 1.0;
    //         v3p = 1.0;
    //         v3m = 1.0;
    //     }

    //     const auto cell = this->cell_geometry(ii, jj, kk);
    //     switch (geometry) {
    //         case simbi::Geometry::CARTESIAN: {
    //             const real x1l = cell.x1L();
    //             const real x1r = cell.x1R();
    //             const real dx1 = x1r - x1l;

    //             const real x2l = cell.x1L();
    //             const real x2r = cell.x1R();
    //             const real dx2 = x2r - x2l;

    //             cfl_dt = std ::min(
    //                 {dx1 / (std::max(v1p, v1m)),
    //                  dx2 / (std::max(v2p, v2m)),
    //                  dx3 / (std::max(v3p, v3m))}
    //             );

    //             break;
    //         }

    //         case simbi::Geometry::SPHERICAL: {
    //             const real rproj = cell.x1mean * std::sin(cell.x2mean);

    //             cfl_dt = std::min(
    //                 {(cell.x1R() - cell.x1L()) / (std::max(v1p, v1m)),
    //                  cell.x1mean * (cell.x2R() - cell.x2L()) /
    //                      (std::max(v2p, v2m)),
    //                  rproj * (cell.x3R() - cell.x3L()) / (std::max(v3p,
    //                  v3m))}
    //             );
    //             break;
    //         }
    //         default: {
    //             const real x1l = cell.x1L();
    //             const real x1r = cell.x1R();
    //             const real dx1 = x1r - x1l;

    //             const real rmean = cell.x1mean;
    //             cfl_dt           = std::min(
    //                 {dx1 / (std::max(v1p, v1m)),
    //                            rmean * dx2 / (std::max(v2p, v2m)),
    //                            dx3 / (std::max(v3p, v3m))}
    //             );
    //             break;
    //         }
    //     }
    //     pooling::update_minimum(min_dt, cfl_dt);
    // });
    // dt = cfl * min_dt;
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL RMHD<dim>::conserved_t
RMHD<dim>::prims2flux(const auto& prims, const luint nhat) const
{
    const real rho   = prims.rho();
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real b1    = prims.bcomponent(1);
    const real b2    = prims.bcomponent(2);
    const real b3    = prims.bcomponent(3);
    const real h     = prims.enthalpy(gamma);
    const real lf    = prims.lorentz_factor();
    const real invlf = 1.0 / lf;
    const real vdotb = prims.vdotb();
    const real bsq   = prims.bsquared();
    const real ptot  = prims.total_pressure();
    const real chi   = prims.chi();
    const real vn    = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
    const real bn    = (nhat == 1) ? b1 : (nhat == 2) ? b2 : b3;
    const real d     = rho * lf;
    const real ed    = d * h * lf;
    const real m1    = (ed + bsq) * v1 - vdotb * b1;
    const real m2    = (ed + bsq) * v2 - vdotb * b2;
    const real m3    = (ed + bsq) * v3 - vdotb * b3;
    const real mn    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;
    const auto bmu   = mag_fourvec_t(prims);
    const real ind1  = (nhat == 1) ? 0.0 : vn * b1 - v1 * bn;
    const real ind2  = (nhat == 2) ? 0.0 : vn * b2 - v2 * bn;
    const real ind3  = (nhat == 3) ? 0.0 : vn * b3 - v3 * bn;
    return {
      d * vn,
      m1 * vn + kronecker(nhat, 1) * ptot - bn * bmu[0] * invlf,
      m2 * vn + kronecker(nhat, 2) * ptot - bn * bmu[1] * invlf,
      m3 * vn + kronecker(nhat, 3) * ptot - bn * bmu[3] * invlf,
      mn - d * vn,
      ind1,
      ind2,
      ind3,
      d * vn * chi
    };
};

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
        net_flux.lamR = aRp;
        net_flux.lamL = aLm;
        if (vface <= aLm) {
            net_flux.aL  = 1.0;
            net_flux.aR  = 0.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = prL.vcomponent(nj);
            net_flux.vkL = prL.vcomponent(nk);
            net_flux.vjR = 0.0;
            net_flux.vkR = 0.0;
        }
        else if (vface >= aRp) {
            net_flux.aL  = 0.0;
            net_flux.aR  = 1.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = 0.0;
            net_flux.vkL = 0.0;
            net_flux.vjR = prR.vcomponent(nj);
            net_flux.vkR = prR.vcomponent(nk);
        }
        else {
            // set the wave coefficients
            const auto afac = 1.0 / (aRp - aLm);
            net_flux.aL     = +aRp * afac;
            net_flux.aR     = -aLm * afac;
            net_flux.dL     = -aRp * aLm * afac;
            net_flux.dR     = net_flux.dL;
            net_flux.vjL    = prL.vcomponent(nj);
            net_flux.vkL    = prL.vcomponent(nk);
            net_flux.vjR    = prR.vcomponent(nj);
            net_flux.vkR    = prR.vcomponent(nk);
        }
    }
    else {
        net_flux.calc_electric_field(unit_vectors::get<dim>(nhat));
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

    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.afL();
    const real aR     = lambda.afR();
    const real aLm    = aL < 0.0 ? aL : 0.0;
    const real aRp    = aR > 0.0 ? aR : 0.0;
    auto net_flux     = [&]() {
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

        if (quirk_smoothing && quirk_strong_shock(prL.press(), prR.press())) {
            return hll_flux - hll_state * vface;
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);
        // the normal component of the magnetic field is assumed to
        // be continuous across the interface, so bnL = bnR = bn
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
        net_flux.lamR = aRp;
        net_flux.lamL = aLm;
        if (vface <= aLm) {
            net_flux.aL  = 1.0;
            net_flux.aR  = 0.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = prL.vcomponent(nj);
            net_flux.vkL = prL.vcomponent(nk);
            net_flux.vjR = 0.0;
            net_flux.vkR = 0.0;
        }
        else if (vface >= aRp) {
            net_flux.aL  = 0.0;
            net_flux.aR  = 1.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = 0.0;
            net_flux.vkL = 0.0;
            net_flux.vjR = prR.vcomponent(nj);
            net_flux.vkR = prR.vcomponent(nk);
        }
        else {
            // if Bn is zero, then the HLLC solver admits a jummp in the
            // transverse magnetic field components across the middle wave.
            // If not, HLLC has the same flux and diffusion coefficients as
            // the HLL solver.
            if (null_normal_field) {
                constexpr auto half = static_cast<real>(0.5);
                const auto aaS      = std::abs(aS);
                net_flux.aL         = half;
                net_flux.aR         = half;
                net_flux.dL = half * (aaS - std::abs(aLm)) * chiL + half * aaS;
                net_flux.dR = half * (aaS - std::abs(aRp)) * chiR + half * aaS;
            }
            else {
                const auto afac = 1.0 / (aRp - aLm);
                net_flux.aL     = +aRp * afac;
                net_flux.aR     = -aLm * afac;
                net_flux.dL     = -aRp * aLm * afac;
                net_flux.dR     = net_flux.dL;
            }
            net_flux.vjL = prL.vcomponent(nj);
            net_flux.vkL = prL.vcomponent(nk);
            net_flux.vjR = prR.vcomponent(nj);
            net_flux.vkR = prR.vcomponent(nk);
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
        return INFINITY;
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

    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.afL();
    const real aR     = lambda.afR();
    const real aLm    = aL < 0.0 ? aL : 0.0;
    const real aRp    = aR > 0.0 ? aR : 0.0;

    auto net_flux = [&]() {
        //---- Check Wave Speeds before wasting computations
        if (vface <= aLm) {
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

        if (quirk_smoothing) {
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
        auto p0 = cons2prim(hll_state).total_pressure();

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
                auto f1 = hlld_vdiff(p1, r, lam, bn, nhat, prAL, prAR, prC);

                dp = (p1 - p0) / (f1 - f0) * f1;
                p0 = p1;
                f0 = f1;
                p1 -= dp;

                if (iter++ > num_tries || !std::isfinite(f1)) {
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
            (vface < laL && vface > aLm) || (vface > laL && vface < vnc);
        const auto at_contact =
            (vface < vnc && vface > laL) || (vface > vnc && vface < laR);

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
        net_flux.lamR       = aRp;
        net_flux.lamL       = aLm;
        if (vface <= aLm) {
            net_flux.aL  = 1.0;
            net_flux.aR  = 0.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = prL.vcomponent(nj);
            net_flux.vkL = prL.vcomponent(nk);
            net_flux.vjR = 0.0;
            net_flux.vkR = 0.0;
        }
        else if (vface >= aRp) {
            net_flux.aL  = 0.0;
            net_flux.aR  = 1.0;
            net_flux.dL  = 0.0;
            net_flux.dR  = 0.0;
            net_flux.vjL = 0.0;
            net_flux.vkL = 0.0;
            net_flux.vjR = prR.vcomponent(nj);
            net_flux.vkR = prR.vcomponent(nk);
        }
        else {
            // if Bn is zero, then the HLLC solver admits a jummp in the
            // transverse magnetic field components across the middle wave.
            // If not, HLLC has the same flux and diffusion coefficients as
            // the HLL solver.
            net_flux.aL = half * (1.0 + veeS);
            net_flux.aR = half * (1.0 - veeS);
            net_flux.dL = half * (veeL - veeS) * chiL +
                          half * (std::abs(laL) - veeS * laL);
            net_flux.dR = half * (veeR - veeS) * chiR +
                          half * (std::abs(laR) - veeS * laR);

            net_flux.vjL = prL.vcomponent(nj);
            net_flux.vkL = prL.vcomponent(nk);
            net_flux.vjR = prR.vcomponent(nj);
            net_flux.vkR = prR.vcomponent(nk);
        }
    }
    else {
        net_flux.calc_electric_field(unit_vectors::get<dim>(nhat));
    }
    return net_flux;
};

template <int dim>
void RMHD<dim>::set_flux_and_fields()
{
    const auto xe      = nxe - 2;
    const auto ye      = nye - 2;
    const auto ze      = nze - 2;
    const auto m2inner = reflect_inner_x2_momentum ? 2 : -1;
    const auto m2outer = reflect_outer_x2_momentum ? 2 : -1;

    // Helper lambda to handle boundary conditions
    auto handle_boundary_conditions = [=, this] DEV(
                                          auto& field,
                                          auto& bstag,
                                          auto idx,
                                          auto real_idx,
                                          auto wrap_idx,
                                          auto bcidx,
                                          auto momentum_idx
                                      ) {
        switch (bcs[bcidx]) {
            case BoundaryCondition::PERIODIC:
                field[idx] = field[wrap_idx];
                bstag[idx] = bstag[wrap_idx];
                break;
            case BoundaryCondition::REFLECTING:
                field[idx] = field[real_idx];
                bstag[idx] = bstag[real_idx];
                if (momentum_idx != -1) {
                    field[idx].mcomponent(momentum_idx) *= -1.0;
                    bstag[idx] *= -1.0;
                }
                break;
            default:   // outflow
                field[idx] = field[real_idx];
                bstag[idx] = bstag[real_idx];
                break;
        }
    };
    // update the flux and field in the x1 direction
    parallel_for(xvertexPolicy, [=, this] DEV(const luint gid) {
        const luint kk = axid<3, BlkAx::K>(gid, nxv, yag);
        const luint jj = axid<3, BlkAx::J>(gid, nxv, yag, kk);
        const luint ii = axid<3, BlkAx::I>(gid, nxv, yag, kk);

        if (global::on_gpu) {
            if ((ii >= nxv) || (jj >= yag) || (kk >= zag)) {
                return;
            }
        }

        // the fluxes only ever need the ghosts zones one cell deep for
        // constrained transport
        constexpr auto hr = 1;
        constexpr auto rr = 0;
        constexpr auto rs = rr + 1;
        const auto jr     = jj + hr;
        const auto kr     = kk + hr;
        // fill the ghost zones for the fluxes at the x1 boundaries
        if (jj < ye && kk < ze) {
            const auto x1jb = idx3(ii, rr, kr, nxv, nye, nze);
            const auto x1je = idx3(ii, nye - rs, kr, nxv, nye, nze);
            const auto x1kb = idx3(ii, jr, rr, nxv, nye, nze);
            const auto x1ke = idx3(ii, jr, nze - rs, nxv, nye, nze);

            // across x2 interface
            handle_boundary_conditions(
                fri,
                bstag1,
                x1jb,
                idx3(ii, hr, kr, nxv, nye, nze),
                idx3(ii, ye, kr, nxv, nye, nze),
                2,
                m2inner
            );
            handle_boundary_conditions(
                fri,
                bstag1,
                x1je,
                idx3(ii, ye, kr, nxv, nye, nze),
                idx3(ii, hr, kr, nxv, nye, nze),
                3,
                m2outer
            );

            // across x3 interface
            handle_boundary_conditions(
                fri,
                bstag1,
                x1kb,
                idx3(ii, jr, hr, nxv, nye, nze),
                idx3(ii, jr, ze, nxv, nye, nze),
                4,
                3
            );
            handle_boundary_conditions(
                fri,
                bstag1,
                x1ke,
                idx3(ii, jr, ze, nxv, nye, nze),
                idx3(ii, jr, hr, nxv, nye, nze),
                5,
                3
            );
        }
    });

    // update the flux and field in the x2 direction
    parallel_for(yvertexPolicy, [=, this] DEV(const luint gid) {
        const luint kk = axid<3, BlkAx::K>(gid, xag, nyv);
        const luint jj = axid<3, BlkAx::J>(gid, xag, nyv, kk);
        const luint ii = axid<3, BlkAx::I>(gid, xag, nyv, kk);

        if (global::on_gpu) {
            if ((ii >= xag) || (jj >= nyv) || (kk >= zag)) {
                return;
            }
        }

        // the fluxes only ever need the ghosts zones one cell deep
        constexpr auto hr = 1;
        constexpr auto rr = 0;
        constexpr auto rs = rr + 1;
        const auto ir     = ii + hr;
        const auto kr     = kk + hr;
        // fill the ghost zones for the fluxes at the x2 boundaries
        if (ii < xe && kk < ze) {
            const auto x2ib = idx3(rr, jj, kr, nxe, nyv, nze);
            const auto x2ie = idx3(nxe - rs, jj, kr, nxe, nyv, nze);
            const auto x2kb = idx3(ir, jj, rr, nxe, nyv, nze);
            const auto x2ke = idx3(ir, jj, nze - rs, nxe, nyv, nze);

            // across x1 interface
            handle_boundary_conditions(
                gri,
                bstag2,
                x2ib,
                idx3(hr, jj, kr, nxe, nyv, nze),
                idx3(xe, jj, kr, nxe, nyv, nze),
                0,
                1
            );
            handle_boundary_conditions(
                gri,
                bstag2,
                x2ie,
                idx3(xe, jj, kr, nxe, nyv, nze),
                idx3(hr, jj, kr, nxe, nyv, nze),
                1,
                1
            );

            // across x3 interface
            handle_boundary_conditions(
                gri,
                bstag2,
                x2kb,
                idx3(ir, jj, hr, nxe, nyv, nze),
                idx3(ir, jj, ze, nxe, nyv, nze),
                4,
                3
            );
            handle_boundary_conditions(
                gri,
                bstag2,
                x2ke,
                idx3(ir, jj, ze, nxe, nyv, nze),
                idx3(ir, jj, hr, nxe, nyv, nze),
                5,
                3
            );
        }
    });

    // update the flux and field in the x3 direction
    parallel_for(zvertexPolicy, [=, this] DEV(const luint gid) {
        const luint kk = axid<3, BlkAx::K>(gid, xag, yag);
        const luint jj = axid<3, BlkAx::J>(gid, xag, yag, kk);
        const luint ii = axid<3, BlkAx::I>(gid, xag, yag, kk);

        if (global::on_gpu) {
            if ((ii >= xag) || (jj >= yag) || (kk >= nzv)) {
                return;
            }
        }

        // the fluxes only ever need the ghosts zones one cell deep
        constexpr auto hr = 1;
        const auto ir     = ii + hr;
        const auto jr     = jj + hr;
        const auto rr     = 0;
        const auto rs     = rr + 1;
        // fill the ghost zones for the fluxes at the x3 boundaries
        if (ii < xe && jj < ye) {
            const auto x3ib = idx3(rr, jr, kk, nxe, nye, nzv);
            const auto x3ie = idx3(nxe - rs, jr, kk, nxe, nye, nzv);
            const auto x3jb = idx3(ir, rr, kk, nxe, nye, nzv);
            const auto x3je = idx3(ir, nye - rs, kk, nxe, nye, nzv);

            // across x1 interface
            handle_boundary_conditions(
                hri,
                bstag3,
                x3ib,
                idx3(hr, jr, kk, nxe, nye, nzv),
                idx3(xe, jr, kk, nxe, nye, nzv),
                0,
                3
            );
            handle_boundary_conditions(
                hri,
                bstag3,
                x3ie,
                idx3(xe, jr, kk, nxe, nye, nzv),
                idx3(hr, jr, kk, nxe, nye, nzv),
                1,
                3
            );

            // across x2 interface
            handle_boundary_conditions(
                hri,
                bstag3,
                x3jb,
                idx3(ir, hr, kk, nxe, nye, nzv),
                idx3(ir, ye, kk, nxe, nye, nzv),
                2,
                m2inner
            );
            handle_boundary_conditions(
                hri,
                bstag3,
                x3je,
                idx3(ir, ye, kk, nxe, nye, nzv),
                idx3(ir, hr, kk, nxe, nye, nzv),
                3,
                m2outer
            );
        }
    });
}

template <int dim>
void RMHD<dim>::riemann_fluxes()
{
    const auto prim_dat = prims.data();
    // Compute the fluxes in the x1 direction
    simbi::parallel_for(xvertexPolicy, [prim_dat, this] DEV(const luint idx) {
        //  primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        const auto prb = sm_or_identity(prim_dat);

        const luint kk = axid<dim, BlkAx::K>(idx, nxv, yag);
        const luint jj = axid<dim, BlkAx::J>(idx, nxv, yag, kk);
        const luint ii = axid<dim, BlkAx::I>(idx, nxv, yag, kk);

        if constexpr (global::on_gpu) {
            if ((ii >= nxv) || (jj >= yag) || (kk >= zag)) {
                return;
            }
        }

        // active zones for primitive variables
        const luint ia = ii + radius;
        const luint ja = jj + radius;
        const luint ka = kk + radius;

        const luint tx  = (global::on_sm) ? threadIdx.x : 0;
        const luint ty  = (global::on_sm) ? threadIdx.y : 0;
        const luint tz  = (global::on_sm) ? threadIdx.z : 0;
        const luint txa = (global::on_sm) ? tx + radius : ia;
        const luint tya = (global::on_sm) ? ty + radius : ja;
        const luint tza = (global::on_sm) ? tz + radius : ka;

        if constexpr (global::on_sm) {
            load_shared_buffer<dim>(
                fullPolicy,
                prb,
                prims,
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

        const auto iobj = get_real_idx(ii, 0, xag);

        // object to left or right ? (x1 - direction)
        const bool object_x = false;
        ib_check<dim>(object_pos, iobj, jj, kk, xag, yag, 1);

        const auto vface = this->cell_geometry(ii, jj, kk).velocity(Side::X1L);
        // active x1 flux perpendicular zone indices
        const auto jaf = jj + 1;
        const auto kaf = kk + 1;
        const auto xf  = idx3(ii, jaf, kaf, nxv, nye, nze);

        // fluxes in i direction
        auto pL = prb[idx3(txa - 1, tya, tza, sx, sy, sz)];
        auto pR = prb[idx3(txa + 0, tya, tza, sx, sy, sz)];

        if (!use_pcm) {
            const auto pLL = prb[idx3(txa - 2, tya, tza, sx, sy, sz)];
            const auto pRR = prb[idx3(txa + 1, tya, tza, sx, sy, sz)];

            pL = pL + plm_gradient(*pL, *pLL, *pR, plm_theta) * 0.5;
            pR = pR - plm_gradient(*pR, *pL, *pRR, plm_theta) * 0.5;
        }
        ib_modify<dim>(pR, pL, object_x, 1);
        fri[xf] = (this->*riemann_solve)(pL, pR, 1, vface, bstag1[xf]);
    });

    // compute the fluxes in the x2 direction
    simbi::parallel_for(yvertexPolicy, [prim_dat, this] DEV(const luint idx) {
        // primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        const auto prb = sm_or_identity(prim_dat);

        const luint kk = axid<dim, BlkAx::K>(idx, xag, nyv);
        const luint jj = axid<dim, BlkAx::J>(idx, xag, nyv, kk);
        const luint ii = axid<dim, BlkAx::I>(idx, xag, nyv, kk);

        if constexpr (global::on_gpu) {
            if ((ii >= xag) || (jj >= nyv) || (kk >= zag)) {
                return;
            }
        }

        // active zones for primitive variables
        const luint ia = ii + radius;
        const luint ja = jj + radius;
        const luint ka = kk + radius;

        const luint tx  = (global::on_sm) ? threadIdx.x : 0;
        const luint ty  = (global::on_sm) ? threadIdx.y : 0;
        const luint tz  = (global::on_sm) ? threadIdx.z : 0;
        const luint txa = (global::on_sm) ? tx + radius : ia;
        const luint tya = (global::on_sm) ? ty + radius : ja;
        const luint tza = (global::on_sm) ? tz + radius : ka;

        if constexpr (global::on_sm) {
            load_shared_buffer<dim>(
                fullPolicy,
                prb,
                prims,
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

        const auto jobj = get_real_idx(jj, 0, yag);

        // object in front or behind? (x2-direction)
        const bool object_y =
            ib_check<dim>(object_pos, ii, jobj, kk, xag, yag, 2);

        const auto vface = this->cell_geometry(ii, jj, kk).velocity(Side::X2L);
        // active x2 flux perpendicular zone indices
        const auto iaf = ii + 1;
        const auto kaf = kk + 1;
        const auto yf  = idx3(iaf, jj, kaf, nxe, nyv, nze);

        // fluxes in j direction
        auto pL = prb[idx3(txa, tya - 1, tza, sx, sy, sz)];
        auto pR = prb[idx3(txa, tya + 0, tza, sx, sy, sz)];
        if (!use_pcm) {
            const auto pLL = prb[idx3(txa, tya - 2, tza, sx, sy, sz)];
            const auto pRR = prb[idx3(txa, tya + 1, tza, sx, sy, sz)];

            pL = pL + plm_gradient(*pL, *pLL, *pR, plm_theta) * 0.5;
            pR = pR - plm_gradient(*pR, *pL, *pRR, plm_theta) * 0.5;
        }
        ib_modify<dim>(pR, pL, object_y, 2);
        gri[yf] = (this->*riemann_solve)(pL, pR, 2, vface, bstag2[yf]);
    });

    // compute the fluxes in the x3 direction
    simbi::parallel_for(zvertexPolicy, [prim_dat, this] DEV(const luint idx) {
        // primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        const auto prb = sm_or_identity(prim_dat);

        const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
        const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
        const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

        if constexpr (global::on_gpu) {
            if ((ii >= xag) || (jj >= yag) || (kk >= nzv)) {
                return;
            }
        }

        // active zones for primitive variables
        const luint ia = ii + radius;
        const luint ja = jj + radius;
        const luint ka = kk + radius;

        const luint tx  = (global::on_sm) ? threadIdx.x : 0;
        const luint ty  = (global::on_sm) ? threadIdx.y : 0;
        const luint tz  = (global::on_sm) ? threadIdx.z : 0;
        const luint txa = (global::on_sm) ? tx + radius : ia;
        const luint tya = (global::on_sm) ? ty + radius : ja;
        const luint tza = (global::on_sm) ? tz + radius : ka;

        if constexpr (global::on_sm) {
            load_shared_buffer<dim>(
                fullPolicy,
                prb,
                prims,
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

        const auto kobj = get_real_idx(kk, 0, zag);
        // object above or below? (x3-direction)
        const bool object_z =
            ib_check<dim>(object_pos, ii, jj, kobj, xag, yag, 3);

        const auto vface = this->cell_geometry(ii, jj, kk).velocity(Side::X3L);
        // active x3 flux perpendicular zone indices
        const auto iaf = ii + 1;
        const auto jaf = jj + 1;
        const auto zf  = idx3(iaf, jaf, kk, nxe, nye, nzv);

        // fluxes in k direction
        auto pL = prb[idx3(txa, tya, tza - 1, sx, sy, sz)];
        auto pR = prb[idx3(txa, tya, tza + 0, sx, sy, sz)];

        if (!use_pcm) {
            const auto pLL = prb[idx3(txa, tya, tza - 2, sx, sy, sz)];
            const auto pRR = prb[idx3(txa, tya, tza + 1, sx, sy, sz)];

            pL = pL + plm_gradient(*pL, *pLL, *pR, plm_theta) * 0.5;
            pR = pR - plm_gradient(*pR, *pL, *pRR, plm_theta) * 0.5;
        }
        ib_modify<dim>(pR, pL, object_z, 3);
        hri[zf] = (this->*riemann_solve)(pL, pR, 3, vface, bstag3[zf]);
    });
}

//===================================================================================================================
//                                           SOURCE TERMS
//===================================================================================================================

template <int dim>
DUAL real RMHD<dim>::div_b(
    const auto b1L,
    const auto b1R,
    const auto b2L,
    const auto b2R,
    const auto b3L,
    const auto b3R,
    const auto& cell
) const
{
    // compute 3D divergence of magnetic field depending on geometry
    switch (geometry) {
        case Geometry::CARTESIAN:
            return (b1R - b1L) * cell.idx1() + (b2R - b2L) * cell.idx2() +
                   (b3R - b3L) * cell.idx3();
        case Geometry::SPHERICAL: {
            const auto r      = cell.x1mean;
            const auto r2     = r * r;
            const auto rrf    = cell.x1R() * cell.x1R();
            const auto rlf    = cell.x1L() * cell.x1L();
            const auto dr     = cell.x1R() - cell.x1L();
            const auto dtheta = cell.x2R() - cell.x2L();
            const auto dphi   = cell.x3R() - cell.x3L();
            const auto sint   = std::sin(cell.x2mean);
            const auto tlf    = std::sin(cell.x2L());
            const auto trf    = std::sin(cell.x2R());
            const auto rsint  = r * sint;
            return (rrf * b1R - rlf * b1L) / (r2 * dr) +
                   (trf * b2R - tlf * b2L) / (rsint * dtheta) +
                   (b3R - b3L) / (rsint * dphi);
        }
        default: {   // Cylindrical
            const auto r    = cell.x1mean;
            const auto dr   = cell.x1R() - cell.x1L();
            const auto dphi = cell.x2R() - cell.x2L();
            const auto dz   = cell.x3R() - cell.x3L();
            return (cell.x1R() * b1R - cell.x1L() * b1L) / (r * dr) +
                   (b2R - b2L) / (dphi * r) + (b3R - b3L) / dz;
        }
    }
}

template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::hydro_sources(const auto& cell) const
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
DUAL RMHD<dim>::conserved_t
RMHD<dim>::gravity_sources(const auto& prims, const auto& cell) const
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
template <size_type i, size_type j>
DUAL void RMHD<dim>::calc_emf_edges(
    real ei[],
    real ej[],
    const auto& cell,
    size_type ii,
    size_type jj,
    size_type kk
) const
{
    auto get_flux_info = [&] DEV(size_type dir) {
        if (dir == 1) {
            return std::make_tuple(fri, bstag1, nxv, nye, nze);
        }
        if (dir == 2) {
            return std::make_tuple(gri, bstag2, nxe, nyv, nze);
        }
        return std::make_tuple(hri, bstag3, nxe, nye, nzv);
    };

    // Get active flux indices
    const auto iaf = ii + 1;
    const auto jaf = jj + 1;
    const auto kaf = kk + 1;

    // process each corner
    detail::for_sequence(detail::make_index_sequence<4>(), [&](auto qidx) {
        constexpr auto q      = static_cast<size_type>(qidx);
        constexpr auto corner = static_cast<Corner>(q);
        // Calculate appropriate EMF components for this edge
        // Based on staggered grid formulation
        // ...EMF calculation logic specific to components i,j

        // Compute shifted indices for this corner
        auto north = corner == Corner::NE || corner == Corner::NW;
        auto east  = corner == Corner::NE || corner == Corner::SE;
        auto south = !north;
        auto west  = !east;

        // Get staggered field components
        auto [flux_i, bstag_i, nx_i, ny_i, nz_i] = get_flux_info(i);
        auto [flux_j, bstag_j, nx_j, ny_j, nz_j] = get_flux_info(j);

        if constexpr (i == 2 && j == 3) {
            // x2-x3 plane
            auto qn = kaf + north;
            auto qs = kaf - south;
            auto qe = jaf + east;
            auto qw = jaf - west;

            auto nidx = idx3(iaf, jj + east, qn, nx_i, ny_i, nz_i);
            auto sidx = idx3(iaf, jj + east, qs, nx_i, ny_i, nz_i);
            auto eidx = idx3(iaf, qe, kk + north, nx_j, ny_j, nz_j);
            auto widx = idx3(iaf, qw, kk + north, nx_j, ny_j, nz_j);

            ei[q] = calc_edge_emf<Plane::JK, corner>(
                flux_j[widx],
                flux_j[eidx],
                flux_i[sidx],
                flux_i[nidx],
                prims.data(),
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                i,
                bstag_j[widx],
                bstag_j[eidx],
                bstag_i[sidx],
                bstag_i[nidx]
            );
        }
        else if constexpr (i == 3 && j == 1) {
            // x3-x1 plane
            auto qn = jaf + north;
            auto qs = jaf - south;
            auto qe = iaf + east;
            auto qw = iaf - west;

            auto nidx = idx3(iaf, jj + north, qn, nx_i, ny_i, nz_i);
            auto sidx = idx3(iaf, jj + north, qs, nx_i, ny_i, nz_i);
            auto eidx = idx3(qe, jj + north, kk + east, nx_j, ny_j, nz_j);
            auto widx = idx3(qw, jj + north, kk + east, nx_j, ny_j, nz_j);

            ei[q] = calc_edge_emf<Plane::IK, corner>(
                flux_i[widx],
                flux_i[eidx],
                flux_j[sidx],
                flux_j[nidx],
                prims.data(),
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                i,
                bstag_i[widx],
                bstag_i[eidx],
                bstag_j[sidx],
                bstag_j[nidx]
            );
        }
        else {
            // x1-x2 plane
            auto qn = iaf + north;
            auto qs = iaf - south;
            auto qe = jaf + east;
            auto qw = jaf - west;

            auto nidx = idx3(qn, jj + east, kk + north, nx_i, ny_i, nz_i);
            auto sidx = idx3(qs, jj + east, kk + north, nx_i, ny_i, nz_i);
            auto eidx = idx3(qe, jj + east, kk + north, nx_j, ny_j, nz_j);
            auto widx = idx3(qw, jj + east, kk + north, nx_j, ny_j, nz_j);

            ei[q] = calc_edge_emf<Plane::IJ, corner>(
                flux_i[widx],
                flux_i[eidx],
                flux_j[sidx],
                flux_j[nidx],
                prims.data(),
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                i,
                bstag_i[widx],
                bstag_i[eidx],
                bstag_j[sidx],
                bstag_j[nidx]
            );
        }
    });
}

template <int dim>
template <int nhat>
void RMHD<dim>::update_magnetic_component(const ExecutionPolicy<>& policy)
{
    const auto& b_stag = (nhat == 1) ? bstag1 : (nhat == 2) ? bstag2 : bstag3;

    // Create contracted views for EMF calculation
    auto magnetic_update = [this] DEV(auto& b_view, const auto& coord) {
        // Get local coordinates for EMF calculation
        const auto [ii, jj, kk] = b_view.position();
        const auto cell         = this->cell_geometry(ii, jj, kk);

        // Calculate EMF components based on nhat
        real ei[4], ej[4];

        if constexpr (nhat == 1) {
            calc_emf_edges<2, 3>(ei, ej, cell, ii, jj, kk);
            // Update B1 using e2,e3
            return b_view.value() -
                   dt * step * curl_e(1, ei, ej, cell, coord.side());
        }
        else if constexpr (nhat == 2) {
            calc_emf_edges<3, 1>(ei, ej, cell, ii, jj, kk);
            // Update B2 using e3,e1
            return b_view.value() -
                   dt * step * curl_e(2, ei, ej, cell, coord.side());
        }
        else {
            calc_emf_edges<1, 2>(ei, ej, cell, ii, jj, kk);
            // Update B3 using e1,e2
            return b_view.value() -
                   dt * step * curl_e(3, ei, ej, cell, coord.side());
        }
    };

    // Transform using stencil view
    b_stag.contract(radius).stencil_transform(magnetic_update, policy);
}

template <int dim>
void RMHD<dim>::advance_magnetic_fields()
{
    // update the magnetic field components
    update_magnetic_component<1>(xvertexPolicy);
    update_magnetic_component<2>(yvertexPolicy);
    update_magnetic_component<3>(zvertexPolicy);
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
            res -= fr.at(q - 1, 0, 0) * cell.idV1() * cell.area(0 + q) * sign;
            if constexpr (dim > 1) {
                res -=
                    gr.at(0, q - 1, 0) * cell.idV2() * cell.area(2 + q) * sign;
                if constexpr (dim > 2) {
                    res -= hr.at(0, 0, q - 1) * cell.idV3() * cell.area(4 + q) *
                           sign;
                }
            }
        }

        res += source_terms;
        res += gravity;
        res += geometrical_sources;

        return res * step * dt;
    };

    auto update_conserved = [this, dcons] DEV(
                                auto& con,
                                const auto& prim,
                                const auto& fr,
                                const auto& gr,
                                const auto& hr
                            ) {
        const auto [ii, jj, kk] = con.position();
        // mesh factors
        const auto cell = this->cell_geometry(ii, jj, kk);

        // compute the change in conserved variables
        const auto dc = dcons(
            fr,
            gr,
            hr,
            hydro_sources(cell),
            gravity_sources(prim.value(), cell),
            cell.geometrical_sources(prim.value()),
            cell
        );

        // update conserved variables
        return con.value() + dc;
    };

    // Transform using stencil operations
    cons.contract(radius).stencil_transform(
        update_conserved,
        activePolicy,
        prims.contract(radius),
        fri.contract(radius),
        gri.contract(radius),
        hri.contract(radius)
    );
}

template <int dim>
void RMHD<dim>::advance()
{
    advance_magnetic_fields();
    advance_conserved();
}

// void old_advance()
// {
//     const auto prim_dat = prims.data();
//     const auto b1const  = bstag1;
//     const auto b2const  = bstag2;
//     const auto b3const  = bstag3;
//     simbi::parallel_for(
//         activePolicy,
//         [prim_dat, b1const, b2const, b3const, this] DEV(const luint idx) {
//             // e1, e2, e3 values at cell edges
//             real e1[4], e2[4], e3[4];

//             // primitive buffer that returns dynamic shared array
//             // if working with shared memory on GPU, identity otherwise
//             const auto prb = sm_or_identity(prim_dat);

//             const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
//             const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
//             const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

//             if constexpr (global::on_gpu) {
//                 if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
//                     return;
//                 }
//             }

//             // active zones for primitive variables
//             const luint ia = ii + radius;
//             const luint ja = jj + radius;
//             const luint ka = kk + radius;
//             // active zones for fluxes
//             const luint iaf = ii + 1;
//             const luint jaf = jj + 1;
//             const luint kaf = kk + 1;
//             const luint tx  = (global::on_sm) ? threadIdx.x : 0;
//             const luint ty  = (global::on_sm) ? threadIdx.y : 0;
//             const luint tz  = (global::on_sm) ? threadIdx.z : 0;
//             const luint txa = (global::on_sm) ? tx + radius : ia;
//             const luint tya = (global::on_sm) ? ty + radius : ja;
//             const luint tza = (global::on_sm) ? tz + radius : ka;
//             const luint aid = idx3(ia, ja, ka, nx, ny, nz);
//             const luint tid = idx3(txa, tya, tza, sx, sy, sz);

//             if constexpr (global::on_sm) {
//                 load_shared_buffer<dim>(
//                     activePolicy,
//                     prb,
//                     prims,
//                     nx,
//                     ny,
//                     nz,
//                     sx,
//                     sy,
//                     tx,
//                     ty,
//                     tz,
//                     txa,
//                     tya,
//                     tza,
//                     ia,
//                     ja,
//                     ka,
//                     radius
//                 );
//             }

//             // mesh factors
//             const auto cell = this->cell_geometry(ii, jj, kk);

//             const auto xlf = idx3(ii + 0, jaf, kaf, nxv, nye, nze);
//             const auto xrf = idx3(ii + 1, jaf, kaf, nxv, nye, nze);
//             const auto ylf = idx3(iaf, jj + 0, kaf, nxe, nyv, nze);
//             const auto yrf = idx3(iaf, jj + 1, kaf, nxe, nyv, nze);
//             const auto zlf = idx3(iaf, jaf, kk + 0, nxe, nye, nzv);
//             const auto zrf = idx3(iaf, jaf, kk + 1, nxe, nye, nzv);

//             // compute edge emfs in clockwise direction wrt cell plane
//             detail::for_sequence(
//                 detail::make_index_sequence<4>(),
//                 [&](auto qidx) {
//                     constexpr auto q      = static_cast<luint>(qidx);
//                     constexpr auto corner = static_cast<Corner>(q);

//                     // calc directional indices for i-j plane
//                     auto north = corner == Corner::NE || corner ==
//                     Corner::NW; auto east  = corner == Corner::NE || corner
//                     == Corner::SE; auto south = !north; auto west  = !east;

//                     auto qn = jaf + north;
//                     auto qs = jaf - south;
//                     auto qe = iaf + east;
//                     auto qw = iaf - west;

//                     auto nidx = idx3(ii + east, qn, kaf, nxv, nye, nze);
//                     auto sidx = idx3(ii + east, qs, kaf, nxv, nye, nze);
//                     auto eidx = idx3(qe, jj + north, kaf, nxe, nyv, nze);
//                     auto widx = idx3(qw, jj + north, kaf, nxe, nyv, nze);

//                     e3[q] = calc_edge_emf<Plane::IJ, corner>(
//                         gri[widx],
//                         gri[eidx],
//                         fri[sidx],
//                         fri[nidx],
//                         prb,
//                         ii,
//                         jj,
//                         kk,
//                         txa,
//                         tya,
//                         tza,
//                         3,
//                         b2const[widx],
//                         b2const[eidx],
//                         b1const[sidx],
//                         b1const[nidx]
//                     );

//                     // calc directional indices for i-k plane
//                     qn   = kaf + north;
//                     qs   = kaf - south;
//                     qe   = iaf + east;
//                     qw   = iaf - west;
//                     nidx = idx3(ii + east, jaf, qn, nxv, nye, nze);
//                     sidx = idx3(ii + east, jaf, qs, nxv, nye, nze);
//                     eidx = idx3(qe, jaf, kk + north, nxe, nye, nzv);
//                     widx = idx3(qw, jaf, kk + north, nxe, nye, nzv);

//                     e2[q] = calc_edge_emf<Plane::IK, corner>(
//                         hri[widx],
//                         hri[eidx],
//                         fri[sidx],
//                         fri[nidx],
//                         prb,
//                         ii,
//                         jj,
//                         kk,
//                         txa,
//                         tya,
//                         tza,
//                         2,
//                         b3const[widx],
//                         b3const[eidx],
//                         b1const[sidx],
//                         b1const[nidx]
//                     );

//                     // calc directional indices for j-k plane
//                     qn   = kaf + north;
//                     qs   = kaf - south;
//                     qe   = jaf + east;
//                     qw   = jaf - west;
//                     nidx = idx3(iaf, jj + east, qn, nxe, nyv, nze);
//                     sidx = idx3(iaf, jj + east, qs, nxe, nyv, nze);
//                     eidx = idx3(iaf, qe, kk + north, nxe, nye, nzv);
//                     widx = idx3(iaf, qw, kk + north, nxe, nye, nzv);

//                     e1[q] = calc_edge_emf<Plane::JK, corner>(
//                         hri[widx],
//                         hri[eidx],
//                         gri[sidx],
//                         gri[nidx],
//                         prb,
//                         ii,
//                         jj,
//                         kk,
//                         txa,
//                         tya,
//                         tza,
//                         1,
//                         b3const[widx],
//                         b3const[eidx],
//                         b2const[sidx],
//                         b2const[nidx]
//                     );
//                 }
//             );

//             auto& b1L = bstag1[xlf];
//             auto& b1R = bstag1[xrf];
//             auto& b2L = bstag2[ylf];
//             auto& b2R = bstag2[yrf];
//             auto& b3L = bstag3[zlf];
//             auto& b3R = bstag3[zrf];
//             auto& b1c = cons[aid].b1();
//             auto& b2c = cons[aid].b2();
//             auto& b3c = cons[aid].b3();

//             b1L = b1const[xlf] - dt * step * curl_e(1, e2, e3, cell, 0);
//             b1R = b1const[xrf] - dt * step * curl_e(1, e2, e3, cell, 1);

//             b2L = b2const[ylf] - dt * step * curl_e(2, e3, e1, cell, 0);
//             b2R = b2const[yrf] - dt * step * curl_e(2, e3, e1, cell, 1);

//             b3L = b3const[zlf] - dt * step * curl_e(3, e1, e2, cell, 0);
//             b3R = b3const[zrf] - dt * step * curl_e(3, e1, e2, cell, 1);

//             if constexpr (global::debug_mode) {
//                 const auto divb = div_b(b1L, b1R, b2L, b2R, b3L, b3R, cell);
//                 if (!goes_to_zero(divb)) {
//                     printf("========================================\n");
//                     printf("DIV.B: %.2e\n", divb);
//                     printf("========================================\n");
//                     printf(
//                         "[%" PRIu64 "] Divergence of B is not zero at: %"
//                         PRIu64
//                         ", %" PRIu64 ", %" PRIu64 "!\n",
//                         global_iter,
//                         ii,
//                         jj,
//                         kk
//                     );
//                     printf("B1L: %.2e, B1R: %.2e\n", b1L, b1R);
//                     printf("B2L: %.2e, B2R: %.2e\n", b2L, b2R);
//                     printf("B3L: %.2e, B3R: %.2e\n", b3L, b3R);
//                     printf("theta_mean: %.2e\n", cell.x2mean);
//                     printf("tL: %.2e, tR: %.2e\n", cell.x2L(), cell.x2R());
//                     std::cin.get();
//                 }
//             }

//             b1c = static_cast<real>(0.5) * (b1R + b1L);
//             b2c = static_cast<real>(0.5) * (b2R + b2L);
//             b3c = static_cast<real>(0.5) * (b3R + b3L);

//             const auto source_terms = hydro_sources(cell);

//             // Gravity
//             const auto gravity = gravity_sources(prb[tid], cell);

//             // geometric source terms
//             const auto geometrical_sources =
//             cell.geometrical_sources(prb[tid]);

//             cons[aid] -=
//                 ((fri[xrf] * cell.a1R() - fri[xlf] * cell.a1L()) *
//                 cell.idV1() +
//                  (gri[yrf] * cell.a2R() - gri[ylf] * cell.a2L()) *
//                  cell.idV2() + (hri[zrf] * cell.a3R() - hri[zlf] *
//                  cell.a3L()) * cell.idV3() - source_terms - gravity -
//                  geometrical_sources) *
//                 dt * step;
//         }
//     );
// }

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void RMHD<dim>::simulate(
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

    // allocate space for Riemann fluxes
    auto total_halos = std::pow(nhalos, dim);
    fri.resize(nze * nye * nxv * total_halos).reshape({nze, nye, nxv});
    gri.resize(nze * nyv * nxe * total_halos).reshape({nze, nyv, nxe});
    hri.resize(nzv * nye * nxe * total_halos).reshape({nzv, nye, nxe});

    // deallocate the old magnetic field
    deallocate_staggered_field();
    // allocate space for volume-average quantities
    cons.reshape({nz, ny, nx});
    prims.reshape({nz, ny, nx});
    if constexpr (global::on_gpu) {
        dt_min.reshape({nz, ny, nx});
    }

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        for (int q = 0; q < conserved_t::nmem; q++) {
            cons[i][q] = state[q][i];
        }
    }
    // set up the problem and release old state from memory
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    init_riemann_solver();
    // this->set_mesh_funcs();

    boundary_manager<conserved_t, dim> bman;
    // true = sync corners too
    bman.sync_boundaries(fullPolicy, cons, cons.contract(2), bcs, true);
    cons2prim();
    adapt_dt<TIMESTEP_TYPE::MINIMUM>();

    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&] {
        riemann_fluxes();
        set_flux_and_fields();
        advance();
        bman.sync_boundaries(fullPolicy, cons, cons.contract(2), bcs, true);
        cons2prim();
        adapt_dt();

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
