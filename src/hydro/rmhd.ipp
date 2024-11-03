#include "util/device_api.hpp"     // for syncrohonize, devSynch, ...
#include "util/idx_sequence.hpp"   // for for_sequence, make_index_sequence
#include "util/logger.hpp"         // for logger
#include "util/parallel_for.hpp"   // for parallel_for
#include "util/printb.hpp"         // for writeln
#include <cmath>                   // for max, min

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
    const InitialConditions& init_conditions
)
    : HydroBase(state, init_conditions)
{
}

// Destructor
template <int dim>
RMHD<dim>::~RMHD() = default;

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
    const RMHD<dim>::conserved_t& fw,
    const RMHD<dim>::conserved_t& fe,
    const RMHD<dim>::conserved_t& fs,
    const RMHD<dim>::conserved_t& fn,
    const RMHD<dim>::primitive_t* prims,
    const luint ii,
    const luint jj,
    const luint kk,
    const luint ia,
    const luint ja,
    const luint ka,
    const luint nhat
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

    // get mean efields
    const real esw        = swp.ecomponent(nhat);
    const real ese        = sep.ecomponent(nhat);
    const real enw        = nwp.ecomponent(nhat);
    const real ene        = nep.ecomponent(nhat);
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
                             ek[IJ::SW] * std::sin(tr)) /
                                dth -
                            (ej[IK::NW] - ej[IK::SW]) / dph) /
                           (cell.x1mean * std::sin(cell.x2mean));
                }
                return ((ek[IJ::NE] * std::sin(tr) - ek[IJ::SE] * std::sin(tr)
                        ) / dth -
                        (ej[IK::NE] - ej[IK::SE]) / dph) /
                       (cell.x1mean * std::sin(cell.x2mean));
            }
            else if (nhat == 2) {
                // compute the curl in the theta-hat direction
                const real dr  = cell.x1R() - cell.x1L();
                const real dph = cell.x3R() - cell.x3L();
                if (side == 0) {
                    return (-(ek[JK::NW] * cell.x1R() - ek[JK::SW] * cell.x1L()
                            ) / dr +
                            (ej[IJ::SE] - ej[IJ::SW]) /
                                (std::sin(cell.x2mean) * dph)) /
                           cell.x1mean;
                }
                return (-(ek[JK::NE] * cell.x1R() - ek[JK::SE] * cell.x1L()) /
                            dr +
                        (ej[IJ::NE] - ej[IJ::NW]) /
                            (std::sin(cell.x2mean) * dph)) /
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

template <int dim>
void RMHD<dim>::emit_troubled_cells() const
{
    for (luint gid = 0; gid < total_zones; gid++) {
        if (troubled_cells[gid] != 0) {
            const luint kk    = get_height(gid, nx, ny);
            const luint jj    = get_row(gid, nx, ny, kk);
            const luint ii    = get_column(gid, nx, ny, kk);
            const lint ireal  = get_real_idx(ii, radius, xag);
            const lint jreal  = get_real_idx(jj, radius, yag);
            const lint kreal  = get_real_idx(kk, radius, zag);
            const auto cell   = this->cell_factors(ireal, jreal, kreal);
            const real x1mean = cell.x1mean;
            const real x2mean = yag == 1 ? 0.0 : cell.x2mean;
            const real x3mean = zag == 1 ? 0.0 : cell.x3mean;
            const real m1     = cons[gid].momentum(1);
            const real m2     = cons[gid].momentum(2);
            const real m3     = cons[gid].momentum(3);
            const real et     = (cons[gid].dens() + cons[gid].nrg());
            const real b1     = cons[gid].bcomponent(1);
            const real b2     = cons[gid].bcomponent(2);
            const real b3     = cons[gid].bcomponent(3);
            const real m      = std::sqrt(m1 * m1 + m2 * m2 + m3 * m3);
            const real vsq    = (m * m) / (et * et);
            const real bsq    = (b1 * b1 + b2 * b2 + b3 * b3);
            const real w      = 1.0 / std::sqrt(1.0 - vsq);
            fprintf(
                stderr,
                "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                "%.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, x2coord: "
                "%.2e, x3coord: %.2e, iter: %" PRIu64 "\n",
                cons[gid].dens() / w,
                prims[gid].p(),
                vsq,
                bsq,
                x1mean,
                x2mean,
                x3mean,
                global_iter
            );
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
    // const auto gr = gamma / (gamma - 1.0);
    simbi::parallel_for(fullP, total_zones, [this] DEV(luint gid) {
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
            const real d    = cons[gid].dens() * invdV;
            const real m1   = cons[gid].momentum(1) * invdV;
            const real m2   = cons[gid].momentum(2) * invdV;
            const real m3   = cons[gid].momentum(3) * invdV;
            const real tau  = cons[gid].nrg() * invdV;
            const real b1   = cons[gid].bcomponent(1) * invdV;
            const real b2   = cons[gid].bcomponent(2) * invdV;
            const real b3   = cons[gid].bcomponent(3) * invdV;
            const real dchi = cons[gid].chi() * invdV;

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
                        troubled_cells[gid] = 1;
                        dt                  = INFINITY;
                        inFailureState      = true;
                        found_failure       = true;
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
                    troubled_cells[gid] = 1;
                    dt                  = INFINITY;
                    inFailureState      = true;
                    found_failure       = true;
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
            prims[gid] =
                primitive_t{rhohat, v1, v2, v3, pg, b1, b2, b3, dchi / d};

            workLeftToDo = false;

            if (!std::isfinite(yy)) {
                troubled_cells[gid] = 1;
                inFailureState      = true;
                found_failure       = true;
                dt                  = INFINITY;
            }
            simbi::gpu::api::synchronize();
        }
    });
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
DUAL RMHD<dim>::primitive_t
RMHD<dim>::cons2prim(const RMHD<dim>::conserved_t& cons) const
{
    const real d    = cons.dens();
    const real m1   = cons.momentum(1);
    const real m2   = cons.momentum(2);
    const real m3   = cons.momentum(3);
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
DUAL void RMHD<dim>::calc_max_wave_speeds(
    const RMHD<dim>::primitive_t& prims,
    const luint nhat,
    real speeds[4]
) const
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
    const real rho   = prims.rho();
    const real h     = prims.enthalpy(gamma);
    const real cs2   = (gamma * prims.p() / (rho * h));
    const auto bmu   = mag_fourvec_t(prims);
    const real bmusq = bmu.inner_product();
    const real bn    = prims.bcomponent(nhat);
    const real bn2   = bn * bn;
    const real vn    = prims.vcomponent(nhat);
    if (prims.vsquared() < global::epsilon) {   // Eq.(57)
        const real fac  = 1.0 / (rho * h + bmusq);
        const real a    = 1.0;
        const real b    = -(bmusq + rho * h * cs2 + bn2 * cs2) * fac;
        const real c    = cs2 * bn2 * fac;
        const real disq = std::sqrt(b * b - 4.0 * a * c);
        speeds[3]       = std::sqrt(0.5 * (-b + disq));
        speeds[0]       = -speeds[3];
    }
    else if (bn2 < global::epsilon) {   // Eq. (58)
        const real g2 = prims.lorentz_factor_squared();
        const real vdbperp =
            prims.vdotb() - prims.vcomponent(nhat) * prims.bcomponent(nhat);
        const real q    = bmusq - cs2 * vdbperp * vdbperp;
        const real a2   = rho * h * (cs2 + g2 * (1.0 - cs2)) + q;
        const real a1   = -2.0 * rho * h * g2 * vn * (1.0 - cs2);
        const real a0   = rho * h * (-cs2 + g2 * vn * vn * (1.0 - cs2)) - q;
        const real disq = a1 * a1 - 4.0 * a2 * a0;
        speeds[3]       = 0.5 * (-a1 + std::sqrt(disq)) / a2;
        speeds[0]       = 0.5 * (-a1 - std::sqrt(disq)) / a2;
    }
    else {   // solve the full quartic Eq. (56)
        const real bmu0 = bmu.zero;
        const real bmun = bmu.normal(nhat);
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

        [[maybe_unused]] const auto nroots = quartic(a3, a2, a1, a0, speeds);

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
    const RMHD<dim>::primitive_t& primsL,
    const RMHD<dim>::primitive_t& primsR,
    const luint nhat
) const
{
    real speeds[4];
    // left side
    calc_max_wave_speeds(primsL, nhat, speeds);
    const real lpL = speeds[3];
    const real lmL = speeds[0];

    // right_side
    calc_max_wave_speeds(primsR, nhat, speeds);
    const real lpR = speeds[3];
    const real lmR = speeds[0];

    return {my_min(lmL, lmR), my_max(lpL, lpR)};
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template <int dim>
DUAL RMHD<dim>::conserved_t
RMHD<dim>::prims2cons(const RMHD<dim>::primitive_t& prims) const
{
    const real rho   = prims.rho();
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real pg    = prims.p();
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
#if GPU_CODE
    if constexpr (dim == 1) {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data());
        compute_dt<primitive_t, dt_type>
            <<<activeP.gridSize, activeP.blockSize>>>(
                this,
                prims.data(),
                dt_min.data()
            );
    }
    else {
        // LAUNCH_ASYNC((compute_dt<primitive_t,dt_type>),
        // p.gridSize, p.blockSize, this, prims.data(), dt_min.data(),
        // geometry);
        compute_dt<primitive_t, dt_type>
            <<<activeP.gridSize, activeP.blockSize>>>(
                this,
                prims.data(),
                dt_min.data(),
                geometry
            );
    }
    // LAUNCH_ASYNC((deviceReduceWarpAtomicKernel<dim>), p.gridSize,
    // p.blockSize, this, dt_min.data(), active_zones);

    deviceReduceWarpAtomicKernel<dim><<<activeP.gridSize, activeP.blockSize>>>(
        this,
        dt_min.data(),
        total_zones
    );
    gpu::api::deviceSynch();
#else
    // singleton instance of thread pool. lazy-evaluated
    static auto& thread_pool =
        simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(active_zones, [&](luint gid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
        real speeds[4];
        const luint kk  = axid<dim, BlkAx::K>(gid, xag, yag);
        const luint jj  = axid<dim, BlkAx::J>(gid, xag, yag, kk);
        const luint ii  = axid<dim, BlkAx::I>(gid, xag, yag, kk);
        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint ka  = kk + radius;
        const luint aid = idx3(ia, ja, ka, nx, ny, nz);
        // Left/Right wave speeds
        if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
            calc_max_wave_speeds(prims[aid], 1, speeds);
            v1p = std::abs(speeds[3]);
            v1m = std::abs(speeds[0]);
            calc_max_wave_speeds(prims[aid], 2, speeds);
            v2p = std::abs(speeds[3]);
            v2m = std::abs(speeds[0]);
            calc_max_wave_speeds(prims[aid], 3, speeds);
            v3p = std::abs(speeds[3]);
            v3m = std::abs(speeds[0]);
        }
        else {
            v1p = 1.0;
            v1m = 1.0;
            v2p = 1.0;
            v2m = 1.0;
            v3p = 1.0;
            v3m = 1.0;
        }

        const auto cell = this->cell_factors(ii, jj, kk);
        switch (geometry) {
            case simbi::Geometry::CARTESIAN: {
                const real x1l = cell.x1L();
                const real x1r = cell.x1R();
                const real dx1 = x1r - x1l;

                const real x2l = cell.x1L();
                const real x2r = cell.x1R();
                const real dx2 = x2r - x2l;

                cfl_dt = std ::min(
                    {dx1 / (std::max(v1p, v1m)),
                     dx2 / (std::max(v2p, v2m)),
                     dx3 / (std::max(v3p, v3m))}
                );

                break;
            }

            case simbi::Geometry::SPHERICAL: {
                const real x1l = cell.x1L();
                const real x1r = cell.x1R();
                const real dx1 = x1r - x1l;

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
                break;
            }
            default: {
                const real x1l = cell.x1L();
                const real x1r = cell.x1R();
                const real dx1 = x1r - x1l;

                const real rmean = cell.x1mean;
                cfl_dt           = std::min(
                    {dx1 / (std::max(v1p, v1m)),
                               rmean * dx2 / (std::max(v2p, v2m)),
                               dx3 / (std::max(v3p, v3m))}
                );
                break;
            }
        }
        pooling::update_minimum(min_dt, cfl_dt);
    });
    dt = cfl * min_dt;
#endif
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::prims2flux(
    const RMHD<dim>::primitive_t& prims,
    const luint nhat
) const
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
      m1 * vn + kronecker(nhat, 1) * ptot - bn * bmu.one * invlf,
      m2 * vn + kronecker(nhat, 2) * ptot - bn * bmu.two * invlf,
      m3 * vn + kronecker(nhat, 3) * ptot - bn * bmu.three * invlf,
      mn - d * vn,
      ind1,
      ind2,
      ind3,
      d * vn * chi
    };
};

template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::calc_hlle_flux(
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL     = prims2cons(prL);
    const auto uR     = prims2cons(prR);
    const auto fL     = prims2flux(prL, nhat);
    const auto fR     = prims2flux(prR, nhat);
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    // Grab the fastest wave speeds
    const real aL  = lambda.afL();
    const real aR  = lambda.afR();
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
    net_flux.calc_electric_field(nhat);
    return net_flux;
};

template <int dim>
DUAL RMHD<dim>::conserved_t RMHD<dim>::calc_hllc_flux(
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL = prims2cons(prL);
    const auto uR = prims2cons(prR);
    const auto fL = prims2flux(prL, nhat);
    const auto fR = prims2flux(prR, nhat);

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

        if (quirk_smoothing) {
            if (quirk_strong_shock(prL.p(), prR.p())) {
                return hll_flux - hll_state * vface;
            }
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);
        // the normal component of the magnetic field is assumed to
        // be continuous across the interface, so bnL = bnR = bn
        const real bn  = hll_state.bcomponent(nhat);
        const real bp1 = hll_state.bcomponent(np1);
        const real bp2 = hll_state.bcomponent(np2);

        // check if normal magnetic field is approaching zero
        const auto bfn = goes_to_zero(bn);

        const real uhlld = hll_state.dens();
        const real uhllm = hll_state.momentum(nhat);
        const real uhlle = hll_state.nrg() + uhlld;

        const real fhlld = hll_flux.dens();
        const real fhllm = hll_flux.momentum(nhat);
        const real fhlle = hll_flux.nrg() + fhlld;
        const real fpb1  = hll_flux.bcomponent(np1);
        const real fpb2  = hll_flux.bcomponent(np2);

        //------Calculate the contact wave velocity and pressure
        const real fdb   = (bp1 * fpb1 + bp2 * fpb2);
        const real bpsq  = bp1 * bp1 + bp2 * bp2;
        const real fbpsq = fpb1 * fpb1 + fpb2 * fpb2;

        const auto [a, b, c] = [&] {
            if (bfn) {
                return std::make_tuple(fhlle, -(fhllm + uhlle), uhllm);
            }

            return std::make_tuple(
                fhlle - fdb,
                -(uhlle + fhllm) + bpsq + fbpsq,
                uhllm - fdb
            );
        }();
        const real quad  = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
        const auto aS    = c / quad;
        const real vp1   = bfn ? 0.0 : (bp1 * aS - fpb1) / bn;   // Eq. (38)
        const real vp2   = bfn ? 0.0 : (bp2 * aS - fpb2) / bn;   // Eq. (38)
        const real invg2 = (1.0 - (aS * aS + vp1 * vp1 + vp2 * vp2));
        const real vsdB  = (aS * bn + vp1 * bp1 + vp2 * bp2);
        const real pS    = -aS * (fhlle - bn * vsdB) + fhllm + bn * bn * invg2;

        const auto on_left = vface < aS;
        const auto u       = on_left ? uL : uR;
        const auto f       = on_left ? fL : fR;
        const auto pr      = on_left ? prL : prR;
        const auto ws      = on_left ? aLm : aRp;

        const real d     = u.dens();
        const real mnorm = u.momentum(nhat);
        const real ump1  = u.momentum(np1);
        const real ump2  = u.momentum(np2);
        const real fmp1  = f.momentum(np1);
        const real fmp2  = f.momentum(np2);
        const real tau   = u.nrg();
        const real et    = tau + d;
        const real cfac  = 1.0 / (ws - aS);

        const real v  = pr.vcomponent(nhat);
        const real vs = cfac * (ws - v);
        const real ds = vs * d;
        const real es = cfac * (ws * et - mnorm + pS * aS - vsdB * bn);
        const real mn = (es + pS) * aS - vsdB * bn;
        const real mp1 =
            bfn ? vs * ump1
                    : cfac * (-bn * (bp1 * invg2 + vsdB * vp1) + ws * ump1 - fmp1);
        const real mp2 =
            bfn ? vs * ump2
                    : cfac * (-bn * (bp2 * invg2 + vsdB * vp2) + ws * ump2 - fmp2);

        // start state
        conserved_t us;
        us.dens()           = ds;
        us.momentum(nhat)   = mn;
        us.momentum(np1)    = mp1;
        us.momentum(np2)    = mp2;
        us.nrg()            = es - ds;
        us.bcomponent(nhat) = bn;
        us.bcomponent(np1)  = bfn ? vs * pr.bcomponent(np1) : bp1;
        us.bcomponent(np2)  = bfn ? vs * pr.bcomponent(np2) : bp2;

        return f + (us - u) * ws - us * vface;
    }();
    // upwind the concentration
    if (net_flux.dens() < 0.0) {
        net_flux.chi() = prR.chi() * net_flux.dens();
    }
    else {
        net_flux.chi() = prL.chi() * net_flux.dens();
    }
    net_flux.calc_electric_field(nhat);
    return net_flux;
};

template <int dim>
DUAL real RMHD<dim>::hlld_vdiff(
    const real p,
    const RMHD<dim>::conserved_t r[2],
    const real lam[2],
    const real bn,
    const luint nhat,
    RMHD<dim>::primitive_t& praL,
    RMHD<dim>::primitive_t& praR,
    RMHD<dim>::primitive_t& prC
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
        const auto rmn  = rS.momentum(nhat);
        const auto rmp1 = rS.momentum(np1);
        const auto rmp2 = rS.momentum(np2);
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
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto uL = prims2cons(prL);
    const auto uR = prims2cons(prR);
    const auto fL = prims2flux(prL, nhat);
    const auto fR = prims2flux(prR, nhat);

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
            if (quirk_strong_shock(prL.p(), prR.p())) {
                return hll_flux - hll_state * vface;
            }
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);
        // the normal component of the magnetic field is assumed to
        // be continuous across the interface, so bnL = bnR = bn
        const real bn = hll_state.bcomponent(nhat);

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
                const real mn_hll  = hll_state.momentum(nhat);
                const real fmn_hll = hll_flux.momentum(nhat);

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

        if (!hlld_success) {
            return hll_flux - hll_state * vface;
        }

        // speed of the contact wave
        const real vnc = prC.vcomponent(nhat);

        // Alfven speeds
        const real laL = prAL.alfven();
        const real laR = prAR.alfven();

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
        ua.momentum(nhat)   = mn;
        ua.momentum(np1)    = mp1;
        ua.momentum(np2)    = mp2;
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
        ut.momentum(nhat)   = mnc;
        ut.momentum(np1)    = mpc1;
        ut.momentum(np2)    = mpc2;
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
    net_flux.calc_electric_field(nhat);
    return net_flux;
};

template <int dim>
void RMHD<dim>::riemann_fluxes()
{
    const auto prim_dat = prims.data();
    simbi::parallel_for(activeP, [prim_dat, this] DEV(const luint idx) {
        primitive_t pL, pLL, pR, pRR;

        // primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        const auto prb = sm_or_identity(prim_dat);

        const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
        const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
        const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

        if constexpr (global::on_gpu) {
            if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
                return;
            }
        }

        const luint ia  = ii + radius;
        const luint ja  = jj + radius;
        const luint ka  = kk + radius;
        const luint tx  = (global::on_sm) ? threadIdx.x : 0;
        const luint ty  = (global::on_sm) ? threadIdx.y : 0;
        const luint tz  = (global::on_sm) ? threadIdx.z : 0;
        const luint txa = (global::on_sm) ? tx + radius : ia;
        const luint tya = (global::on_sm) ? ty + radius : ja;
        const luint tza = (global::on_sm) ? tz + radius : ka;

        if constexpr (global::on_sm) {
            load_shared_buffer<dim>(
                fullP,
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

        const auto cell   = this->cell_factors(ii, jj, kk);
        const real vfs[2] = {cell.v1fL(), cell.v1fR()};
        // Calc Rimeann Flux at all interfaces
        for (int q = 0; q < 2; q++) {
            const auto xf = idx3(ii + q, jj, kk, nxv, yag, zag);
            const auto yf = idx3(ii, jj + q, kk, xag, nyv, zag);
            const auto zf = idx3(ii, jj, kk + q, xag, yag, nzv);
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
            fri[xf] = (this->*riemann_solve)(pL, pR, 1, vfs[q]);

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
            gri[yf] = (this->*riemann_solve)(pL, pR, 2, 0);

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
            hri[zf] = (this->*riemann_solve)(pL, pR, 3, 0);
        }
    });
}

//===================================================================================================================
//                                           SOURCE TERMS
//===================================================================================================================
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
DUAL RMHD<dim>::conserved_t RMHD<dim>::gravity_sources(
    const RMHD<dim>::primitive_t& prims,
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
void RMHD<dim>::advance()
{
    const auto prim_dat = prims.data();
    // copy the bfield vectors as to not modify the original
    const auto b1_data = bstag1;
    const auto b2_data = bstag2;
    const auto b3_data = bstag3;
    simbi::parallel_for(
        activeP,
        [prim_dat, b1_data, b2_data, b3_data, this] DEV(const luint idx) {
            // e1, e2, e3 values at cell edges
            real e1[4], e2[4], e3[4];

            // primitive buffer that returns dynamic shared array
            // if working with shared memory on GPU, identity otherwise
            const auto prb = sm_or_identity(prim_dat);

            const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
            const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
            const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

            if constexpr (global::on_gpu) {
                if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
                    return;
                }
            }

            const luint ia  = ii + radius;
            const luint ja  = jj + radius;
            const luint ka  = kk + radius;
            const luint tx  = (global::on_sm) ? threadIdx.x : 0;
            const luint ty  = (global::on_sm) ? threadIdx.y : 0;
            const luint tz  = (global::on_sm) ? threadIdx.z : 0;
            const luint txa = (global::on_sm) ? tx + radius : ia;
            const luint tya = (global::on_sm) ? ty + radius : ja;
            const luint tza = (global::on_sm) ? tz + radius : ka;
            const luint aid = idx3(ia, ja, ka, nx, ny, nz);
            const luint tid = idx3(txa, tya, tza, sx, sy, sz);

            if constexpr (global::on_sm) {
                load_shared_buffer<dim>(
                    activeP,
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

            // mesh factors
            const auto cell = this->cell_factors(ii, jj, kk);

            const auto xlf = idx3(ii + 0, jj, kk, nxv, yag, zag);
            const auto xrf = idx3(ii + 1, jj, kk, nxv, yag, zag);
            const auto ylf = idx3(ii, jj + 0, kk, xag, nyv, zag);
            const auto yrf = idx3(ii, jj + 1, kk, xag, nyv, zag);
            const auto zlf = idx3(ii, jj, kk + 0, xag, yag, nzv);
            const auto zrf = idx3(ii, jj, kk + 1, xag, yag, nzv);

            // compute edge emfs in clockwise direction wrt cell plane
            detail::for_sequence(
                detail::make_index_sequence<4>(),
                [&](auto qidx) {
                    constexpr auto q      = static_cast<luint>(qidx);
                    constexpr auto corner = static_cast<Corner>(q);

                    // calc directional indices for i-j plane
                    auto north = corner == Corner::NE || corner == Corner::NW;
                    auto east  = corner == Corner::NE || corner == Corner::SE;
                    auto south = !north;
                    auto west  = !east;
                    auto qn    = my_min<lint>(jj + north, yag - 1);
                    auto qs    = my_max<lint>(jj - south, 0);
                    auto qe    = my_min<lint>(ii + east, xag - 1);
                    auto qw    = my_max<lint>(ii - west, 0);

                    // printf(
                    //     "I-J --- qn: %lld, qs: %lld, qe: %lld, qw: %lld\n",
                    //     qn,
                    //     qs,
                    //     qe,
                    //     qw
                    // );

                    auto nidx = idx3(ii + east, qn, kk, nxv, yag, zag);
                    auto sidx = idx3(ii + east, qs, kk, nxv, yag, zag);
                    auto eidx = idx3(qe, jj + north, kk, xag, nyv, zag);
                    auto widx = idx3(qw, jj + north, kk, xag, nyv, zag);

                    e3[q] = calc_edge_emf<Plane::IJ, corner>(
                        gri[widx],
                        gri[eidx],
                        fri[sidx],
                        fri[nidx],
                        prb,
                        ii,
                        jj,
                        kk,
                        txa,
                        tya,
                        tza,
                        3
                    );

                    // calc directional indices for i-k plane
                    qn   = my_min<lint>(kk + north, zag - 1);
                    qs   = my_max<lint>(kk - south, 0);
                    qe   = my_min<lint>(ii + east, xag - 1);
                    qw   = my_max<lint>(ii - west, 0);
                    nidx = idx3(ii + east, jj, qn, nxv, yag, zag);
                    sidx = idx3(ii + east, jj, qs, nxv, yag, zag);
                    eidx = idx3(qe, jj, kk + north, xag, yag, nzv);
                    widx = idx3(qw, jj, kk + north, xag, yag, nzv);

                    // printf(
                    //     "I-K --- qn: %lld, qs: %lld, qe: %lld, qw: %lld\n",
                    //     qn,
                    //     qs,
                    //     qe,
                    //     qw
                    // );
                    e2[q] = calc_edge_emf<Plane::IK, corner>(
                        hri[widx],
                        hri[eidx],
                        fri[sidx],
                        fri[nidx],
                        prb,
                        ii,
                        jj,
                        kk,
                        txa,
                        tya,
                        tza,
                        2
                    );

                    // calc directional indices for j-k plane
                    qn   = my_min<lint>(kk + north, zag - 1);
                    qs   = my_max<lint>(kk - south, 0);
                    qe   = my_min<lint>(jj + east, yag - 1);
                    qw   = my_max<lint>(jj - west, 0);
                    nidx = idx3(ii, jj + east, qn, xag, nyv, zag);
                    sidx = idx3(ii, jj + east, qs, xag, nyv, zag);
                    eidx = idx3(ii, qe, kk + north, xag, yag, nzv);
                    widx = idx3(ii, qe, kk + north, xag, yag, nzv);

                    // printf(
                    //     "J-K --- qn: %lld, qs: %lld, qe: %lld, qw: %lld\n",
                    //     qn,
                    //     qs,
                    //     qe,
                    //     qw
                    // );
                    e1[q] = calc_edge_emf<Plane::JK, corner>(
                        hri[widx],
                        hri[eidx],
                        gri[sidx],
                        gri[nidx],
                        prb,
                        ii,
                        jj,
                        kk,
                        txa,
                        tya,
                        tza,
                        1
                    );
                }
            );
            // std::cin.get();

            auto& b1L = bstag1[xlf];
            auto& b1R = bstag1[xrf];
            auto& b2L = bstag2[ylf];
            auto& b2R = bstag2[yrf];
            auto& b3L = bstag3[zlf];
            auto& b3R = bstag3[zrf];
            auto& b1c = cons[aid].b1();
            auto& b2c = cons[aid].b2();
            auto& b3c = cons[aid].b3();

            b1L = b1_data[xlf] - dt * step * curl_e(1, e2, e3, cell, 0);
            b1R = b1_data[xrf] - dt * step * curl_e(1, e2, e3, cell, 1);

            b2L = b2_data[ylf] - dt * step * curl_e(2, e3, e1, cell, 0);
            b2R = b2_data[yrf] - dt * step * curl_e(2, e3, e1, cell, 1);

            b3L = b3_data[zlf] - dt * step * curl_e(3, e1, e2, cell, 0);
            b3R = b3_data[zrf] - dt * step * curl_e(3, e1, e2, cell, 1);

            if constexpr (global::debug_mode) {
                const auto divb = (b1R - b1L) * invdx1 + (b2R - b2L) * invdx2 +
                                  (b3R - b3L) * invdx3;

                if (!goes_to_zero(divb)) {
                    if (kk == 0 && jj == 2 && ii == 4) {
                        printf("========================================\n");
                        printf("DIV.B: %.2e\n", divb);
                        printf("========================================\n");
                        printf("Divergence of B is not zero!\n");
                    }
                }
            }
            b1c = static_cast<real>(0.5) * (b1R + b1L);
            b2c = static_cast<real>(0.5) * (b2R + b2L);
            b3c = static_cast<real>(0.5) * (b3R + b3L);

            // TODO: implement functional source and gravity
            const auto source_terms = hydro_sources(cell);
            // Gravity
            const auto gravity = gravity_sources(prb[tid], cell);

            // geometric source terms
            const auto geom_source = cell.geom_sources(prb[tid]);

            cons[aid] -=
                ((fri[xrf] * cell.a1R() - fri[xlf] * cell.a1L()) * cell.idV1() +
                 (gri[yrf] * cell.a2R() - gri[ylf] * cell.a2L()) * cell.idV2() +
                 (hri[zrf] * cell.a3R() - hri[zlf] * cell.a3L()) * cell.idV3() -
                 source_terms - gravity - geom_source) *
                dt * step;
        }
    );
}

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void RMHD<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot,
    const std::vector<std::optional<RMHD<dim>::function_t>>& bsources,
    const std::vector<std::optional<RMHD<dim>::function_t>>& hsources,
    const std::vector<std::optional<RMHD<dim>::function_t>>& gsources
)
{
    anyDisplayProps();
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

    // check if ~all~ boundary sources have been set.
    // if the user forgot one, the code will run with
    // and outflow outer boundary condition
    this->all_outer_bounds =
        std::all_of(this->bsources.begin(), this->bsources.end(), [](auto q) {
            return q != nullptr;
        });

    this->null_gravity =
        std::all_of(this->gsources.begin(), this->gsources.end(), [](auto q) {
            return q == nullptr;
        });

    this->null_sources =
        std::all_of(this->hsources.begin(), this->hsources.end(), [](auto q) {
            return q == nullptr;
        });

    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (x2max == 0.5 * M_PI) {
        this->half_sphere = true;
    }

    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
    }

    // allocate space for face-centered magnetic fields
    bstag1.resize(nxv * yag * zag);
    bstag2.resize(xag * nyv * zag);
    bstag3.resize(xag * yag * nzv);
    // allocate space for Riemann fluxes
    fri.resize(nxv * yag * zag);
    gri.resize(xag * nyv * zag);
    hri.resize(xag * yag * nzv);
    // set the staggered magnetic fields to ics
    bstag1 = bfield[0];
    bstag2 = bfield[1];
    bstag3 = bfield[2];

    // allocate space for volume-average quantities
    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    if constexpr (global::BuildPlatform == global::Platform::GPU) {
        dt_min.resize(active_zones);
    }

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        const real d   = state[0][i];
        const real m1  = state[1][i];
        const real m2  = state[2][i];
        const real m3  = state[3][i];
        const real tau = state[4][i];
        // volume-averaged magnetic fields pre-baked into
        // initial conserved from python initialization
        const real b1   = state[5][i];
        const real b2   = state[6][i];
        const real b3   = state[7][i];
        const real dchi = state[8][i];
        cons[i]         = conserved_t{d, m1, m2, m3, tau, b1, b2, b3, dchi};
    }

    // set up the problem and release old state from memory
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    print_shared_mem();
    set_the_riemann_solver();
    this->set_mesh_funcs();

    config_ghosts(this);
    cons2prim();
    adapt_dt<TIMESTEP_TYPE::MINIMUM>();

    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) {
        write_to_file(*this);
    }

    // Simulate :)
    try {
        simbi::detail::logger::with_logger(*this, tend, [&] {
            riemann_fluxes();
            advance();
            config_ghosts(this);
            cons2prim();
            adapt_dt();

            t += step * dt;
            if (mesh_motion) {
                // update x1 endpoints
                const real vmin =
                    (homolog) ? x1min * hubble_param : hubble_param;
                const real vmax =
                    (homolog) ? x1max * hubble_param : hubble_param;
                x1max += step * dt * vmax;
                x1min += step * dt * vmin;
                hubble_param = adot(t) / a(t);
            }
        });
    }
    catch (const SimulationFailureException& e) {
        std::cerr << std::string(80, '=') << "\n";
        std::cerr << e.what() << '\n';
        std::cerr << std::string(80, '=') << "\n";
        troubled_cells.copyFromGpu();
        cons.copyFromGpu();
        prims.copyFromGpu();
        hasCrashed = true;
        write_to_file(*this);
        emit_troubled_cells();
    }
};
