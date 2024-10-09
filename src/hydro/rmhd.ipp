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

// Helpers
template <int dim>
DUAL constexpr real RMHD<dim>::get_x1face(const lint ii, const int side) const
{
    switch (x1_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x1l = my_max<real>(x1min + (ii - 0.5) * dx1, x1min);
                if (side == 0) {
                    return x1l;
                }
                return my_min<real>(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
            }
        default:
            {
                const real x1l = my_max<real>(
                    x1min * std::pow(10.0, (ii - 0.5) * dlogx1),
                    x1min
                );
                if (side == 0) {
                    return x1l;
                }
                return my_min<real>(
                    x1l * std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0)),
                    x1max
                );
            }
    }
}

template <int dim>
DUAL constexpr real RMHD<dim>::get_x2face(const lint ii, const int side) const
{
    switch (x2_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x2l = my_max<real>(x2min + (ii - 0.5) * dx2, x2min);
                if (side == 0) {
                    return x2l;
                }
                return my_min<real>(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
            }
        default:
            {
                const real x2l = my_max<real>(
                    x2min * std::pow(10.0, (ii - 0.5) * dlogx2),
                    x2min
                );
                if (side == 0) {
                    return x2l;
                }
                return my_min<real>(
                    x2l * std::pow(10.0, dlogx2 * (ii == 0 ? 0.5 : 1.0)),
                    x2max
                );
            }
    }
}

template <int dim>
DUAL constexpr real RMHD<dim>::get_x3face(const lint ii, const int side) const
{
    switch (x3_cell_spacing) {
        case simbi::Cellspacing::LINSPACE:
            {
                const real x3l = my_max<real>(x3min + (ii - 0.5) * dx3, x3min);
                if (side == 0) {
                    return x3l;
                }
                return my_min<real>(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
            }
        default:
            {
                const real x3l = my_max<real>(
                    x3min * std::pow(10.0, (ii - 0.5) * dlogx3),
                    x3min
                );
                if (side == 0) {
                    return x3l;
                }
                return my_min<real>(
                    x3l * std::pow(10.0, dlogx3 * (ii == 0 ? 0.5 : 1.0)),
                    x3max
                );
            }
    }
}

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
    const ndarray<real>& bstagp1,
    const ndarray<real>& bstagp2,
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
                const real de_dq2L = ew - esw + ee - ese;
                const real de_dq2R = enw - ew + ene - ee;
                const real de_dq1L = es - esw + en - enw;
                const real de_dq1R = ese - es + ene - en;
                return (
                    eavg + one_eighth * (de_dq2L - de_dq2R + de_dq1L - de_dq1R)
                );
            }
        case CTTYPE::CONTACT:   // Eq. (51)
            {
                const real de_dq2L = [&]() {
                    if (fs.den > 0) {
                        return static_cast<real>(2.0) * (ew - esw);
                    }
                    else if (fs.den < 0) {
                        return static_cast<real>(2.0) * (ee - ese);
                    }
                    return ew - esw + ee - ese;
                }();

                const real de_dq2R = [&]() {
                    if (fn.den > 0) {
                        return static_cast<real>(2.0) * (enw - ew);
                    }
                    else if (fn.den < 0) {
                        return static_cast<real>(2.0) * (ene - ee);
                    }
                    return enw - ew + ene - ee;
                }();

                const real de_dq1L = [&]() {
                    if (fw.den > 0) {
                        return static_cast<real>(2.0) * (es - esw);
                    }
                    else if (fw.den < 0) {
                        return static_cast<real>(2.0) * (en - enw);
                    }
                    return es - esw + en - enw;
                }();

                const real de_dq1R = [&]() {
                    if (fe.den > 0) {
                        return static_cast<real>(2.0) * (ese - es);
                    }
                    else if (fe.den < 0) {
                        return static_cast<real>(2.0) * (ene - en);
                    }
                    return ese - es + ene - en;
                }();

                return (
                    eavg + one_eighth * (de_dq2L - de_dq2R + de_dq1L - de_dq1R)
                );
            }
        default:   // ALPHA, Eq. (49)
            {
                constexpr real alpha = 0.1;
                // compute permutation indices
                const auto np1 = (P == Plane::JK) ? 2 : 1;
                const auto np2 = (P == Plane::IJ) ? 2 : 3;

                // face-center magnetic field indices
                const auto [nx1, ny1, nz1] = [&] {
                    if constexpr (P == Plane::JK) {
                        return std::make_tuple(xag + 2, nyv, zag + 2);   // B2
                    }
                    return std::make_tuple(nxv, yag + 2, zag + 2);   // B1
                }();
                const auto sidx = cidx<P, C, Dir::S>(ii, jj, kk, nx1, ny1, nz1);
                const auto nidx = cidx<P, C, Dir::N>(ii, jj, kk, nx1, ny1, nz1);

                const auto [nx2, ny2, nz2] = [&] {
                    if constexpr (P == Plane::IJ) {   // B2
                        return std::make_tuple(xag + 2, nyv, zag + 2);
                    }
                    return std::make_tuple(xag + 2, yag + 2, nzv);   // B3
                }();
                const auto eidx = cidx<P, C, Dir::E>(ii, jj, kk, nx2, ny2, nz2);
                const auto widx = cidx<P, C, Dir::W>(ii, jj, kk, nx2, ny2, nz2);

                // perpendicular mean field 1
                const auto bp1sw = swp.bcomponent(np1);
                const auto bp1nw = nwp.bcomponent(np1);
                const auto bp1se = sep.bcomponent(np1);
                const auto bp1ne = nep.bcomponent(np1);

                // perpendicular mean field 2
                const auto bp2sw = swp.bcomponent(np2);
                const auto bp2nw = nwp.bcomponent(np2);
                const auto bp2se = sep.bcomponent(np2);
                const auto bp2ne = nep.bcomponent(np2);

                // perpendicular staggered field 1
                const auto bp1s = bstagp1[sidx];
                const auto bp1n = bstagp1[nidx];
                // perpendicular staggered field 2
                const auto bp2e = bstagp2[eidx];
                const auto bp2w = bstagp2[widx];

                const real de_dq2L = (ew - esw + ee - ese) +
                                     alpha * (bp2e - bp2se - bp2w + bp2sw);
                const real de_dq2R = (enw - ew + ene - ee) +
                                     alpha * (bp2ne - bp2e - bp2nw + bp2w);
                const real de_dq1L = (es - esw + en - enw) +
                                     alpha * (bp1s - bp1sw - bp1n + bp1nw);
                const real de_dq1R = (ese - es + ene - en) +
                                     alpha * (bp1se - bp1s - bp1ne + bp1n);

                return (
                    eavg + one_eighth * (de_dq2L - de_dq2R + de_dq1L - de_dq1R)
                );
            }
    }
};

template <int dim>
DUAL real RMHD<dim>::curl_e(
    const luint nhat,
    const real ejl,
    const real ejr,
    const real ekl,
    const real ekr
) const
{
    switch (geometry) {
        case Geometry::CARTESIAN:
            {
                if (nhat == 1) {
                    return invdx2 * (ekr - ekl) - invdx3 * (ejr - ejl);
                }
                else if (nhat == 2) {
                    return invdx3 * (ejr - ejl) - invdx1 * (ekr - ekl);
                }
                else {
                    return invdx1 * (ekr - ekl) - invdx2 * (ejr - ejl);
                }
            }
        default:
            return 0.0;
            // case Geometry::SPHERICAL:
            //     {
            //         curl_e = [] {
            //             if constexpr (dim == 1) {
            //                 if (nhat == 2) {
            //                     return (1.0 / r / dr) * (rl * e3l - rr *
            //                     e3r);
            //                 }
            //                 else {
            //                     return (1.0 / r / dr) * (rr * e2r - rl *
            //                     e2l);
            //                 }
            //             }
            //             else if constexpr (dim == 2) {
            //                 if (nhat == 1) {
            //                     (1.0 / (r * dth * std::sin(th))) *
            //                         (e3r * std::sin(thr) - e3l *
            //                         std::sin(thl));
            //                 }
            //                 else if (nhat == 2) {
            //                     return (1.0 / r / dr) * (rl * e3l - rr *
            //                     e3r);
            //                 }
            //                 else {
            //                     return (1.0 / r) * ((rr * e2r - rl * e2l) /
            //                     dr +
            //                                         (e1l - e1r) / dth);
            //                 }
            //             }
            //             else {
            //                 if (nhat == 1) {
            //                     (1.0 / (r * std::sin(th))) *
            //                         ((e3r * std::sin(thr) - e3l *
            //                         std::sin(thl)) *
            //                              invdx2 +
            //                          (e2l - e2r) * invdx3);
            //                 }
            //                 else if (nhat == 2) {
            //                     return (1.0 / r) *
            //                            (1.0 / std::sin(th) * invdx3 * (e1r -
            //                            e1l) +
            //                             (rl * e3l - rr * e3r) / dr);
            //                 }
            //                 else {
            //                     return (1.0 / r) * ((rr * e2r - rl * e2l) /
            //                     dr +
            //                                         (e1l - e1r) / dth);
            //                 }
            //             }
            //         }();
            //         return b -= dt * step * curl_e;
            //     }
            // default:
            //     {
            //         curl_e = [] {
            //             if constexpr (dim == 1) {
            //                 if (nhat == 2) {
            //                     return (e3l - e3r) / dr;
            //                 }
            //                 else {
            //                     return (1.0 / r) * (rr * e2r - rl * e2l) /
            //                     dr;
            //                 }
            //             }
            //             else if constexpr (dim == 2) {
            //                 switch (geometry) {
            //                     case Geometry::AXIS_CYLINDRICAL:
            //                         {
            //                             if (nhat == 1) {
            //                                 (e2l - e2r) * invdx3;
            //                             }
            //                             else if (nhat == 2) {
            //                                 return (e1r - e1l) * invdx3 +
            //                                        (e3l - e3r) * invdx1;
            //                             }
            //                             else {
            //                                 return (1.0 / r / dr) *
            //                                        (rr * e2r - rl e2l);
            //                             }
            //                         }
            //                     default:
            //                         if (nhat == 1) {
            //                             (e3r - e3l) * invdx2 / r;
            //                         }
            //                         else if (nhat == 2) {
            //                             return (e3l - e3r) * invdx1;
            //                         }
            //                         else {
            //                             return (1.0 / r) *
            //                                    ((rr * e2r - rl e2l) / dr +
            //                                     (e1l - e1r) * invdx2);
            //                         }
            //                 }
            //             }
            //             else {
            //                 if (nhat == 1) {
            //                     (e3r - e3l) * invdx2 / r + (e2l - e2r) *
            //                     invdx3;
            //                 }
            //                 else if (nhat == 2) {
            //                     return (e1r - e1l) * invdx3 + (e3l - e3r) *
            //                     invdx1;
            //                 }
            //                 else {
            //                     return (1.0 / r) * ((rr * e2r - rl e2l) / dr
            //                     +
            //                                         (e1l - e1r) * invdx2);
            //                 }
            //             }
            //         }();
            //         return b -= dt * step * curl_e;
            //     }
    }
}

template <int dim>
DUAL constexpr real RMHD<dim>::get_x1_differential(const lint ii) const
{
    const real x1l   = get_x1face(ii, 0);
    const real x1r   = get_x1face(ii, 1);
    const real xmean = get_cell_centroid(x1r, x1l, geometry);
    switch (geometry) {
        case Geometry::SPHERICAL:
            return xmean * xmean * (x1r - x1l);
        default:
            return xmean * (x1r - x1l);
    }
}

template <int dim>
DUAL constexpr real RMHD<dim>::get_x2_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return 2.0;
            default:
                return (2.0 * M_PI);
        }
    }
    else {
        switch (geometry) {
            case Geometry::SPHERICAL:
                {
                    const real x2l  = get_x2face(ii, 0);
                    const real x2r  = get_x2face(ii, 1);
                    const real dcos = std::cos(x2l) - std::cos(x2r);
                    return dcos;
                }
            default:
                {
                    return dx2;
                }
        }
    }
}

template <int dim>
DUAL constexpr real RMHD<dim>::get_x3_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return (2.0 * M_PI);
            default:
                return 1.0;
        }
    }
    else if constexpr (dim == 2) {
        switch (geometry) {
            case Geometry::PLANAR_CYLINDRICAL:
                return 1.0;
            default:
                return (2.0 * M_PI);
        }
    }
    else {
        return dx3;
    }
}

template <int dim>
DUAL real
RMHD<dim>::get_cell_volume(const lint ii, const lint jj, const lint kk) const
{
    // the volume in cartesian coordinates is only nominal
    if (geometry == Geometry::CARTESIAN) {
        return 1.0;
    }
    return get_x1_differential(ii) * get_x2_differential(jj) *
           get_x3_differential(kk);
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
            const real x1l    = get_x1face(ireal, 0);
            const real x1r    = get_x1face(ireal, 1);
            const real x2l    = get_x2face(jreal, 0);
            const real x2r    = get_x2face(jreal, 1);
            const real x3l    = get_x3face(kreal, 0);
            const real x3r    = get_x3face(kreal, 1);
            const real x1mean = calc_any_mean(x1l, x1r, x1_cell_spacing);
            const real x2mean =
                yag == 1 ? 0.0 : calc_any_mean(x2l, x2r, x2_cell_spacing);
            const real x3mean =
                zag == 1 ? 0.0 : calc_any_mean(x3l, x3r, x3_cell_spacing);
            const real m1  = cons[gid].momentum(1);
            const real m2  = cons[gid].momentum(2);
            const real m3  = cons[gid].momentum(3);
            const real et  = (cons[gid].den + cons[gid].nrg);
            const real b1  = cons[gid].bcomponent(1);
            const real b2  = cons[gid].bcomponent(2);
            const real b3  = cons[gid].bcomponent(3);
            const real m   = std::sqrt(m1 * m1 + m2 * m2 + m3 * m3);
            const real vsq = (m * m) / (et * et);
            const real bsq = (b1 * b1 + b2 * b2 + b3 * b3);
            const real w   = 1.0 / std::sqrt(1.0 - vsq);
            fprintf(
                stderr,
                "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                "%.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, x2coord: "
                "%.2e, x3coord: %.2e, iter: %" PRIu64 "\n",
                cons[gid].den / w,
                prims[gid].p,
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
void RMHD<dim>::cons2prim(const ExecutionPolicy<>& p)
{
    // const auto gr = gamma / (gamma - 1.0);
    simbi::parallel_for(p, total_zones, [this] DEV(luint gid) {
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
                const real dV    = get_cell_volume(ireal, jreal, kreal);
                invdV            = 1.0 / dV;
            }
            const real d    = cons[gid].den * invdV;
            const real m1   = cons[gid].momentum(1) * invdV;
            const real m2   = cons[gid].momentum(2) * invdV;
            const real m3   = cons[gid].momentum(3) * invdV;
            const real tau  = cons[gid].nrg * invdV;
            const real b1   = cons[gid].bcomponent(1) * invdV;
            const real b2   = cons[gid].bcomponent(2) * invdV;
            const real b3   = cons[gid].bcomponent(3) * invdV;
            const real dchi = cons[gid].chi * invdV;

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
            const real bee1   = b1 * isqrtd;
            const real bee2   = b2 * isqrtd;
            const real bee3   = b3 * isqrtd;
            const real beesq  = bee1 * bee1 + bee2 * bee2 + bee3 * bee3;
            const real rdb    = r1 * bee1 + r2 * bee2 + r3 * bee3;
            const real rdbsq  = rdb * rdb;
            // r-parallel
            const real rp1 = (rdb / beesq) * bee1;
            const real rp2 = (rdb / beesq) * bee2;
            const real rp3 = (rdb / beesq) * bee3;
            // r-perpendicular
            const real rperp1 = r1 - rp1;
            const real rperp2 = r2 - rp2;
            const real rperp3 = r3 - rp3;

            const real rparr = std::sqrt(rp1 * rp1 + rp2 * rp2 + rp3 * rp3);
            const real rperp =
                std::sqrt(rperp1 * rperp1 + rperp2 * rperp2 + rperp3 * rperp3);

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
            f1   = kkc_fmu44(y1, rmag, rparr, rperp, beesq, rdbsq, q, d, gamma);
            f2   = kkc_fmu44(y2, rmag, rparr, rperp, beesq, rdbsq, q, d, gamma);
            iter = 0.0;
            do {
                yy = (y1 * f2 - y2 * f1) / (f2 - f1);
                f  = kkc_fmu44(
                    yy,
                    rmag,
                    rparr,
                    rperp,
                    beesq,
                    rdbsq,
                    q,
                    d,
                    gamma
                );
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
                q - 0.5 * (beesq + yy * yy * x * x * beesq * rperp * rperp);

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
            real v1 = yy * x * (r1 + bee1 * rdb * yy);
            real v2 = yy * x * (r2 + bee2 * rdb * yy);
            real v3 = yy * x * (r3 + bee3 * rdb * yy);

            if constexpr (global::VelocityType ==
                          global::Velocity::FourVelocity) {
                v1 *= w;
                v2 *= w;
                v3 *= w;
            }
            prims[gid] =
                primitive_t{d / w, v1, v2, v3, pg, b1, b2, b3, dchi / d};

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
    const real d    = cons.den;
    const real m1   = cons.momentum(1);
    const real m2   = cons.momentum(2);
    const real m3   = cons.momentum(3);
    const real tau  = cons.nrg;
    const real b1   = cons.bcomponent(1);
    const real b2   = cons.bcomponent(2);
    const real b3   = cons.bcomponent(3);
    const real dchi = cons.chi;

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
    const real bee1   = b1 * isqrtd;
    const real bee2   = b2 * isqrtd;
    const real bee3   = b3 * isqrtd;
    const real beesq  = bee1 * bee1 + bee2 * bee2 + bee3 * bee3;
    const real rdb    = r1 * bee1 + r2 * bee2 + r3 * bee3;
    const real rdbsq  = rdb * rdb;
    // r-parallel
    const real rp1 = (rdb / beesq) * bee1;
    const real rp2 = (rdb / beesq) * bee2;
    const real rp3 = (rdb / beesq) * bee3;
    // r-perpendicular
    const real rperp1 = r1 - rp1;
    const real rperp2 = r2 - rp2;
    const real rperp3 = r3 - rp3;

    const real rparr = std::sqrt(rp1 * rp1 + rp2 * rp2 + rp3 * rp3);
    const real rperp =
        std::sqrt(rperp1 * rperp1 + rperp2 * rperp2 + rperp3 * rperp3);

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
    f1   = kkc_fmu44(y1, rmag, rparr, rperp, beesq, rdbsq, q, d, gamma);
    f2   = kkc_fmu44(y2, rmag, rparr, rperp, beesq, rdbsq, q, d, gamma);
    iter = 0.0;
    do {
        yy = (y1 * f2 - y2 * f1) / (f2 - f1);
        f  = kkc_fmu44(yy, rmag, rparr, rperp, beesq, rdbsq, q, d, gamma);
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
    const real qbar =
        q - 0.5 * (beesq + yy * yy * x * x * beesq * rperp * rperp);

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
    real v1 = yy * x * (r1 + bee1 * rdb * yy);
    real v2 = yy * x * (r2 + bee2 * rdb * yy);
    real v3 = yy * x * (r3 + bee3 * rdb * yy);

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
    real speeds[],
    real& cs2
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
    const real rho   = prims.rho;
    const real h     = prims.gas_enthalpy(gamma);
    cs2              = (gamma * prims.p / (rho * h));
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
    real cs2L, cs2R;
    real speeds[4];

    // left side
    calc_max_wave_speeds(primsL, nhat, speeds, cs2L);
    const real lpL = speeds[3];
    const real lmL = speeds[0];

    // right_side
    calc_max_wave_speeds(primsR, nhat, speeds, cs2R);
    const real lpR = speeds[3];
    const real lmR = speeds[0];

    return {
      my_min(lmL, lmR),
      my_max(lpL, lpR),
      std::sqrt(cs2L),
      std::sqrt(cs2R)
    };
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template <int dim>
DUAL RMHD<dim>::conserved_t
RMHD<dim>::prims2cons(const RMHD<dim>::primitive_t& prims) const
{
    const real rho   = prims.rho;
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real pg    = prims.p;
    const real b1    = prims.bcomponent(1);
    const real b2    = prims.bcomponent(2);
    const real b3    = prims.bcomponent(3);
    const real lf    = prims.lorentz_factor();
    const real h     = prims.gas_enthalpy(gamma);
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
      d * prims.chi
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
    // singleton instance of thread pool. lazy-evaluated
    static auto& thread_pool =
        simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(total_zones, [&](luint gid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt, cs;
        real speeds[4];
        const luint kk    = axid<dim, BlkAx::K>(gid, nx, ny);
        const luint jj    = axid<dim, BlkAx::J>(gid, nx, ny, kk);
        const luint ii    = axid<dim, BlkAx::I>(gid, nx, ny, kk);
        const luint ireal = get_real_idx(ii, radius, xag);
        // Left/Right wave speeds
        if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
            calc_max_wave_speeds(prims[gid], 1, speeds, cs);
            v1p = std::abs(speeds[3]);
            v1m = std::abs(speeds[0]);
            calc_max_wave_speeds(prims[gid], 2, speeds, cs);
            v2p = std::abs(speeds[3]);
            v2m = std::abs(speeds[0]);
            calc_max_wave_speeds(prims[gid], 3, speeds, cs);
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

        const real x1l = get_x1face(ireal, 0);
        const real x1r = get_x1face(ireal, 1);
        const real dx1 = x1r - x1l;
        switch (geometry) {
            case simbi::Geometry::CARTESIAN:
                cfl_dt = std ::min(
                    {dx1 / (std::max(v1p, v1m)),
                     dx2 / (std::max(v2p, v2m)),
                     dx3 / (std::max(v3p, v3m))}
                );

                break;
            case simbi::Geometry::SPHERICAL:
                {
                    const real x2l = get_x2face(jj, 0);
                    const real x2r = get_x2face(jj, 1);
                    const real rmean =
                        get_cell_centroid(x1r, x1l, simbi::Geometry::SPHERICAL);
                    const real th    = 0.5 * (x2r + x2l);
                    const real rproj = rmean * std::sin(th);
                    cfl_dt           = std::min(
                        {dx1 / (std::max(v1p, v1m)),
                                   rmean * dx2 / (std::max(v2p, v2m)),
                                   rproj * dx3 / (std::max(v3p, v3m))}
                    );
                    break;
                }
            default:
                {
                    const real rmean = get_cell_centroid(
                        x1r,
                        x1l,
                        simbi::Geometry::CYLINDRICAL
                    );
                    cfl_dt = std::min(
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
};

template <int dim>
template <TIMESTEP_TYPE dt_type>
void RMHD<dim>::adapt_dt(const ExecutionPolicy<>& p)
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
    gpu::api::deviceSynch();
#endif
}

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template <int dim>
DUAL RMHD<dim>::conserved_t
RMHD<dim>::prims2flux(const RMHD<dim>::primitive_t& prims, const luint nhat)
    const
{
    const real rho   = prims.rho;
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real b1    = prims.bcomponent(1);
    const real b2    = prims.bcomponent(2);
    const real b3    = prims.bcomponent(3);
    const real h     = prims.gas_enthalpy(gamma);
    const real lf    = prims.lorentz_factor();
    const real invlf = 1.0 / lf;
    const real vdotb = prims.vdotb();
    const real bsq   = prims.bsquared();
    const real p     = prims.total_pressure();
    const real chi   = prims.chi;
    const real vn    = prims.vcomponent(nhat);
    const real bn    = prims.bcomponent(nhat);
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
      m1 * vn + kronecker(nhat, 1) * p - bn * bmu.one * invlf,
      m2 * vn + kronecker(nhat, 2) * p - bn * bmu.two * invlf,
      m3 * vn + kronecker(nhat, 3) * p - bn * bmu.three * invlf,
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
    const real aL  = lambda.afL;
    const real aR  = lambda.afR;
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
    if (net_flux.den < 0.0) {
        net_flux.chi = prR.chi * net_flux.den;
    }
    else {
        net_flux.chi = prL.chi * net_flux.den;
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
    const real aL     = lambda.afL;
    const real aR     = lambda.afR;
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
            if (quirk_strong_shock(prL.p, prR.p)) {
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

        const real uhlld = hll_state.den;
        const real uhllm = hll_state.momentum(nhat);
        const real uhlle = hll_state.nrg + uhlld;

        const real fhlld = hll_flux.den;
        const real fhllm = hll_flux.momentum(nhat);
        const real fhlle = hll_flux.nrg + fhlld;
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

        const real d     = u.den;
        const real mnorm = u.momentum(nhat);
        const real ump1  = u.momentum(np1);
        const real ump2  = u.momentum(np2);
        const real fmp1  = f.momentum(np1);
        const real fmp2  = f.momentum(np2);
        const real tau   = u.nrg;
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
        us.den              = ds;
        us.momentum(nhat)   = mn;
        us.momentum(np1)    = mp1;
        us.momentum(np2)    = mp2;
        us.nrg              = es - ds;
        us.bcomponent(nhat) = bn;
        us.bcomponent(np1)  = bfn ? vs * pr.bcomponent(np1) : bp1;
        us.bcomponent(np2)  = bfn ? vs * pr.bcomponent(np2) : bp2;

        return f + (us - u) * ws - us * vface;
    }();
    // upwind the concentration
    if (net_flux.den < 0.0) {
        net_flux.chi = prR.chi * net_flux.den;
    }
    else {
        net_flux.chi = prL.chi * net_flux.den;
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
    const real aL     = lambda.afL;
    const real aR     = lambda.afR;
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
            if (quirk_strong_shock(prL.p, prR.p)) {
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
        const real da  = rc.den * fac;
        const real ea  = (rc.total_energy() + p * vna - vdba * bn) * fac;
        const real mn  = (ea + p) * vna - vdba * bn;
        const real mp1 = (ea + p) * vp1 - vdba * bp1;
        const real mp2 = (ea + p) * vp2 - vdba * bp2;

        conserved_t ua;
        ua.den              = da;
        ua.momentum(nhat)   = mn;
        ua.momentum(np1)    = mp1;
        ua.momentum(np2)    = mp2;
        ua.nrg              = ea - da;
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
        ut.den              = dc;
        ut.momentum(nhat)   = mnc;
        ut.momentum(np1)    = mpc1;
        ut.momentum(np2)    = mpc2;
        ut.nrg              = ec - dc;
        ut.bcomponent(nhat) = bnC;
        ut.bcomponent(np1)  = bp1C;
        ut.bcomponent(np2)  = bp2C;

        return fa + (ut - ua) * la - ut * vface;
    }();

    // upwind the concentration
    if (net_flux.den < 0.0) {
        net_flux.chi = prR.chi * net_flux.den;
    }
    else {
        net_flux.chi = prL.chi * net_flux.den;
    }
    net_flux.calc_electric_field(nhat);
    return net_flux;
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void RMHD<dim>::advance(const ExecutionPolicy<>& p)
{
    const luint extent  = p.get_full_extent();
    const auto prim_dat = prims.data();
    simbi::parallel_for(p, extent, [p, prim_dat, this] DEV(const luint idx) {
        // x1,x2,x3 hydro riemann fluxes
        conserved_t f[10];
        conserved_t g[10];
        conserved_t h[10];
        primitive_t pL, pLL, pR, pRR;

        // e1, e2, e3 values at cell edges
        real e1[4], e2[4], e3[4];

        // primitive buffer that returns dynamic shared array
        // if working with shared memory on GPU, identity otherwise
        const auto prim_buff = sm_or_identity(prim_dat);

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
                p,
                prim_buff,
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
        else {
            // cast away unused lambda capture
            (void) p;
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

        const real x1l    = get_x1face(ii, 0);
        const real x1r    = get_x1face(ii, 1);
        const real vfaceL = (homolog) ? x1l * hubble_param : hubble_param;
        const real vfaceR = (homolog) ? x1r * hubble_param : hubble_param;

        const auto xlf = idx3(ii + 0, jj, kk, nxv, yag, zag);
        const auto xrf = idx3(ii + 1, jj, kk, nxv, yag, zag);
        const auto ylf = idx3(ii, jj + 0, kk, xag, nyv, zag);
        const auto yrf = idx3(ii, jj + 1, kk, xag, nyv, zag);
        const auto zlf = idx3(ii, jj, kk + 0, xag, yag, nzv);
        const auto zrf = idx3(ii, jj, kk + 1, xag, yag, nzv);

        // // Calc Rimeann Flux at all interfaces
        for (luint q = 0; q < 10; q++) {
            const auto vdir = 1 * ((luint) (q - 2) < (4 - 2)) -
                              1 * ((luint) (q - 6) < (8 - 6));
            const auto hdir = 1 * ((luint) (q - 4) < (6 - 4)) -
                              1 * ((luint) (q - 8) < (10 - 8));

            // fluxes in i direction
            pL = prim_buff
                [idx3(txa + (q % 2) - 1, tya + vdir, tza + hdir, sx, sy, 0)];
            pR = prim_buff
                [idx3(txa + (q % 2) + 0, tya + vdir, tza + hdir, sx, sy, 0)];

            if (!use_pcm) {
                pLL = prim_buff[idx3(
                    txa + (q % 2) - 2,
                    tya + vdir,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];
                pRR = prim_buff[idx3(
                    txa + (q % 2) + 1,
                    tya + vdir,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];

                pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
            }
            ib_modify<dim>(pR, pL, object_x[(q % 2)], 1);
            f[q] = (this->*riemann_solve)(pL, pR, 1, 0);

            // fluxes in j direction
            pL = prim_buff
                [idx3(txa + vdir, tya + (q % 2) - 1, tza + hdir, sx, sy, 0)];
            pR = prim_buff
                [idx3(txa + vdir, tya + (q % 2) + 0, tza + hdir, sx, sy, 0)];

            if (!use_pcm) {
                pLL = prim_buff[idx3(
                    txa + vdir,
                    tya + (q % 2) - 2,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];
                pRR = prim_buff[idx3(
                    txa + vdir,
                    tya + (q % 2) + 1,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];

                pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
            }
            ib_modify<dim>(pR, pL, object_y[(q % 2)], 2);
            g[q] = (this->*riemann_solve)(pL, pR, 2, 0);

            // fluxes in k direction
            pL = prim_buff
                [idx3(txa + vdir, tya + hdir, tza + (q % 2) - 1, sx, sy, 0)];
            pR = prim_buff
                [idx3(txa + vdir, tya + hdir, tza + (q % 2) + 0, sx, sy, 0)];

            if (!use_pcm) {
                pLL = prim_buff[idx3(
                    txa + vdir,
                    tya + hdir,
                    tza + (q % 2) - 2,
                    sx,
                    sy,
                    0
                )];
                pRR = prim_buff[idx3(
                    txa + vdir,
                    tya + hdir,
                    tza + (q % 2) + 1,
                    sx,
                    sy,
                    0
                )];

                pL = pL + plm_gradient(pL, pLL, pR, plm_theta) * 0.5;
                pR = pR - plm_gradient(pR, pL, pRR, plm_theta) * 0.5;
            }
            ib_modify<dim>(pR, pL, object_z[(q % 2)], 3);
            h[q] = (this->*riemann_solve)(pL, pR, 3, 0);
        }

        // compute edge emfs in clockwise direction wrt cell plane
        detail::for_sequence(detail::make_index_sequence<4>(), [&](auto qidx) {
            constexpr auto q      = static_cast<luint>(qidx);
            constexpr auto corner = static_cast<Corner>(q);
            auto widx             = q == 0 ? 1 : q == 1 ? 0 : q == 2 ? 6 : 7;
            auto eidx             = q == 0 ? 3 : q == 1 ? 2 : q == 2 ? 0 : 1;
            auto sidx             = q == 0 ? 1 : q == 1 ? 7 : q == 2 ? 6 : 0;
            auto nidx             = q == 0 ? 3 : q == 1 ? 1 : q == 2 ? 0 : 2;

            e3[q] = calc_edge_emf<Plane::IJ, corner>(
                g[widx],
                g[eidx],
                f[sidx],
                f[nidx],
                bstag1,
                bstag2,
                prim_buff,
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                3
            );

            widx  = q == 0 ? 1 : q == 1 ? 0 : q == 2 ? 6 : 7;
            eidx  = q == 0 ? 3 : q == 1 ? 2 : q == 2 ? 0 : 1;
            sidx  = q == 0 ? 1 : q == 1 ? 9 : q == 2 ? 8 : 0;
            nidx  = q == 0 ? 5 : q == 1 ? 1 : q == 2 ? 0 : 4;
            e2[q] = calc_edge_emf<Plane::IK, corner>(
                h[widx],
                h[eidx],
                f[sidx],
                f[nidx],
                bstag1,
                bstag3,
                prim_buff,
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                2
            );

            widx  = q == 0 ? 1 : q == 1 ? 0 : q == 2 ? 8 : 9;
            eidx  = q == 0 ? 5 : q == 1 ? 4 : q == 2 ? 0 : 1;
            sidx  = q == 0 ? 1 : q == 1 ? 9 : q == 2 ? 8 : 0;
            nidx  = q == 0 ? 5 : q == 1 ? 1 : q == 2 ? 0 : 4;
            e1[q] = calc_edge_emf<Plane::JK, corner>(
                h[widx],
                h[eidx],
                g[sidx],
                g[nidx],
                bstag2,
                bstag3,
                prim_buff,
                ii,
                jj,
                kk,
                txa,
                tya,
                tza,
                1
            );
        });

        auto& b1L = bstag1[xlf];
        auto& b1R = bstag1[xrf];
        auto& b2L = bstag2[ylf];
        auto& b2R = bstag2[yrf];
        auto& b3L = bstag3[zlf];
        auto& b3R = bstag3[zrf];
        auto& b1c = cons[aid].b1;
        auto& b2c = cons[aid].b2;
        auto& b3c = cons[aid].b3;

        b1L -= dt * step *
               curl_e(1, e2[IK::SW], e2[IK::NW], e3[IJ::SW], e3[IJ::NW]);
        b1R -= dt * step *
               curl_e(1, e2[IK::SE], e2[IK::NE], e3[IJ::SE], e3[IJ::NE]);

        b2L -= dt * step *
               curl_e(2, e1[JK::SW], e1[JK::NW], e3[IJ::SW], e3[IJ::SE]);
        b2R -= dt * step *
               curl_e(2, e1[JK::SE], e1[JK::NE], e3[IJ::NW], e3[IJ::NE]);

        b3L -= dt * step *
               curl_e(3, e1[JK::SW], e1[JK::SE], e2[IK::SW], e2[IK::SE]);
        b3R -= dt * step *
               curl_e(3, e1[JK::NW], e1[JK::NE], e2[IK::NW], e2[IK::NE]);

        const auto divb =
            (b1R - b1L) * invdx1 + (b2R - b2L) * invdx2 + (b3R - b3L) * invdx3;

        if (!goes_to_zero(divb)) {
            printf("Divergence of B is not zero\n");
        }
        b1c = static_cast<real>(0.5) * (b1L + b1R);
        b2c = static_cast<real>(0.5) * (b2L + b2R);
        b3c = static_cast<real>(0.5) * (b3L + b3R);

        // TODO: implement functional source and gravity
        const auto source_terms = conserved_t{};
        // Gravity
        const auto gravity = conserved_t{};

        // Advance depending on geometry
        switch (geometry) {
            case simbi::Geometry::CARTESIAN:
                {
                    cons[aid] -=
                        ((f[RF] - f[LF]) * invdx1 + (g[RF] - g[LF]) * invdx2 +
                         (h[RF] - h[LF]) * invdx3 - source_terms - gravity) *
                        dt * step;
                    break;
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl = x1l + vfaceL * step * dt;
                    const real rr = x1r + vfaceR * step * dt;
                    const real tl = get_x2face(jj, 0);
                    const real tr = get_x2face(jj, 1);
                    const real ql = get_x3face(kk, 0);
                    const real qr = get_x3face(kk, 1);
                    const real rmean =
                        get_cell_centroid(rr, rl, simbi::Geometry::SPHERICAL);
                    const real s1R    = rr * rr;
                    const real s1L    = rl * rl;
                    const real s2R    = std::sin(tr);
                    const real s2L    = std::sin(tl);
                    const real thmean = 0.5 * (tl + tr);
                    const real sint   = std::sin(thmean);
                    const real dV1    = rmean * rmean * (rr - rl);
                    const real dV2    = rmean * sint * (tr - tl);
                    const real dV3    = rmean * sint * (qr - ql);
                    const real cot    = std::cos(thmean) / sint;

                    // Grab central primitives
                    const real rhoc = prim_buff[tid].rho;
                    const real uc   = prim_buff[tid].get_v1();
                    const real vc   = prim_buff[tid].get_v2();
                    const real wc   = prim_buff[tid].get_v3();
                    const real pc   = prim_buff[tid].total_pressure();
                    const auto bmuc = mag_fourvec_t(prim_buff[tid]);

                    const real hc   = prim_buff[tid].gas_enthalpy(gamma);
                    const real gam2 = prim_buff[tid].lorentz_factor_squared();

                    const auto geom_source = conserved_t{
                      0.0,
                      (rhoc * hc * gam2 * (vc * vc + wc * wc) -
                       bmuc.two * bmuc.two - bmuc.three * bmuc.three) /
                              rmean +
                          pc * (s1R - s1L) / dV1,
                      (rhoc * hc * gam2 * (wc * wc * cot - uc * vc) -
                       bmuc.three * bmuc.three * cot + bmuc.one * bmuc.two) /
                              rmean +
                          pc * (s2R - s2L) / dV2,
                      -(rhoc * hc * gam2 * wc * (uc + vc * cot) -
                        bmuc.three * bmuc.one - bmuc.three * bmuc.two * cot) /
                          rmean,
                      0.0,
                      0.0,
                      0.0,
                      0.0
                    };
                    cons[aid] -= ((f[RF] * s1R - f[LF] * s1L) / dV1 +
                                  (g[RF] * s2R - g[LF] * s2L) / dV2 +
                                  (h[RF] - h[LF]) / dV3 - geom_source -
                                  source_terms - gravity) *
                                 dt * step;
                    break;
                }
            default:
                {
                    const real rl = x1l + vfaceL * step * dt;
                    const real rr = x1r + vfaceR * step * dt;
                    const real ql = get_x2face(jj, 0);
                    const real qr = get_x2face(jj, 1);
                    const real zl = get_x3face(kk, 0);
                    const real zr = get_x3face(kk, 1);
                    const real rmean =
                        get_cell_centroid(rr, rl, simbi::Geometry::CYLINDRICAL);
                    const real s1R = rr * (zr - zl) * (qr - ql);
                    const real s1L = rl * (zr - zl) * (qr - ql);
                    const real s2R = (rr - rl) * (zr - zl);
                    const real s2L = (rr - rl) * (zr - zl);
                    const real s3L = rmean * (rr - rl) * (zr - zl);
                    const real s3R = s3L;
                    // const real thmean = 0.5 * (tl + tr);
                    const real dV = rmean * (rr - rl) * (zr - zl) * (qr - ql);
                    const real invdV = 1.0 / dV;

                    // Grab central primitives
                    const real rhoc = prim_buff[tid].rho;
                    const real uc   = prim_buff[tid].get_v1();
                    const real vc   = prim_buff[tid].get_v2();
                    // const real wc   = prim_buff[tid].get_v3();
                    const real pc   = prim_buff[tid].total_pressure();
                    const auto bmuc = mag_fourvec_t(prim_buff[tid]);

                    const real hc   = prim_buff[tid].gas_enthalpy(gamma);
                    const real gam2 = prim_buff[tid].lorentz_factor_squared();

                    const auto geom_source = conserved_t{
                      0.0,
                      (rhoc * hc * gam2 * (vc * vc) - bmuc.two * bmuc.two -
                       bmuc.three * bmuc.three) /
                              rmean +
                          pc * (s1R - s1L) * invdV,
                      -(rhoc * hc * gam2 * uc * vc - bmuc.one * bmuc.two) /
                          rmean,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0
                    };
                    cons[aid] -= ((f[RF] * s1R - f[LF] * s1L) * invdV +
                                  (g[RF] * s2R - g[LF] * s2L) * invdV +
                                  (h[RF] * s3R - h[LF] * s3L) * invdV -
                                  geom_source - source_terms) *
                                 dt * step;
                    break;
                }
        }   // end switch
    });
}

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void RMHD<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot,
    std::optional<RMHD<dim>::function_t> const& d_outer,
    std::optional<RMHD<dim>::function_t> const& s1_outer,
    std::optional<RMHD<dim>::function_t> const& s2_outer,
    std::optional<RMHD<dim>::function_t> const& s3_outer,
    std::optional<RMHD<dim>::function_t> const& e_outer
)
{
    anyDisplayProps();
    // set the primitive functionals
    this->dens_outer = d_outer.value_or(nullptr);
    this->mom1_outer = s1_outer.value_or(nullptr);
    this->mom2_outer = s2_outer.value_or(nullptr);
    this->mom3_outer = s3_outer.value_or(nullptr);
    this->enrg_outer = e_outer.value_or(nullptr);

    if constexpr (dim == 1) {
        this->all_outer_bounds =
            (d_outer.has_value() && s1_outer.has_value() && e_outer.has_value()
            );
    }
    else if constexpr (dim == 2) {
        this->all_outer_bounds =
            (d_outer.has_value() && s1_outer.has_value() &&
             s2_outer.has_value() && e_outer.has_value());
    }
    else {
        this->all_outer_bounds =
            (d_outer.has_value() && s1_outer.has_value() &&
             s2_outer.has_value() && s3_outer.has_value() &&
             e_outer.has_value());
    }

    // Stuff for moving mesh
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->homolog      = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (mesh_motion && all_outer_bounds) {
        if constexpr (dim == 1) {
            outer_zones.resize(spatial_order == "pcm" ? 1 : 2);
            const real dV = get_cell_volume(active_zones - 1);
            outer_zones[0] =
                conserved_t{
                  dens_outer(x1max),
                  mom1_outer(x1max),
                  mom2_outer(x1max),
                  mom3_outer(x1max),
                  enrg_outer(x1max),
                  mag1_outer(x1max),
                  mag2_outer(x1max),
                  mag3_outer(x1max)
                } *
                dV;
            outer_zones.copyToGpu();
        }
        else if constexpr (dim == 2) {
            outer_zones.resize(ny);
            for (luint jj = 0; jj < ny; jj++) {
                const auto jreal = get_real_idx(jj, radius, yag);
                const real dV    = get_cell_volume(nxv - 1, jreal);
                outer_zones[jj] =
                    conserved_t{
                      dens_outer(x1max, x2[jreal]),
                      mom1_outer(x1max, x2[jreal]),
                      mom2_outer(x1max, x2[jreal]),
                      mom3_outer(x1max, x2[jreal]),
                      enrg_outer(x1max, x2[jreal]),
                      mag1_outer(x1max, x2[jreal]),
                      mag2_outer(x1max, x2[jreal]),
                      mag3_outer(x1max, x2[jreal])
                    } *
                    dV;
            }
            outer_zones.copyToGpu();
        }
        else {
            outer_zones.resize(ny * nz);
            for (luint kk = 0; kk < nz; kk++) {
                const auto kreal = get_real_idx(kk, radius, zag);
                for (luint jj = 0; jj < ny; jj++) {
                    const auto jreal = get_real_idx(jj, radius, yag);
                    const real dV    = get_cell_volume(nxv - 1, jreal, kreal);
                    outer_zones[kk * ny + jj] =
                        conserved_t{
                          dens_outer(x1max, x2[jreal], x3[kreal]),
                          mom1_outer(x1max, x2[jreal], x3[kreal]),
                          mom2_outer(x1max, x2[jreal], x3[kreal]),
                          mom3_outer(x1max, x2[jreal], x3[kreal]),
                          enrg_outer(x1max, x2[jreal], x3[kreal]),
                          mag1_outer(x1max, x2[jreal], x3[kreal]),
                          mag2_outer(x1max, x2[jreal], x3[kreal]),
                          mag3_outer(x1max, x2[jreal], x3[kreal])
                        } *
                        dV;
                }
            }
            outer_zones.copyToGpu();
        }
    }

    if (x2max == 0.5 * M_PI) {
        this->half_sphere = true;
    }

    inflow_zones.resize(dim * 2);
    bcs.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs[i]          = boundary_cond_map.at(boundary_conditions[i]);
        this->inflow_zones[i] = conserved_t{
          boundary_sources[i][0],
          boundary_sources[i][1],
          boundary_sources[i][2],
          boundary_sources[i][3],
          boundary_sources[i][4],
          boundary_sources[i][5],
          boundary_sources[i][6],
          boundary_sources[i][7]
        };
    }

    // allocate space for face-centered magnetic fields
    bstag1.resize(nxv * (yag + 2) * (zag + 2));
    bstag2.resize((xag + 2) * nyv * (zag + 2));
    bstag3.resize((xag + 2) * (yag + 2) * nzv);
    bstag1 = bfield[0];
    bstag2 = bfield[1];
    bstag3 = bfield[2];

    // allocate space for volume-average quantities
    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(total_zones);
    edens_guess.resize(total_zones);

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

        // take the positive root of A(27) from
        // Mignone & McKinney (2007):
        // https://articles.adsabs.harvard.edu/pdf/2007MNRAS.378.1118M (we
        // take vsq = 1)
        const real bsq  = (b1 * b1 + b2 * b2 + b3 * b3);
        const real msq  = (m1 * m1 + m2 * m2 + m3 * m3);
        const real et   = tau + d;
        const real a    = 3.0;
        const real b    = -4.0 * (et - bsq);
        const real c    = msq - 2.0 * et * bsq + bsq * bsq;
        const real disc = std::max(b * b - 4.0 * a * c, 0.0);
        const real qq   = (-b + std::sqrt(disc)) / (2.0 * a);
        const real qr   = std::max(qq, d);
        edens_guess[i]  = qr - d;
        cons[i]         = conserved_t{d, m1, m2, m3, tau, b1, b2, b3, dchi};
    }

    // set up the problem and release old state from memory
    set_output_params(dim, "srmhd");
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    print_shared_mem();
    set_the_riemann_solver();
    cons2prim(fullP);
    if constexpr (global::on_gpu) {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(fullP);
    }
    else {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }

    // Using a sigmoid decay function to represent when the source terms
    // turn off.
    time_constant = sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) {
        write_to_file(*this);
        config_ghosts3D(
            fullP,
            cons.data(),
            nx,
            ny,
            nz,
            spatial_order == "pcm",
            bcs.data(),
            inflow_zones.data(),
            half_sphere,
            geometry
        );
    }

    // Simulate :)
    try {
        simbi::detail::logger::with_logger(*this, tend, [&] {
            advance(activeP);
            config_ghosts3D(
                fullP,
                cons.data(),
                nx,
                ny,
                nz,
                spatial_order == "pcm",
                bcs.data(),
                inflow_zones.data(),
                half_sphere,
                geometry
            );
            cons2prim(fullP);

            if constexpr (global::on_gpu) {
                adapt_dt(fullP);
            }
            else {
                adapt_dt();
            }
            time_constant =
                sigmoid(t, engine_duration, step * dt, constant_sources);

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
