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
GPU_CALLABLE_MEMBER constexpr real
RMHD<dim>::get_x1face(const lint ii, const int side) const
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
GPU_CALLABLE_MEMBER constexpr real
RMHD<dim>::get_x2face(const lint ii, const int side) const
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
GPU_CALLABLE_MEMBER constexpr real
RMHD<dim>::get_x3face(const lint ii, const int side) const
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
 * @return          GPU_CALLABLE_MEMBER
 * @retval
 */
template <int dim>
template <Plane P, Corner C>
GPU_CALLABLE_MEMBER real RMHD<dim>::calc_edge_emf(
    const RMHD<dim>::conserved_t& fw,
    const RMHD<dim>::conserved_t& fe,
    const RMHD<dim>::conserved_t& fs,
    const RMHD<dim>::conserved_t& fn,
    const real* const bstagp1,
    const real* const bstagp2,
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
    const real esw       = swp.ecomponent(nhat);
    const real ese       = sep.ecomponent(nhat);
    const real enw       = nwp.ecomponent(nhat);
    const real ene       = nep.ecomponent(nhat);
    const real one_eigth = static_cast<real>(0.125);
    const real eavg      = static_cast<real>(0.25) * (ew + ee + es + en);

    // Decides at compile time which method to use
    switch (comp_ct_type) {
        case CTTYPE::ZERO:   // Eq. (40)
            {
                const real de_dq2L = ew - esw + ee - ese;
                const real de_dq2R = enw - ew + ene - ee;
                const real de_dq1L = es - esw + en - enw;
                const real de_dq1R = ese - es + ene - en;
                // result is extremely sensitive to how the
                // output value is written. floating points ops
                // and all that...
                return (
                    eavg + one_eigth * (de_dq2L - de_dq2R) +
                    one_eigth * (de_dq1L - de_dq1R)
                );
            }
        case CTTYPE::UPWIND:   // Eq. (51)
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
                    eavg + one_eigth * (de_dq2L - de_dq2R + de_dq1L - de_dq1R)
                );
            }
        default:   // ALPHA, Eq. (49)
            {
                constexpr real alpha = 0.0;
                constexpr auto fac   = static_cast<real>(0.125) * alpha;
                // compute permutation indices
                const auto np1 = (P == Plane::JK) ? 2 : 1;
                const auto np2 = (P == Plane::IJ) ? 2 : 3;

                // face-center magnetic field indices
                const auto [nx1, ny1, nz1] =
                    [&]() -> std::tuple<luint, luint, luint> {
                    if constexpr (P == Plane::JK) {
                        return {xag + 2, nyv, zag + 2};   // B2
                    }
                    return {nxv, yag + 2, zag + 2};   // B1
                }();
                const auto sidx = cidx<P, C, Dir::S>(ii, jj, kk, nx1, ny1, nz1);
                const auto nidx = cidx<P, C, Dir::N>(ii, jj, kk, nx1, ny1, nz1);

                const auto [nx2, ny2, nz2] =
                    [&]() -> std::tuple<luint, luint, luint> {
                    if constexpr (P == Plane::IJ) {   // B2
                        return {xag + 2, nyv, zag + 2};
                    }
                    return {xag + 2, yag + 2, nzv};   // B3
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
                    eavg + one_eigth * (de_dq2L - de_dq2R) +
                    one_eigth * (de_dq1L - de_dq1R)
                );
            }
    }
};

template <int dim>
GPU_CALLABLE_MEMBER real RMHD<dim>::curl_e(
    const luint nhat,
    const real e1l,
    const real e1r,
    const real e2l,
    const real e2r,
    const real e3l,
    const real e3r
) const
{
    switch (geometry) {
        case Geometry::CARTESIAN:
            {
                if (nhat == 1) {
                    return invdx2 * (e3r - e3l) - invdx3 * (e2r - e2l);
                }
                else if (nhat == 2) {
                    return invdx3 * (e1r - e1l) - invdx1 * (e3r - e3l);
                }
                else {
                    return invdx1 * (e2r - e2l) - invdx2 * (e1r - e1l);
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
GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x1_differential(const lint ii
) const
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
GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x2_differential(const lint ii
) const
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
GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x3_differential(const lint ii
) const
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
GPU_CALLABLE_MEMBER real
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
            const real x2mean = calc_any_mean(x2l, x2r, x2_cell_spacing);
            const real x3mean = calc_any_mean(x3l, x3r, x3_cell_spacing);
            const real m1     = cons[gid].momentum(1);
            const real m2     = cons[gid].momentum(2);
            const real m3     = cons[gid].momentum(3);
            const real et     = (cons[gid].den + cons[gid].nrg);
            const real b1     = cons[gid].bcomponent(1);
            const real b2     = cons[gid].bcomponent(2);
            const real b3     = cons[gid].bcomponent(3);
            const real m      = std::sqrt(m1 * m1 + m2 * m2 + m3 * m3);
            const real vsq    = (m * m) / (et * et);
            const real bsq    = (b1 * b1 + b2 * b2 + b3 * b3);
            const real w      = 1.0 / std::sqrt(1.0 - vsq);
            if constexpr (dim == 1) {
                fprintf(
                    stderr,
                    "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, iter: "
                    "%" PRIu64 "\n",
                    cons[gid].den / w,
                    prims[gid].p,
                    vsq,
                    bsq,
                    x1mean,
                    n
                );
            }
            else if constexpr (dim == 2) {
                fprintf(
                    stderr,
                    "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, Bsq: %.2e, x1coord: %.2e, x2coord: "
                    "%.2e, iter: %" PRIu64 "\n",
                    cons[gid].den / w,
                    prims[gid].p,
                    vsq,
                    bsq,
                    x1mean,
                    x2mean,
                    n
                );
            }
            else {
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
                    n
                );
            }
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
    const auto gr               = gamma / (gamma - 1.0);
    const auto* const cons_data = cons.data();
    auto* const prim_data       = prims.data();
    auto* const edens_data      = edens_guess.data();
    auto* const troubled_data   = troubled_cells.data();
    simbi::parallel_for(
        p,
        total_zones,
        [prim_data, cons_data, edens_data, troubled_data, gr, this] GPU_LAMBDA(
            luint gid
        ) {
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
                    if constexpr (dim == 1) {
                        const auto ireal =
                            get_real_idx(gid, radius, active_zones);
                        const real dV = get_cell_volume(ireal);
                        invdV         = 1.0 / dV;
                    }
                    else if constexpr (dim == 2) {
                        const luint ii   = gid % nx;
                        const luint jj   = gid / nx;
                        const auto ireal = get_real_idx(ii, radius, xag);
                        const auto jreal = get_real_idx(jj, radius, yag);
                        const real dV    = get_cell_volume(ireal, jreal);
                        invdV            = 1.0 / dV;
                    }
                    else {
                        const luint kk   = get_height(gid, xag, yag);
                        const luint jj   = get_row(gid, xag, yag, kk);
                        const luint ii   = get_column(gid, xag, yag, kk);
                        const auto ireal = get_real_idx(ii, radius, xag);
                        const auto jreal = get_real_idx(jj, radius, yag);
                        const auto kreal = get_real_idx(kk, radius, zag);
                        const real dV    = get_cell_volume(ireal, jreal, kreal);
                        invdV            = 1.0 / dV;
                    }
                }

                const real d    = cons_data[gid].den * invdV;
                const real m1   = cons_data[gid].momentum(1) * invdV;
                const real m2   = cons_data[gid].momentum(2) * invdV;
                const real m3   = cons_data[gid].momentum(3) * invdV;
                const real tau  = cons_data[gid].nrg * invdV;
                const real b1   = cons_data[gid].bcomponent(1) * invdV;
                const real b2   = cons_data[gid].bcomponent(2) * invdV;
                const real b3   = cons_data[gid].bcomponent(3) * invdV;
                const real dchi = cons_data[gid].chi * invdV;
                const real s    = (m1 * b1 + m2 * b2 + m3 * b3);
                const real ssq  = s * s;
                const real msq  = (m1 * m1 + m2 * m2 + m3 * m3);
                const real bsq  = (b1 * b1 + b2 * b2 + b3 * b3);

                int iter       = 0;
                real qq        = edens_data[gid];
                const real tol = d * global::tol_scale;
                real f, g, dqq;
                do {
                    f   = newton_f_mhd(gr, tau, d, ssq, bsq, msq, qq);
                    g   = newton_g_mhd(gr, d, ssq, bsq, msq, qq);
                    dqq = f / g;
                    qq -= dqq;

                    if (iter >= global::MAX_ITER || std::isnan(qq)) {
                        troubled_data[gid] = 1;
                        dt                 = INFINITY;
                        inFailureState     = true;
                        found_failure      = true;
                        break;
                    }
                    iter++;

                } while (std::abs(dqq) >= tol);

                const real qqd  = qq + d;
                const real rat  = s / qqd;
                const real fac  = 1.0 / (qqd + bsq);
                const real v1   = fac * (m1 + rat * b1);
                const real v2   = fac * (m2 + rat * b2);
                const real v3   = fac * (m3 + rat * b3);
                const real vsq  = v1 * v2 + v2 * v2 + v3 * v3;
                const real wsq  = 1.0 / (1.0 - vsq);
                const real w    = std::sqrt(wsq);
                const real usq  = wsq * vsq;
                const real chi  = qq / wsq - d * usq / (wsq * (1.0 + w));
                const real pg   = (1.0 / gr) * chi;
                edens_data[gid] = qq;
#if FOUR_VELOCITY
                prim_data[gid] =
                    {d / w, v1 * w, v2 * w, v3 * w, pg, b1, b2, b3, dchi / d};
#else
                prim_data[gid] = {d / w, v1, v2, v3, pg, b1, b2, b3, dchi / d};
#endif
                workLeftToDo = false;

                if (qq < 0) {
                    troubled_data[gid] = 1;
                    inFailureState     = true;
                    found_failure      = true;
                    dt                 = INFINITY;
                }
                simbi::gpu::api::synchronize();
            }
        }
    );
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
RMHD<dim>::primitive_t
RMHD<dim>::cons2prim(const RMHD<dim>::conserved_t& cons, const luint gid)
{
    const auto gr          = gamma / (gamma - 1.0);
    auto* const edens_data = edens_guess.data();
    real invdV             = 1.0;
    if (homolog) {
        if constexpr (dim == 1) {
            const auto ireal = get_real_idx(gid, radius, active_zones);
            const real dV    = get_cell_volume(ireal);
            invdV            = 1.0 / dV;
        }
        else if constexpr (dim == 2) {
            const luint ii   = gid % nx;
            const luint jj   = gid / nx;
            const auto ireal = get_real_idx(ii, radius, xag);
            const auto jreal = get_real_idx(jj, radius, yag);
            const real dV    = get_cell_volume(ireal, jreal);
            invdV            = 1.0 / dV;
        }
        else {
            const luint kk   = get_height(gid, xag, yag);
            const luint jj   = get_row(gid, xag, yag, kk);
            const luint ii   = get_column(gid, xag, yag, kk);
            const auto ireal = get_real_idx(ii, radius, xag);
            const auto jreal = get_real_idx(jj, radius, yag);
            const auto kreal = get_real_idx(kk, radius, zag);
            const real dV    = get_cell_volume(ireal, jreal, kreal);
            invdV            = 1.0 / dV;
        }
    }

    const real d    = cons.den * invdV;
    const real m1   = cons.momentum(1) * invdV;
    const real m2   = cons.momentum(2) * invdV;
    const real m3   = cons.momentum(3) * invdV;
    const real tau  = cons.nrg * invdV;
    const real b1   = cons.bcomponent(1) * invdV;
    const real b2   = cons.bcomponent(2) * invdV;
    const real b3   = cons.bcomponent(3) * invdV;
    const real dchi = cons.chi * invdV;
    const real s    = (m1 * b1 + m2 * b2 + m3 * b3);
    const real ssq  = s * s;
    const real msq  = (m1 * m1 + m2 * m2 + m3 * m3);
    const real bsq  = (b1 * b1 + b2 * b2 + b3 * b3);

    int iter       = 0;
    real qq        = edens_data[gid];
    const real tol = d * global::tol_scale;
    real f, g, dqq;
    do {
        f   = newton_f_mhd(gr, tau, d, ssq, bsq, msq, qq);
        g   = newton_g_mhd(gr, d, ssq, bsq, msq, qq);
        dqq = f / g;
        qq -= dqq;

        if (iter >= global::MAX_ITER || std::isnan(qq)) {
            dt             = INFINITY;
            inFailureState = true;
            break;
        }
        iter++;

    } while (std::abs(dqq) >= tol);

    const real qqd = qq + d;
    const real rat = s / qqd;
    const real fac = 1.0 / (qqd + bsq);
    const real v1  = fac * (m1 + rat * b1);
    const real v2  = fac * (m2 + rat * b2);
    const real v3  = fac * (m3 + rat * b3);
    const real vsq = v1 * v2 + v2 * v2 + v3 * v3;
    const real w   = std::sqrt(1.0 / (1.0 - vsq));
    const real usq = w * w * vsq;
    const real chi = qq / (w * w) - (d * usq) / (w * w * (1.0 + w));
    const real pg  = (1.0 / gr) * chi;
#if FOUR_VELOCITY
    return {d / w, v1 * w, v2 * w, v3 * w, pg, b1, b2, b3, dchi / d};
#else
    return {d / w, v1, v2, v3, pg, b1, b2, b3, dchi / d};
#endif
}

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
/*
    Compute the outer wave speeds as discussed in Mignone and Bodo (2006)
*/

template <int dim>
GPU_CALLABLE_MEMBER void RMHD<dim>::calc_max_wave_speeds(
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
    if (prims.vsquared() < global::tol_scale) {   // Eq.(57)
        const real fac  = 1.0 / (rho * h + bmusq);
        const real a    = 1.0;
        const real b    = -(bmusq + rho * h * cs2 + bn2 * cs2) * fac;
        const real c    = cs2 * bn2 * fac;
        const real disq = std::sqrt(b * b - 4.0 * a * c);
        speeds[3]       = std::sqrt(0.5 * (-b + disq));
        speeds[0]       = -speeds[3];
    }
    else if (bn2 < global::tol_scale) {   // Eq. (58)
        const real g2 = prims.lorentz_factor_squared();
        const real vdbperp =
            prims.vdotb() - prims.vcomponent(nhat) * prims.bcomponent(nhat);
        const real q    = bmusq - cs2 * vdbperp * vdbperp;
        const real a2   = rho * h * (cs2 + g2 * (1.0 - cs2)) + q;
        const real a1   = -2 * rho * h * g2 * vn * (1.0 - cs2);
        const real a0   = rho * h * (-cs2 + g2 * vn * vn * (1 - cs2)) - q;
        const real disq = a1 * a1 - 4 * a2 * a0;
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
GPU_CALLABLE_MEMBER RMHD<dim>::eigenvals_t RMHD<dim>::calc_eigenvals(
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
GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t
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
    const real vdotb = (v1 * b1 + v2 * b2 + v3 * b3);
    const real bsq   = (b1 * b1 + b2 * b2 + b3 * b3);
    const real vsq   = (v1 * v1 + v2 * v2 + v3 * v3);
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
            if constexpr (dim > 1) {
                calc_max_wave_speeds(prims[gid], 2, speeds, cs);
                v2p = std::abs(speeds[3]);
                v2m = std::abs(speeds[0]);
            }
            if constexpr (dim > 2) {
                calc_max_wave_speeds(prims[gid], 3, speeds, cs);
                v3p = std::abs(speeds[3]);
                v3m = std::abs(speeds[0]);
            }
        }
        else {
            v1p = 1.0;
            v1m = 1.0;
            if constexpr (dim > 1) {
                v2p = 1.0;
                v2m = 1.0;
            }
            if constexpr (dim > 2) {
                v3p = 1.0;
                v3m = 1.0;
            }
        }

        const real x1l = get_x1face(ireal, 0);
        const real x1r = get_x1face(ireal, 1);
        const real dx1 = x1r - x1l;
        switch (geometry) {
            case simbi::Geometry::CARTESIAN:
                if constexpr (dim == 1) {
                    cfl_dt = dx1 / (std::max(v1p, v1m));
                }
                else if constexpr (dim == 2) {
                    cfl_dt = std::min(
                        {dx1 / (std::max(v1p, v1m)), dx2 / (std::max(v2p, v2m))}
                    );
                }
                else {
                    cfl_dt = std::min(
                        {dx1 / (std::max(v1p, v1m)),
                         dx2 / (std::max(v2p, v2m)),
                         dx3 / (std::max(v3p, v3m))}
                    );
                }
                break;

            case simbi::Geometry::SPHERICAL:
                {
                    if constexpr (dim == 1) {
                        cfl_dt = dx1 / (std::max(v1p, v1m));
                    }
                    else if constexpr (dim == 2) {
                        const real rmean = get_cell_centroid(
                            x1r,
                            x1l,
                            simbi::Geometry::SPHERICAL
                        );
                        cfl_dt = std::min(
                            {dx1 / (std::max(v1p, v1m)),
                             rmean * dx2 / (std::max(v2p, v2m))}
                        );
                    }
                    else {
                        const real x2l   = get_x2face(jj, 0);
                        const real x2r   = get_x2face(jj, 1);
                        const real rmean = get_cell_centroid(
                            x1r,
                            x1l,
                            simbi::Geometry::SPHERICAL
                        );
                        const real th    = 0.5 * (x2r + x2l);
                        const real rproj = rmean * std::sin(th);
                        cfl_dt           = std::min(
                            {dx1 / (std::max(v1p, v1m)),
                                       rmean * dx2 / (std::max(v2p, v2m)),
                                       rproj * dx3 / (std::max(v3p, v3m))}
                        );
                    }
                    break;
                }
            default:
                {
                    if constexpr (dim == 1) {
                        cfl_dt = dx1 / (std::max(v1p, v1m));
                    }
                    else if constexpr (dim == 2) {
                        switch (geometry) {
                            case Geometry::AXIS_CYLINDRICAL:
                                {
                                    cfl_dt = std::min(
                                        {dx1 / (std::max(v1p, v1m)),
                                         dx2 / (std::max(v2p, v2m))}
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
                                         rmean * dx2 / (std::max(v2p, v2m))}
                                    );
                                    break;
                                }
                        }
                    }
                    else {
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
                    }
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
GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t
RMHD<dim>::prims2flux(const RMHD<dim>::primitive_t& prims, const luint nhat)
    const
{
    const real rho   = prims.rho;
    const real v1    = prims.vcomponent(1);
    const real v2    = prims.vcomponent(2);
    const real v3    = prims.vcomponent(3);
    const real p     = prims.total_pressure();
    const real b1    = prims.bcomponent(1);
    const real b2    = prims.bcomponent(2);
    const real b3    = prims.bcomponent(3);
    const real chi   = prims.chi;
    const real vn    = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
    const real bn    = (nhat == 1) ? b1 : (nhat == 2) ? b2 : b3;
    const real lf    = prims.lorentz_factor();
    const real h     = prims.gas_enthalpy(gamma);
    const real bsq   = (b1 * b1 + b2 * b2 + b3 * b3);
    const real vdotb = prims.vdotb();
    const real d     = rho * lf;
    const real ed    = d * h * lf;
    const real m1    = (ed + bsq) * v1 - vdotb * b1;
    const real m2    = (ed + bsq) * v2 - vdotb * b2;
    const real m3    = (ed + bsq) * v3 - vdotb * b3;
    const real mn    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;
    const auto bmu   = mag_fourvec_t(prims);
    const auto invlf = 1.0 / lf;
    return {
      d * vn,
      m1 * vn + kronecker(nhat, 1) * p - bn * bmu.one * invlf,
      m2 * vn + kronecker(nhat, 2) * p - bn * bmu.two * invlf,
      m3 * vn + kronecker(nhat, 3) * p - bn * bmu.three * invlf,
      mn - d * vn,
      vn * b1 - v1 * bn,
      vn * b2 - v2 * bn,
      vn * b3 - v3 * bn,
      d * vn * chi
    };
};

template <int dim>
GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hll_flux(
    const RMHD<dim>::conserved_t& uL,
    const RMHD<dim>::conserved_t& uR,
    const RMHD<dim>::conserved_t& fL,
    const RMHD<dim>::conserved_t& fR,
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
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
GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hllc_flux(
    const RMHD<dim>::conserved_t& uL,
    const RMHD<dim>::conserved_t& uR,
    const RMHD<dim>::conserved_t& fL,
    const RMHD<dim>::conserved_t& fR,
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    static auto construct_the_state = [](const luint nhat,
                                         const luint np1,
                                         const luint np2,
                                         const real d,
                                         const real mnorm,
                                         const real mt1,
                                         const real mt2,
                                         const real tau,
                                         const real bnorm,
                                         const real bt1,
                                         const real bt2) {
        conserved_t u;
        u.den              = d;
        u.momentum(nhat)   = mnorm;
        u.momentum(np1)    = mt1;
        u.momentum(np2)    = mt2;
        u.nrg              = tau;
        u.bcomponent(nhat) = bnorm;
        u.bcomponent(np1)  = bt1;
        u.bcomponent(np2)  = bt2;
        return u;
    };

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
                return hll_flux;
            }
        }

        // get the perpendicular directional unit vectors
        const auto np1 = next_perm(nhat, 1);
        const auto np2 = next_perm(nhat, 2);

        // the normal component of the magnetic field is assumed to
        // be continuos across the interface, so bnL = bnR = bnStar
        const real bnStar  = hll_state.bcomponent(nhat);
        const real bt1Star = hll_state.bcomponent(np1);
        const real bt2Star = hll_state.bcomponent(np2);

        const real uhlld   = hll_state.den;
        const real uhllm1  = hll_state.momentum(1);
        const real uhllm2  = hll_state.momentum(2);
        const real uhllm3  = hll_state.momentum(3);
        const real uhlltau = hll_state.nrg;

        const real fhlld   = hll_flux.den;
        const real fhllm1  = hll_flux.momentum(1);
        const real fhllm2  = hll_flux.momentum(2);
        const real fhllm3  = hll_flux.momentum(3);
        const real fhlltau = hll_flux.nrg;
        const real fhllb1  = hll_flux.bcomponent(1);
        const real fhllb2  = hll_flux.bcomponent(2);
        const real fhllb3  = hll_flux.bcomponent(3);

        const real e    = uhlltau + uhlld;
        const real s    = (nhat == 1) ? uhllm1 : (nhat == 2) ? uhllm2 : uhllm3;
        const real fe   = fhlltau + fhlld;
        const real fs   = (nhat == 1) ? fhllm1 : (nhat == 2) ? fhllm2 : fhllm3;
        const real fpb1 = (np1 == 1) ? fhllb1 : (np1 == 2) ? fhllb2 : fhllb3;
        const real fpb2 = (np2 == 1) ? fhllb1 : (np2 == 2) ? fhllb2 : fhllb3;

        //------Calculate the contact wave velocity and pressure
        const real fdb2 = (bt1Star * fpb1 + bt2Star * fpb2);
        const real a    = fe - fdb2;
        const real b    = -(e + fs) + (bt1Star * bt1Star + bt2Star * bt2Star) +
                       (fpb1 * fpb1 + fpb2 * fpb2);
        const real c     = s - fdb2;
        const real quad  = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
        const real aStar = c * (1.0 / quad);
        const real vt1Star =
            (bnStar == 0.0) ? 0.0
                                : (bt1Star * aStar - fpb1) / bnStar;   // Eq. (38)
        const real vt2Star =
            (bnStar == 0.0) ? 0.0
                                : (bt2Star * aStar - fpb2) / bnStar;   // Eq. (38)
        const real invg2 =
            (1 - (aStar * aStar + vt1Star * vt1Star + vt2Star * vt2Star));
        const real vsdB =
            (aStar * bnStar + vt1Star * bt1Star + vt2Star * bt2Star);
        const real pStar = (bnStar != 0.0) ? -aStar * (fe - bnStar * vsdB) +
                                                 fs + bnStar * bnStar * invg2
                                               : -fe * aStar + fs;

        if (vface <= aStar) {
            // const real pressure = prL.p;
            const real d   = uL.den;
            const real m1  = uL.momentum(1);
            const real m2  = uL.momentum(2);
            const real m3  = uL.momentum(3);
            const real tau = uL.nrg;
            // const real chi      = uL.chi;
            const real e        = tau + d;
            const real cofactor = 1.0 / (aL - aStar);
            const real mnorm    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;

            const real vL = prL.vcomponent(nhat);
            // Left Star State in x-direction of coordinate lattice
            const real dStar = cofactor * (aL - vL) * d;
            const real eStar =
                cofactor * (aL * e - mnorm + pStar * aStar - vsdB * bnStar);
            const real mnStar = (eStar + pStar) * aStar - vsdB * bnStar;
            const real mt1Star =
                cofactor * (-bnStar * (bt1Star * invg2 + vsdB * vt1Star) +
                            aL * uL.momentum(np1) - fL.momentum(np1));
            const real mt2Star =
                cofactor * (-bnStar * (bt2Star * invg2 + vsdB * vt2Star) +
                            aL * uL.momentum(np2) - fL.momentum(np2));
            const real tauStar    = eStar - dStar;
            const auto starStateL = construct_the_state(
                nhat,
                np1,
                np2,
                d,
                mnStar,
                mt1Star,
                mt2Star,
                tauStar,
                bnStar,
                bt1Star,
                bt2Star
            );
            return fL + (starStateL - uL) * aL - starStateL * vface;
        }
        else {
            // const real pressure = prR.p;
            const real d   = uR.den;
            const real m1  = uR.momentum(1);
            const real m2  = uR.momentum(2);
            const real m3  = uR.momentum(3);
            const real tau = uR.nrg;
            // const real chi      = uR.chi;
            const real e        = tau + d;
            const real cofactor = 1.0 / (aR - aStar);
            const real mnorm    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;

            const real vR = prR.vcomponent(nhat);
            // Right Star State in x-direction of coordinate lattice
            const real dStar = cofactor * (aR - vR) * d;
            const real eStar =
                cofactor * (aR * e - mnorm + pStar * aStar - vsdB * bnStar);
            const real mnStar = (eStar + pStar) * aStar - vsdB * bnStar;
            const real mt1Star =
                cofactor * (-bnStar * (bt1Star * invg2 + vsdB * vt1Star) +
                            aR * uR.momentum(np1) - fR.momentum(np1));
            const real mt2Star =
                cofactor * (-bnStar * (bt2Star * invg2 + vsdB * vt2Star) +
                            aR * uR.momentum(np2) - fR.momentum(np2));
            const real tauStar    = eStar - dStar;
            const auto starStateR = construct_the_state(
                nhat,
                np1,
                np2,
                d,
                mnStar,
                mt1Star,
                mt2Star,
                tauStar,
                bnStar,
                bt1Star,
                bt2Star
            );
            return fR + (starStateR - uR) * aR - starStateR * vface;
        }
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
GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hlld_flux(
    const RMHD<dim>::conserved_t& uL,
    const RMHD<dim>::conserved_t& uR,
    const RMHD<dim>::conserved_t& fL,
    const RMHD<dim>::conserved_t& fR,
    const RMHD<dim>::primitive_t& prL,
    const RMHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const {
    // conserved_t ua, uc;
    // const auto lambda = calc_eigenvals(prL, prR,
    // nhat); const real aL  = lambda.afL; const real aR  = lambda.afR;
    // const
    // real
    // aLm = aL < 0 ? aL : 0; const real aRp = aR > 0 ? aR : 0;

    // //---- Check wave speeds before wasting computations
    // if (vface <= aLm) {
    //     return fL - uL * vface;
    // } else if (vface >= aRp) {
    //     return fR - uR * vface;
    // }

    //  //-------------------Calculate the HLL Intermediate State
    // const auto hll_state =
    //     (uR * aRp - uL * aLm - fR + fL) /
    //     (aRp
    //     - aLm);

    // //------------------Calculate the RHLLE Flux---------------
    // const auto hll_flux
    //     = (fL * aRp - fR * aLm + (uR - uL) *
    //     aRp * aLm)
    //         / (aRp - aLm);

    // // define the magnetic field normal to the zone
    // const auto bn = hll_state.bcomponent(nhat);

    // // Eq. (12)
    // const auto rL = uL * aLm - fL;
    // const auto rR = uR * aRp - fR;

    // //==================================================================
    // // Helper functions to ease repetition
    // //==================================================================
    // const real qfunc = [](const conserved_t &r, const luint nhat,
    // const
    // real a, const real p) {
    //     return r.total_energy() * a + p * (1.0 - a * a);
    // };
    // const real gfunc =[](const luint np1, const luint np2, const
    // conserved_t &r) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     } else if constexpr(dim == 2) {
    //         return (r.bcomponent(np1) * r.bcomponent(np1));
    //     } else {
    //         return (r.bcomponent(np1) * r.bcomponent(np1) +
    //         r.bcomponent(np2)
    //         *
    //         r.bcomponent(np2));
    //     }
    // };
    // const real yfunc = [](const luint np1, const luint np2, const
    // conserved_t &r) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     } else if constexpr(dim == 2) {
    //         return r.bcomponent(np1) * r.momentum(np1);
    //     } else {
    //         return r.bcomponent(np1) * r.momentum(np1) +
    //         r.bcomponent(np2) *
    //         r.momentum(np2);
    //     }
    // };
    // const real ofunc = [](const real q, const real g, const real bn,
    // const
    // real
    // a) {
    //     return q - g + bn * bn * (1.0 - a * a);
    // };
    // const real xfunc = [](const real q, const real y, const real g, const
    // real
    // bn, const real a, const real p, const real et) {
    //     return bn * (q * a * bn + y) - (q + g) * (a * p + et);
    // };
    // const real vnfunc = [](const real bn, const real q, const real a,
    // const
    // real y, const real g, const real p, const real mn, const real x) {
    //     return (bn * (q* bn + a * y) - (q + g) * (p + mn)) / x;
    // };
    // const real vt1func = [](const real o, const real mt1, const real bt1,
    // const
    // real y, const real bn, const real a, const real mn, const real et,
    // const
    // real x) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     };
    //     return (o * mt1 + bt1 * (y + bn * (a * mn - et))) / x;
    // };
    // const real vt2func = [](const real o, const real mt2, const real bt2,
    // const
    // real y, const real bn, const real a, const real mn, const real et,
    // const
    // real x) {
    //     if constexpr(dim < 3) {
    //         return 0;
    //     };
    //     return (o * mt1 + bt2 * (y + bn * (a * mn - et))) / x;
    // };
    // const real btanfunc = [](const real rbk, const real bn, const real
    // vn,
    // const real a) {
    //     if constexpr(dim == 1) {
    //         return 0;
    //     };
    //     return (rbk - bn * vn) / (a - vn);
    // };

    // const real total_enthalpy(const real p, const real et, const real
    // vdr,
    // const real a, const real vn) {
    //     return p + (et - vdr) / (a - vn);
    // };

    // const real bkc = [](const real bkL, const real bkR, const real vaL,
    // const
    // real vaR, const real vnL, const real vnR, const real bn, const real
    // vkL,
    // const real vkR) {
    //     return (
    //           bkR * (vaR - vnR)
    //         - bkL * (vaL - vnL)
    //         + bn  * (vkR - vkL)
    //     ) / (vaR - vaL);
    // };

    // const real vec_dot = [](const real x1, const real x2, const real x3,
    // const
    // real y1, const real y2, const real y3) {
    //     x1 * y1 + x2 * y2 + x3 * y3;
    // };

    // const real vec_sq = [](const real x1, const real x2, const real x3) {
    //     return x1 *x1 + x2 * x2 + x3 * x3;
    // };

    // const conserved_t construct_the_state = [](
    //     const luint nhat,
    //     const luint np1,
    //     const luint np2
    //     const real d,
    //     const real vfac,
    //     const real et,
    //     const real p,
    //     const real vn,
    //     const real vdb,
    //     const real bn,
    //     const real bp1,
    //     const real bp2,
    //     const real vp1,
    //     const real vp2
    // ) {
    //     conserved_t u;
    //     u.den= d * vfac;
    //     u.momentum(nhat) = (et + p) * vn - vdb * bn;
    //     if constexpr(dim > 1) {
    //         u.momentum(np1 > dim ? 1 : np1) = (et + p) * vp1 - vdb * bp1;
    //     }
    //     if constexpr(dim > 2) {
    //         u.momentum(np2) = (et + p) * vp2 - vdb * bp2;
    //     }
    //     u.nrg = et - u.den;
    //     u.bcomponent(nhat) = bn;
    //     if constexpr(dim > 1) {
    //         u.bcomponent(np1 > dim ? 1 : np1) = bp1;
    //     }
    //     if constexpr(dim > 2) {
    //         u.bcomponent(np2) = bp2;
    //     }
    //     return u;
    // };

    // //==============================================================================
    // // initial pressure guess
    // real p0 = 0;
    // if (bn * bn / (pguess * pguess) < 0.01) {
    //     const real a = aRp - aLm;
    //     const real b = rR.total_energy() - rL.total_energy() + aRp * rL -
    //     aLm
    //     *
    //     rR; const real c = rL.momentum(nhat) * rR.total_energy() -
    //     rR.momentum(nhat) * rL.total_energy(); const real quad =
    //     std::max((0.0), b * b - 4 * a * c); p0 = 0.5 * (-b +
    //     std::sqrt(quad))
    //     /
    //     (aRp - aLm);
    // } else {
    //     const auto phll = cons2prim(hll_state, gid);
    //     p0 = phll.total_pressure();
    // }
    // //----------------- Jump conditions across the fast waves
    // (section 3.1)
    // const auto np1  = next_perm(nhat, 1);
    // const auto np2  = next_perm(nhat, 2);

    // // left side
    // const auto pL   = prL.total_pressure();
    // const auto qL   = qfunc(rL, nhat, aLm, pL);;
    // const auto gL   = gfunc(np1, np2, rL);
    // const auto yL   = yfunc(np1, np2, rL);
    // const auto oL   = ofunc(qL, gL, bn, aLm);
    // const auto xL   = xfunc(qL, yL, gL, bn, aLm, pL, rL.total_energ());
    // // velocity components
    // const auto vnL   = vnfunc(bn, qL, aLm, yL, gL, pL, mnL, xL);
    // const auto vt1L  = vt1func(oL, rL.momentum(np1), rL.bcomponent(np1),
    // yL,
    // bn, aLm, rL.momentum(nhat), rL.total_energy(), xL); const auto vt2L =
    // vt2func(oL, rL.momentum(np2), rL.bcomponent(np2), yL, bn, aLm,
    // rL.momentum(nhat), rL.total_energy(), xL); const auto bp1L  =
    // btanfunc(rL.bcomponent(np1), bn, vnL, vt1L, aLm); const auto bp2L  =
    // btanfunc(rL.bcomponent(np2), bn, vnL, vt2L, aLm); const auto vdrL  =
    // vnL
    // *
    // rL.momentum(nhat) + vt1L * rL.momentum(np1) + vt2L *
    // rL.momentum(np2);
    // const auto wL    = total_enthalpy(pL, rL.total_energy(), vdr, aLm,
    // vnL);
    // const auto vdbL  = (vnL * bn + vnL1 * bp1 + vnL2 * bp2);
    // const auto vfacL = 1.0 /(aLm - vnL);

    // // right side
    // const auto pR   = prR.total_pressure();
    // const auto qR   = qfunc(rR, nhat, aRm, pR);;
    // const auto gR   = gfunc(np1, np2, rR);
    // const auto yR   = yfunc(np1, np2, rR);
    // const auto oR   = ofunc(qR, gR, bn, aRm);
    // const auto xR   = xfunc(qR, yR, gR, bn, aRm, pR, rR.total_energ());
    // // velocity components
    // const auto vnR   = vnfunc(bn, qR, aRm, yR, gR, pR, mnR, xR);
    // const auto vt1R  = vt1func(oR, rR.momentum(np1), rR.bcomponent(np1),
    // yR,
    // bn, aRm, rR.momentum(nhat), rR.total_energy(), xR); const auto vt2R =
    // vt2func(oR, rR.momentum(np2), rR.bcomponent(np2), yR, bn, aRm,
    // rR.momentum(nhat), rR.total_energy(), xR); const auto bp1R  =
    // btanfunc(rR.bcomponent(np1), bn, vnR, vt1R, aRm); const auto bp2R  =
    // btanfunc(rR.bcomponent(np2), bn, vnR, vt2R, aRm); const auto vdrR  =
    // vnR
    // *
    // rR.momentum(nhat) + vt1R * rR.momentum(np1) + vt2R *
    // rR.momentum(np2);
    // const auto wR    = total_enthalpy(pR, rR.total_energy(), vdr, aRm,
    // vnR);
    // const auto vdbR  = (vnR * bn + vnR1 * bp1 + vnR2 * bp2);
    // const auto vfacR = 1.0 /(aRm - vnR);

    // //--------------Jump conditions across the Alfven waves (section 3.2)
    // const auto etaL = - sgn(bn) * std::sqrt(wL);
    // const auto etaR =   sgn(bn) * std::sqrt(wR);
    // const auto calc_kcomp = (const int nhat, const int ehat, const
    // conserved_t &r, const real p, const real a, const real eta) {
    //     return (r.momentum(nhat) + p * kronecker(ehat, nhat) +
    //     r.bcomponent(ehat) * eta) / (a * p + r.total_energy() + bn *
    //     eta);
    // }
    // const auto knL  = calc_kcomp(nhat, nhat, rL, pL, aLm, etaL);
    // const auto knR  = calc_kcomp(nhat, nhat, rR, pR, aRm, etaR);
    // const auto kt1L = calc_kcomp(nhat, np1, rL,  pL, aLm, etaL);
    // const auto kt1R = calc_kcomp(nhat, np1, rR,  pR, aRm, etaR);
    // const auto kt2L = calc_kcomp(nhat, np2, rL,  pL, aLm, etaL);
    // const auto kt2R = calc_kcomp(nhat, np2, rR,  pR, aRp, etaR);
    // // the k-normal is the Alfven wave speed
    // const auto vaL = knL;
    // const auto vaR = knR;
    // if (aLm - vaL < vface) { // return FaL
    //     ua = construct_the_state(
    //         nhat,
    //         np1,
    //         np2,
    //         rL.d,
    //         vfacL,
    //         rL.total_energy(),
    //         pL,
    //         vnL,
    //         vdbL,
    //         bn,
    //         bp1L,
    //         bp2L,
    //         vt1L,
    //         vt2L
    //     );
    //     return fL + (ua - uL) * vaL - ua * vface;
    // } else if (vaR - aRp < vface) { // return FaR
    //     ua = construct_the_state(
    //         nhat,
    //         np1,
    //         np2,
    //         rR.d,
    //         vfacR,
    //         rR.total_energy(),
    //         pR,
    //         vnR,
    //         vdbR,
    //         bn,
    //         bp1R,
    //         bp2R,
    //         vt1R,
    //         vt2R
    //     );

    //     return fR + (ua - uR) * vaR - ua * vface;
    // } else {
    //     dK  = 1.0 /(vaR - vaL);
    //     //---------------Jump conditions across the contact wave
    //     (section 3.3)
    //     const auto bkxn  = bn;
    //     const auto bkc1  = bkc(uaL.bcomponent(np1), uaR.bcomponent(np1),
    //     vaL,
    //     vaR, vnL, vnR, vt1L, vt1R) * dK; const auto bkc2  =
    //     bkc(uaL.bcomponent(np2), uaR.bcomponent(np2), vaL, vaR, vnL, vnR,
    //     vt2L,
    //     vt2R) * dK; const auto kdbL  = vec_dot(bkxn, bkc1, bkc2, knL,
    //     kt1L,
    //     kt2L); const auto kdbR  = vec_dot(bkxn, bkc1, bkc2, knR, kt1R,
    //     kt2R);
    //     const auto ksqL  = vec_sq(knL, kt1L, kt2L);
    //     const auto ksqR  = vec_sq(knR, kt1R, kt2R);
    //     const auto kfacL = (1.0 - ksqL) / (etaL - kdbL);
    //     const auto kfacR = (1.0 - ksqR) / (etaR - kdbR);
    //     const auto vanL  = knL  -  bn * kfacL;
    //     const auto vat1L = kt1L - bkc1 * kfacL;
    //     const auto vat2L = kt2L - bkc2 * kfacL;
    //     const auto vanR  = knR  -  bn * kfacR;
    //     const auto vat1R = kt1R - bkc1 * kfacR;
    //     const auto vat2R = kt2R - bkc2 * kfacR;
    //     const auto vakn = 0.5 * (vanL + vanR);
    //     const auto vat1 = 0.5 * (vat1L + vat1R);
    //     const auto vat2 = 0.5 * (vat2L + vat2R);
    //     const auto vdbc = vec_dot(vakn, vat1, vat2, bkxn, bkc1, bkc2);
    //     if (vakn > 0) {
    //         ua = construct_the_state(
    //             nhat,
    //             np1,
    //             np2,
    //             rL.d,
    //             vfacL,
    //             rL.total_energy(),
    //             pL,
    //             vnL,
    //             vdbL,
    //             bn,
    //             bp1L,
    //             bp2L,
    //             vt1L,
    //             vt2L
    //         );
    //         const real etc  = (vaL * ua.total_energy() -
    //         ua.momentum(nhat) +
    //         pL
    //         * vakn - vdbc * bn) / (vaL - vakn); uc = construct_the_state(
    //             nhat,
    //             np1,
    //             np2,
    //             ua.d,
    //             (vaL - vnL) / (vaL - vakn),
    //             etc,
    //             pL,
    //             vnL,
    //             vdbc,
    //             bn,
    //             bkc1,
    //             bkc2,
    //             vat1L,
    //             vat2L
    //         );

    //         const auto fa = fL + (ua - uL) * vaL;
    //         return fa + (uc - ua) * vakn - uc * vface;
    //     } else {
    //         ua = construct_the_state(
    //             nhat,
    //             np1,
    //             np2,
    //             rL.d,
    //             vfacR,
    //             rR.total_energy(),
    //             pR,
    //             vnR,
    //             vdbR,
    //             bn,
    //             bp1R,
    //             bp2R,
    //             vt1R,
    //             vt2R
    //         );
    //         const real etc  = (vaR * uaR.total_energy() -
    //         uaR.momentum(nhat)
    //         +
    //         pR * vakn - vdbc * bnR) / (vaR - vakn); uc =
    //         construct_the_state(
    //             nhat,
    //             np1,
    //             np2,
    //             ua.d,
    //             (vaR - vnR) / (vaR - vakn),
    //             etc,
    //             pR,
    //             vnR,
    //             vdbc,
    //             bn,
    //             bkc1,
    //             bkc2,
    //             vat1R,
    //             vat2R
    //         );
    //         const auto fa = fR + (ua - uR) * vaR;
    //         return fa + (uc - ua) * vakn - uc * vface;
    //     }
    // }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void RMHD<dim>::advance(const ExecutionPolicy<>& p)
{
    const luint extent            = p.get_full_extent();
    auto* const cons_data         = cons.data();
    auto* const bstag1_dat        = bstag1.data();
    auto* const bstag2_dat        = bstag2.data();
    auto* const bstag3_dat        = bstag3.data();
    const auto* const prim_data   = prims.data();
    const auto* const dens_source = density_source.data();
    const auto* const mom1_source = m1_source.data();
    const auto* const mom2_source = m2_source.data();
    const auto* const mom3_source = m3_source.data();
    const auto* const mag1_source = sourceB1.data();
    const auto* const mag2_source = sourceB2.data();
    const auto* const mag3_source = sourceB3.data();
    const auto* const erg_source  = energy_source.data();
    const auto* const object_data = object_pos.data();
    const auto* const g1_source   = sourceG1.data();
    const auto* const g2_source   = sourceG2.data();
    const auto* const g3_source   = sourceG3.data();

    simbi::parallel_for(
        p,
        extent,
        [p,
         prim_data,
         cons_data,
         dens_source,
         mom1_source,
         mom2_source,
         mom3_source,
         mag1_source,
         mag2_source,
         mag3_source,
         erg_source,
         object_data,
         g1_source,
         g2_source,
         g3_source,
         bstag1_dat,
         bstag2_dat,
         bstag3_dat,
         this] GPU_LAMBDA(const luint idx) {
            conserved_t f[10];
            conserved_t g[10];
            conserved_t h[10];
            primitive_t pL, pLL, pR, pRR;
            real e1[4], e2[4], e3[4];

            auto prim_buff = sm_proxy<primitive_t>(prim_data);

            const luint kk = axid<dim, BlkAx::K>(idx, xag, yag);
            const luint jj = axid<dim, BlkAx::J>(idx, xag, yag, kk);
            const luint ii = axid<dim, BlkAx::I>(idx, xag, yag, kk);

            if constexpr (global::on_gpu) {
                if constexpr (dim == 1) {
                    if (ii >= xag) {
                        return;
                    }
                }
                else if constexpr (dim == 2) {
                    if ((ii >= xag) || (jj >= yag)) {
                        return;
                    }
                }
                else {
                    if ((ii >= xag) || (jj >= yag) || (kk >= zag)) {
                        return;
                    }
                }
            }

            const luint ia  = ii + radius;
            const luint ja  = dim < 2 ? 0 : jj + radius;
            const luint ka  = dim < 3 ? 0 : kk + radius;
            const luint tx  = (global::on_sm) ? threadIdx.x : 0;
            const luint ty  = dim < 2 ? 0 : (global::on_sm) ? threadIdx.y : 0;
            const luint tz  = dim < 3 ? 0 : (global::on_sm) ? threadIdx.z : 0;
            const luint txa = (global::on_sm) ? tx + radius : ia;
            const luint tya = dim < 2 ? 0 : (global::on_sm) ? ty + radius : ja;
            const luint tza = dim < 3 ? 0 : (global::on_sm) ? tz + radius : ka;
            const luint aid = idx3(ia, ja, ka, nx, ny, nz);

            if constexpr (global::on_sm) {
                load_shared_buffer<dim>(
                    p,
                    prim_buff,
                    prim_data,
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
            const bool object_to_left =
                ib_check<dim>(object_data, il, jj, kk, xag, yag, 1);
            const bool object_to_right =
                ib_check<dim>(object_data, ir, jj, kk, xag, yag, 1);
            const bool object_in_front =
                ib_check<dim>(object_data, ii, jr, kk, xag, yag, 2);
            const bool object_behind =
                ib_check<dim>(object_data, ii, jl, kk, xag, yag, 2);
            const bool object_above =
                ib_check<dim>(object_data, ii, jj, kr, xag, yag, 3);
            const bool object_below =
                ib_check<dim>(object_data, ii, jj, kl, xag, yag, 3);

            const real x1l    = get_x1face(ii, 0);
            const real x1r    = get_x1face(ii, 1);
            const real vfaceL = (homolog) ? x1l * hubble_param : hubble_param;
            const real vfaceR = (homolog) ? x1r * hubble_param : hubble_param;

            // Calc Rimeann Flux at all interfaces
            // printf("aid: %llu, ii: %llu\n", aid, ii);
            for (luint q = 0; q < 10; q++) {
                const auto vdir = 1 * ((luint) (q - 2) < (4 - 2)) -
                                  1 * ((luint) (q - 6) < (8 - 6));
                const auto hdir = 1 * ((luint) (q - 4) < (6 - 4)) -
                                  1 * ((luint) (q - 8) < (10 - 8));

                // fluxes in i direction
                pL = prim_buff[idx3(
                    txa + (q % 2) - 1,
                    tya + vdir,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];
                pR = prim_buff[idx3(
                    txa + (q % 2) + 0,
                    tya + vdir,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];

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

                f[q] = (this->*riemann_solve)(
                    prims2cons(pL),
                    prims2cons(pR),
                    prims2flux(pL, 1),
                    prims2flux(pR, 1),
                    pL,
                    pR,
                    1,
                    0
                );

                // fluxes in j direction
                pL = prim_buff[idx3(
                    txa + vdir,
                    tya + (q % 2) - 1,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];
                pR = prim_buff[idx3(
                    txa + vdir,
                    tya + (q % 2) + 0,
                    tza + hdir,
                    sx,
                    sy,
                    0
                )];

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

                g[q] = (this->*riemann_solve)(
                    prims2cons(pL),
                    prims2cons(pR),
                    prims2flux(pL, 2),
                    prims2flux(pR, 2),
                    pL,
                    pR,
                    2,
                    0
                );

                // fluxes in k direction
                pL = prim_buff[idx3(
                    txa + vdir,
                    tya + hdir,
                    tza + (q % 2) - 1,
                    sx,
                    sy,
                    0
                )];
                pR = prim_buff[idx3(
                    txa + vdir,
                    tya + hdir,
                    tza + (q % 2) + 0,
                    sx,
                    sy,
                    0
                )];

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

                h[q] = (this->*riemann_solve)(
                    prims2cons(pL),
                    prims2cons(pR),
                    prims2flux(pL, 3),
                    prims2flux(pR, 3),
                    pL,
                    pR,
                    3,
                    0
                );

                // printf("f[%llu] = %2.e\n", q, f[q].den);
                // printf("g[%llu] = %2.e\n", q, g[q].den);
                // printf("h[%llu] = %2.e\n", q, h[q].den);
            }
            // std::cin.get();

            const luint real_loc = kk * xag * yag + jj * xag + ii;
            const real d_source  = null_den ? 0.0 : dens_source[real_loc];
            const real s1_source = null_mom1 ? 0.0 : mom1_source[real_loc];
            const real e_source  = null_nrg ? 0.0 : erg_source[real_loc];
            const real b1_source = null_mag1 ? 0.0 : mag1_source[real_loc];

            const auto source_terms = [&] {
                const real s2_source = null_mom2 ? 0.0 : mom2_source[real_loc];
                const real s3_source = null_mom3 ? 0.0 : mom3_source[real_loc];
                const real b2_source = null_mag2 ? 0.0 : mag2_source[real_loc];
                const real b3_source = null_mag3 ? 0.0 : mag3_source[real_loc];
                return conserved_t{
                         d_source,
                         s1_source,
                         s2_source,
                         s3_source,
                         e_source,
                         b1_source,
                         b2_source,
                         b3_source
                       } *
                       time_constant;
            }();

            // Gravity
            const auto gs1_source =
                nullg1 ? 0 : g1_source[real_loc] * cons_data[aid].den;
            const auto tid = tza * sx * sy + tya * sx + txa;

            const auto gravity = [&] {
                const auto gs2_source =
                    nullg2 ? 0 : g2_source[real_loc] * cons_data[aid].den;
                const auto gs3_source =
                    nullg3 ? 0 : g3_source[real_loc] * cons_data[aid].den;
                const auto ge_source = gs1_source * prim_buff[tid].v1 +
                                       gs2_source * prim_buff[tid].v2 +
                                       gs3_source * prim_buff[tid].v3;
                return conserved_t{
                  0.0,
                  gs1_source,
                  gs2_source,
                  gs3_source,
                  ge_source,
                  0.0,
                  0.0,
                  0.0
                };
            }();

            // compute edge emfs in clockwise direction wrt cell plane
            detail::for_sequence(
                detail::make_index_sequence<4>(),
                [&](auto qidx) {
                    constexpr auto q      = luint(qidx);
                    constexpr auto corner = static_cast<Corner>(q);
                    auto widx = q == 0 ? 1 : q == 1 ? 0 : q == 2 ? 6 : 7;
                    auto eidx = q == 0 ? 3 : q == 1 ? 2 : q == 2 ? 0 : 1;
                    auto sidx = q == 0 ? 1 : q == 1 ? 7 : q == 2 ? 6 : 0;
                    auto nidx = q == 0 ? 3 : q == 1 ? 1 : q == 2 ? 0 : 2;
                    e3[q]     = calc_edge_emf<Plane::IJ, corner>(
                        g[widx],
                        g[eidx],
                        f[sidx],
                        f[nidx],
                        bstag1_dat,
                        bstag2_dat,
                        prim_buff,
                        ii,
                        jj,
                        kk,
                        ia,
                        ja,
                        ka,
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
                        bstag1_dat,
                        bstag3_dat,
                        prim_buff,
                        ii,
                        jj,
                        kk,
                        ia,
                        ja,
                        ka,
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
                        bstag2_dat,
                        bstag3_dat,
                        prim_buff,
                        ii,
                        jj,
                        kk,
                        ia,
                        ja,
                        ka,
                        1
                    );
                }
            );

            const auto xg  = xag + 2;
            const auto yg  = yag + 2;
            const auto zg  = zag + 2;
            const auto ks  = kk + 1;
            const auto js  = jj + 1;
            const auto is  = ii + 1;
            const auto xlf = idx3(ii + 0, js, ks, nxv, yg, zg);
            const auto xrf = idx3(ii + 1, js, ks, nxv, yg, zg);
            const auto ylf = idx3(is, jj + 0, ks, xg, nyv, zg);
            const auto yrf = idx3(is, jj + 1, ks, xg, nyv, zg);
            const auto zlf = idx3(is, js, kk + 0, xg, yg, nzv);
            const auto zrf = idx3(is, js, kk + 1, xg, yg, nzv);

            auto& b1L = bstag1_dat[xlf];
            auto& b1R = bstag1_dat[xrf];
            auto& b2L = bstag2_dat[ylf];
            auto& b2R = bstag2_dat[yrf];
            auto& b3L = bstag3_dat[zlf];
            auto& b3R = bstag3_dat[zrf];

            b1L -= dt * step *
                   curl_e(1, 0.0, 0.0, e2[IMKM], e2[IMKP], e3[IMJM], e3[IMJP]);
            b1R -= dt * step *
                   curl_e(1, 0.0, 0.0, e2[IPKM], e2[IPKP], e3[IPJM], e3[IPJP]);
            b2L -= dt * step *
                   curl_e(2, e1[JMKM], e1[JMKP], 0.0, 0.0, e3[IMJM], e3[IPJM]);
            b2R -= dt * step *
                   curl_e(2, e1[JPKM], e1[JPKP], 0.0, 0.0, e3[IMJP], e3[IPJP]);
            b3L -= dt * step *
                   curl_e(3, e1[JMKM], e1[JPKM], e2[IMKM], e2[IPKM], 0.0, 0.0);
            b3R -= dt * step *
                   curl_e(3, e1[JMKP], e1[JPKP], e2[IMKP], e2[IPKP], 0.0, 0.0);

            cons_data[aid].b1 = static_cast<real>(0.5) * (b1L + b1R);
            cons_data[aid].b2 = static_cast<real>(0.5) * (b2L + b2R);
            cons_data[aid].b3 = static_cast<real>(0.5) * (b3L + b3R);

            // printf(
            //     "e1[JPKP]: %.3e, e1[JMKP]: %.3e, e1[JMKM]: %.3e, e1[JMKP]: "
            //     "%.3e, b1: "
            //     "%.3e, bxL: %.3e, bxR: %.3e\n",
            //     e1[JPKP],
            //     e1[JMKP],
            //     e1[JMKM],
            //     e1[JMKP],
            //     cons_data[aid].b1,
            //     b1L,
            //     b1R
            // );

            // printf(
            //     "e2[IPKP]: %.3e, e2[IPKM]: %.3e, e2[IMKM]: %.3e, e2[IMKP]: "
            //     "%.3e, b2: "
            //     "%.3e, byL: %.3e, byR: %.3e\n",
            //     e2[IPKP],
            //     e2[IPKM],
            //     e2[IMKM],
            //     e2[IMKP],
            //     cons_data[aid].b2,
            //     b2L,
            //     b2R
            // );

            // printf(
            //     "e3[IPJP]: %.3e, e3[IPJM]: %.3e, e3[IMJM]: %.3e, e3[IMJP]: "
            //     "%.3e, b3: "
            //     "%.3e, bzL: %.3e, bzR: %.3e\n",
            //     e3[IPJP],
            //     e3[IPJM],
            //     e3[IMJM],
            //     e3[IMJP],
            //     cons_data[aid].b3,
            //     b3L,
            //     b3R
            // );
            // std::cin.get();

            // Advance depending on geometry
            switch (geometry) {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons_data[aid] -= ((f[RF] - f[LF]) * invdx1 +
                                           (g[RF] - g[LF]) * invdx2 +
                                           (h[RF] - h[LF]) * invdx3 -
                                           source_terms - gravity) *
                                          dt * step;
                        break;
                    }
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real tl    = get_x2face(jj, 0);
                        const real tr    = get_x2face(jj, 1);
                        const real ql    = get_x3face(kk, 0);
                        const real qr    = get_x3face(kk, 1);
                        const real rmean = get_cell_centroid(
                            rr,
                            rl,
                            simbi::Geometry::SPHERICAL
                        );
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

                        const real hc = prim_buff[tid].gas_enthalpy(gamma);
                        const real gam2 =
                            prim_buff[tid].lorentz_factor_squared();

                        const auto geom_source = conserved_t{
                          0.0,
                          (rhoc * hc * gam2 * (vc * vc + wc * wc) -
                           bmuc.two * bmuc.two - bmuc.three * bmuc.three) /
                                  rmean +
                              pc * (s1R - s1L) / dV1,
                          (rhoc * hc * gam2 * (wc * wc * cot - uc * vc) -
                           bmuc.three * bmuc.three * cot + bmuc.one * bmuc.two
                          ) / rmean +
                              pc * (s2R - s2L) / dV2,
                          -(rhoc * hc * gam2 * wc * (uc + vc * cot) -
                            bmuc.three * bmuc.one - bmuc.three * bmuc.two * cot
                          ) / rmean,
                          0.0,
                          0.0,
                          0.0,
                          0.0
                        };
                        cons_data[aid] -= ((f[RF] * s1R - f[LF] * s1L) / dV1 +
                                           (g[RF] * s2R - g[LF] * s2L) / dV2 +
                                           (h[RF] - h[LF]) / dV3 - geom_source -
                                           source_terms - gravity) *
                                          dt * step;
                        break;
                    }
                default:
                    {
                        const real rl    = x1l + vfaceL * step * dt;
                        const real rr    = x1r + vfaceR * step * dt;
                        const real ql    = get_x2face(jj, 0);
                        const real qr    = get_x2face(jj, 1);
                        const real zl    = get_x3face(kk, 0);
                        const real zr    = get_x3face(kk, 1);
                        const real rmean = get_cell_centroid(
                            rr,
                            rl,
                            simbi::Geometry::CYLINDRICAL
                        );
                        const real s1R = rr * (zr - zl) * (qr - ql);
                        const real s1L = rl * (zr - zl) * (qr - ql);
                        const real s2R = (rr - rl) * (zr - zl);
                        const real s2L = (rr - rl) * (zr - zl);
                        const real s3L = rmean * (rr - rl) * (zr - zl);
                        const real s3R = s3L;
                        // const real thmean = 0.5 * (tl + tr);
                        const real dV =
                            rmean * (rr - rl) * (zr - zl) * (qr - ql);
                        const real invdV = 1.0 / dV;

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].get_v1();
                        const real vc   = prim_buff[tid].get_v2();
                        // const real wc   = prim_buff[tid].get_v3();
                        const real pc   = prim_buff[tid].total_pressure();
                        const auto bmuc = mag_fourvec_t(prim_buff[tid]);

                        const real hc = prim_buff[tid].gas_enthalpy(gamma);
                        const real gam2 =
                            prim_buff[tid].lorentz_factor_squared();

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
                        cons_data[aid] -= ((f[RF] * s1R - f[LF] * s1L) * invdV +
                                           (g[RF] * s2R - g[LF] * s2L) * invdV +
                                           (h[RF] * s3R - h[LF] * s3L) * invdV -
                                           geom_source - source_terms) *
                                          dt * step;
                        break;
                    }
            }   // end switch
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
                const real dV    = get_cell_volume(xag - 1, jreal);
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
                    const real dV    = get_cell_volume(xag - 1, jreal, kreal);
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
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs.push_back(boundary_cond_map.at(boundary_conditions[i]));
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

    // Write some info about the setup for writeup later
    setup.x1max = x1[xag - 1];
    setup.x1min = x1[0];
    setup.x1    = x1;
    if constexpr (dim > 1) {
        setup.x2max = x2[yag - 1];
        setup.x2min = x2[0];
        setup.x2    = x2;
    }
    if constexpr (dim > 2) {
        setup.x3max = x3[zag - 1];
        setup.x3min = x3[0];
        setup.x3    = x3;
    }
    setup.nx              = nx;
    setup.ny              = ny;
    setup.nz              = nz;
    setup.xactive_zones   = xag;
    setup.yactive_zones   = yag;
    setup.zactive_zones   = zag;
    setup.x1_cell_spacing = cell2str.at(x1_cell_spacing);
    setup.x2_cell_spacing = cell2str.at(x2_cell_spacing);
    setup.x3_cell_spacing = cell2str.at(x3_cell_spacing);
    setup.ad_gamma        = gamma;
    setup.spatial_order   = spatial_order;
    setup.time_order      = time_order;
    setup.coord_system    = coord_system;
    setup.using_fourvelocity =
        (global::VelocityType == global::Velocity::FourVelocity);
    setup.regime              = "srmhd";
    setup.mesh_motion         = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions          = dim;

    // allocate space for face-centered magnetic fields
    bstag1.resize(nxv * yag * zag);
    bstag2.resize(xag * nyv * zag);
    bstag3.resize(xag * yag * nzv);
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
        const real bsq = (b1 * b1 + b2 * b2 + b3 * b3);
        const real msq = (m1 * m1 + m2 * m2 + b3 * m3);
        const real et  = tau + d;
        const real a   = 3.0;
        const real b   = -4.0 * (et - bsq);
        const real c   = msq - 2.0 * et * bsq + bsq * bsq;
        const real qq  = (-b + std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
        edens_guess[i] = std::max(qq - d, d);
        cons[i]        = conserved_t{d, m1, m2, m3, tau, b1, b2, b3, dchi};
    }

    // Deallocate duplicate memory and setup the system
    deallocate_state();
    offload();
    compute_bytes_and_strides<primitive_t>(dim);
    print_shared_mem();

    cons2prim(fullP);
    if constexpr (global::on_gpu) {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>(fullP);
    }
    else {
        adapt_dt<TIMESTEP_TYPE::MINIMUM>();
    }

    // Choose the Riemann solver
    switch (sim_solver) {
        case Solver::HLLE:
            this->riemann_solve = &RMHD<dim>::calc_hll_flux;
            break;
        case Solver::HLLC:
            this->riemann_solve = &RMHD<dim>::calc_hllc_flux;
            break;
        default:
            // TODO: implement hlld
            inFailureState = true;
            // this->riemann_solve = &RMHD<dim>::calc_hlld_flux;
            break;
    }

    // Using a sigmoid decay function to represent when the source terms
    // turn off.
    time_constant = sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) {
        write_to_file(*this);
        if constexpr (dim == 1) {
            config_ghosts1D(
                fullP,
                cons.data(),
                nx,
                spatial_order == "pcm",
                bcs.data(),
                outer_zones.data(),
                inflow_zones.data()
            );
        }
        else if constexpr (dim == 2) {
            config_ghosts2D(
                fullP,
                cons.data(),
                nx,
                ny,
                spatial_order == "pcm",
                geometry,
                bcs.data(),
                outer_zones.data(),
                inflow_zones.data(),
                half_sphere
            );
        }
        else {
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
    }
    // Simulate :)
    try {
        simbi::detail::logger::with_logger(*this, tend, [&] {
            advance(activeP);
            cons2prim(fullP);
            if constexpr (dim == 1) {
                config_ghosts1D(
                    fullP,
                    cons.data(),
                    nx,
                    spatial_order == "pcm",
                    bcs.data(),
                    outer_zones.data(),
                    inflow_zones.data()
                );
            }
            else if constexpr (dim == 2) {
                config_ghosts2D(
                    fullP,
                    cons.data(),
                    nx,
                    ny,
                    spatial_order == "pcm",
                    geometry,
                    bcs.data(),
                    outer_zones.data(),
                    inflow_zones.data(),
                    half_sphere
                );
            }
            else {
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
        std::cout << std::string(80, '=') << "\n";
        std::cerr << e.what() << '\n';
        std::cout << std::string(80, '=') << "\n";
        troubled_cells.copyFromGpu();
        cons.copyFromGpu();
        prims.copyFromGpu();
        hasCrashed = true;
        write_to_file(*this);
        emit_troubled_cells();
    }
};
