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
    const InitialConditions& init_conditions
)
    : HydroBase(state, init_conditions)
{
}

// Destructor
template <int dim>
SRHD<dim>::~SRHD() = default;

// Helpers
template <int dim>
HD constexpr real SRHD<dim>::get_x1face(const lint ii, const int side) const
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
HD constexpr real SRHD<dim>::get_x2face(const lint ii, const int side) const
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
HD constexpr real SRHD<dim>::get_x3face(const lint ii, const int side) const
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

template <int dim>
HD constexpr real SRHD<dim>::get_x1_differential(const lint ii) const
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
HD constexpr real SRHD<dim>::get_x2_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return 2.0;
            default:
                return 2.0 * M_PI;
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
HD constexpr real SRHD<dim>::get_x3_differential(const lint ii) const
{
    if constexpr (dim == 1) {
        switch (geometry) {
            case Geometry::SPHERICAL:
                return 2.0 * M_PI;
            default:
                return 1.0;
        }
    }
    else if constexpr (dim == 2) {
        switch (geometry) {
            case Geometry::PLANAR_CYLINDRICAL:
                return 1.0;
            default:
                return 2.0 * M_PI;
        }
    }
    else {
        return dx3;
    }
}

template <int dim>
HD real
SRHD<dim>::get_cell_volume(const lint ii, const lint jj, const lint kk) const
{
    // the volume in cartesian coordinates is only nominal
    if (geometry == Geometry::CARTESIAN) {
        return 1.0;
    }
    return get_x1_differential(ii) * get_x2_differential(jj) *
           get_x3_differential(kk);
}

template <int dim>
void SRHD<dim>::emit_troubled_cells() const
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
            const real s1     = cons[gid].momentum(1);
            const real s2     = cons[gid].momentum(2);
            const real s3     = cons[gid].momentum(3);
            const real et     = (cons[gid].den + cons[gid].nrg + prims[gid].p);
            const real s      = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);
            const real v2     = (s * s) / (et * et);
            const real w      = 1.0 / std::sqrt(1.0 - v2);
            if constexpr (dim == 1) {
                fprintf(
                    stderr,
                    "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, iter: %" PRIu64 "\n",
                    cons[gid].den / w,
                    prims[gid].p,
                    v2,
                    x1mean,
                    n
                );
            }
            else if constexpr (dim == 2) {
                fprintf(
                    stderr,
                    "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, iter: "
                    "%" PRIu64 "\n",
                    cons[gid].den / w,
                    prims[gid].p,
                    v2,
                    x1mean,
                    x2mean,
                    n
                );
            }
            else {
                fprintf(
                    stderr,
                    "\nCons2Prim cannot converge\nDensity: %.2e, Pressure: "
                    "%.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, "
                    "x3coord: %.2e, iter: %" PRIu64 "\n",
                    cons[gid].den / w,
                    prims[gid].p,
                    v2,
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
void SRHD<dim>::cons2prim(const ExecutionPolicy<>& p)
{
    const auto* const cons_data = cons.data();
    auto* const prim_data       = prims.data();
    auto* const press_data      = pressure_guess.data();
    auto* const troubled_data   = troubled_cells.data();
    simbi::parallel_for(
        p,
        total_zones,
        [prim_data, cons_data, press_data, troubled_data, this] DEV(luint gid) {
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
                const real s1   = cons_data[gid].momentum(1) * invdV;
                const real s2   = cons_data[gid].momentum(2) * invdV;
                const real s3   = cons_data[gid].momentum(3) * invdV;
                const real tau  = cons_data[gid].nrg * invdV;
                const real dchi = cons_data[gid].chi * invdV;
                const real s    = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);

                // Perform modified Newton Raphson based on
                // https://www.sciencedirect.com/science/article/pii/S0893965913002930
                // so far, the convergence rate is the same, but perhaps I need
                // a slight tweak

                // compute f(x_0)
                // f = newton_f(gamma, tau, d, S, peq);
                int iter       = 0;
                real peq       = press_data[gid];
                const real tol = d * global::tol_scale;
                real f, g;
                do {
                    // compute x_[k+1]
                    f = newton_f(gamma, tau, d, s, peq);
                    g = newton_g(gamma, tau, d, s, peq);
                    peq -= f / g;

                    // compute x*_k
                    // f     = newton_f(gamma, tau, d, S, peq);
                    // pstar = peq - f / g;

                    if (iter >= global::MAX_ITER || std::isnan(peq)) {
                        troubled_data[gid] = 1;
                        dt                 = INFINITY;
                        inFailureState     = true;
                        found_failure      = true;
                        break;
                    }
                    iter++;

                } while (std::abs(f / g) >= tol);

                const real inv_et = 1.0 / (tau + d + peq);
                const real v1     = s1 * inv_et;
                press_data[gid]   = peq;
#if FOUR_VELOCITY
                if constexpr (dim == 1) {
                    const real w   = 1.0 / std::sqrt(1.0 - v1 * v1);
                    prim_data[gid] = {d / w, v1 * w, peq, dchi / d};
                }
                else if constexpr (dim == 2) {
                    const real v2  = s2 * inv_et;
                    const real w   = 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2));
                    prim_data[gid] = {d / w, v1 * w, v2 * w, peq, dchi / d};
                }
                else {
                    const real v2 = s2 * inv_et;
                    const real v3 = s3 * inv_et;
                    const real w =
                        1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
                    prim_data[gid] =
                        {d / w, v1 * w, v2 * w, v3 * w, peq, dchi / d};
                }
#else
                if constexpr (dim == 1) {
                    const real w   = 1.0 / std::sqrt(1.0 - (v1 * v1));
                    prim_data[gid] = {d / w, v1, peq, dchi / d};
                }
                else if constexpr (dim == 2) {
                    const real v2  = s2 * inv_et;
                    const real w   = 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2));
                    prim_data[gid] = {d / w, v1, v2, peq, dchi / d};
                }
                else {
                    const real v2 = s2 * inv_et;
                    const real v3 = s3 * inv_et;
                    const real w =
                        1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
                    prim_data[gid] = {d / w, v1, v2, v3, peq, dchi / d};
                }
#endif
                workLeftToDo = false;

                if (peq < 0) {
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

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template <int dim>
HD SRHD<dim>::eigenvals_t SRHD<dim>::calc_eigenvals(
    const SRHD<dim>::primitive_t& primsL,
    const SRHD<dim>::primitive_t& primsR,
    const luint nhat
) const
{
    // Separate the left and right Primitive
    const real rhoL = primsL.rho;
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p;
    const real hL   = primsL.get_enthalpy(gamma);

    const real rhoR = primsR.rho;
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.p;
    const real hR   = primsR.get_enthalpy(gamma);

    const real csR = std::sqrt(gamma * pR / (hR * rhoR));
    const real csL = std::sqrt(gamma * pL / (hL * rhoL));

    switch (comp_wave_speed) {
        //-----------Calculate wave speeds based on Schneider et al. 1993
        case simbi::WaveSpeeds::SCHNEIDER_ET_AL_93:
            {
                const real vbar = 0.5 * (vL + vR);
                const real cbar = 0.5 * (csL + csR);
                const real bl   = (vbar - cbar) / (1.0 - cbar * vbar);
                const real br   = (vbar + cbar) / (1.0 + cbar * vbar);
                const real aL   = my_min(bl, (vL - csL) / (1.0 - vL * csL));
                const real aR   = my_max(br, (vR + csR) / (1.0 + vR * csR));

                return {aL, aR, csL, csR};
            }
        //-----------Calculate wave speeds based on Mignone & Bodo 2005
        case simbi::WaveSpeeds::MIGNONE_AND_BODO_05:
            {
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
        case simbi::WaveSpeeds::HUBER_AND_KISSMANN_2021:
            {
                const real gammaL = 1.0 / std::sqrt(1.0 - (vL * vL));
                const real gammaR = 1.0 / std::sqrt(1.0 - (vR * vR));
                const real uL     = gammaL * vL;
                const real uR     = gammaR * vR;
                const real sL     = csL * csL / (1.0 - csL * csL);
                const real sR     = csR * csR / (1.0 - csR * csR);
                const real sqrtR =
                    std::sqrt(sR * (gammaR * gammaR - uR * uR + sR));
                const real sqrtL =
                    std::sqrt(sL * (gammaL * gammaL - uL * uL + sL));
                const real qfL = 1.0 / (gammaL * gammaL + sL);
                const real qfR = 1.0 / (gammaR * gammaR + sR);

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

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template <int dim>
HD SRHD<dim>::conserved_t
SRHD<dim>::prims2cons(const SRHD<dim>::primitive_t& prims) const
{
    const real rho      = prims.rho;
    const real v1       = prims.vcomponent(1);
    const real v2       = prims.vcomponent(2);
    const real v3       = prims.vcomponent(3);
    const real pressure = prims.p;
    const real lf       = prims.lorentz_factor();
    const real h        = prims.get_enthalpy(gamma);
    const real d        = rho * lf;
    const real ed       = d * lf * h;
    if constexpr (dim == 1) {
        return {d, ed * v1, ed - pressure - d};
    }
    else if constexpr (dim == 2) {
        return {d, ed * v1, ed * v2, ed - pressure - d};
    }
    else {
        return {d, ed * v1, ed * v2, ed * v3, ed - pressure - d};
    }
};

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditional time step
template <int dim>
template <TIMESTEP_TYPE dt_type>
void SRHD<dim>::adapt_dt()
{
    // singleton instance of thread pool. lazy-evaluated
    static auto& thread_pool =
        simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(total_zones, [&](luint gid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
        const luint kk    = axid<dim, BlkAx::K>(gid, nx, ny);
        const luint jj    = axid<dim, BlkAx::J>(gid, nx, ny, kk);
        const luint ii    = axid<dim, BlkAx::I>(gid, nx, ny, kk);
        const luint ireal = get_real_idx(ii, radius, xag);
        // Left/Right wave speeds
        if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
            const real rho = prims[gid].rho;
            const real v1  = prims[gid].vcomponent(1);
            const real v2  = prims[gid].vcomponent(2);
            const real v3  = prims[gid].vcomponent(3);
            const real pre = prims[gid].p;
            const real h   = prims[gid].get_enthalpy(gamma);
            const real cs  = std::sqrt(gamma * pre / (rho * h));
            v1p            = std::abs(v1 + cs) / (1.0 + v1 * cs);
            v1m            = std::abs(v1 - cs) / (1.0 - v1 * cs);
            if constexpr (dim > 1) {
                v2p = std::abs(v2 + cs) / (1.0 + v2 * cs);
                v2m = std::abs(v2 - cs) / (1.0 - v2 * cs);
            }
            if constexpr (dim > 2) {
                v3p = std::abs(v3 + cs) / (1.0 + v3 * cs);
                v3m = std::abs(v3 - cs) / (1.0 - v3 * cs);
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
                    cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
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
                        cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
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
                        cfl_dt = std::min({dx1 / (std::max(v1p, v1m))});
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
void SRHD<dim>::adapt_dt(const ExecutionPolicy<>& p)
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
HD SRHD<dim>::conserved_t
SRHD<dim>::prims2flux(const SRHD<dim>::primitive_t& prims, const luint nhat)
    const
{
    const real rho      = prims.rho;
    const real v1       = prims.vcomponent(1);
    const real v2       = prims.vcomponent(2);
    const real v3       = prims.vcomponent(3);
    const real pressure = prims.p;
    const real chi      = prims.chi;
    const real vn       = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
    const real lf       = prims.lorentz_factor();

    const real h  = prims.get_enthalpy(gamma);
    const real d  = rho * lf;
    const real ed = d * lf * h;
    const real s1 = ed * v1;
    const real s2 = ed * v2;
    const real s3 = ed * v3;
    const real mn = (nhat == 1) ? s1 : (nhat == 2) ? s2 : s3;
    if constexpr (dim == 1) {
        return {
          d * vn,
          s1 * vn + kronecker(nhat, 1) * pressure,
          mn - d * vn,
          d * vn * chi
        };
    }
    else if constexpr (dim == 2) {
        return {
          d * vn,
          s1 * vn + kronecker(nhat, 1) * pressure,
          s2 * vn + kronecker(nhat, 2) * pressure,
          mn - d * vn,
          d * vn * chi
        };
    }
    else {
        return {
          d * vn,
          s1 * vn + kronecker(nhat, 1) * pressure,
          s2 * vn + kronecker(nhat, 2) * pressure,
          s3 * vn + kronecker(nhat, 3) * pressure,
          mn - d * vn,
          d * vn * chi
        };
    }
};

template <int dim>
HD SRHD<dim>::conserved_t SRHD<dim>::calc_hll_flux(
    const SRHD<dim>::conserved_t& uL,
    const SRHD<dim>::conserved_t& uR,
    const SRHD<dim>::conserved_t& fL,
    const SRHD<dim>::conserved_t& fR,
    const SRHD<dim>::primitive_t& prL,
    const SRHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    // Grab the necessary wave speeds
    const real aL  = lambda.aL;
    const real aR  = lambda.aR;
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

    // Compute the HLL Flux component-wise
    return net_flux;
};

template <int dim>
HD SRHD<dim>::conserved_t SRHD<dim>::calc_hllc_flux(
    const SRHD<dim>::conserved_t& uL,
    const SRHD<dim>::conserved_t& uR,
    const SRHD<dim>::conserved_t& fL,
    const SRHD<dim>::conserved_t& fR,
    const SRHD<dim>::primitive_t& prL,
    const SRHD<dim>::primitive_t& prR,
    const luint nhat,
    const real vface
) const
{
    const auto lambda = calc_eigenvals(prL, prR, nhat);
    const real aL     = lambda.aL;
    const real aR     = lambda.aR;
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
        if (quirk_strong_shock(prL.p, prR.p)) {
            return hll_flux;
        }
    }
    const real uhlld   = hll_state.den;
    const real uhlls1  = hll_state.momentum(1);
    const real uhlls2  = hll_state.momentum(2);
    const real uhlls3  = hll_state.momentum(3);
    const real uhlltau = hll_state.nrg;
    const real fhlld   = hll_flux.den;
    const real fhlls1  = hll_flux.momentum(1);
    const real fhlls2  = hll_flux.momentum(2);
    const real fhlls3  = hll_flux.momentum(3);
    const real fhlltau = hll_flux.nrg;
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
            const real v        = prL.get_v();
            const real pressure = prL.p;
            const real d        = uL.den;
            const real s        = uL.m1;
            const real tau      = uL.nrg;
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
            const real v        = prR.get_v();
            const real pressure = prR.p;
            const real d        = uR.den;
            const real s        = uR.m1;
            const real tau      = uR.nrg;
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
            case HLLCTYPE::FLEISCHMANN:
                {
                    // Apply the low-Mach HLLC fix found in Fleischmann et al
                    // 2020:
                    // https://www.sciencedirect.com/science/article/pii/S0021999120305362
                    const real csL        = lambda.csL;
                    const real csR        = lambda.csR;
                    constexpr real ma_lim = 5.0;

                    // --------------Compute the L Star State----------
                    real pressure = prL.p;
                    real d        = uL.den;
                    real s1       = uL.momentum(1);
                    real s2       = uL.momentum(2);
                    real s3       = uL.momentum(3);
                    real tau      = uL.nrg;
                    real e        = tau + d;
                    real cofactor = 1.0 / (aL - aStar);

                    const real vL = prL.vcomponent(nhat);
                    const real vR = prR.vcomponent(nhat);
                    // Left Star State in x-direction of coordinate lattice
                    real dStar = cofactor * (aL - vL) * d;
                    real s1star =
                        cofactor * (s1 * (aL - vL) +
                                    kronecker(nhat, 1) * (-pressure + pStar));
                    real s2star =
                        cofactor * (s2 * (aL - vL) +
                                    kronecker(nhat, 2) * (-pressure + pStar));
                    real s3star =
                        cofactor * (s3 * (aL - vL) +
                                    kronecker(nhat, 3) * (-pressure + pStar));
                    real eStar   = cofactor * (e * (aL - vL) + pStar * aStar -
                                             pressure * vL);
                    real tauStar = eStar - dStar;
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

                    pressure = prR.p;
                    d        = uR.den;
                    s1       = uR.momentum(1);
                    s2       = uR.momentum(2);
                    s3       = uR.momentum(3);
                    tau      = uR.nrg;
                    e        = tau + d;
                    cofactor = 1.0 / (aR - aStar);

                    dStar = cofactor * (aR - vR) * d;
                    s1star =
                        cofactor * (s1 * (aR - vR) +
                                    kronecker(nhat, 1) * (-pressure + pStar));
                    s2star =
                        cofactor * (s2 * (aR - vR) +
                                    kronecker(nhat, 2) * (-pressure + pStar));
                    s3star =
                        cofactor * (s3 * (aR - vR) +
                                    kronecker(nhat, 3) * (-pressure + pStar));
                    eStar = cofactor *
                            (e * (aR - vR) + pStar * aStar - pressure * vR);
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
                        vL / csL *
                        std::sqrt((1.0 - csL * csL) / (1.0 - vL * vL));
                    const real ma_right =
                        vR / csR *
                        std::sqrt((1.0 - csR * csR) / (1.0 - vR * vR));
                    const real ma_local =
                        my_max(std::abs(ma_left), std::abs(ma_right));
                    const real phi = std::sin(
                        my_min<real>(1, ma_local / ma_lim) * M_PI * 0.5
                    );
                    const real aL_lm = phi == 0 ? aL : phi * aL;
                    const real aR_lm = phi == 0 ? aR : phi * aR;

                    const auto face_starState =
                        (aStar <= 0) ? starStateR : starStateL;
                    auto net_flux =
                        (fL + fR) * 0.5 +
                        ((starStateL - uL) * aL_lm +
                         (starStateL - starStateR) * std::abs(aStar) +
                         (starStateR - uR) * aR_lm) *
                            0.5 -
                        face_starState * vface;

                    // upwind the concentration
                    if (net_flux.den < 0.0) {
                        net_flux.chi = prR.chi * net_flux.den;
                    }
                    else {
                        net_flux.chi = prL.chi * net_flux.den;
                    }
                    return net_flux;
                }

            default:
                {
                    if (vface <= aStar) {
                        const real pressure = prL.p;
                        const real d        = uL.den;
                        const real s1       = uL.momentum(1);
                        const real s2       = uL.momentum(2);
                        const real s3       = uL.momentum(3);
                        const real tau      = uL.nrg;
                        const real chi      = uL.chi;
                        const real e        = tau + d;
                        const real cofactor = 1.0 / (aL - aStar);

                        const real vL = prL.vcomponent(nhat);
                        // Left Star State in x-direction of coordinate lattice
                        const real dStar   = cofactor * (aL - vL) * d;
                        const real chistar = cofactor * (aL - vL) * chi;
                        const real s1star =
                            cofactor *
                            (s1 * (aL - vL) +
                             kronecker(nhat, 1) * (-pressure + pStar));
                        const real s2star =
                            cofactor *
                            (s2 * (aL - vL) +
                             kronecker(nhat, 2) * (-pressure + pStar));
                        const real s3star =
                            cofactor *
                            (s3 * (aL - vL) +
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
                        if (hllc_flux.den < 0.0) {
                            hllc_flux.chi = prR.chi * hllc_flux.den;
                        }
                        else {
                            hllc_flux.chi = prL.chi * hllc_flux.den;
                        }

                        return hllc_flux;
                    }
                    else {
                        const real pressure = prR.p;
                        const real d        = uR.den;
                        const real s1       = uR.momentum(1);
                        const real s2       = uR.momentum(2);
                        const real s3       = uR.momentum(3);
                        const real tau      = uR.nrg;
                        const real chi      = uR.chi;
                        const real e        = tau + d;
                        const real cofactor = 1.0 / (aR - aStar);

                        const real vR      = prR.vcomponent(nhat);
                        const real dStar   = cofactor * (aR - vR) * d;
                        const real chistar = cofactor * (aR - vR) * chi;
                        const real s1star =
                            cofactor *
                            (s1 * (aR - vR) +
                             kronecker(nhat, 1) * (-pressure + pStar));
                        const real s2star =
                            cofactor *
                            (s2 * (aR - vR) +
                             kronecker(nhat, 2) * (-pressure + pStar));
                        const real s3star =
                            cofactor *
                            (s3 * (aR - vR) +
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
                        if (hllc_flux.den < 0.0) {
                            hllc_flux.chi = prR.chi * hllc_flux.den;
                        }
                        else {
                            hllc_flux.chi = prL.chi * hllc_flux.den;
                        }

                        return hllc_flux;
                    }
                }
        }   // end switch
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template <int dim>
void SRHD<dim>::advance(const ExecutionPolicy<>& p)
{
    const luint extent            = p.get_full_extent();
    auto* const cons_data         = cons.data();
    const auto* const prim_data   = prims.data();
    const auto* const dens_source = density_source.data();
    const auto* const mom1_source = m1_source.data();
    const auto* const mom2_source = m2_source.data();
    const auto* const mom3_source = m3_source.data();
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
         erg_source,
         object_data,
         g1_source,
         g2_source,
         g3_source,
         this] DEV(const luint idx) {
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

            conserved_t uxL, uxR, fL, fR, flf, frf;
            primitive_t xprimsL, xprimsR;
            // Compiler optimizes these out if unused since they are of trivial
            // type
            [[maybe_unused]] conserved_t uyL, uyR, gL, gR, glf, grf;
            [[maybe_unused]] conserved_t uzL, uzR, hL, hR, hlf, hrf;
            [[maybe_unused]] primitive_t yprimsL, yprimsR;
            [[maybe_unused]] primitive_t zprimsL, zprimsR;

            const luint aid = ka * nx * ny + ja * nx + ia;
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
                // cast away lambda capture;
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

            if (use_pcm) [[unlikely]] {
                xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
                xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
                if constexpr (dim > 1) {
                    // j+1/2
                    yprimsL = prim_buff[tza * sx * sy + (tya + 0) * sx + txa];
                    yprimsR = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
                }
                if constexpr (dim > 2) {
                    // k+1/2
                    zprimsL = prim_buff[(tza + 0) * sx * sy + tya * sx + txa];
                    zprimsR = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
                }

                ib_modify<dim>(xprimsR, xprimsL, object_to_right, 1);
                ib_modify<dim>(yprimsR, yprimsL, object_in_front, 2);
                ib_modify<dim>(zprimsR, zprimsL, object_above, 3);

                uxL = prims2cons(xprimsL);
                uxR = prims2cons(xprimsR);
                if constexpr (dim > 1) {
                    uyL = prims2cons(yprimsL);
                    uyR = prims2cons(yprimsR);
                }
                if constexpr (dim > 2) {
                    uzL = prims2cons(zprimsL);
                    uzR = prims2cons(zprimsR);
                }

                fL = prims2flux(xprimsL, 1);
                fR = prims2flux(xprimsR, 1);
                if constexpr (dim > 1) {
                    gL = prims2flux(yprimsL, 2);
                    gR = prims2flux(yprimsR, 2);
                }
                if constexpr (dim > 2) {
                    hL = prims2flux(zprimsL, 3);
                    hR = prims2flux(zprimsR, 3);
                }
                // Calc HLL Flux at i+1/2 interface
                switch (sim_solver) {
                    case Solver::HLLC:
                        if constexpr (dim == 1) {
                            frf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceR
                            );
                            break;
                        }
                        else {
                            frf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceR
                            );
                            grf = calc_hllc_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2
                            );

                            if constexpr (dim > 2) {
                                hrf = calc_hllc_flux(
                                    uzL,
                                    uzR,
                                    hL,
                                    hR,
                                    zprimsL,
                                    zprimsR,
                                    3
                                );
                            }
                            break;
                        }
                    default:
                        frf = calc_hll_flux(
                            uxL,
                            uxR,
                            fL,
                            fR,
                            xprimsL,
                            xprimsR,
                            1,
                            vfaceR
                        );
                        if constexpr (dim > 1) {
                            grf = calc_hll_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2
                            );
                        }
                        if constexpr (dim > 2) {
                            hrf = calc_hll_flux(
                                uzL,
                                uzR,
                                hL,
                                hR,
                                zprimsL,
                                zprimsR,
                                3
                            );
                        }
                        break;
                }

                // Set up the left and right state interfaces for i-1/2
                xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
                xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
                if constexpr (dim > 1) {
                    // j+1/2
                    yprimsL = prim_buff[tza * sx * sy + (tya - 1) * sx + txa];
                    yprimsR = prim_buff[tza * sx * sy + (tya + 0) * sx + txa];
                }
                if constexpr (dim > 2) {
                    // k+1/2
                    zprimsL = prim_buff[(tza - 1) * sx * sy + tya * sx + txa];
                    zprimsR = prim_buff[(tza - 0) * sx * sy + tya * sx + txa];
                }

                ib_modify<dim>(xprimsL, xprimsR, object_to_left, 1);
                ib_modify<dim>(yprimsL, yprimsR, object_behind, 2);
                ib_modify<dim>(zprimsL, zprimsR, object_below, 3);

                uxL = prims2cons(xprimsL);
                uxR = prims2cons(xprimsR);
                if constexpr (dim > 1) {
                    uyL = prims2cons(yprimsL);
                    uyR = prims2cons(yprimsR);
                }
                if constexpr (dim > 2) {
                    uzL = prims2cons(zprimsL);
                    uzR = prims2cons(zprimsR);
                }
                fL = prims2flux(xprimsL, 1);
                fR = prims2flux(xprimsR, 1);
                if constexpr (dim > 1) {
                    gL = prims2flux(yprimsL, 2);
                    gR = prims2flux(yprimsR, 2);
                }
                if constexpr (dim > 2) {
                    hL = prims2flux(zprimsL, 3);
                    hR = prims2flux(zprimsR, 3);
                }

                // Calc HLL Flux at i-1/2 interface
                switch (sim_solver) {
                    case Solver::HLLC:
                        if constexpr (dim == 1) {
                            flf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceL
                            );
                            break;
                        }
                        else {
                            flf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceL
                            );
                            glf = calc_hllc_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2
                            );

                            if constexpr (dim > 2) {
                                hlf = calc_hllc_flux(
                                    uzL,
                                    uzR,
                                    hL,
                                    hR,
                                    zprimsL,
                                    zprimsR,
                                    3
                                );
                            }
                            break;
                        }
                    default:
                        flf = calc_hll_flux(
                            uxL,
                            uxR,
                            fL,
                            fR,
                            xprimsL,
                            xprimsR,
                            1,
                            vfaceL
                        );
                        if constexpr (dim > 1) {
                            glf = calc_hll_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2
                            );
                        }
                        if constexpr (dim > 2) {
                            hlf = calc_hll_flux(
                                uzL,
                                uzR,
                                hL,
                                hR,
                                zprimsL,
                                zprimsR,
                                3
                            );
                        }
                        break;
                }
            }
            else {
                // Coordinate X
                const primitive_t xlm =
                    prim_buff[tza * sx * sy + tya * sx + (txa - 2)];
                const primitive_t xlc =
                    prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
                const primitive_t center =
                    prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
                const primitive_t xrc =
                    prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
                const primitive_t xrm =
                    prim_buff[tza * sx * sy + tya * sx + (txa + 2)];
                primitive_t ylm, ylc, yrc, yrm;
                primitive_t zlm, zlc, zrc, zrm;
                // Reconstructed left X primitive_t vector at the i+1/2
                // interface
                xprimsL =
                    center + plm_gradient(center, xlc, xrc, plm_theta) * 0.5;
                xprimsR = xrc - plm_gradient(xrc, center, xrm, plm_theta) * 0.5;

                // Coordinate Y
                if constexpr (dim > 1) {
                    ylm     = prim_buff[tza * sx * sy + (tya - 2) * sx + txa];
                    ylc     = prim_buff[tza * sx * sy + (tya - 1) * sx + txa];
                    yrc     = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
                    yrm     = prim_buff[tza * sx * sy + (tya + 2) * sx + txa];
                    yprimsL = center +
                              plm_gradient(center, ylc, yrc, plm_theta) * 0.5;
                    yprimsR =
                        yrc - plm_gradient(yrc, center, yrm, plm_theta) * 0.5;
                }

                // Coordinate z
                if constexpr (dim > 2) {
                    zlm     = prim_buff[(tza - 2) * sx * sy + tya * sx + txa];
                    zlc     = prim_buff[(tza - 1) * sx * sy + tya * sx + txa];
                    zrc     = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
                    zrm     = prim_buff[(tza + 2) * sx * sy + tya * sx + txa];
                    zprimsL = center +
                              plm_gradient(center, zlc, zrc, plm_theta) * 0.5;
                    zprimsR =
                        zrc - plm_gradient(zrc, center, zrm, plm_theta) * 0.5;
                }

                ib_modify<dim>(xprimsR, xprimsL, object_to_right, 1);
                ib_modify<dim>(yprimsR, yprimsL, object_in_front, 2);
                ib_modify<dim>(zprimsR, zprimsL, object_above, 3);

                // Calculate the left and right states using the reconstructed
                // PLM Primitive
                uxL = prims2cons(xprimsL);
                uxR = prims2cons(xprimsR);
                if constexpr (dim > 1) {
                    uyL = prims2cons(yprimsL);
                    uyR = prims2cons(yprimsR);
                }
                if constexpr (dim > 2) {
                    uzL = prims2cons(zprimsL);
                    uzR = prims2cons(zprimsR);
                }

                fL = prims2flux(xprimsL, 1);
                fR = prims2flux(xprimsR, 1);
                if constexpr (dim > 1) {
                    gL = prims2flux(yprimsL, 2);
                    gR = prims2flux(yprimsR, 2);
                }
                if constexpr (dim > 2) {
                    hL = prims2flux(zprimsL, 3);
                    hR = prims2flux(zprimsR, 3);
                }

                switch (sim_solver) {
                    case Solver::HLLC:
                        if constexpr (dim == 1) {
                            frf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceR
                            );
                            break;
                        }
                        else {
                            frf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceR
                            );
                            grf = calc_hllc_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2,
                                0
                            );

                            if constexpr (dim > 2) {
                                hrf = calc_hllc_flux(
                                    uzL,
                                    uzR,
                                    hL,
                                    hR,
                                    zprimsL,
                                    zprimsR,
                                    3,
                                    0
                                );
                            }
                            break;
                        }
                    default:
                        frf = calc_hll_flux(
                            uxL,
                            uxR,
                            fL,
                            fR,
                            xprimsL,
                            xprimsR,
                            1,
                            vfaceR
                        );
                        if constexpr (dim > 1) {
                            grf = calc_hll_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2,
                                0
                            );
                        }
                        if constexpr (dim > 2) {
                            hrf = calc_hll_flux(
                                uzL,
                                uzR,
                                hL,
                                hR,
                                zprimsL,
                                zprimsR,
                                3,
                                0
                            );
                        }
                        break;
                }

                // Do the same thing, but for the left side interface [i - 1/2]
                xprimsL = xlc + plm_gradient(xlc, xlm, center, plm_theta) * 0.5;
                xprimsR =
                    center - plm_gradient(center, xlc, xrc, plm_theta) * 0.5;
                if constexpr (dim > 1) {
                    yprimsL =
                        ylc + plm_gradient(ylc, ylm, center, plm_theta) * 0.5;
                    yprimsR = center -
                              plm_gradient(center, ylc, yrc, plm_theta) * 0.5;
                }
                if constexpr (dim > 2) {
                    zprimsL =
                        zlc + plm_gradient(zlc, zlm, center, plm_theta) * 0.5;
                    zprimsR = center -
                              plm_gradient(center, zlc, zrc, plm_theta) * 0.5;
                }

                ib_modify<dim>(xprimsL, xprimsR, object_to_left, 1);
                ib_modify<dim>(yprimsL, yprimsR, object_behind, 2);
                ib_modify<dim>(zprimsL, zprimsR, object_below, 3);

                // Calculate the left and right states using the reconstructed
                // PLM Primitive
                uxL = prims2cons(xprimsL);
                uxR = prims2cons(xprimsR);
                if constexpr (dim > 1) {
                    uyL = prims2cons(yprimsL);
                    uyR = prims2cons(yprimsR);
                }
                if constexpr (dim > 2) {
                    uzL = prims2cons(zprimsL);
                    uzR = prims2cons(zprimsR);
                }
                fL = prims2flux(xprimsL, 1);
                fR = prims2flux(xprimsR, 1);
                if constexpr (dim > 1) {
                    gL = prims2flux(yprimsL, 2);
                    gR = prims2flux(yprimsR, 2);
                }
                if constexpr (dim > 2) {
                    hL = prims2flux(zprimsL, 3);
                    hR = prims2flux(zprimsR, 3);
                }

                switch (sim_solver) {
                    case Solver::HLLC:
                        if constexpr (dim == 1) {
                            flf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceL
                            );
                            break;
                        }
                        else {
                            flf = calc_hllc_flux(
                                uxL,
                                uxR,
                                fL,
                                fR,
                                xprimsL,
                                xprimsR,
                                1,
                                vfaceL
                            );
                            glf = calc_hllc_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2,
                                0
                            );

                            if constexpr (dim > 2) {
                                hlf = calc_hllc_flux(
                                    uzL,
                                    uzR,
                                    hL,
                                    hR,
                                    zprimsL,
                                    zprimsR,
                                    3,
                                    0
                                );
                            }
                            break;
                        }

                    default:
                        flf = calc_hll_flux(
                            uxL,
                            uxR,
                            fL,
                            fR,
                            xprimsL,
                            xprimsR,
                            1,
                            vfaceL
                        );
                        if constexpr (dim > 1) {
                            glf = calc_hll_flux(
                                uyL,
                                uyR,
                                gL,
                                gR,
                                yprimsL,
                                yprimsR,
                                2,
                                0
                            );
                        }
                        if constexpr (dim > 2) {
                            hlf = calc_hll_flux(
                                uzL,
                                uzR,
                                hL,
                                hR,
                                zprimsL,
                                zprimsR,
                                3,
                                0
                            );
                        }
                        break;
                }
            }   // end else

            // Advance depending on geometry
            const luint real_loc = kk * xag * yag + jj * xag + ii;
            const real d_source  = null_den ? 0.0 : dens_source[real_loc];
            const real s1_source = null_mom1 ? 0.0 : mom1_source[real_loc];
            const real e_source  = null_nrg ? 0.0 : erg_source[real_loc];

            const auto source_terms = [&] {
                if constexpr (dim == 1) {
                    // cast away unused lambda capture
                    (void) mom2_source;
                    (void) mom3_source;
                    return conserved_t{d_source, s1_source, e_source} *
                           time_constant;
                }
                else if constexpr (dim == 2) {
                    // cast away unused lambda capture
                    (void) mom3_source;
                    const real s2_source =
                        null_mom2 ? 0.0 : mom2_source[real_loc];
                    return conserved_t{
                             d_source,
                             s1_source,
                             s2_source,
                             e_source
                           } *
                           time_constant;
                }
                else {
                    const real s2_source =
                        null_mom2 ? 0.0 : mom2_source[real_loc];
                    const real s3_source =
                        null_mom3 ? 0.0 : mom3_source[real_loc];
                    return conserved_t{
                             d_source,
                             s1_source,
                             s2_source,
                             s3_source,
                             e_source
                           } *
                           time_constant;
                }
            }();

            // Gravity
            const auto gs1_source =
                nullg1 ? 0 : g1_source[real_loc] * cons_data[aid].den;
            const auto tid     = tza * sx * sy + tya * sx + txa;
            const auto gravity = [&] {
                if constexpr (dim == 1) {
                    // cast away unused lambda capture
                    (void) g2_source;
                    (void) g3_source;
                    const auto ge_source = gs1_source * prim_buff[tid].v1;
                    return conserved_t{0.0, gs1_source, ge_source};
                }
                else if constexpr (dim == 2) {
                    // cast away unused lambda capture
                    (void) g3_source;
                    const auto gs2_source =
                        nullg2 ? 0 : g2_source[real_loc] * cons_data[aid].den;
                    const auto ge_source = gs1_source * prim_buff[tid].v1 +
                                           gs2_source * prim_buff[tid].v2;
                    return conserved_t{0.0, gs1_source, gs2_source, ge_source};
                }
                else {
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
                      ge_source
                    };
                }
            }();

            if constexpr (dim == 1) {
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        {
                            cons_data[ia] -= ((frf - flf) * invdx1 -
                                              source_terms - gravity) *
                                             dt * step;
                            break;
                        }
                    default:
                        {
                            const real rlf = x1l + vfaceL * step * dt;
                            const real rrf = x1r + vfaceR * step * dt;
                            const real rmean =
                                get_cell_centroid(rrf, rlf, geometry);
                            const real sR = 4.0 * M_PI * rrf * rrf;
                            const real sL = 4.0 * M_PI * rlf * rlf;
                            const real dV =
                                4.0 * M_PI * rmean * rmean * (rrf - rlf);
                            const real factor = (mesh_motion) ? dV : 1;
                            const real pc     = prim_buff[txa].p;
                            const real invdV  = 1.0 / dV;
                            const auto geom_sources =
                                conserved_t{0.0, pc * (sR - sL) * invdV, 0.0};

                            cons_data[ia] -=
                                ((frf * sR - flf * sL) * invdV - geom_sources -
                                 source_terms - gravity) *
                                step * dt * factor;
                            break;
                        }
                }   // end switch
            }
            else if constexpr (dim == 2) {
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        {
                            cons_data[aid] -=
                                ((frf - flf) * invdx1 + (grf - glf) * invdx2 -
                                 source_terms - gravity) *
                                step * dt;
                            break;
                        }

                    case simbi::Geometry::SPHERICAL:
                        {
                            const real rl = x1l + vfaceL * step * dt;
                            const real rr = x1r + vfaceR * step * dt;
                            const real rmean =
                                get_cell_centroid(rr, rl, geometry);
                            const real tl =
                                my_max<real>(x2min + (jj - 0.5) * dx2, x2min);
                            const real tr = my_min<real>(
                                tl + dx2 * (jj == 0 ? 0.5 : 1.0),
                                x2max
                            );
                            const real dcos = std::cos(tl) - std::cos(tr);
                            const real dV   = 2.0 * M_PI * (1.0 / 3.0) *
                                            (rr * rr * rr - rl * rl * rl) *
                                            dcos;
                            const real invdV = 1.0 / dV;
                            const real s1R   = 2.0 * M_PI * rr * rr * dcos;
                            const real s1L   = 2.0 * M_PI * rl * rl * dcos;
                            const real s2R   = 2.0 * M_PI * 0.5 *
                                             (rr * rr - rl * rl) * std::sin(tr);
                            const real s2L = 2.0 * M_PI * 0.5 *
                                             (rr * rr - rl * rl) * std::sin(tl);
                            const real factor = (mesh_motion) ? dV : 1;

                            // Grab central primitives
                            const real rhoc = prim_buff[tid].rho;
                            const real uc   = prim_buff[tid].get_v1();
                            const real vc   = prim_buff[tid].get_v2();
                            const real pc   = prim_buff[tid].p;
                            const real hc = prim_buff[tid].get_enthalpy(gamma);
                            const real gam2 =
                                prim_buff[tid].lorentz_factor_squared();

                            const conserved_t geom_source = {
                              0.0,
                              (rhoc * hc * gam2 * vc * vc) / rmean +
                                  pc * (s1R - s1L) * invdV,
                              -(rhoc * hc * gam2 * uc * vc) / rmean +
                                  pc * (s2R - s2L) * invdV,
                              0.0
                            };

                            cons_data[aid] -=
                                ((frf * s1R - flf * s1L) * invdV +
                                 (grf * s2R - glf * s2L) * invdV - geom_source -
                                 source_terms - gravity) *
                                dt * step * factor;
                            break;
                        }
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        {
                            const real rl    = x1l + vfaceL * step * dt;
                            const real rr    = x1r + vfaceR * step * dt;
                            const real rmean = get_cell_centroid(
                                rr,
                                rl,
                                simbi::Geometry::PLANAR_CYLINDRICAL
                            );
                            // const real tl           = my_max(x2min +
                            // (jj - 0.5) * dx2 , x2min); const real tr =
                            // my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0),
                            // x2max);
                            const real dV    = rmean * (rr - rl) * dx2;
                            const real invdV = 1.0 / dV;
                            const real s1R   = rr * dx2;
                            const real s1L   = rl * dx2;
                            const real s2R   = (rr - rl);
                            const real s2L   = (rr - rl);

                            // Grab central primitives
                            const real rhoc = prim_buff[tid].rho;
                            const real uc   = prim_buff[tid].get_v1();
                            const real vc   = prim_buff[tid].get_v2();
                            const real pc   = prim_buff[tid].p;

                            const real hc = prim_buff[tid].get_enthalpy(gamma);
                            const real gam2 =
                                prim_buff[tid].lorentz_factor_squared();

                            const conserved_t geom_source = {
                              0.0,
                              (rhoc * hc * gam2 * vc * vc) / rmean +
                                  pc * (s1R - s1L) * invdV,
                              -(rhoc * hc * gam2 * uc * vc) / rmean,
                              0.0
                            };
                            cons_data[aid] -=
                                ((frf * s1R - flf * s1L) * invdV +
                                 (grf * s2R - glf * s2L) * invdV - geom_source -
                                 source_terms - gravity) *
                                dt * step;
                            break;
                        }
                    default:
                        {
                            const real rl    = x1l + vfaceL * step * dt;
                            const real rr    = x1r + vfaceR * step * dt;
                            const real rmean = get_cell_centroid(
                                rl,
                                rr,
                                simbi::Geometry::AXIS_CYLINDRICAL
                            );
                            const real dV    = rmean * (rr - rl) * dx2;
                            const real invdV = 1.0 / dV;
                            const real s1R   = rr * dx2;
                            const real s1L   = rl * dx2;
                            const real s2R   = rmean * (rr - rl);
                            const real s2L   = rmean * (rr - rl);

                            // Grab central primitives
                            const real pc          = prim_buff[tid].p;
                            const auto geom_source = conserved_t{
                              0.0,
                              pc * (s1R - s1L) * invdV,
                              0.0,
                              0.0
                            };
                            cons_data[aid] -=
                                ((frf * s1R - flf * s1L) * invdV +
                                 (grf * s2R - glf * s2L) * invdV - geom_source -
                                 source_terms - gravity) *
                                dt * step;
                            break;
                        }
                }   // end switch
            }
            else {
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        {
                            cons_data[aid] -=
                                ((frf - flf) * invdx1 + (grf - glf) * invdx2 +
                                 (hrf - hlf) * invdx3 - source_terms -
                                 gravity) *
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
                            const real pc   = prim_buff[tid].p;

                            const real hc = prim_buff[tid].get_enthalpy(gamma);
                            const real gam2 =
                                prim_buff[tid].lorentz_factor_squared();

                            const auto geom_source = conserved_t{
                              0.0,
                              (rhoc * hc * gam2 * (vc * vc + wc * wc)) / rmean +
                                  pc * (s1R - s1L) / dV1,
                              rhoc * hc * gam2 * (wc * wc * cot - uc * vc) /
                                      rmean +
                                  pc * (s2R - s2L) / dV2,
                              -rhoc * hc * gam2 * wc * (uc + vc * cot) / rmean,
                              0.0
                            };
                            cons_data[aid] -= ((frf * s1R - flf * s1L) / dV1 +
                                               (grf * s2R - glf * s2L) / dV2 +
                                               (hrf - hlf) / dV3 - geom_source -
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
                            const real pc = prim_buff[tid].p;

                            const real hc = prim_buff[tid].get_enthalpy(gamma);
                            const real gam2 =
                                prim_buff[tid].lorentz_factor_squared();

                            const auto geom_source = conserved_t{
                              0.0,
                              (rhoc * hc * gam2 * (vc * vc)) / rmean +
                                  pc * (s1R - s1L) * invdV,
                              -(rhoc * hc * gam2 * uc * vc) / rmean,
                              0.0,
                              0.0
                            };
                            cons_data[aid] -= ((frf * s1R - flf * s1L) * invdV +
                                               (grf * s2R - glf * s2L) * invdV +
                                               (hrf * s3R - hlf * s3L) * invdV -
                                               geom_source - source_terms) *
                                              dt * step;
                            break;
                        }
                }   // end switch
            }
        }
    );
}

// //===================================================================================================================
// //                                            SIMULATE
// //===================================================================================================================
template <int dim>
void SRHD<dim>::simulate(
    std::function<real(real)> const& a,
    std::function<real(real)> const& adot,
    std::optional<SRHD<dim>::function_t> const& d_outer,
    std::optional<SRHD<dim>::function_t> const& s1_outer,
    std::optional<SRHD<dim>::function_t> const& s2_outer,
    std::optional<SRHD<dim>::function_t> const& s3_outer,
    std::optional<SRHD<dim>::function_t> const& e_outer
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
                  enrg_outer(x1max)
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
                      enrg_outer(x1max, x2[jreal])
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
                          enrg_outer(x1max, x2[jreal], x3[kreal])
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
        this->bcs[i] = boundary_cond_map.at(boundary_conditions[i]);
        if constexpr (dim == 1) {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2]
            };
        }
        else if constexpr (dim == 2) {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2],
              boundary_sources[i][3]
            };
        }
        else {
            this->inflow_zones[i] = conserved_t{
              boundary_sources[i][0],
              boundary_sources[i][1],
              boundary_sources[i][2],
              boundary_sources[i][3],
              boundary_sources[i][4]
            };
        }
    }

    // Write some info about the setup for writeup later
    setup.x1max = x1[nxv - 1];
    setup.x1min = x1[0];
    setup.x1    = x1;
    if constexpr (dim > 1) {
        setup.x2max = x2[nyv - 1];
        setup.x2min = x2[0];
        setup.x2    = x2;
    }
    if constexpr (dim > 2) {
        setup.x3max = x3[nzv - 1];
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
    setup.regime              = "srhd";
    setup.mesh_motion         = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions          = dim;

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(total_zones);
    pressure_guess.resize(total_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++) {
        const real d  = state[0][i];
        const real s1 = state[1][i];
        const real s2 = [&] {
            if constexpr (dim < 2) {
                return static_cast<real>(0.0);
            }
            return state[2][i];
        }();
        const real s3 = [&] {
            if constexpr (dim < 3) {
                return static_cast<real>(0.0);
            }
            return state[3][i];
        }();
        const real E = [&] {
            if constexpr (dim == 1) {
                return state[2][i];
            }
            else if constexpr (dim == 2) {
                return state[3][i];
            }
            else {
                return state[4][i];
            }
        }();
        const real S = std::sqrt(s1 * s1 + s2 * s2 + s3 * s3);
        if constexpr (dim == 1) {
            cons[i] = conserved_t{d, s1, E};
        }
        else if constexpr (dim == 2) {
            cons[i] = conserved_t{d, s1, s2, E};
        }
        else {
            cons[i] = conserved_t{d, s1, s2, s3, E};
        }
        pressure_guess[i] = std::abs(S - d - E);
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

    // Using a sigmoid decay function to represent when the source terms turn
    // off.
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

    this->n = 0;
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
