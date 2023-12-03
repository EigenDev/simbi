/*
 * C++ Source to perform Newtonian Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */
#include <cmath>                  // for max, min
#include "util/device_api.hpp"    // for syncrohonize, devSynch, ...
#include "util/parallel_for.hpp"  // for parallel_for
#include "util/printb.hpp"        // for writeln
#include "util/logger.hpp"        // for logger

using namespace simbi;
using namespace simbi::util;

// Default Constructor
template<int dim>
Newtonian<dim>::Newtonian() {
    
}

// Overloaded Constructor
template<int dim>
Newtonian<dim>::Newtonian(
    std::vector<std::vector<real>> &state, 
    const InitialConditions &init_conditions)
:
    HydroBase(
        state,
        init_conditions
    )
{
}

// Destructor
template<int dim>
Newtonian<dim>::~Newtonian() {
    
}


// Helpers
template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x1face(const lint ii, const int side) const
{
    switch (x1_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x1l = helpers::my_max<real>(x1min  + (ii - 0.5) * dx1,  x1min);
            if (side == 0) {
                return x1l;
            }
            return helpers::my_min<real>(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
        }
    default:
        {
            const real x1l = helpers::my_max<real>(x1min * std::pow(10, (ii - 0.5) * dlogx1),  x1min);
            if (side == 0) {
                return x1l;
            }
            return helpers::my_min<real>(x1l * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
        }
    }
}


template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x2face(const lint ii, const int side) const
{
    switch (x2_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x2l = helpers::my_max<real>(x2min  + (ii - 0.5) * dx2,  x2min);
            if (side == 0) {
                return x2l;
            }
            return helpers::my_min<real>(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }
    default:
        {
            const real x2l = helpers::my_max<real>(x2min * std::pow(10, (ii - 0.5) * dlogx2),  x2min);
            if (side == 0) {
                return x2l;
            }
            return helpers::my_min<real>(x2l * std::pow(10, dlogx2 * (ii == 0 ? 0.5 : 1.0)), x2max);
        }
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x3face(const lint ii, const int side) const
{
    switch (x3_cell_spacing)
    {
    case simbi::Cellspacing::LINSPACE:
        {
            const real x3l = helpers::my_max<real>(x3min  + (ii - 0.5) * dx3,  x3min);
            if (side == 0) {
                return x3l;
            }
            return helpers::my_min<real>(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
        }
    default:
        {
            const real x3l = helpers::my_max<real>(x3min * std::pow(10, (ii - 0.5) * dlogx3),  x3min);
            if (side == 0) {
                return x3l;
            }
            return helpers::my_min<real>(x3l * std::pow(10, dlogx3 * (ii == 0 ? 0.5 : 1.0)), x3max);
        }
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x1_differential(const lint ii) const {
    const real x1l   = get_x1face(ii, 0);
    const real x1r   = get_x1face(ii, 1);
    const real xmean = helpers::get_cell_centroid(x1r, x1l, geometry);
    switch (geometry)
    {
    case Geometry::SPHERICAL:
        return xmean * xmean * (x1r - x1l);
    default:
        return xmean * (x1r - x1l);
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x2_differential(const lint ii) const {
    if constexpr(dim == 1) {
        switch (geometry)
        {
        case Geometry::SPHERICAL:
            return 2;
        default:
            return (2.0 * M_PI);
        }
    } else {
        switch (geometry)
        {
            case Geometry::SPHERICAL:
            {
                const real x2l = get_x2face(ii, 0);
                const real x2r = get_x2face(ii, 1);
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

template<int dim>
GPU_CALLABLE_MEMBER
constexpr real Newtonian<dim>::get_x3_differential(const lint ii) const {
    if constexpr(dim == 1) {
        switch (geometry)
        {
        case Geometry::SPHERICAL:
            return (2.0 * M_PI);
        default:
            return 1;
        }
    } else if constexpr(dim == 2) {
        switch (geometry)
        {
            case Geometry::PLANAR_CYLINDRICAL:
                return 1;
            default:
                return (2.0 * M_PI);
        }
    } else {
        return dx3;
    }
}

template<int dim>
GPU_CALLABLE_MEMBER
real Newtonian<dim>::get_cell_volume(const lint ii, const lint jj, const lint kk) const
{
    if (geometry == Geometry::CARTESIAN){
        return 1;
    }
    return get_x1_differential(ii) * get_x2_differential(jj) * get_x3_differential(kk);
}

template<int dim>
void Newtonian<dim>::emit_troubled_cells() {
    troubled_cells.copyFromGpu();
    cons.copyFromGpu();
    prims.copyFromGpu();
    for (luint gid = 0; gid < total_zones; gid++)
    {
        if (troubled_cells[gid] != 0) {
            const luint xpg   = xactive_grid;
            const luint ypg   = yactive_grid;
            const luint kk    = helpers::get_height(gid, xpg, ypg);
            const luint jj    = helpers::get_row(gid, xpg, ypg, kk);
            const luint ii    = helpers::get_column(gid, xpg, ypg, kk);
            const lint ireal  = helpers::get_real_idx(ii, radius, xactive_grid);
            const lint jreal  = helpers::get_real_idx(jj, radius, yactive_grid); 
            const lint kreal  = helpers::get_real_idx(kk, radius, zactive_grid); 
            const real x1l    = get_x1face(ireal, 0);
            const real x1r    = get_x1face(ireal, 1);
            const real x2l    = get_x2face(jreal, 0);
            const real x2r    = get_x2face(jreal, 1);
            const real x3l    = get_x3face(kreal, 0);
            const real x3r    = get_x3face(kreal, 1);
            const real x1mean = helpers::calc_any_mean(x1l, x1r, x1_cell_spacing);
            const real x2mean = helpers::calc_any_mean(x2l, x2r, x2_cell_spacing);
            const real x3mean = helpers::calc_any_mean(x3l, x3r, x3_cell_spacing);
            const real m1 = cons[gid].momentum(1);
            const real m2 = cons[gid].momentum(2);
            const real m3 = cons[gid].momentum(3);
            const real et = cons[gid].e_dens;
            const real s  = std::sqrt(m1 * m1 + m2 * m2 + m3 * m3);
            const real v2 = (s * s) / (et * et);
            if constexpr(dim == 1) {
                printf("\nSimulation in bad state\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, x1coord: %.2e, iter: %" PRIu64 "\n", 
                        cons[gid].rho, prims[gid].p, v2, x1mean, n
                );
            } else if constexpr(dim == 2) {
                printf("\nSimulation in bad state\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, iter: %" PRIu64 "\n", 
                        cons[gid].rho, prims[gid].p, v2, x1mean, x2mean, n
                );
            } else {
                printf("\nSimulation in bad state\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, x3coord: %.2e, iter: %" PRIu64 "\n", 
                        cons[gid].rho, prims[gid].p, v2, x1mean, x2mean, x3mean, n
                );
            }
        }
    }
}
//-----------------------------------------------------------------------------------------
//                          GET THE nt::Primitive
//-----------------------------------------------------------------------------------------
/**
 * Return the primitive
 * variables density , three-velocity, pressure
 * 
 * @param  p executation policy class  
 * @return none
 */
template<int dim>
void Newtonian<dim>::cons2prim(const ExecutionPolicy<> &p)
{
    const auto* const cons_data = cons.data();
    auto* const prim_data = prims.data();
    auto* const troubled_data = troubled_cells.data();
    simbi::parallel_for(p, (luint)0, total_zones, [
        cons_data,
        prim_data,
        troubled_data,
        this
    ]   GPU_LAMBDA (const luint gid) {

        real invdV = 1.0;
        if (changing_volume)
        {
            if constexpr(dim == 1) {
                const auto ireal = helpers::get_real_idx(gid, radius, active_zones);
                const real dV    = get_cell_volume(ireal);
                invdV            = 1 / dV;
            } else if constexpr(dim == 2) {
                const luint ii   = gid % nx;
                const luint jj   = gid / nx;
                const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
                const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
                const real dV    = get_cell_volume(ireal, jreal);
                invdV = 1.0 / dV;
            } else {
                const luint kk  = simbi::helpers::get_height(gid, xactive_grid, yactive_grid);
                const luint jj  = simbi::helpers::get_row(gid, xactive_grid, yactive_grid, kk);
                const luint ii  = simbi::helpers::get_column(gid, xactive_grid, yactive_grid, kk);
                const auto ireal = helpers::get_real_idx(ii, radius, xactive_grid);
                const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid); 
                const auto kreal = helpers::get_real_idx(kk, radius, zactive_grid); 
                const real dV    = get_cell_volume(ireal, jreal, kreal);
                invdV = 1.0 / dV;
            }
        }
        const real rho     =  cons_data[gid].rho * invdV;
        const real v1      = (cons_data[gid].momentum(1) / rho) * invdV;
        const real v2      = (cons_data[gid].momentum(2) / rho) * invdV;
        const real v3      = (cons_data[gid].momentum(3) / rho) * invdV;
        const real rho_chi =  cons_data[gid].chi * invdV;
        const real pre     = (gamma - 1)*(
            cons_data[gid].e_dens - 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3)
        );
        if constexpr(dim == 1) {
            prim_data[gid] = nt::Primitive<1>{rho, v1, pre, rho_chi / rho};
        } else if constexpr(dim == 2) {
            prim_data[gid] = nt::Primitive<2>{rho, v1, v2, pre, rho_chi / rho};
        } else {
            prim_data[gid] = nt::Primitive<3>{rho, v1, v2, v3, pre, rho_chi / rho};
        }

        if (pre < 0 || std::isnan(pre)) {
            troubled_data[gid] = 1;
            inFailureState = true;
            dt = INFINITY;
        }
    });
}
//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
template<int dim>
GPU_CALLABLE_MEMBER
Newtonian<dim>::eigenvals_t Newtonian<dim>::calc_eigenvals(
    const Newtonian<dim>::primitive_t &primsL,
    const Newtonian<dim>::primitive_t &primsR,
    const luint nhat) const
{
    const real rhoL = primsL.rho;
    const real vL   = primsL.vcomponent(nhat);
    const real pL   = primsL.p;
    
    const real rhoR = primsR.rho;
    const real vR   = primsR.vcomponent(nhat);
    const real pR   = primsR.p;

    const real csR = std::sqrt(gamma * pR/rhoR);
    const real csL = std::sqrt(gamma * pL/rhoL);
    switch (sim_solver)
    {
    case Solver::HLLC:
        {
            // real cbar   = 0.5*(csL + csR);
            // real rhoBar = 0.5*(rhoL + rhoR);
            // real pStar  = 0.5*(pL + pR) + 0.5*(vL - vR)*cbar*rhoBar;

            // Steps to Compute HLLC as described in Toro et al. 2019
            const real num    = csL + csR- ( gamma-1.) * 0.5 * (vR- vL);
            const real denom  = csL * std::pow(pL, -hllc_z) + csR * std::pow(pR, -hllc_z);
            const real p_term = num/denom;
            const real pStar  = std::pow(p_term, (1/hllc_z));

            const real qL = 
                (pStar <= pL) ? 1 : std::sqrt(1. + ( (gamma + 1)/(2*gamma))*(pStar/pL - 1));

            const real qR = 
                (pStar <= pR) ? 1 : std::sqrt(1. + ( (gamma + 1)/(2*gamma))*(pStar/pR- 1));

            const real aL = vL - qL*csL;
            const real aR = vR + qR*csR;

            const real aStar = ( (pR- pL + rhoL*vL*(aL - vL) - rhoR*vR*(aR - vR))/
                            (rhoL*(aL - vL) - rhoR*(aR - vR) ) );

            if constexpr(dim == 1) {
                return nt::Eigenvals<dim>(aL, aR, aStar, pStar);
            } else {
                return nt::Eigenvals<dim>(aL, aR, csL, csR, aStar, pStar);
            }
        }

    default:
        {
            const real aR = helpers::my_max<real>(helpers::my_max<real>(vL + csL, vR + csR), (0.0)); 
            const real aL = helpers::my_min<real>(helpers::my_min<real>(vL - csL, vR - csR), (0.0));
            return nt::Eigenvals<dim>{aL, aR};
        }

    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------
template<int dim>
GPU_CALLABLE_MEMBER 
Newtonian<dim>::conserved_t Newtonian<dim>::prims2cons(const Newtonian<dim>::primitive_t &prims) const
{
    const real rho = prims.rho;
    const real v1  = prims.vcomponent(1);
    const real v2  = prims.vcomponent(2);
    const real v3  = prims.vcomponent(3);
    const real et  = prims.get_energy_density(gamma);
    if constexpr(dim == 1) { 
        return {
            rho, 
            rho * v1,
            et
        };
    } else if constexpr(dim == 2) {
        return {
            rho, 
            rho * v1,
            rho * v2,
            et
        };
    } else {
        return {
            rho, 
            rho * v1,
            rho * v2,
            rho * v3,
            et
        };
    }
};
//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------
// Adapt the cfl conditonal timestep
template<int dim>
void Newtonian<dim>::adapt_dt()
{
    // singleton instance of thread pool. lazy-evaluated
    static auto &thread_pool = simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
    std::atomic<real> min_dt = INFINITY;
    thread_pool.parallel_for(static_cast<luint>(0), active_zones, [&](luint aid) {
        real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
        const luint kk = dim < 3 ? 0 : simbi::helpers::get_height(aid, xactive_grid, yactive_grid);
        const luint jj = dim < 2 ? 0 : simbi::helpers::get_row(aid, xactive_grid, yactive_grid, kk);
        const luint ii = simbi::helpers::get_column(aid, xactive_grid, yactive_grid, kk);
        // Left/Right wave speeds
        const real rho = prims[aid].rho;
        const real v1  = prims[aid].vcomponent(1);
        const real v2  = prims[aid].vcomponent(2);
        const real v3  = prims[aid].vcomponent(3);
        const real pre = prims[aid].p;
        const real cs  = std::sqrt(gamma * pre / rho);
        v1p = std::abs(v1 + cs);
        v1m = std::abs(v1 - cs);
        if constexpr(dim > 1) {
            v2p = std::abs(v2 + cs);
            v2m = std::abs(v2 - cs);
        }
        if constexpr(dim > 2) {
            v3p = std::abs(v3 + cs);
            v3m = std::abs(v3 - cs);
        }                        

        const real x1l = get_x1face(ii, 0);
        const real x1r = get_x1face(ii, 1);
        const real dx1 = x1r - x1l; 
        switch (geometry)
        {
        case simbi::Geometry::CARTESIAN:
            if constexpr(dim == 1) {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m))
                });

            } else if constexpr(dim == 2) {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m)),
                    dx2 / (std::max(v2p, v2m))
                });
            } else {
                cfl_dt = std::min({
                    dx1 / (std::max(v1p, v1m)),
                    dx2 / (std::max(v2p, v2m)),
                    dx3 / (std::max(v3p, v3m))
                });
            }   
            break;
        
        case simbi::Geometry::SPHERICAL:
            {
                if constexpr(dim == 1) {
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m))
                    });
                } else if constexpr(dim == 2) {
                    const real rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::SPHERICAL);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m))
                    });
                } else {
                    const real x2l   = get_x2face(jj, 0);
                    const real x2r   = get_x2face(jj, 1);
                    const real rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::SPHERICAL);
                    const real th    = 0.5 * (x2r + x2l);
                    const real rproj = rmean * std::sin(th);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m)),
                        rproj * dx3 / (std::max(v3p, v3m))
                    });
                }
                break;
            }
        default:
            {
                if constexpr(dim == 1) {
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m))
                    });
                } else if constexpr(dim == 2) {
                    switch (geometry)
                    {
                    case Geometry::AXIS_CYLINDRICAL:
                    {
                        cfl_dt = std::min({       
                            dx1 / (std::max(v1p, v1m)),
                            dx2 / (std::max(v2p, v2m))
                        });
                        break;
                    }
                    
                    default:
                    {
                        const real rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::CYLINDRICAL);
                        cfl_dt = std::min({       
                            dx1 / (std::max(v1p, v1m)),
                            rmean * dx2 / (std::max(v2p, v2m))
                        });
                        break;
                    }
                    }
                } else {
                    const real rmean = helpers::get_cell_centroid(x1r, x1l, simbi::Geometry::CYLINDRICAL);
                    cfl_dt = std::min({       
                        dx1 / (std::max(v1p, v1m)),
                        rmean * dx2 / (std::max(v2p, v2m)),
                        dx3 / (std::max(v3p, v3m))
                    });
                }
                break;
            }
        }
        pooling::update_minimum(min_dt, cfl_dt);
    });
    dt = cfl * min_dt;
};

template<int dim>
void Newtonian<dim>::adapt_dt(const ExecutionPolicy<> &p)
{
    #if GPU_CODE
        if constexpr(dim == 1) {
            // LAUNCH_ASYNC((helpers::compute_dt<nt::Primitive<1>,dt_type>), p.gridSize, p.blockSize, this, prims.data(), dt_min.data());
            helpers::compute_dt<nt::Primitive<1>><<<p.gridSize, p.blockSize>>>(this, prims.data(), dt_min.data());
        } else {
            // LAUNCH_ASYNC((helpers::compute_dt<nt::Primitive<dim>,dt_type>), p.gridSize, p.blockSize, this, prims.data(), dt_min.data(), geometry);
            helpers::compute_dt<nt::Primitive<dim>><<<p.gridSize,p.blockSize>>>(this, prims.data(), dt_min.data(), geometry);
        }
        // LAUNCH_ASYNC((helpers::deviceReduceWarpAtomicKernel<dim>), p.gridSize, p.blockSize, this, dt_min.data(), active_zones);
        helpers::deviceReduceWarpAtomicKernel<dim><<<p.gridSize, p.blockSize>>>(this, dt_min.data(), active_zones);
        gpu::api::deviceSynch();
    #endif
}
//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================
template<int dim>
GPU_CALLABLE_MEMBER
Newtonian<dim>::conserved_t Newtonian<dim>::prims2flux(const Newtonian<dim>::primitive_t &prims, const luint nhat) const
{
    const real rho      = prims.rho;
    const real v1       = prims.vcomponent(1);
    const real v2       = prims.vcomponent(2);
    const real v3       = prims.vcomponent(3);
    const real pressure = prims.p;
    const real chi      = prims.chi;
    const real vn       = nhat == 1 ? v1 : nhat == 2 ? v2 : v3;
    const auto et       = prims.get_energy_density(gamma);
    const real m1       = rho * v1;
    if constexpr(dim == 1) {
        return {
            rho  * vn, 
            m1   * vn + helpers::kronecker(nhat, 1) * pressure, 
            (et + pressure) * vn, 
            rho  * vn * chi
        };
    } else if constexpr(dim == 2) {
        const real m2 = rho * v2;
        return {
            rho * vn, 
            m1  * vn + helpers::kronecker(nhat, 1) * pressure, 
            m2  * vn + helpers::kronecker(nhat, 2) * pressure, 
            (et + pressure) * vn, 
            rho * vn * chi
        };
    } else {
        const real m2 = rho * v2;
        const real m3 = rho * v3;
        return {
            rho * vn, 
            m1  * vn + helpers::kronecker(nhat, 1) * pressure, 
            m2  * vn + helpers::kronecker(nhat, 2) * pressure, 
            m3  * vn + helpers::kronecker(nhat, 3) * pressure,
            (et + pressure) * vn, 
            rho * vn * chi
        };
    }
};

template<int dim>
GPU_CALLABLE_MEMBER
Newtonian<dim>::conserved_t Newtonian<dim>::calc_hll_flux(
    const Newtonian<dim>::conserved_t &left_state, 
    const Newtonian<dim>::conserved_t &right_state,
    const Newtonian<dim>::conserved_t &left_flux, 
    const Newtonian<dim>::conserved_t &right_flux,
    const Newtonian<dim>::primitive_t &left_prims, 
    const Newtonian<dim>::primitive_t &right_prims,
    const luint nhat,
    const real vface) const
{
    const auto lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL = lambda.aL;
    const real aR = lambda.aR;

    auto net_flux = [&] {
        // Compute the HLL Flux component-wise
        if (vface <= aL) {
            return left_flux - left_state * vface;
        }
        else if (vface >= aR) {
            return right_flux - right_state * vface;
        }
        else {    
            const auto f_hll = (left_flux * aR - right_flux * aL + (right_state - left_state) * aR * aL) / (aR - aL);
            const auto u_hll = (right_state * aR - left_state * aL - right_flux + left_flux) / (aR - aL);
            return f_hll - u_hll * vface;
        }
    }();

    // Upwind the scalar concentration flux
    if (net_flux.rho < 0)
        net_flux.chi = right_prims.chi * net_flux.rho;
    else
        net_flux.chi = left_prims.chi  * net_flux.rho;

    // Compute the HLL Flux component-wise
    return net_flux;
};

template<int dim>
GPU_CALLABLE_MEMBER
Newtonian<dim>::conserved_t Newtonian<dim>::calc_hllc_flux(
    const Newtonian<dim>::conserved_t &left_state,
    const Newtonian<dim>::conserved_t &right_state,
    const Newtonian<dim>::conserved_t &left_flux,
    const Newtonian<dim>::conserved_t &right_flux,
    const Newtonian<dim>::primitive_t &left_prims,
    const Newtonian<dim>::primitive_t &right_prims,
    const luint nhat,
    const real vface) const 
{
    const auto lambda = calc_eigenvals(left_prims, right_prims, nhat);
    const real aL    = lambda.aL;
    const real aR    = lambda.aR;

    // Quick checks before moving on with rest of computation
    if (vface <= aL){
        return left_flux - left_state * vface;
    } else if (vface >= aR){
        return right_flux - right_state * vface;
    }

    if constexpr(dim == 1) {
        const real aStar = lambda.aStar;
        const real pStar = lambda.pStar;
        const real ap  = helpers::my_max<real>(0, aR);
        const real am  = helpers::my_min<real>(0, aL);
        auto hll_flux  = (left_flux * ap + right_flux * am - (right_state - left_state) * am * ap)  / (am + ap) ;
        auto hll_state = (right_state * aR - left_state * aL - right_flux + left_flux)/(aR - aL);
        
        if (vface <= aStar){
            real pressure = left_prims.p;
            real v        = left_prims.v1;
            real rho      = left_state.rho;
            real m        = left_state.m1;
            real energy   = left_state.e_dens;
            real cofac    = 1/(aL - aStar);

            real rhoStar = cofac * (aL - v)*rho;
            real mstar   = cofac * (m*(aL - v) - pressure + pStar);
            real eStar   = cofac * (energy*(aL - v) + pStar*aStar - pressure*v);

            auto star_state = nt::Conserved<1>{rhoStar, mstar, eStar};

            // Compute the luintermediate left flux
            return left_flux + (star_state - left_state) * aL - star_state * vface;
        } else {
            real pressure = right_prims.p;
            real v        = right_prims.v1;
            real rho      = right_state.rho;
            real m        = right_state.m1;
            real energy   = right_state.e_dens;
            real cofac    = 1./(aR - aStar);

            real rhoStar = cofac * (aR - v)*rho;
            real mstar   = cofac * (m*(aR - v) - pressure + pStar);
            real eStar   = cofac * (energy*(aR - v) + pStar*aStar - pressure*v);

            auto star_state = nt::Conserved<1>{rhoStar, mstar, eStar};

            // Compute the luintermediate right flux
            return right_flux + (star_state - right_state) * aR - star_state * vface;
        }
    } else {
        const real cL = lambda.csL;
        const real cR = lambda.csR;
        const real aStar = lambda.aStar;
        const real pStar = lambda.pStar;
        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020: 
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        constexpr real ma_lim   = 0.10;

        // --------------Compute the L Star State----------
        real pressure = left_prims.p;
        real rho      = left_state.rho;
        real m1       = left_state.momentum(1);
        real m2       = left_state.momentum(2);
        real m3       = left_state.momentum(3);
        real edens    = left_state.e_dens;
        real cofactor = 1 / (aL - aStar);

        const real vL           = left_prims.vcomponent(nhat);
        const real vR           = right_prims.vcomponent(nhat);

        // Left Star State in x-direction of coordinate lattice
        real rhostar            = cofactor * (aL - vL) * rho;
        real m1star             = cofactor * (m1 * (aL - vL) + helpers::kronecker(nhat, 1) * (-pressure + pStar) );
        real m2star             = cofactor * (m2 * (aL - vL) + helpers::kronecker(nhat, 2) * (-pressure + pStar) );
        real m3star             = cofactor * (m3 * (aL - vL) + helpers::kronecker(nhat, 3) * (-pressure + pStar) );
        real estar              = cofactor * (edens  * (aL - vL) + pStar * aStar - pressure * vL);
        const auto starStateL = [=] {
            if constexpr(dim == 2) {
                return nt::Conserved<2>{rhostar, m1star, m2star, estar};
            } else {
                return nt::Conserved<3>{rhostar, m1star, m2star, m3star, estar};
            }
        }();

        pressure = right_prims.p;
        rho      = right_state.rho;
        m1       = right_state.m1;
        m2       = right_state.m2;
        edens    = right_state.e_dens;
        cofactor = 1 / (aR - aStar);

        rhostar               = cofactor * (aR - vR) * rho;
        m1star                = cofactor * (m1 * (aR - vR) + helpers::kronecker(nhat, 1) * (-pressure + pStar) );
        m2star                = cofactor * (m2 * (aR - vR) + helpers::kronecker(nhat, 2) * (-pressure + pStar) );
        m3star                = cofactor * (m3 * (aR - vR) + helpers::kronecker(nhat, 3) * (-pressure + pStar) );
        estar                 = cofactor * (edens  * (aR - vR) + pStar * aStar - pressure * vR);
        const auto starStateR = [=] {
            if constexpr(dim == 2) {
                return nt::Conserved<2>{rhostar, m1star, m2star, estar};
            } else {
                return nt::Conserved<3>{rhostar, m1star, m2star, m3star, estar};
            }
        }();

        const real ma_local = helpers::my_max(std::abs(vL / cL), std::abs(vR / cR));
        const real phi      = std::sin(helpers::my_min<real>(1.0, ma_local / ma_lim) * M_PI * 0.5);
        const real aL_lm    = phi * aL;
        const real aR_lm    = phi * aR;
        const auto face_starState = (aStar <= 0) ? starStateR : starStateL;
        auto net_flux = (left_flux + right_flux) * 0.5 + ( (starStateL - left_state) * aL_lm
                            + (starStateL - starStateR) * std::abs(aStar) + (starStateR - right_state) * aR_lm) * 0.5 - face_starState * vface;

        // upwind the concentration flux 
        if (net_flux.rho < 0)
            net_flux.chi = right_prims.chi * net_flux.rho;
        else
            net_flux.chi = left_prims.chi  * net_flux.rho;

        return net_flux;
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================
template<int dim>
void Newtonian<dim>::advance(
    const ExecutionPolicy<> &p,
    const luint sx,
    const luint sy)
{
    const luint xpg = this->xactive_grid;
    const luint ypg = this->yactive_grid;
    const luint zpg = this->zactive_grid;

    const luint extent      = p.get_full_extent();
    auto* const cons_data   = cons.data();
    const auto* const prim_data   = prims.data();
    const auto* const dens_source = density_source.data();
    const auto* const mom1_source = m1_source.data();
    const auto* const mom2_source = m2_source.data();
    const auto* const mom3_source = m3_source.data();
    const auto* const erg_source  = energy_source.data();
    const auto* const object_data = object_pos.data();
    const auto* const grav1_source = sourceG1.data();
    const auto* const grav2_source = sourceG2.data();
    const auto* const grav3_source = sourceG3.data();

    simbi::parallel_for(p, (luint)0, extent, [
        sx,
        sy,
        p,
        prim_data,
        cons_data,
        dens_source,
        mom1_source,
        mom2_source,
        mom3_source,
        erg_source,
        object_data,
        grav1_source,
        grav2_source,
        grav3_source,
        xpg,
        ypg,
        zpg,
        this
    ] GPU_LAMBDA (const luint idx){
        #if GPU_CODE 
        auto prim_buff = global::shared_memory_proxy<nt::Primitive<dim>>();
        #else 
        auto *const prim_buff = prim_data;
        #endif 

        const luint kk  = dim < 3 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::helpers::get_height(idx, xpg, ypg);
        const luint jj  = dim < 2 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::helpers::get_row(idx, xpg, ypg, kk);
        const luint ii  = (global::BuildPlatform == global::Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::helpers::get_column(idx, xpg, ypg, kk);
        #if GPU_CODE
        if constexpr(dim == 1) {
            if (ii >= xpg) return;
        } else if constexpr(dim == 2) {
            if ((ii >= xpg) || (jj >= ypg)) return;
        } else {
            if ((ii >= xpg) || (jj >= ypg) || (kk >= zpg)) return;
        }
        #endif 

        const luint ia  = ii + radius;
        const luint ja  = dim < 2 ? 0 : jj + radius;
        const luint ka  = dim < 3 ? 0 : kk + radius;
        const luint tx  = (global::BuildPlatform == global::Platform::GPU) ? threadIdx.x : 0;
        const luint ty  = dim < 2 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? threadIdx.y : 0;
        const luint tz  = dim < 3 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? threadIdx.z : 0;
        const luint txa = (global::BuildPlatform == global::Platform::GPU) ? tx + radius : ia;
        const luint tya = dim < 2 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? ty + radius : ja;
        const luint tza = dim < 3 ? 0 : (global::BuildPlatform == global::Platform::GPU) ? tz + radius : ka;

        nt::Conserved<dim> uxL, uxR, uyL, uyR, uzL, uzR;
        nt::Conserved<dim> fL, fR, gL, gR, hL, hR, frf, flf, grf, glf, hrf, hlf;
        nt::Primitive<dim> xprimsL, xprimsR, yprimsL, yprimsR, zprimsL, zprimsR;

        const luint aid = ka * nx * ny + ja * nx + ia;
        #if GPU_CODE
            if constexpr(dim == 1) {
                luint txl = p.blockSize.x;
                // Check if the active index exceeds the active zones
                // if it does, then this thread buffer will taken on the
                // ghost index at the very end and return
                prim_buff[txa] = prim_data[ia];
                if (threadIdx.x < radius)
                {
                    if (blockIdx.x == p.gridSize.x - 1 && (ia + p.blockSize.x > nx - radius + threadIdx.x)) {
                        txl = nx - radius - ia + threadIdx.x;
                    }
                    prim_buff[txa - radius] = prim_data[ia - radius];
                    prim_buff[txa + txl]        = prim_data[ia + txl];
                }
                simbi::gpu::api::synchronize();
            } else if constexpr(dim == 2) {
                luint txl = p.blockSize.x;
                luint tyl = p.blockSize.y;
                // Load Shared memory into buffer for active zones plus ghosts
                prim_buff[tya * sx + txa * sy] = prim_data[aid];
                if (ty < radius)
                {
                    if (blockIdx.y == p.gridSize.y - 1 && (ja + p.blockSize.y > ny - radius + ty)) {
                        tyl = ny - radius - ja + ty;
                    }
                    prim_buff[(tya - radius) * sx + txa] = prim_data[(ja - radius) * nx + ia];
                    prim_buff[(tya + tyl) * sx + txa]    = prim_data[(ja + tyl) * nx + ia]; 
                }
                if (tx < radius)
                {   
                    if (blockIdx.x == p.gridSize.x - 1 && (ia + p.blockSize.x > nx - radius + tx)) {
                        txl = nx - radius - ia + tx;
                    }
                    prim_buff[tya * sx + txa - radius] =  prim_data[ja * nx + (ia - radius)];
                    prim_buff[tya * sx + txa +    txl] =  prim_data[ja * nx + (ia + txl)]; 
                }
                simbi::gpu::api::synchronize();

            } else {
                luint txl = p.blockSize.x;
                luint tyl = p.blockSize.y;
                luint tzl = p.blockSize.z;
                // Load Shared memory into buffer for active zones plus ghosts
                prim_buff[tza * sx * sy + tya * sx + txa] = prim_data[aid];
                if (tz == 0)    
                {
                    if ((blockIdx.z == p.gridSize.z - 1) && (ka + p.blockSize.z > nz - radius + tz)) {
                        tzl = nz - radius - ka + tz;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = tzl + q - 1;
                        prim_buff[(tza - q) * sx * sy + tya * sx + txa]  = prim_data[(ka - q) * nx * ny + ja * nx + ia];
                        prim_buff[(tza + re) * sx * sy + tya * sx + txa] = prim_data[(ka + re) * nx * ny + ja * nx + ia];
                    } 
                }
                if (ty == 0)    
                {
                    if ((blockIdx.y == p.gridSize.y - 1) && (ja + p.blockSize.y > ny - radius + ty)) {
                        tyl = ny - radius - ja + ty;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = tyl + q - 1;
                        prim_buff[tza * sx * sy + (tya - q) * sx + txa]  = prim_data[ka * nx * ny + (ja - q) * nx + ia];
                        prim_buff[tza * sx * sy + (tya + re) * sx + txa] = prim_data[ka * nx * ny + (ja + re) * nx + ia];
                    } 
                }
                if (tx == 0)
                {   
                    if ((blockIdx.x == p.gridSize.x - 1) && (ia + p.blockSize.x > nx - radius + tx)) {
                        txl = nx - radius - ia + tx;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const auto re = txl + q - 1;
                        prim_buff[tza * sx * sy + tya * sx + txa - q]  =  prim_data[ka * nx * ny + ja * nx + ia - q];
                        prim_buff[tza * sx * sy + tya * sx + txa + re] =  prim_data[ka * nx * ny + ja * nx + ia + re]; 
                    }
                }
                simbi::gpu::api::synchronize();
            }
        #endif

        const bool object_to_my_left  = dim < 2 ? false : object_data[kk * xpg * ypg + jj * xpg +  helpers::my_max(static_cast<lint>(ii - 1), static_cast<lint>(0))];
        const bool object_to_my_right = dim < 2 ? false : object_data[kk * xpg * ypg + jj * xpg +  helpers::my_min(ii + 1,  xpg - 1)];
        const bool object_in_front    = dim < 2 ? false : object_data[kk * xpg * ypg + helpers::my_min(jj + 1, ypg - 1) * xpg +  ii];
        const bool object_behind      = dim < 2 ? false : object_data[kk * xpg * ypg + helpers::my_max(static_cast<lint>(jj - 1), static_cast<lint>(0)) * xpg + ii];
        const bool object_above_me    = dim < 3 ? false : object_data[helpers::my_min(kk + 1, zpg - 1)  * xpg * ypg + jj * xpg +  ii];
        const bool object_below_me    = dim < 3 ? false : object_data[helpers::my_max(static_cast<lint>(kk - 1), static_cast<lint>(0)) * xpg * ypg + jj * xpg +  ii];

        const real x1l    = get_x1face(ii, 0);
        const real x1r    = get_x1face(ii, 1);
        const real vfaceL = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1l * hubble_param;
        const real vfaceR = (geometry == simbi::Geometry::CARTESIAN) ? hubble_param : x1r * hubble_param;

        if (first_order) [[unlikely]] {
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            if constexpr(dim > 1) {
                //j+1/2
                yprimsL = prim_buff[tza * sx * sy + (tya + 0) * sx + txa];
                yprimsR = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
            }
            if constexpr(dim > 2) {
                //k+1/2
                zprimsL = prim_buff[(tza + 0) * sx * sy + tya * sx + txa];
                zprimsR = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
            }

            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                if constexpr(dim > 1) {
                    xprimsR.v2  =  xprimsL.v2;
                }
                if constexpr(dim > 2){
                    xprimsR.v3  =  xprimsL.v3;
                }
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                if constexpr(dim > 1) {
                    yprimsR.v2  = -yprimsL.v2;
                }
                if constexpr(dim > 2){
                    yprimsR.v3  =  yprimsL.v3;
                }
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                if constexpr(dim == 3) {
                    zprimsR.v2  =  zprimsL.v2;
                    zprimsR.v3  = -zprimsL.v3;
                }
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            }
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }
            // Calc HLL Flux at i+1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1){
                    grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1) {
                    grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            }

            // Set up the left and right state interfaces for i-1/2
            xprimsL = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            xprimsR = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            if constexpr(dim > 1) {
                //j+1/2
                yprimsL = prim_buff[tza * sx * sy + (tya - 1) * sx + txa]; 
                yprimsR = prim_buff[tza * sx * sy + (tya + 0) * sx + txa]; 
            }
            if constexpr(dim > 2) {
                //k+1/2
                zprimsL = prim_buff[(tza - 1) * sx * sy + tya * sx + txa]; 
                zprimsR = prim_buff[(tza - 0) * sx * sy + tya * sx + txa]; 
            }

            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                if constexpr(dim > 1){
                    xprimsL.v2  =  xprimsR.v2;
                }
                if constexpr(dim > 2){
                    xprimsL.v3  =  xprimsR.v3;
                }
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                if constexpr(dim > 1) {
                    yprimsL.v2  = -yprimsR.v2;
                }
                if constexpr(dim > 2){
                    yprimsL.v3  =  yprimsR.v3;
                }
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                if constexpr(dim == 3) {
                    zprimsL.v2  =  zprimsR.v2;
                    zprimsL.v3  = -zprimsR.v3;
                }
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }
            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            } 
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            // Calc HLL Flux at i-1/2 interface
            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1){
                    glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }  
                if constexpr(dim > 2){
                    hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                } 
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1) {
                    glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0); 
                }
                break;
            }
        } else{
            // Coordinate X
            const nt::Primitive<dim> xleft_most  = prim_buff[tza * sx * sy + tya * sx + (txa - 2)];
            const nt::Primitive<dim> xleft_mid   = prim_buff[tza * sx * sy + tya * sx + (txa - 1)];
            const nt::Primitive<dim> center      = prim_buff[tza * sx * sy + tya * sx + (txa + 0)];
            const nt::Primitive<dim> xright_mid  = prim_buff[tza * sx * sy + tya * sx + (txa + 1)];
            const nt::Primitive<dim> xright_most = prim_buff[tza * sx * sy + tya * sx + (txa + 2)];
            nt::Primitive<dim> yleft_most, yleft_mid, yright_mid, yright_most;
            nt::Primitive<dim> zleft_most, zleft_mid, zright_mid, zright_most;
            // Reconstructed left X nt::Primitive<dim> vector at the i+1/2 interface
            xprimsL  = center     + helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta)   * 0.5; 
            xprimsR  = xright_mid - helpers::plm_gradient(xright_mid, center, xright_most, plm_theta) * 0.5;

            // Coordinate Y
            if constexpr(dim > 1){
                yleft_most  = prim_buff[tza * sx * sy + (tya - 2) * sx + txa];
                yleft_mid   = prim_buff[tza * sx * sy + (tya - 1) * sx + txa];
                yright_mid  = prim_buff[tza * sx * sy + (tya + 1) * sx + txa];
                yright_most = prim_buff[tza * sx * sy + (tya + 2) * sx + txa];
                yprimsL  = center     + helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta)   * 0.5;  
                yprimsR  = yright_mid - helpers::plm_gradient(yright_mid, center, yright_most, plm_theta) * 0.5;
            }

            // Coordinate z
            if constexpr(dim > 2){
                zleft_most  = prim_buff[(tza - 2) * sx * sy + tya * sx + txa];
                zleft_mid   = prim_buff[(tza - 1) * sx * sy + tya * sx + txa];
                zright_mid  = prim_buff[(tza + 1) * sx * sy + tya * sx + txa];
                zright_most = prim_buff[(tza + 2) * sx * sy + tya * sx + txa];
                zprimsL  = center     + helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta)   * 0.5;  
                zprimsR  = zright_mid - helpers::plm_gradient(zright_mid, center, zright_most, plm_theta) * 0.5;
            }

            if (object_to_my_right){
                xprimsR.rho =  xprimsL.rho;
                xprimsR.v1  = -xprimsL.v1;
                if constexpr(dim > 1) {
                    xprimsR.v2  =  xprimsL.v2;
                }
                if constexpr(dim > 2){
                    xprimsR.v3  =  xprimsL.v3;
                }
                xprimsR.p   =  xprimsL.p;
                xprimsR.chi =  xprimsL.chi;
            }

            if (object_in_front){
                yprimsR.rho =  yprimsL.rho;
                yprimsR.v1  =  yprimsL.v1;
                if constexpr(dim > 1) {
                    yprimsR.v2  = -yprimsL.v2;
                }
                if constexpr(dim > 2) {
                    yprimsR.v3  =  yprimsL.v3;
                }
                yprimsR.p   =  yprimsL.p;
                yprimsR.chi =  yprimsL.chi;
            }

            if (object_above_me) {
                zprimsR.rho =  zprimsL.rho;
                zprimsR.v1  =  zprimsL.v1;
                if constexpr(dim == 3) {
                    zprimsR.v2  =  zprimsL.v2;
                    zprimsR.v3  = -zprimsL.v3;
                }
                zprimsR.p   =  zprimsL.p;
                zprimsR.chi =  zprimsL.chi;
            }

            // Calculate the left and right states using the reconstructed PLM
            // nt::Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }

            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            }
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            switch (sim_solver)
            {
            case Solver::HLLC:
                frf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1){
                    grf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            
            default:
                frf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceR);
                if constexpr(dim > 1) {
                    grf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hrf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                }
                break;
            }

            // Do the same thing, but for the left side interface [i - 1/2]
            xprimsL  = xleft_mid  + helpers::plm_gradient(xleft_mid, xleft_most, center, plm_theta) * 0.5; 
            xprimsR  = center     - helpers::plm_gradient(center, xleft_mid, xright_mid, plm_theta) * 0.5;
            if constexpr(dim > 1) {
                yprimsL  = yleft_mid  + helpers::plm_gradient(yleft_mid, yleft_most, center, plm_theta) * 0.5; 
                yprimsR  = center     - helpers::plm_gradient(center, yleft_mid, yright_mid, plm_theta) * 0.5;
            }
            if constexpr(dim > 2) {
                zprimsL  = zleft_mid  + helpers::plm_gradient(zleft_mid, zleft_most, center, plm_theta) * 0.5;
                zprimsR  = center     - helpers::plm_gradient(center, zleft_mid, zright_mid, plm_theta) * 0.5;
            }

            
            if (object_to_my_left){
                xprimsL.rho =  xprimsR.rho;
                xprimsL.v1  = -xprimsR.v1;
                if constexpr(dim > 1){
                    xprimsL.v2  =  xprimsR.v2;
                }
                if constexpr(dim > 2){
                    xprimsL.v3  =  xprimsR.v3;
                }
                xprimsL.p   =  xprimsR.p;
                xprimsL.chi =  xprimsR.chi;
            }

            if (object_behind){
                yprimsL.rho =  yprimsR.rho;
                yprimsL.v1  =  yprimsR.v1;
                if constexpr(dim > 1) {
                    yprimsL.v2  = -yprimsR.v2;
                }
                if constexpr(dim > 2){
                    yprimsL.v3  =  yprimsR.v3;
                }
                yprimsL.p   =  yprimsR.p;
                yprimsL.chi =  yprimsR.chi;
            }

            if (object_below_me) {
                zprimsL.rho =  zprimsR.rho;
                zprimsL.v1  =  zprimsR.v1;
                if constexpr(dim == 3) {
                    zprimsL.v2  =  zprimsR.v2;
                    zprimsL.v3  = -zprimsR.v3;
                }
                zprimsL.p   =  zprimsR.p;
                zprimsL.chi =  zprimsR.chi;
            }

            // Calculate the left and right states using the reconstructed PLM nt::Primitive
            uxL = prims2cons(xprimsL);
            uxR = prims2cons(xprimsR);
            if constexpr(dim > 1) {
                uyL = prims2cons(yprimsL);
                uyR = prims2cons(yprimsR);
            }
            if constexpr(dim > 2) {
                uzL = prims2cons(zprimsL);
                uzR = prims2cons(zprimsR);
            }
            fL = prims2flux(xprimsL, 1);
            fR = prims2flux(xprimsR, 1);
            if constexpr(dim > 1) {
                gL = prims2flux(yprimsL, 2);
                gR = prims2flux(yprimsR, 2);
            } 
            if constexpr(dim > 2) {
                hL = prims2flux(zprimsL, 3);
                hR = prims2flux(zprimsR, 3);
            }

            switch (sim_solver)
            {
            case Solver::HLLC:
                flf = calc_hllc_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1){
                    glf = calc_hllc_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }  
                if constexpr(dim > 2){
                    hlf = calc_hllc_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0);
                } 
                break;
            
            default:
                flf = calc_hll_flux(uxL, uxR, fL, fR, xprimsL, xprimsR, 1, vfaceL);
                if constexpr(dim > 1) {
                    glf = calc_hll_flux(uyL, uyR, gL, gR, yprimsL, yprimsR, 2, 0);
                }
                if constexpr(dim > 2) {
                    hlf = calc_hll_flux(uzL, uzR, hL, hR, zprimsL, zprimsR, 3, 0); 
                }
                break;
            }
        }// end else 

        //Advance depending on geometry
        const luint real_loc = kk * xpg * ypg + jj * xpg + ii;
        const real d_source  = den_source_all_zeros     ? 0.0 : dens_source[real_loc];
        const real m1_source = mom1_source_all_zeros    ? 0.0 : mom1_source[real_loc];
        const real e_source  = energy_source_all_zeros  ? 0.0 : erg_source[real_loc];
        
        const auto source_terms = [&]{
            if constexpr(dim == 1) {
                return nt::Conserved<1>{d_source, m1_source, e_source} * time_constant;
            } else if constexpr(dim == 2) {
                const real m2_source = mom2_source_all_zeros ? 0.0 : mom2_source[real_loc];
                return nt::Conserved<2>{d_source, m1_source, m2_source, e_source} * time_constant;
            } else {
                const real m2_source = mom2_source_all_zeros ? 0.0 : mom2_source[real_loc];
                const real m3_source = mom3_source_all_zeros ? 0.0 : mom3_source[real_loc];
                return nt::Conserved<3>{d_source, m1_source, m2_source, m3_source, e_source} * time_constant;
            }
        }();

        // Gravity
        const auto gm1_source = zero_gravity1 ? 0 : grav1_source[real_loc] * cons_data[aid].rho;
        const auto tid = tza * sx * sy + tya * sx + txa;
        const auto gravity = [&]{
            if constexpr(dim == 1) {
                const auto ge_source  = gm1_source * prim_buff[tid].v1;
                return nt::Conserved<1>{0, gm1_source, ge_source};
            } else if constexpr(dim == 2) {
                const auto gm2_source = zero_gravity2 ? 0 : grav2_source[real_loc] * cons_data[aid].rho;
                const auto ge_source  = gm1_source * prim_buff[tid].v1 + gm2_source * prim_buff[tid].v2;
                return nt::Conserved<2>{0, gm1_source, gm2_source, ge_source};
            } else {
                const auto gm2_source = zero_gravity2 ? 0 : grav2_source[real_loc] * cons_data[aid].rho;
                const auto gm3_source = zero_gravity3 ? 0 : grav3_source[real_loc] * cons_data[aid].rho;
                const auto ge_source  = gm1_source * prim_buff[tid].v1 + gm2_source * prim_buff[tid].v2 + gm3_source * prim_buff[tid].v3;
                return nt::Conserved<3>{0, gm1_source, gm2_source, gm3_source, ge_source};
            }
        }();

        if constexpr(dim == 1) {
            switch(geometry)
            {
                case simbi::Geometry::CARTESIAN:
                {
                    cons_data[ia] -= ((frf - flf) * invdx1 - source_terms - gravity) * dt * step;
                    break;
                }
                default:
                {
                    const real rlf    = x1l + vfaceL * step * dt; 
                    const real rrf    = x1r + vfaceR * step * dt;
                    const real rmean  = helpers::get_cell_centroid(rrf, rlf, geometry);
                    const real sR     = 4.0 * M_PI * rrf * rrf; 
                    const real sL     = 4.0 * M_PI * rlf * rlf; 
                    const real dV     = 4.0 * M_PI * rmean * rmean * (rrf - rlf);    
                    const real factor = (mesh_motion) ? dV : 1;         
                    const real pc     = prim_buff[txa].p;
                    const real invdV  = 1 / dV;
                    const auto geom_sources = nt::Conserved<1>{0.0, pc * (sR - sL) * invdV, 0.0};
                    cons_data[ia] -= ( (frf * sR - flf * sL) * invdV - geom_sources - source_terms - gravity) * step * dt * factor;
                    break;
                }
            } // end switch
        } else if constexpr(dim == 2) {
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                {
                    cons_data[aid] -= ( (frf - flf) * invdx1 + (grf - glf) * invdx2 - source_terms - gravity) * step * dt;
                    break;
                }
                
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl           = x1l + vfaceL * step * dt; 
                        const real rr           = x1r + vfaceR * step * dt;
                        const real rmean        = helpers::get_cell_centroid(rr, rl, geometry);
                        const real tl           = helpers::my_max<real>(x2min + (jj - 0.5) * dx2 , x2min);
                        const real tr           = helpers::my_min<real>(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dcos         = std::cos(tl) - std::cos(tr);
                        const real dV           = 2.0 * M_PI * (1.0 / 3.0) * (rr * rr * rr - rl * rl * rl) * dcos;
                        const real invdV        = 1.0 / dV;
                        const real s1R          = 2.0 * M_PI * rr * rr * dcos; 
                        const real s1L          = 2.0 * M_PI * rl * rl * dcos; 
                        const real s2R          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tr);
                        const real s2L          = 2.0 * M_PI * 0.5 * (rr * rr - rl * rl) * std::sin(tl);
                        const real factor       = (mesh_motion) ? dV : 1;  

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].v1;
                        const real vc   = prim_buff[tid].v2;
                        const real pc   = prim_buff[tid].p;

                        const nt::Conserved<2> geom_source  = {
                            0, 
                              (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * uc * vc) / rmean + pc * (s2R - s2L) * invdV, 
                            0
                        };

                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step * factor;
                        break;
                    }
                case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        const real rl    = x1l + vfaceL * step * dt; 
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = helpers::get_cell_centroid(rr, rl, simbi::Geometry::PLANAR_CYLINDRICAL);
                        // const real tl           = helpers::my_max(x2min + (jj - 0.5) * dx2 , x2min);
                        // const real tr           = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                        const real dV    = rmean * (rr - rl) * dx2;
                        const real invdV = 1.0 / dV;
                        const real s1R   = rr * dx2; 
                        const real s1L   = rl * dx2; 
                        const real s2R   = (rr - rl); 
                        const real s2L   = (rr - rl); 

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].v1;
                        const real vc   = prim_buff[tid].v2;
                        const real pc   = prim_buff[tid].p;

                        const nt::Conserved<2> geom_source  = {
                            0, 
                              (rhoc * vc * vc) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * uc * vc) / rmean, 
                            0
                        };
                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms 
                            - gravity
                        ) * dt * step;
                        break;
                    }
                default:
                    {
                        const real rl    = x1l + vfaceL * step * dt; 
                        const real rr    = x1r + vfaceR * step * dt;
                        const real rmean = helpers::get_cell_centroid(rr, rl, simbi::Geometry::AXIS_CYLINDRICAL);
                        const real dV    = rmean * (rr - rl) * dx2;
                        const real invdV = 1.0 / dV;
                        const real s1R   = rr * dx2; 
                        const real s1L   = rl * dx2; 
                        const real s2R   = rmean * (rr - rl); 
                        const real s2L   = rmean * (rr - rl);  

                        // Grab central primitives
                        const real pc   = prim_buff[tid].p;
                        const auto geom_source  = nt::Conserved<2>{0, pc * (s1R - s1L) * invdV, 0, 0};
                        cons_data[aid] -= ( 
                              (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step;
                        break;
                    }
            } // end switch
        } else {
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    {
                        cons_data[aid] -= ( (frf  - flf ) * invdx1 + (grf - glf) * invdx2 + (hrf - hlf) * invdx3 - source_terms - gravity) * dt * step;
                        break;
                    }
                case simbi::Geometry::SPHERICAL:
                    {
                        const real rl     = x1l + vfaceL * step * dt; 
                        const real rr     = x1r + vfaceR * step * dt;
                        const real tl     = get_x2face(jj, 0);
                        const real tr     = get_x2face(jj, 1); 
                        const real ql     = get_x3face(kk, 0);
                        const real qr     = get_x3face(kk, 1); 
                        const real rmean  = helpers::get_cell_centroid(rr, rl, simbi::Geometry::SPHERICAL);
                        const real s1R    = rr * rr; 
                        const real s1L    = rl * rl; 
                        const real s2R    = std::sin(tr);
                        const real s2L    = std::sin(tl);
                        const real thmean = 0.5 * (tl + tr);
                        const real sint   = std::sin(thmean);
                        const real dV1    = rmean * rmean * (rr - rl);             
                        const real dV2    = rmean * sint  * (tr - tl); 
                        const real dV3    = rmean * sint  * (qr - ql); 
                        const real cot    = std::cos(thmean) / sint;

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].v1;
                        const real vc   = prim_buff[tid].v2;
                        const real wc   = prim_buff[tid].v3;
                        const real pc   = prim_buff[tid].p;

                        const auto geom_source  = nt::Conserved<3>{0, 
                            ( rhoc * (vc * vc + wc * wc)) / rmean + pc * (s1R - s1L) / dV1,
                              rhoc * (wc * wc * cot - uc * vc) / rmean + pc * (s2R - s2L)/dV2 , 
                            - rhoc * wc * (uc + vc * cot) / rmean, 
                            0
                        };
                        cons_data[aid] -= ( 
                            (frf * s1R - flf * s1L) / dV1 
                            + (grf * s2R - glf * s2L) / dV2 
                            + (hrf - hlf) / dV3 
                            - geom_source 
                            - source_terms
                            - gravity
                        ) * dt * step;
                        break;
                    }
                default:
                    {
                        const real rl     = x1l + vfaceL * step * dt; 
                        const real rr     = x1r + vfaceR * step * dt;
                        const real ql     = get_x2face(jj, 0);
                        const real qr     = get_x2face(jj, 1); 
                        const real zl     = get_x3face(kk, 0);
                        const real zr     = get_x3face(kk, 1); 
                        const real rmean  = helpers::get_cell_centroid(rr, rl, simbi::Geometry::CYLINDRICAL);
                        const real s1R    = rr * (zr - zl) * (qr - ql); 
                        const real s1L    = rl * (zr - zl) * (qr - ql); 
                        const real s2R    = (rr - rl) * (zr - rl);
                        const real s2L    = (rr - rl) * (zr - rl);
                        // const real s3L          = rmean * (rr - rl) * (tr - tl);
                        // const real s3R          = s3L;
                        // const real thmean       = 0.5 * (tl + tr);
                        const real dV    = rmean  * (rr - rl) * (zr - zl) * (qr - ql);
                        const real invdV = 1/ dV;

                        // Grab central primitives
                        const real rhoc = prim_buff[tid].rho;
                        const real uc   = prim_buff[tid].v1;
                        const real vc   = prim_buff[tid].v2;
                        // const real wc   = prim_buff[tid].v3;
                        const real pc   = prim_buff[tid].p;

                        const auto geom_source  = nt::Conserved<3>{
                            0, 
                            (rhoc * (vc * vc)) / rmean + pc * (s1R - s1L) * invdV, 
                            - (rhoc * uc * vc) / rmean , 
                            0, 
                            0
                        };
                        cons_data[aid] -= ( 
                              (frf * s1R - flf * s1L) * invdV 
                            + (grf * s2R - glf * s2L) * invdV 
                            + (hrf - hlf) * invdV 
                            - geom_source 
                            - source_terms
                        ) * dt * step;
                        break;
                    }
            } // end switch
        }
    });
}
//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
template<int dim>
void Newtonian<dim>::simulate(
    std::function<real(real)> const &a,
    std::function<real(real)> const &adot,
    std::optional<Newtonian<dim>::function_t> const &d_outer,
    std::optional<Newtonian<dim>::function_t> const &m1_outer,
    std::optional<Newtonian<dim>::function_t> const &m2_outer,
    std::optional<Newtonian<dim>::function_t> const &m3_outer,
    std::optional<Newtonian<dim>::function_t> const &e_outer
)
{   
    helpers::anyDisplayProps();
    // set the primtive functionals
    this->dens_outer = d_outer.value_or(nullptr);
    this->mom1_outer = m1_outer.value_or(nullptr);
    this->mom2_outer = m2_outer.value_or(nullptr);
    this->mom3_outer = m3_outer.value_or(nullptr);
    this->enrg_outer = e_outer.value_or(nullptr);

    if constexpr(dim == 1) {
        this->all_outer_bounds = (
            d_outer.has_value() 
            && m1_outer.has_value()
            && enrg_outer
        );
    } else if constexpr(dim == 2) {
        this->all_outer_bounds = (
            d_outer.has_value() 
            && m1_outer.has_value()
            && m2_outer.has_value() 
            && e_outer.has_value()
        );
    } else {
        this->all_outer_bounds = (
            d_outer.has_value() 
            && m1_outer.has_value()
            && m2_outer.has_value() 
            && m3_outer.has_value()
            && e_outer.has_value()
        );
    }

    // Stuff for moving mesh 
    this->hubble_param = adot(t) / a(t);
    this->mesh_motion  = (hubble_param != 0);
    this->changing_volume = mesh_motion && geometry != simbi::Geometry::CARTESIAN;

    if (mesh_motion && all_outer_bounds) {
        if constexpr(dim == 1) {
            outer_zones.resize(first_order ? 1 : 2);
            const real dV  = get_cell_volume(active_zones - 1);
            outer_zones[0] = conserved_t{
                dens_outer(x1max), 
                mom1_outer(x1max), 
                enrg_outer(x1max)} * dV;
            outer_zones.copyToGpu();
        } else if constexpr(dim == 2) {
            outer_zones.resize(ny);
            for (luint jj = 0; jj < ny; jj++) {
                const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid);
                const real dV    = get_cell_volume(xactive_grid - 1, jreal);
                outer_zones[jj]  = conserved_t{
                    dens_outer(x1max, x2[jreal]), 
                    mom1_outer(x1max, x2[jreal]), 
                    mom2_outer(x1max, x2[jreal]), 
                    enrg_outer(x1max, x2[jreal])} * dV;
            }
            outer_zones.copyToGpu();
        } else {
            outer_zones.resize(ny * nz);
            for (luint kk = 0; kk < nz; kk++)
            {       
                const auto kreal = helpers::get_real_idx(kk, radius, zactive_grid);    
                for (luint jj = 0; jj < ny; jj++) {
                    const auto jreal = helpers::get_real_idx(jj, radius, yactive_grid);
                    const real dV    = get_cell_volume(xactive_grid - 1, jreal, kreal);
                    outer_zones[kk * ny + jj]  = conserved_t{
                        dens_outer(x1max, x2[jreal], x3[kreal]), 
                        mom1_outer(x1max, x2[jreal], x3[kreal]), 
                        mom2_outer(x1max, x2[jreal], x3[kreal]),
                        mom3_outer(x1max, x2[jreal], x3[kreal]), 
                        enrg_outer(x1max, x2[jreal], x3[kreal])} * dV;
                }
            }
            outer_zones.copyToGpu();
        }
    }

    if (x2max == 0.5 * M_PI){
        this->half_sphere = true;
    }

    inflow_zones.resize(dim * 2);
    for (int i = 0; i < 2 * dim; i++) {
        this->bcs.push_back(helpers::boundary_cond_map.at(boundary_conditions[i]));
        if constexpr(dim == 1) {
            this->inflow_zones[i] = nt::Conserved<1>{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2]};
        } else if constexpr(dim == 2) {
            this->inflow_zones[i] = nt::Conserved<2>{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2], boundary_sources[i][3]};
        } else {
            this->inflow_zones[i] = nt::Conserved<3>{boundary_sources[i][0], boundary_sources[i][1], boundary_sources[i][2], boundary_sources[i][3], boundary_sources[i][4]};
        }
    }

    // Write some info about the setup for writeup later
    setup.x1max = x1[xactive_grid - 1];
    setup.x1min = x1[0];
    setup.x1    = x1;
    if constexpr(dim > 1) {
        setup.x2max = x2[yactive_grid - 1];
        setup.x2min = x2[0];
        setup.x2    = x2;
    }
    if constexpr(dim > 2) {
        setup.x3max = x3[zactive_grid - 1];
        setup.x3min = x3[0];
        setup.x3    = x3;
    }
    
    setup.nx                  = nx;
    setup.ny                  = ny;
    setup.nz                  = nz;
    setup.xactive_zones       = xactive_grid;
    setup.yactive_zones       = yactive_grid;
    setup.zactive_zones       = zactive_grid;
    setup.x1_cell_spacing      = cell2str.at(x1_cell_spacing);
    setup.x2_cell_spacing      = cell2str.at(x2_cell_spacing);
    setup.x3_cell_spacing      = cell2str.at(x3_cell_spacing);
    setup.ad_gamma            = gamma;
    setup.first_order         = first_order;
    setup.coord_system        = coord_system;
    setup.using_fourvelocity  = false;
    setup.regime              = "classical";
    setup.mesh_motion         = mesh_motion;
    setup.boundary_conditions = boundary_conditions;
    setup.dimensions          = dim;

    cons.resize(total_zones);
    prims.resize(total_zones);
    troubled_cells.resize(total_zones, 0);
    dt_min.resize(active_zones);

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < total_zones; i++)
    {
        const real rho  = state[0][i];
        const real m1 = state[1][i];
        const real m2 = [&]{
            if constexpr(dim < 2) {
                return static_cast<real>(0.0);
            }
            return state[2][i];
        }();
        const real m3 = [&]{
            if constexpr(dim < 3) {
                return static_cast<real>(0.0);
            }
            return state[3][i];
        }();
        const real E = [&] {
            if constexpr(dim == 1) {
                return state[2][i];
            } else if constexpr(dim == 2) {
                return state[3][i];
            } else {
                return state[4][i];
            }
        }(); 
        if constexpr(dim == 1) {
            cons[i] = nt::Conserved<1>{rho, m1, E};
        } else if constexpr(dim == 2) {
            cons[i] = nt::Conserved<2>{rho, m1, m2, E};
        } else {
            cons[i] = nt::Conserved<3>{rho, m1, m2, m3, E};
        }
    }

    cons.copyToGpu();
    prims.copyToGpu();
    dt_min.copyToGpu();
    density_source.copyToGpu();
    m1_source.copyToGpu();
    if constexpr(dim > 1) {
        m2_source.copyToGpu();
    }
    if constexpr(dim > 2) {
        m3_source.copyToGpu();
    }
    if constexpr(dim > 1) {
        object_pos.copyToGpu();
    }
    energy_source.copyToGpu();
    inflow_zones.copyToGpu();
    bcs.copyToGpu();
    troubled_cells.copyToGpu();
    sourceG1.copyToGpu();
    if constexpr(dim > 1) {
        sourceG2.copyToGpu();
    }
    if constexpr(dim > 2) {
        sourceG3.copyToGpu();
    }

    // Setup the system
    const luint xblockdim    = xactive_grid > gpu_block_dimx ? gpu_block_dimx : xactive_grid;
    const luint yblockdim    = yactive_grid > gpu_block_dimy ? gpu_block_dimy : yactive_grid;
    const luint zblockdim    = zactive_grid > gpu_block_dimz ? gpu_block_dimz : zactive_grid;
    this->radius             = (first_order) ? 1 : 2;
    this->step               = (first_order) ? 1 : 0.5;
    const luint xstride      = (global::BuildPlatform == global::Platform::GPU) ? xblockdim + 2 * radius: nx;
    const luint ystride      = (dim < 3) ? 1 : (global::BuildPlatform == global::Platform::GPU) ? yblockdim + 2 * radius: ny;
    const auto  xblockspace  =  xblockdim + 2 * radius;
    const auto  yblockspace  = (dim < 2) ? 1 : yblockdim + 2 * radius;
    const auto  zblockspace  = (dim < 3) ? 1 : zblockdim + 2 * radius;
    const luint shBlockSpace = xblockspace * yblockspace * zblockspace;
    const luint shBlockBytes = shBlockSpace * sizeof(nt::Primitive<dim>);
    const auto fullP         = simbi::ExecutionPolicy({nx, ny, nz}, {xblockdim, yblockdim, zblockdim});
    const auto activeP       = simbi::ExecutionPolicy({xactive_grid, yactive_grid, zactive_grid}, {xblockdim, yblockdim, zblockdim}, shBlockBytes);
    
    if constexpr(global::BuildPlatform == global::Platform::GPU){
        writeln("Requested shared memory: {} bytes", shBlockBytes);
    }

    cons2prim(fullP);
    if constexpr(global::BuildPlatform == global::Platform::GPU) {
        adapt_dt(activeP);
    } else {
        adapt_dt();
    }

    // Using a sigmoid decay function to represent when the source terms turn off.
    time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
    // Save initial condition
    if (t == 0 || init_chkpt_idx == 0) { 
        nt::write2file<dim>(*this, setup, data_directory, t, 0, chkpt_interval, checkpoint_zones);
        if constexpr(dim == 1) {
            helpers::config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
        } else if constexpr(dim == 2) {
            helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
        } else {
            helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
        }
    }

    this->n = 0;
    // Simulate :)
    simbi::detail::logger::with_logger(*this, tend, [&] {
        if (inFailureState){
            return;
        }
        advance(activeP, xstride, ystride);
        cons2prim(fullP);
        if constexpr(dim == 1) {
            helpers::config_ghosts1D(fullP, cons.data(), nx, first_order, bcs.data(), outer_zones.data(), inflow_zones.data());
        } else if constexpr(dim == 2) {
            helpers::config_ghosts2D(fullP, cons.data(), nx, ny, first_order, geometry, bcs.data(), outer_zones.data(), inflow_zones.data(), half_sphere);
        } else {
            helpers::config_ghosts3D(fullP, cons.data(), nx, ny, nz, first_order, bcs.data(), inflow_zones.data(), half_sphere, geometry);
        }

        if constexpr(global::BuildPlatform == global::Platform::GPU) {
            adapt_dt(activeP);
        } else {
            adapt_dt();
        }
        time_constant = helpers::sigmoid(t, engine_duration, step * dt, constant_sources);
        t += step * dt;
    });

    if (inFailureState){
        emit_troubled_cells();
    }
};
