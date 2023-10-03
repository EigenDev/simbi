/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO3D_HIP_HPP
#define SRHYDRO3D_HIP_HPP

#include <vector>
#include "common/hydro_structs.hpp"
#include "common/helpers.hip.hpp"
#include "base.hpp"

namespace simbi
{
    struct SRHD3D : public HydroBase
    {
        using primitive_t     = sr3d::Primitive;
        using conserved_t     = sr3d::Conserved;
        using primitive_soa_t = sr3d::PrimitiveSOA;
        const static int dimensions = 3;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> sourceD, sourceS1, sourceS2, sourceS3, sourceTau, pressure_guess, dt_min;
        ndarray<bool> object_pos;
        bool scalar_all_zeros, quirk_smoothing;

        /* Methods */
        SRHD3D();
        SRHD3D(
            std::vector<std::vector<real>> &state3D, 
            luint nx, 
            luint ny,
            luint nz, 
            real gamma, 
            std::vector<real> &x1,
            std::vector<real> &x2, 
            std::vector<real> &x3,
            real cfl, 
            std::string coord_system);
        ~SRHD3D();

        void cons2prim(const ExecutionPolicy<> &p);

        void advance(
            const ExecutionPolicy<> &p,
            const luint xstride,
            const luint ystride);

        GPU_CALLABLE_MEMBER
        sr3d::Eigenvals calc_eigenvals(
            const sr3d::Primitive &primsL,
            const sr3d::Primitive &primsR,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2cons(const sr3d::Primitive &prims);
        
        sr3d::Conserved calc_hll_state(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        sr3d::Conserved calc_luintermed_statesSR2D(const sr3d::Primitive &prims,
                                                 const sr3d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hllc_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2flux(const sr3d::Primitive &prims, const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hll_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);  

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<> &p);

        std::vector<std::vector<real>> simulate3D(
            const std::vector<std::vector<real>> &sources,
            const std::vector<bool> &object_cells,
            real tstart, 
            real tend, 
            real dlogt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_interval,
            int  chkpt_idx,
            std::string data_directory, 
            std::vector<std::string> boundary_conditions,
            bool first_order,
            bool linspace, 
            const std::string solver,
            bool constant_sources,
            std::vector<std::vector<real>> boundary_sources);

        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                    const real x1l = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                    if (side == 0) {
                        return x1l;
                    }
                    return helpers::my_min(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
            default:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
                break;
            }
        }


        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side)
        {
            const real x2l = helpers::my_max(x2min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return x2l;
            } 
            return helpers::my_min(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x3face(const lint ii, const int side)
        {

            const real x3l = helpers::my_max(x3min  + (ii - static_cast<real>(0.5)) * dx3,  x3min);
            if (side == 0) {
                return x3l;
            } 
            return helpers::my_min(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
        }

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const real step)
        {
            const real x1l     = get_x1face(ii, 0);
            const real x1r     = get_x1face(ii, 1);
            // const real x1lf    = x1l * (1.0 + step * dt * hubble_param);
            // const real x1rf    = x1r * (1.0 + step * dt * hubble_param);
            const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
            const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
            const real dcos   = std::cos(tl) - std::cos(tr);
            const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * dcos);
            return dV;
            
        }

        void emit_troubled_cells() {
            troubled_cells.copyFromGpu();
            cons.copyFromGpu();
            prims.copyFromGpu();
            for (luint gid = 0; gid < total_zones; gid++)
            {
                if (troubled_cells[gid] != 0) {
                    const luint xpg   = xphysical_grid;
                    const luint ypg   = yphysical_grid;
                    const luint kk    = detail::get_height(gid, xpg, ypg);
                    const luint jj    = detail::get_row(gid, xpg, ypg, kk);
                    const luint ii    = detail::get_column(gid, xpg, ypg, kk);
                    const lint ireal  = helpers::get_real_idx(ii, radius, xphysical_grid);
                    const lint jreal  = helpers::get_real_idx(jj, radius, yphysical_grid); 
                    const lint kreal  = helpers::get_real_idx(kk, radius, zphysical_grid); 
                    const real x1l    = get_x1face(ireal, 0);
                    const real x1r    = get_x1face(ireal, 1);
                    const real x2l    = get_x2face(jreal, 0);
                    const real x2r    = get_x2face(jreal, 1);
                    const real x3l    = get_x3face(kreal, 0);
                    const real x3r    = get_x3face(kreal, 1);
                    const real x1mean = helpers::calc_any_mean(x1l, x1r, x1cell_spacing);
                    const real x2mean = helpers::calc_any_mean(x2l, x2r, x2cell_spacing);
                    const real x3mean = helpers::calc_any_mean(x3l, x3r, x3cell_spacing);

                    const real et  = (cons[gid].d + cons[gid].tau + prims[gid].p);
                    const real s   = std::sqrt(cons[gid].s1 * cons[gid].s1 + cons[gid].s2 * cons[gid].s2 + cons[gid].s3 * cons[gid].s3);
                    const real v2  = (s * s) / (et * et);
                    const real w   = 1 / std::sqrt(1 - v2);
                    printf("\nCons2Prim cannot converge\nDensity: %.2e, Pressure: %.2e, Vsq: %.2e, x1coord: %.2e, x2coord: %.2e, x3coord: %.2e, iter: %d\n", 
                    cons[gid].d / w, prims[gid].p, v2, x1mean, x2mean, x3mean, troubled_cells[gid]);
                }
            }
        }
    };
}

template<>
struct is_relativistic<simbi::SRHD3D>
{
    static constexpr bool value = true;
};
#endif