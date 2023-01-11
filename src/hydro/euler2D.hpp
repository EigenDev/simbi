/* 
* interface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER2D_HPP
#define EULER2D_HPP

#include <vector>
#include "util/exec_policy.hpp"
#include "common/hydro_structs.hpp"
#include "common/enums.hpp"
#include "common/helpers.hpp"
#include "build_options.hpp"
#include "base.hpp"
namespace simbi {
    struct Newtonian2D : public HydroBase {

        using primitive_t = hydro2d::Primitive;
        using conserved_t = hydro2d::Conserved;
        using primitive_soa_t = hydro2d::PrimitiveSOA;
        const static int dimensions = 2;

        // Simulation Param
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> sourceRho, sourceM1, sourceM2, sourceE, dt_min;
        bool rho_all_zeros, m1_all_zeros, m2_all_zeros, e_all_zeros;
        
        Newtonian2D();
        Newtonian2D(
            std::vector<std::vector<real>> state, 
            luint nx, 
            luint ny,
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            real cfl, 
            std::string coord_system);

        ~Newtonian2D();

        void cons2prim(const ExecutionPolicy<> &p);

        GPU_CALLABLE_MEMBER
        hydro2d::Eigenvals calc_eigenvals(
            const hydro2d::Primitive &left_state, 
            const hydro2d::Primitive &right_state,
            const luint ehat);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved prims2cons(const hydro2d::Primitive &prims);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved prims2flux(const hydro2d::Primitive &prims, const luint ehat);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved calc_hll_flux(
            const hydro2d::Conserved &left_state,
            const hydro2d::Conserved &right_state,
            const hydro2d::Conserved &left_flux,
            const hydro2d::Conserved &right_flux,
            const hydro2d::Primitive &left_prims,
            const hydro2d::Primitive &right_prims,
            const luint ehat);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved calc_hllc_flux(
            const hydro2d::Conserved &left_state,
            const hydro2d::Conserved &right_state,
            const hydro2d::Conserved &left_flux,
            const hydro2d::Conserved &right_flux,
            const hydro2d::Primitive &left_prims,
            const hydro2d::Primitive &right_prims,
            const luint ehat = 1);
        
        GPU_CALLABLE_INLINE
        constexpr real get_xface(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                        return 1.0;
                }
            
            case simbi::Geometry::SPHERICAL:
                {
                        const real rl = (ii > 0 ) ? x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1) :  x1min;
                        if (side == 0) {
                            return rl;
                        } else {
                            return (ii < static_cast<lint>(xphysical_grid - 1)) ? rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                        }
                        break;
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement
                break;
            }
        }

        void adapt_dt();
        void adapt_dt(
            const ExecutionPolicy<> &p, 
            luint bytes);

        void advance(
               const ExecutionPolicy<> &p, 
               const luint bx,
               const luint by);

        std::vector<std::vector<real> > simulate2D(
            const std::vector<std::vector<real>> sources,
            real tstart, 
            real tend, 
            real dlogt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_luinterval,
            int  chkpt_idx,
            std::string data_directory, 
            std::vector<std::string> boundary_conditions,
            bool first_order,
            bool linspace, 
            bool hllc,
            bool constant_sources,
            std::vector<std::vector<real>> boundary_sources);


        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::AXIS_CYLINDRICAL:
            case simbi::Geometry::CARTESIAN:
                {
                        const real xl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                        if (side == 0) {
                            return xl;
                        } else {
                            return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                        }
                }
            case simbi::Geometry::PLANAR_CYLINDRICAL:
            case simbi::Geometry::SPHERICAL:
                {
                        const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                        if (side == 0) {
                            return rl;
                        } else {
                            return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                        }
                }
            }
        }


        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side)
        {

            const real yl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return yl;
            } else {
                return helpers::my_min(yl + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
            }
        }

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const simbi::Geometry geometry)
        {
            switch (geometry)
            {
            case simbi::Geometry::SPHERICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dcos   = std::cos(tl) - std::cos(tr);
                const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (xr * xr * xr - xl * xl * xl) * dcos);
                return dV;
            }
            case simbi::Geometry::PLANAR_CYLINDRICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dx2    = tr - tl;
                const real dV     = (1.0 / 2.0) * (xr * xr - xl * xl) * dx2;
                return dV;
            }

            case simbi::Geometry::AXIS_CYLINDRICAL:
            {
                const real xl     = get_x1face(ii, geometry, 0);
                const real xr     = get_x1face(ii, geometry, 1);
                const real zl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                const real zr     = helpers::my_min(zl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                const real dx2    = zr - zl;
                const real dV     = (1.0 / 2.0) * (xr * xr - xl * xl) * dx2;
                return dV;
            }
            
            default:
                break;
            }
            
        }
        
    };
}

#endif