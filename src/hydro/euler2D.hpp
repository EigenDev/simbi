/* 
* luinterface between python construction of the 2D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER2D_HPP
#define EULER2D_HPP

#include <vector>
#include <string>
#include "util/exec_policy.hpp"
#include "common/hydro_structs.hpp"
#include "common/clattice2D.hpp"
#include "common/enums.hpp"
#include "common/helpers.hpp"
#include "build_options.hpp"

namespace simbi {
    struct Newtonian2D {

        // Init Params (order matters)
        std::vector< std::vector<real> > init_state;
        luint nx; 
        luint ny;
        real gamma;
        std::vector<real> x1;
        std::vector<real> x2;
        real cfl;
        std::string coord_system;

        // Simulation Params
        std::vector<std::vector<real> > sources;
        std::vector<hydro2d::Conserved> cons;
        std::vector<hydro2d::Primitive> prims;
        std::vector<real> sourceRho, sourceM1, sourceM2, sourceE;
        real plm_theta, tend, dt, decay_const, hubble_param;
        bool first_order, periodic, hllc, linspace, inFailureState, mesh_motion, quirk_smoothing, reflecting_theta;
        luint nzones, active_zones, idx_active, n;
        luint xphysical_grid, yphysical_grid, total_zones;
        CLattice2D coord_lattice;
        simbi::Solver solver;
        simbi::Geometry geometry;
        simbi::BoundaryCondition bc;
        simbi::Cellspacing x1cell_spacing, x2cell_spacing;

        real x2max, x2min, x1min, x1max, dx2, dx1, dlogx1, dlogt;
        bool rho_all_zeros, m1_all_zeros, m2_all_zeros, e_all_zeros;
        
        //==============================================================
        // Create dynamic array instances that will live on device
        //==============================================================
        //             GPU RESOURCES
        //==============================================================
        luint blockSize;
        real *gpu_sourceRho,*gpu_sourceM1, *gpu_sourceM2, *gpu_sourceE, *dt_min;
        hydro2d::Primitive  *gpu_prims;
        hydro2d::Conserved  *gpu_cons;


        Newtonian2D();
        Newtonian2D(
            std::vector< std::vector<real> > init_state, 
            luint nx, 
            luint ny,
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            real cfl, 
            std::string coord_system);

        ~Newtonian2D();

        void cons2prim(
            ExecutionPolicy<> p, 
            Newtonian2D *dev = nullptr, 
            simbi::MemSide user = simbi::MemSide::Host);

        GPU_CALLABLE_MEMBER
        hydro2d::Eigenvals calc_eigenvals(
            const hydro2d::Primitive &left_state, 
            const hydro2d::Primitive &right_state,
            const luint ehat = 1);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved prims2cons(const hydro2d::Primitive &prims);

        GPU_CALLABLE_MEMBER
        hydro2d::Conserved prims2flux(const hydro2d::Primitive &prims, const luint ehat = 1);

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
            Newtonian2D *dev, 
            const simbi::Geometry geometry, 
            const ExecutionPolicy<> p, 
            luint bytes);

        void advance(
               Newtonian2D *s, 
               const ExecutionPolicy<> p, 
               const luint bx,
               const luint by,
               const luint radius, 
               const simbi::Geometry geometry, 
               const simbi::MemSide user = simbi::MemSide::Host);

        std::vector<std::vector<real> > simulate2D(
            const std::vector<std::vector<real>> sources,
            real tstart, 
            real tend, 
            real dlogt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_luinterval,
            std::string data_directory, 
            std::string boundary_condition,
            bool first_order,
            bool linspace, 
            bool hllc);


        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                        const real xl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                        if (side == 0) {
                            return xl;
                        } else {
                            return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                        }
                }
            case simbi::Geometry::SPHERICAL:
                {
                        const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                        if (side == 0) {
                            return rl;
                        } else {
                            return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                        }
                }
            case simbi::Geometry::CYLINDRICAL:
                // TODO: Implement
                break;
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
        real get_cell_volume(const lint ii, const lint jj, const simbi::Geometry geometry, const real step)
        {
            const real xl     = get_x1face(ii, geometry, 0);
            const real xr     = get_x1face(ii, geometry, 1);
            // const real xlf    = xl * (1.0 + step * dt * hubble_param);
            // const real xrf    = xr * (1.0 + step * dt * hubble_param);
            const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
            const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
            const real dcos   = std::cos(tl) - std::cos(tr);
            const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (xr * xr * xr - xl * xl * xl) * dcos);
            return dV;
            
        }
        
    };
}

#endif