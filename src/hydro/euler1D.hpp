/* 
* luinterface between python construction of the 1D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER1D_HPP
#define EULER1D_HPP

#include <vector>
#include <string>
#include "common/hydro_structs.hpp"
#include "common/clattice1D.hpp"
#include "common/helpers.hpp"
#include "common/enums.hpp"
#include "build_options.hpp"
#include "util/exec_policy.hpp"

namespace simbi {
    enum class SOLVER{HLLE, HLLC};

    struct Newtonian1D {
        // Initializer list args
        std::vector<std::vector<real>> state;
        real gamma;
        real cfl;
        std::vector<real> x1;
        std::string coord_system;

        real plm_theta, tend, dt, engine_duration, t, decay_constant, hubble_param, x1min , x1max, dlogx1, dx1, dlogt;
        bool first_order, periodic, linspace, hllc, inFailureState, mesh_motion;

        
        std::vector<hydro1d::Conserved> cons, cons_n; 
        std::vector<hydro1d::Primitive> prims;
        std::vector<real> xvertices, sourceRho, sourceMom, sourceE;
        luint nzones, active_zones, idx_active, total_zones, n, nx;
        simbi::SOLVER sim_solver;
        CLattice1D coord_lattice;
        simbi::BoundaryCondition bc;
        simbi::Geometry geometry;
        simbi::Cellspacing xcell_spacing;

        //==============================================================
        // Create dynamic array instances that will live on device
        //==============================================================
        //             GPU RESOURCES
        //==============================================================
        luint blockSize;
        hydro1d::Conserved *gpu_cons;
        hydro1d::Primitive *gpu_prims;
        real               *gpu_sourceRho, *gpu_sourceMom, *gpu_sourceE, *dt_min;
        
        Newtonian1D();
        Newtonian1D(
            std::vector<std::vector<real>> state, 
            real gamma, 
            real cfl,
            std::vector<real> x1, 
            std::string coord_system);
        ~Newtonian1D();

        // Calculate the wave speeds from the Jacobian Matrix formed by the Euler Eqns
        GPU_CALLABLE_MEMBER
        hydro1d::Eigenvals calc_eigenvals(const hydro1d::Primitive &left_state, const hydro1d::Primitive &right_state);

        void cons2prim(ExecutionPolicy<> p, Newtonian1D *dev = nullptr, simbi::MemSide user = simbi::MemSide::Host);
        void adapt_dt();
        void adapt_dt(Newtonian1D *dev, luint blockSize, luint tblock);
        
        GPU_CALLABLE_MEMBER
        hydro1d::Conserved prims2cons(const hydro1d::Primitive &prims);

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved prims2flux(const hydro1d::Primitive &prims);

        GPU_CALLABLE_INLINE
        constexpr real get_xface(const lint ii, const simbi::Geometry geometry, const int side) const
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
            {
                //  TODO: Implement
                break;
            }
            }
        }

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved calc_hll_flux(
            const hydro1d::Primitive &left_prims,
            const hydro1d::Primitive &right_prims,
            const hydro1d::Conserved &left_state,
            const hydro1d::Conserved &right_state,
            const hydro1d::Conserved &left_flux,
            const hydro1d::Conserved &right_flux);

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved calc_hllc_flux(
            const hydro1d::Primitive &left_prims,
            const hydro1d::Primitive &right_prims,
            const hydro1d::Conserved &left_state,
            const hydro1d::Conserved &right_state,
            const hydro1d::Conserved &left_flux,
            const hydro1d::Conserved &right_flux);

        void advance(
            const luint radius,
            const simbi::Geometry geometry,
            const ExecutionPolicy<> p,
            Newtonian1D *dev = nullptr,  
            const luint sh_block_size = 0,
            const simbi::MemSide user = simbi::MemSide::Host);

        std::vector<std::vector<real> > simulate1D(
            std::vector<std::vector<real>> &sources,
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
    };
}

#endif