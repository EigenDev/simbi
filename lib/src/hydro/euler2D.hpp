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
#include "common/config.hpp"
#include "common/helpers.hpp"


namespace simbi {
    class Newtonian2D {
        public:
        std::vector<std::vector<real> > init_state, sources;
        std::vector<hydro2d::Conserved> cons, cons_n;
        std::vector<hydro2d::Primitive> prims;
        std::vector<real> sourceRho, sourceM1, sourceM2, sourceE;
        real plm_theta, gamma, tend, CFL, dt, decay_const;
        bool first_order, periodic, hllc, linspace, inFailureState;
        std::string coord_system;
        std::vector<real> x1, x2;
        luint nzones, ny, nx, active_zones, idx_active, n;
        luint xphysical_grid, yphysical_grid, x_bound, y_bound;
        CLattice2D coord_lattice;
        simbi::Solver solver;

        real x2max, x2min, x1min, x1max, dx2, dx1, dlogx1;
        bool rho_all_zeros, m1_all_zeros, m2_all_zeros, e_all_zeros;
        
        // GPU Mirrors
        real *gpu_sourceRho, *gpu_sourceM1, *gpu_sourceM2, *gpu_sourceE, *dt_min;
        hydro2d::Primitive *gpu_prims;
        hydro2d::Conserved *gpu_cons;


        Newtonian2D();
        Newtonian2D(std::vector< std::vector<real> > init_state, 
            luint nx, 
            luint ny,
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            real CFL, 
            std::string coord_system);

        ~Newtonian2D();

        void cons2prim();
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

        void adapt_dt();
        void adapt_dt(Newtonian2D *dev, const simbi::Geometry geometry, const ExecutionPolicy<> p, luint bytes);

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
            real tstart = 0., 
            real tend = 0.1, 
            real init_dt = 1.e-4, 
            real plm_theta = 1.5,
            real engine_duration = 10, 
            real chkpt_luinterval = 0.1,
            std::string data_directory = "data/", 
            bool first_order = true,
            bool periodic = false, 
            bool linspace = true, 
            bool hllc = false);
        
    };
}

#endif