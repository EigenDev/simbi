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
#include "common/config.hpp"
#include "util/exec_policy.hpp"

namespace simbi {
    enum class SOLVER{HLLE, HLLC};

    struct Newtonian1D {
        // Initializer list args
        std::vector<std::vector<real>> init_state;
        real gamma, cfl;
        std::vector<real> r;
        std::string coord_system;

        real plm_theta, tend, dt, engine_duration, t, decay_constant;
        bool first_order, periodic, linspace, hllc, inFailureState;

        
        std::vector<hydro1d::Conserved> cons, cons_n; 
        std::vector<hydro1d::Primitive> prims;
        std::vector<real> xvertices, sourceRho, sourceMom, sourceE;
        luint nzones, active_zones, idx_active, i_start, i_bound, n, nx;
        simbi::SOLVER sim_solver;
        CLattice1D coord_lattice;

        //==============================================================
        // Create dynamic array instances that will live on device
        //==============================================================
        //             GPU RESOURCES
        //==============================================================
        luint blockSize;
        hydro1d::Conserved *gpu_cons, *gpu_du_dt, *gpu_u1;
        hydro1d::Primitive *gpu_prims;
        real               *gpu_pressure_guess, *gpu_sourceD, *gpu_sourceS, *gpu_source0, *dt_min;
        CLattice1D         *gpu_coord_lattice;
        
        Newtonian1D();
        Newtonian1D(
            std::vector<std::vector<real>> init_state, 
            real gamma, 
            real cfl,
            std::vector<real> r, 
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
            real tstart = 0.0,
            real tend = 0.1,
            real init_dt = 1.e-4,
            real plm_theta = 1.5,
            real engine_duration = 10,
            real chkpt_luinterval = 0.1,
            std::string data_directory = "data/",
            std::string boundary_condition = "outflow",
            bool first_order = true,
            bool linspace = true,
            bool hllc = false);
    };
}

#endif