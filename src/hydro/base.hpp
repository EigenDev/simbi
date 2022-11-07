#ifndef BASE_HPP
#define BASE_HPP

#include <string>
#include "common/hydro_structs.hpp"
#include "build_options.hpp"
namespace simbi
{
    struct HydroBase
    {
        // Initializer members
        std::vector<std::vector<real>> state;
        real gamma;
        real cfl;
        std::vector<real> x1, x2, x3;
        std::string coord_system, data_directory;

        // Common mmembers
        DataWriteMembers setup;
        real dt, t, tend, t_interval, chkpt_interval, plm_theta, decay_constant, hubble_param; 
        real x1min, x1max, x2min, x2max, x3min, x3max;
        real dlogx1, dx1, dx2, dx3, dlogt, tstart, engine_duration;
        bool first_order, periodic, linspace, hllc, mesh_motion, reflecting_theta, quirk_smoothing;
        luint nzones, active_zones, idx_active, total_zones, n, nx, ny, nz, init_chkpt_idx, radius, pseudo_radius;
        luint xphysical_grid, yphysical_grid, zphysical_grid;
        simbi::Solver sim_solver;
        simbi::BoundaryCondition bc;
        simbi::Geometry geometry;
        simbi::Cellspacing x1cell_spacing, x2cell_spacing, x3cell_spacing;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real> > sources;
        volatile bool inFailureState; 

        protected:
        HydroBase(){}
        ~HydroBase(){}

        HydroBase(
            std::vector<std::vector<real>> state, 
            real gamma, 
            real cfl, 
            std::vector<real> x1,
            std::string coord_system)
            :
            state(state),
            gamma(gamma),
            cfl(cfl),
            x1(x1),
            coord_system(coord_system),
            inFailureState(false),
            nx(state[0].size())
        {

        }

        HydroBase(
            std::vector<std::vector<real>> state, 
            luint nx, 
            luint ny, 
            real gamma,
            std::vector<real> x1, 
            std::vector<real> x2, 
            real cfl,
            std::string coord_system = "cartesian")
        :
            state(state),
            nx(nx),
            ny(ny),
            gamma(gamma),
            x1(x1),
            x2(x2),
            cfl(cfl),
            coord_system(coord_system),
            inFailureState(false),
            nzones(state[0].size())
        {
        }

         HydroBase(
            std::vector<std::vector<real>> state, 
            luint nx, 
            luint ny, 
            luint nz,
            real gamma,
            std::vector<real> x1, 
            std::vector<real> x2, 
            std::vector<real> x3, 
            real cfl,
            std::string coord_system = "cartesian")
        :
            state(state),
            nx(nx),
            ny(ny),
            nz(nz),
            gamma(gamma),
            x1(x1),
            x2(x2),
            x3(x3),
            cfl(cfl),
            coord_system(coord_system),
            inFailureState(false),
            nzones(state[0].size())
        {
        }
    };
    
} // namespace simbi
#endif