#ifndef BASE_HPP
#define BASE_HPP

#include "common/hydro_structs.hpp"
#include "util/managed.hpp"
#include "build_options.hpp"
#include "util/ndarray.hpp"

namespace simbi
{
    struct HydroBase : public Managed<managed_memory>
    {
        // Initializer members
        std::vector<std::vector<real>> state;
        real gamma;
        real cfl;
        std::vector<real> x1, x2, x3;
        std::string coord_system;
        volatile bool inFailureState; 
        luint nzones;
        real hllc_z;

        // Common members
        DataWriteMembers setup;
        real dt, t, tend, t_interval, chkpt_interval, plm_theta, time_constant, hubble_param; 
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dx2, dx3, dlogt, tstart, engine_duration, invdx1, invdx2, invdx3;
        bool first_order, periodic, linspace, hllc, mesh_motion, half_sphere, quirk_smoothing, constant_sources, all_outer_bounds;
        bool den_source_all_zeros, mom1_source_all_zeros, mom2_source_all_zeros, mom3_source_all_zeros, energy_source_all_zeros; 
        luint active_zones, idx_active, total_zones, n, nx, ny, nz, init_chkpt_idx, radius, pseudo_radius;
        luint xphysical_grid, yphysical_grid, zphysical_grid;
        simbi::Solver sim_solver;
        ndarray<simbi::BoundaryCondition> bcs;
        simbi::Geometry geometry;
        simbi::Cellspacing x1cell_spacing, x2cell_spacing, x3cell_spacing;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> sources;
        std::string data_directory;

        int BLOCK_DIMX, BLOCK_DIMY, BLOCK_DIMZ;
        char* err_reason;
        char err_location[100];

        void check_state(){
            if (inFailureState) {
                throw helpers::SimulationFailureException(err_reason, err_location);
            }
        }

        const auto get_xblock_dims() const {
            return std::stoi(getEnvVar("GPUXBLOCK_SIZE"));
        }

        const auto get_yblock_dims() const {
            return std::stoi(getEnvVar("GPUYBLOCK_SIZE"));
        }

        const auto get_zblock_dims() const {
            return std::stoi(getEnvVar("GPUZBLOCK_SIZE"));
        }

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
            nx(state[0].size()),
            hllc_z((gamma - 1)/ (2 * gamma)),
            BLOCK_DIMX(get_xblock_dims()),
            BLOCK_DIMY(1),
            BLOCK_DIMZ(1)
        {
            if constexpr(BuildPlatform == Platform::GPU) {
                std::cout << "GPU Thread Block Geometry: (" << BLOCK_DIMX 
                          << ", " << BLOCK_DIMY << ", " << BLOCK_DIMZ << ")" << std::endl; 
            }
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
            nzones(state[0].size()),
            hllc_z((gamma - 1)/ (2 * gamma)),
            BLOCK_DIMX(get_xblock_dims()),
            BLOCK_DIMY(get_yblock_dims()),
            BLOCK_DIMZ(1)
        {
            if constexpr(BuildPlatform == Platform::GPU) {
                std::cout << "GPU Thread Block Geometry: (" << BLOCK_DIMX << ", " 
                << BLOCK_DIMY << ", " << BLOCK_DIMZ << ")" << std::endl; 
            }
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
            nzones(state[0].size()),
            hllc_z((gamma - 1)/ (2 * gamma)),
            BLOCK_DIMX(get_xblock_dims()),
            BLOCK_DIMY(get_yblock_dims()),
            BLOCK_DIMZ(get_zblock_dims())
        {
            if constexpr(BuildPlatform == Platform::GPU) {
                std::cout << "GPU Thread Block Geometry: (" << BLOCK_DIMX << ", "
                 << BLOCK_DIMY << ", " << BLOCK_DIMZ << ")" << std::endl; 
            }
        }
    };
    
} // namespace simbi
#endif