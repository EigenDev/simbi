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
        std::string coord_system;
        volatile bool inFailureState; 
        real hllc_z;
        luint nx, ny, nz, nzones;
        std::vector<real> x1, x2, x3;
        luint gpu_block_dimx, gpu_block_dimy, gpu_block_dimz;

        // Common members
        DataWriteMembers setup;
        real dt, t, tend, t_interval, chkpt_interval, plm_theta, time_constant, hubble_param; 
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dx2, dx3, dlogt, tstart, engine_duration, invdx1, invdx2, invdx3;
        bool first_order, linspace, mesh_motion, adaptive_mesh_motion, half_sphere, quirk_smoothing, constant_sources, all_outer_bounds;
        bool den_source_all_zeros, mom1_source_all_zeros, mom2_source_all_zeros, mom3_source_all_zeros, energy_source_all_zeros; 
        bool grav_source_all_zeros;
        luint active_zones, idx_active, total_zones, n, init_chkpt_idx, radius;
        luint xphysical_grid, yphysical_grid, zphysical_grid;
        simbi::Solver sim_solver;
        ndarray<simbi::BoundaryCondition> bcs;
        ndarray<int> troubled_cells;
        ndarray<real> sourceG;
        simbi::Geometry geometry;
        simbi::Cellspacing x1cell_spacing, x2cell_spacing, x3cell_spacing;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> sources;
        std::string data_directory;
        ndarray<bool> object_pos;
        
        const auto get_xblock_dims() const {
            return static_cast<luint>(std::stoi(getEnvVar("GPUXBLOCK_SIZE")));
        }

        const auto get_yblock_dims() const {
            return static_cast<luint>(std::stoi(getEnvVar("GPUYBLOCK_SIZE")));
        }

        const auto get_zblock_dims() const {
            return static_cast<luint>(std::stoi(getEnvVar("GPUZBLOCK_SIZE")));
        }

        void define_tinterval(real t, real dlogt, real chkpt_interval, real chkpt_idx) {
            real round_place = 1 / chkpt_interval;
            t_interval = 
                 dlogt != 0 ? tstart * std::pow(10, dlogt)
               : floor(tstart * round_place + static_cast<real>(0.5)) / round_place + chkpt_interval;
        }

        void define_total_zones() {
            this->total_zones = nx * ny * nz;
        }

        void define_chkpt_idx(int chkpt_idx) {
            init_chkpt_idx = chkpt_idx + (chkpt_idx > 0);
        }

        void define_xphysical_grid(bool is_first_order) {
            xphysical_grid = (is_first_order) ? nx -2 : nx - 4;
        }

        void define_yphysical_grid(bool is_first_order) {
            yphysical_grid =  (is_first_order) ? ny -2 : ny - 4;
        }

        void define_zphysical_grid(bool is_first_order) {
            zphysical_grid = (is_first_order) ? nz -2 : nz - 4;
        }

        void define_active_zones(){
            this->active_zones = xphysical_grid * yphysical_grid * zphysical_grid;
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
            coord_system(coord_system),
            inFailureState(false),
            hllc_z((gamma - 1)/ (2 * gamma)),
            nx(state[0].size()),
            ny(1),
            nz(1),
            nzones(state[0].size()),
            x1(x1),
            x2({}),
            x3({}),
            gpu_block_dimx(get_xblock_dims()),
            gpu_block_dimy(1),
            gpu_block_dimz(1)
        {
            // if constexpr(BuildPlatform == Platform::GPU) {
            //     std::cout << "GPU Thread Block Geometry: (" << gpu_block_dimx 
            //               << ", " << gpu_block_dimy << ", " << gpu_block_dimz << ")" << std::endl; 
            // }
            if (std::getenv("USE_OMP")) {
                use_omp = true;
                if (char * omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
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
            gamma(gamma),
            cfl(cfl),
            coord_system(coord_system),
            inFailureState(false),
            hllc_z((gamma - 1)/ (2 * gamma)),
            nx(nx),
            ny(ny),
            nz(1),
            nzones(state[0].size()),
            x1(x1),
            x2(x2),
            x3({}),
            gpu_block_dimx(get_xblock_dims()),
            gpu_block_dimy(get_yblock_dims()),
            gpu_block_dimz(1)
        {
            // if constexpr(BuildPlatform == Platform::GPU) {
            //     std::cout << "GPU Thread Block Geometry: (" << gpu_block_dimx << ", " 
            //     << gpu_block_dimy << ", " << gpu_block_dimz << ")" << std::endl; 
            // }
            if (std::getenv("USE_OMP")) {
                use_omp = true;
                if (char * omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
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
            gamma(gamma),
            cfl(cfl),
            coord_system(coord_system),
            inFailureState(false),
            hllc_z((gamma - 1)/ (2 * gamma)),
            nx(nx),
            ny(ny),
            nz(nz),
            nzones(state[0].size()),
            x1(x1),
            x2(x2),
            x3(x3),
            gpu_block_dimx(get_xblock_dims()),
            gpu_block_dimy(get_yblock_dims()),
            gpu_block_dimz(get_zblock_dims())
        {
            // if constexpr(BuildPlatform == Platform::GPU) {
            //     std::cout << "GPU Thread Block Geometry: (" << gpu_block_dimx << ", "
            //      << gpu_block_dimy << ", " << gpu_block_dimz << ")" << std::endl; 
            // }
            if (std::getenv("USE_OMP")) {
                use_omp = true;
                if (char * omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
            }
            
        }
    };
    
} // namespace simbi
#endif