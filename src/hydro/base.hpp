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
        std::vector<std::string> boundary_conditions;
        simbi::Solver sim_solver;
        ndarray<simbi::BoundaryCondition> bcs;
        ndarray<int> troubled_cells;
        ndarray<real> sourceG1, sourceG2, sourceG3;
        ndarray<real> dens_source, m1_source, m2_source, m3_source, erg_source;
        simbi::Geometry geometry;
        simbi::Cellspacing x1cell_spacing, x2cell_spacing, x3cell_spacing;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> sources;
        std::vector<bool> object_cells;
        std::string data_directory;
        std::vector<std::vector<real>> boundary_sources;
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
            std::string coord_system)
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
            std::string coord_system)
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
            if (std::getenv("USE_OMP")) {
                use_omp = true;
                if (char * omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
            }
            
        }

        HydroBase(
            std::vector<std::vector<real>> state,
            InitialConditions &init_conditions)
        :
            state(state),
            gamma(init_conditions.gamma),
            cfl(init_conditions.cfl),
            coord_system(init_conditions.coord_system),
            inFailureState(false),
            hllc_z((gamma - 1)/ (2 * gamma)),
            nx(init_conditions.nx),
            ny(init_conditions.ny),
            nz(init_conditions.nz),
            nzones(state[0].size()),
            x1(init_conditions.x1),
            x2(init_conditions.x2),
            x3(init_conditions.x3),
            gpu_block_dimx(get_xblock_dims()),
            gpu_block_dimy(get_yblock_dims()),
            gpu_block_dimz(get_zblock_dims())
        {
            initialize(init_conditions);
            if (std::getenv("USE_OMP")) {
                use_omp = true;
                if (char * omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
            }
            
        }

        void initialize(const InitialConditions &init_conditions) {
            // Define the source terms
            this->dens_source = init_conditions.sources[0];
            this->m1_source   = init_conditions.sources[1];
            this->m2_source   = init_conditions.sources[2];
            this->m3_source   = init_conditions.sources[3];
            this->erg_source  = init_conditions.sources[4];
            this->sourceG1    = init_conditions.gsource[0];
            this->sourceG2    = init_conditions.gsource[1];
            this->sourceG3    = init_conditions.gsource[2];
            
            // Define simulation params
            this->boundary_conditions = init_conditions.boundary_conditions;
            this->quirk_smoothing  = init_conditions.quirk_smoothing;
            this->t                = init_conditions.tstart;
            this->object_pos       = init_conditions.object_cells;
            this->chkpt_interval   = init_conditions.chkpt_interval;
            this->data_directory   = init_conditions.data_directory;
            this->tstart           = init_conditions.tstart;
            this->engine_duration  = init_conditions.engine_duration;
            this->total_zones      = nx * ny * nz;
            this->first_order      = init_conditions.first_order;
            this->sim_solver       = helpers::solver_map.at(init_conditions.solver);
            this->dlogt            = init_conditions.dlogt;
            this->linspace         = init_conditions.linspace;
            this->plm_theta        = init_conditions.plm_theta;
            this->geometry         = helpers::geometry_map.at(init_conditions.coord_system);
            this->xphysical_grid   = (init_conditions.first_order) ? nx - 2: nx - 4;
            this->yphysical_grid   = (ny == 1) ? 1 : (init_conditions.first_order) ? ny - 2: ny - 4;
            this->zphysical_grid   = (nz == 1) ? 1 : (init_conditions.first_order) ? nz - 2: nz - 4;
            this->idx_active       = (init_conditions.first_order) ? 1     : 2;
            this->active_zones     = xphysical_grid * yphysical_grid * zphysical_grid;
            this->x1cell_spacing   = (init_conditions.linspace) ? simbi::Cellspacing::LINSPACE : simbi::Cellspacing::LOGSPACE;
            this->x2cell_spacing   = simbi::Cellspacing::LINSPACE;
            this->x3cell_spacing   = simbi::Cellspacing::LINSPACE;
            this->dx3              = (x3[zphysical_grid - 1] - x3[0]) / (zphysical_grid - 1);
            this->dx2              = (x2[yphysical_grid - 1] - x2[0]) / (yphysical_grid - 1);
            this->dlogx1           = std::log10(x1[xphysical_grid - 1]/ x1[0]) / (xphysical_grid - 1);
            this->dx1              = (x1[xphysical_grid - 1] - x1[0]) / (xphysical_grid - 1);
            this->invdx1           = 1 / dx1;
            this->invdx2           = 1 / dx2;
            this->invdx3           = 1 / dx3;
            this->x1min            = x1[0];
            this->x1max            = x1[xphysical_grid - 1];
            this->x2min            = x2[0];
            this->x2max            = x2[yphysical_grid - 1];
            this->x3min            = x3[0];
            this->x3max            = x3[zphysical_grid - 1];
            this->den_source_all_zeros     = std::all_of(dens_source.begin(),   dens_source.end(),   [](real i) {return i == 0;});
            this->mom1_source_all_zeros    = std::all_of(m1_source.begin(),  m1_source.end(),  [](real i) {return i == 0;});
            this->mom2_source_all_zeros    = std::all_of(m2_source.begin(),  m2_source.end(),  [](real i) {return i == 0;});
            this->mom3_source_all_zeros    = std::all_of(m3_source.begin(),  m3_source.end(),  [](real i) {return i == 0;});
            this->energy_source_all_zeros  = std::all_of(erg_source.begin(), erg_source.end(), [](real i) {return i == 0;});

            if (nz > 1) {
                this->checkpoint_zones = zphysical_grid;
            } else if (ny > 1) {
                this->checkpoint_zones = yphysical_grid;
            } else {
                this->checkpoint_zones = xphysical_grid;
            }
            
            define_tinterval(t, dlogt, chkpt_interval, init_conditions.chkpt_idx);
            define_chkpt_idx(init_conditions.chkpt_idx);
        }
    };
    
} // namespace simbi
#endif