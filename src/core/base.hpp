/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       base.hpp
 * @brief      base state for all hydro states to derive from
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 */
#ifndef BASE_HPP
#define BASE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use...
#include "core/types/enums.hpp"   // for Cellspacing, BoundaryCondition (...
#include "core/types/init_conditions.hpp"   // for InitialConditions
#include "core/types/managed.hpp"           // for Managed
#include "core/types/ndarray.hpp"           // for ndarray
#include "util/tools/helpers.hpp"           // for geometry_map, solver_map
#include <algorithm>                        // for all_of
#include <cmath>                            // for log10, floor, pow
#include <cstdlib>                          // for getenv
#include <map>                              // for map
#include <memory>                           // for allocator, swap
#include <omp.h>                            // for omp_set_num_threads
#include <string>                           // for stoi, string, operator<=>
#include <utility>                          // for swap
#include <vector>                           // for vector

namespace simbi {
    struct HydroBase : public Managed<global::managed_memory> {
        // Initializer members
        std::vector<std::vector<real>> state;
        real gamma;
        real cfl;
        std::string coord_system;
        atomic_bool inFailureState;
        real hllc_z;
        luint nx, ny, nz, nzones;

        ndarray<real> x1, x2, x3;
        luint gpu_block_dimx, gpu_block_dimy, gpu_block_dimz, global_iter;
        real t, tend, chkpt_interval, plm_theta, dlogt, tstart, engine_duration;
        std::string spatial_order, time_order;
        bool use_pcm, use_rk1, quirk_smoothing, constant_sources;
        luint total_zones;
        std::vector<std::string> boundary_conditions;
        simbi::Solver sim_solver;
        simbi::Geometry geometry;
        simbi::Cellspacing x1_cell_spacing, x2_cell_spacing, x3_cell_spacing;
        std::string data_directory;
        ndarray<real> bstag1, bstag2, bstag3;
        ndarray<bool> object_pos;
        bool using_fourvelocity;
        std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;

        // Common members
        real dt, t_interval, time_constant, hubble_param;
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dlogx2, dx2, dlogx3, dx3, invdx1, invdx2, invdx3;
        bool linspace, mesh_motion, null_gravity, null_sources;
        bool half_sphere, all_outer_bounds, reflect_inner_x2_momentum,
            reflect_outer_x2_momentum;
        bool homolog, hasCrashed, wasInterrupted;
        luint active_zones, idx_active, radius, nhalos;
        luint xag, yag, zag, init_chkpt_idx, chkpt_idx;
        ndarray<simbi::BoundaryCondition> bcs;
        ndarray<int> troubled_cells;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> bfield;

        luint sx, sy, sz;
        luint nxv, nyv, nzv, nxe, nye, nze, nv;
        ExecutionPolicy<> fullPolicy, activePolicy, xvertexPolicy,
            yvertexPolicy, zvertexPolicy;

        //=========================== GPU Threads Per Dimension
        std::string readGpuEnvVar(const std::string& key) const
        {
            if (const char* val = std::getenv(key.c_str())) {
                return std::string(val);
            }
            return "1";
        }

        luint get_block_dims(const std::string& key) const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar(key)));
        }

        void define_tinterval(real dlogt, real chkpt_interval)
        {
            // Set the initial time interval
            // based on the current time, advanced
            // by the checkpoint interval to the nearest
            // place in the log10 scale. If dlogt is 0
            // then the interval is set to the current time
            // shifted towards the nearest checkpoint interval
            // if the checkpoint interval is 0 then the interval
            // is set to the current time
            if (dlogt != 0) {
                t_interval = std::pow(10.0, std::floor(std::log10(t) + dlogt));
            }
            else if (chkpt_interval != 0) {
                const real round_place = 1.0 / chkpt_interval;
                t_interval =
                    std::floor(tstart * round_place + 0.5) / round_place +
                    chkpt_interval;
            }
            else {
                t_interval = t;
            }
        }

        void define_chkpt_idx(int chkpt_idx)
        {
            init_chkpt_idx = chkpt_idx + (chkpt_idx > 0);
        }

        void deallocate_staggered_field()
        {
            std::vector<std::vector<real>>().swap(bfield);
        }

        void deallocate_state()
        {
            std::vector<std::vector<real>>().swap(state);
        }

        void print_shared_mem(const size_t shBlockBytes) const
        {
            if constexpr (global::on_sm) {
                printf(
                    "Requested shared memory: %.2f kB\n",
                    static_cast<real>(shBlockBytes / 1024)
                );
            }
        }

        void update_mesh_motion(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
        )
        {
            if (!mesh_motion) {
                return;
            }

            auto update = [this](real x, real h) {
                return x + step * dt * (homolog ? x * h : h);
            };

            hubble_param = adot(t) / a(t);
            x1max        = update(x1max, hubble_param);
            x1min        = update(x1min, hubble_param);
        }

        template <typename P>
        void compute_bytes_and_strides(int dim)
        {
            auto xblockdim = std::min(xag, gpu_block_dimx);
            auto yblockdim = std::min(yag, gpu_block_dimy);
            auto zblockdim = std::min(zag, gpu_block_dimz);

            if constexpr (global::on_gpu) {
                if (xblockdim * yblockdim * zblockdim < global::WARP_SIZE) {
                    if (nz > 1) {
                        xblockdim = yblockdim = zblockdim = 4;
                    }
                    else if (ny > 1) {
                        xblockdim = yblockdim = 16;
                        zblockdim             = 1;
                    }
                    else {
                        xblockdim = 128;
                        yblockdim = zblockdim = 1;
                    }
                }
            }

            step = (time_order == "rk1") ? 1.0 : 0.5;
            sx   = (global::on_sm) ? xblockdim + nhalos : nx;
            sy   = (dim < 2) ? 1 : (global::on_sm) ? yblockdim + nhalos : ny;
            sz   = (dim < 3) ? 1 : (global::on_sm) ? zblockdim + nhalos : nz;
            const auto xblockspace    = xblockdim + nhalos;
            const auto yblockspace    = (dim < 2) ? 1 : yblockdim + nhalos;
            const auto zblockspace    = (dim < 3) ? 1 : zblockdim + nhalos;
            const size_t shBlockSpace = xblockspace * yblockspace * zblockspace;
            const size_t shBlockBytes =
                shBlockSpace * sizeof(P) * global::on_sm;

            const simbiStream_t stream = nullptr;
            fullPolicy                 = ExecutionPolicy(
                {nx, ny, nz},
                {xblockdim, yblockdim, zblockdim},
                {0, 0, 0},   // strides
                {.sharedMemBytes = shBlockBytes,
                                 .streams        = {stream},
                                 .devices        = {0}}
            );
            activePolicy = ExecutionPolicy(
                {xag, yag, zag},
                {xblockdim, yblockdim, zblockdim},
                {sx, sy, sz},
                {.sharedMemBytes = shBlockBytes,
                 .streams        = {stream},
                 .devices        = {0}}

            );
            xvertexPolicy = ExecutionPolicy(
                {nxv, yag, zag},
                {xblockdim, yblockdim, zblockdim},
                {0, 0, 0},
                {.sharedMemBytes = 0, .streams = {stream}, .devices = {0}}
            );
            yvertexPolicy = ExecutionPolicy(
                {xag, nyv, zag},
                {xblockdim, yblockdim, zblockdim},
                {0, 0, 0},
                {.sharedMemBytes = 0, .streams = {stream}, .devices = {0}}
            );
            zvertexPolicy = ExecutionPolicy(
                {xag, yag, nzv},
                {xblockdim, yblockdim, zblockdim},
                {0, 0, 0},
                {.sharedMemBytes = 0, .streams = {stream}, .devices = {0}}
            );

            print_shared_mem(0);
        }

        // void load_hydro_source_lib()
        // {
        //     // Load the symbol based on the dimension
        //     using f2arg = void (*)(real, real, real[]);
        //     using f3arg = void (*)(real, real, real, real[]);
        //     using f4arg = void (*)(real, real, real, real, real[]);

        //     //=================================================================
        //     // Check if the hydro source library is set
        //     //=================================================================
        //     null_sources = true;
        //     if (!hydro_source_lib.empty()) {
        //         // Load the shared library
        //         hsource_handle = dlopen(hydro_source_lib.c_str(), RTLD_LAZY);
        //         if (!hsource_handle) {
        //             std::cerr << "Cannot open library: " << dlerror() <<
        //             '\n'; return;
        //         }

        //         // Clear any existing error
        //         dlerror();

        //         const std::vector<std::pair<const char*, function_t&>>
        //         symbols =
        //             {
        //               {"hydro_source", hydro_source},
        //             };

        //         bool success = true;
        //         for (const auto& [symbol, func] : symbols) {
        //             void* source            = dlsym(hsource_handle, symbol);
        //             const char* dlsym_error = dlerror();
        //             // if can't load symbol, print error and
        //             // set null_sources to true
        //             if (dlsym_error) {
        //                 std::cerr << "Cannot load symbol '" << symbol
        //                           << "': " << dlsym_error << '\n';
        //                 success = false;
        //                 dlclose(hsource_handle);
        //                 break;
        //             }

        //             // Assign the function pointer based on the dimension
        //             if constexpr (dim == 1) {
        //                 func = reinterpret_cast<f2arg>(source);
        //             }
        //             else if constexpr (dim == 2) {
        //                 func = reinterpret_cast<f3arg>(source);
        //             }
        //             else if constexpr (dim == 3) {
        //                 func = reinterpret_cast<f4arg>(source);
        //             }
        //         }
        //         if (success) {
        //             null_sources = false;
        //         }
        //     }
        // }

        // void load_gravity_source_lib()
        // {
        //     //=================================================================
        //     // Check if the gravity source library is set
        //     //=================================================================
        //     null_gravity = true;
        //     if (!gravity_source_lib.empty()) {
        //         gsource_handle = dlopen(gravity_source_lib.c_str(),
        //         RTLD_LAZY); if (!gsource_handle) {
        //             std::cerr << "Cannot open library: " << dlerror() <<
        //             '\n'; return;
        //         }

        //         // Clear any existing error
        //         dlerror();

        //         // Load the symbol based on the dimension
        //         const std::vector<std::pair<const char*, function_t&>>
        //             g_symbols = {
        //               {"gravity_source", gravity_source},
        //             };

        //         bool success = true;
        //         for (const auto& [symbol, func] : g_symbols) {
        //             void* source            = dlsym(gsource_handle, symbol);
        //             const char* dlsym_error = dlerror();
        //             // if can't load symbol, print error
        //             if (dlsym_error) {
        //                 std::cerr << "Cannot load symbol '" << symbol
        //                           << "': " << dlsym_error << '\n';
        //                 success = false;
        //                 dlclose(gsource_handle);
        //                 break;
        //             }

        //             // Assign the function pointer based on the dimension
        //             if constexpr (dim == 1) {
        //                 func = reinterpret_cast<f2arg>(source);
        //             }
        //             else if constexpr (dim == 2) {
        //                 func = reinterpret_cast<f3arg>(source);
        //             }
        //             else if constexpr (dim == 3) {
        //                 func = reinterpret_cast<f4arg>(source);
        //             }
        //         }
        //         if (success) {
        //             null_gravity = false;
        //         }
        //     }
        // }

        // void load_boundary_source_lib()
        // {
        //     //=================================================================
        //     // Check if the boundary source library is set
        //     //=================================================================
        //     if (!boundary_source_lib.empty()) {
        //         bsource_handle = dlopen(boundary_source_lib.c_str(),
        //         RTLD_LAZY); if (!bsource_handle) {
        //             std::cerr << "Cannot open library: " << dlerror() <<
        //             '\n'; return;
        //         }

        //         // Clear any existing error
        //         dlerror();

        //         // Load the symbol based on the dimension
        //         const std::vector<std::pair<const char*, function_t&>>
        //             b_symbols = {
        //               {"bx1_inner_source", bx1_inner_source},
        //               {"bx1_outer_source", bx1_outer_source},
        //               {"bx2_inner_source", bx2_inner_source},
        //               {"bx2_outer_source", bx2_outer_source},
        //               {"bx3_inner_source", bx3_inner_source},
        //               {"bx3_outer_source", bx3_outer_source},
        //             };

        //         for (const auto& [symbol, func] : b_symbols) {
        //             void* source            = dlsym(bsource_handle, symbol);
        //             const char* dlsym_error = dlerror();
        //             // if can't load symbol, print error
        //             if (dlsym_error) {
        //                 // erro out  only  if the boundary
        //                 // condition is set to dynamic
        //                 for (int i = 0; i < 2 * dim; ++i) {
        //                     if (symbol == b_symbols[i].first &&
        //                         bcs[i] == BoundaryCondition::DYNAMIC) {
        //                         std::cerr << "Cannot load symbol '" << symbol
        //                                   << "': " << dlsym_error << '\n';
        //                         bcs[i] = BoundaryCondition::OUTFLOW;
        //                         dlclose(bsource_handle);
        //                     }
        //                 }
        //             }
        //             else {
        //                 // Assign the function pointer based on the dimension
        //                 if constexpr (dim == 1) {
        //                     func = reinterpret_cast<f2arg>(source);
        //                 }
        //                 else if constexpr (dim == 2) {
        //                     func = reinterpret_cast<f3arg>(source);
        //                 }
        //                 else if constexpr (dim == 3) {
        //                     func = reinterpret_cast<f4arg>(source);
        //                 }
        //             }
        //         }
        //     }
        // }

        // void load_functions()
        // {
        //     load_hydro_source_lib();
        //     load_gravity_source_lib();
        //     load_boundary_source_lib();
        // }

      protected:
        HydroBase() = default;

        ~HydroBase() = default;

        HydroBase(
            std::vector<std::vector<real>> state,
            InitialConditions& init_conditions
        )
            : state(std::move(state)),
              gamma(init_conditions.gamma),
              cfl(init_conditions.cfl),
              coord_system(init_conditions.coord_system),
              inFailureState(false),
              hllc_z((gamma - 1.0) / (2.0 * gamma)),
              nx(init_conditions.nx),
              ny(init_conditions.ny),
              nz(init_conditions.nz),
              x1(std::move(init_conditions.x1)),
              x2(std::move(init_conditions.x2)),
              x3(std::move(init_conditions.x3)),
              gpu_block_dimx(get_block_dims("GPUXBLOCK_SIZE")),
              gpu_block_dimy(get_block_dims("GPUYBLOCK_SIZE")),
              gpu_block_dimz(get_block_dims("GPUZBLOCK_SIZE")),
              global_iter(0),
              t(init_conditions.tstart),
              tend(init_conditions.tend),
              chkpt_interval(init_conditions.chkpt_interval),
              plm_theta(init_conditions.plm_theta),
              dlogt(init_conditions.dlogt),
              tstart(init_conditions.tstart),
              engine_duration(init_conditions.engine_duration),
              spatial_order(init_conditions.spatial_order),
              time_order(init_conditions.time_order),
              use_pcm(spatial_order == "pcm"),
              use_rk1(time_order == "rk1"),
              quirk_smoothing(init_conditions.quirk_smoothing),
              constant_sources(init_conditions.constant_sources),
              total_zones(nx * ny * nz),
              boundary_conditions(std::move(init_conditions.boundary_conditions)
              ),
              sim_solver(helpers::solver_map.at(init_conditions.solver)),
              geometry(helpers::geometry_map.at(init_conditions.coord_system)),
              x1_cell_spacing(str2cell.at(init_conditions.x1_cell_spacing)),
              x2_cell_spacing(str2cell.at(init_conditions.x2_cell_spacing)),
              x3_cell_spacing(str2cell.at(init_conditions.x3_cell_spacing)),
              data_directory(init_conditions.data_directory),
              bstag1(std::move(init_conditions.bfield[0])),
              bstag2(std::move(init_conditions.bfield[1])),
              bstag3(std::move(init_conditions.bfield[2])),
              object_pos(std::move(init_conditions.object_cells)),
              using_fourvelocity(
                  global::VelocityType == global::Velocity::FourVelocity
              ),
              hydro_source_lib(init_conditions.hydro_source_lib),
              gravity_source_lib(init_conditions.gravity_source_lib),
              boundary_source_lib(init_conditions.boundary_source_lib)
        {
            initialize(init_conditions);
            if (std::getenv("USE_OMP")) {
                global::use_omp = true;
                if (const char* omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
            }
        }

        void initialize(InitialConditions& init_conditions)
        {
            const bool pcm = (spatial_order == "pcm");
            radius         = pcm ? 1 : 2;
            nhalos         = 2 * radius;
            xag            = nx - nhalos;
            yag            = (ny == 1) ? 1 : ny - nhalos;
            zag            = (nz == 1) ? 1 : nz - nhalos;
            nxv            = xag + 1;
            nyv            = yag + 1;
            nzv            = zag + 1;
            nxe            = xag + nhalos;
            nye            = yag + nhalos;
            nze            = zag + nhalos;

            nv           = nxv * nyv * nzv;
            idx_active   = pcm ? 1 : 2;
            active_zones = xag * yag * zag;
            x1min        = x1[0];
            x1max        = x1[nxv - 1];
            if (x1_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                dlogx1 = std::log10(x1[nxv - 1] / x1[0]) / (nxv - 1);
            }
            else {
                dx1 = (x1[nxv - 1] - x1[0]) / (nxv - 1);
            }
            invdx1 = 1.0 / dx1;

            bfield = std::move(init_conditions.bfield);

            if (ny > 1) {   // 2D check
                x2min = x2[0];
                x2max = x2[nyv - 1];
                if (x2_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                    dlogx2 = std::log10(x2[nyv - 1] / x2[0]) / (nyv - 1);
                }
                else {
                    dx2    = (x2[nyv - 1] - x2[0]) / (nyv - 1);
                    invdx2 = 1.0 / dx2;
                }

                if (nz > 1) {   // 3D Check
                    x3min = x3[0];
                    x3max = x3[nzv - 1];
                    if (x3_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                        dlogx3 = std::log10(x3[nzv - 1] / x3[0]) / (nzv - 1);
                    }
                    else {
                        dx3    = (x3[nzv - 1] - x3[0]) / (nzv - 1);
                        invdx3 = 1.0 / dx3;
                    }
                }
            }

            half_sphere = (x2max == 0.5 * M_PI) && coord_system == "spherical";
            reflect_inner_x2_momentum = (coord_system != "spherical");
            reflect_outer_x2_momentum =
                (coord_system != "spherical") ||
                ((coord_system == "spherical") && half_sphere);
            checkpoint_zones = (zag > 1) ? zag : (yag > 1) ? yag : xag;
            define_tinterval(dlogt, chkpt_interval);
            define_chkpt_idx(init_conditions.chkpt_idx);
        }
    };
}   // namespace simbi
#endif