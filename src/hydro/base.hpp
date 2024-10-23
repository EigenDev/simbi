/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       base.hpp
 * @brief      base state for all hydro states to derive from
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef BASE_HPP
#define BASE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use...
#include "common/enums.hpp"    // for Cellspacing, BoundaryCondition (...
#include "common/helpers.hpp"         // for geometry_map, solver_map
#include "common/hydro_structs.hpp"   // for InitialConditions
#include "util/managed.hpp"           // for Managed
#include "util/ndarray.hpp"           // for ndarray
#include <algorithm>                  // for all_of
#include <cmath>                      // for log10, floor, pow
#include <cstdlib>                    // for getenv
#include <map>                        // for map
#include <memory>                     // for allocator, swap
#include <omp.h>                      // for omp_set_num_threads
#include <string>                     // for stoi, string, operator<=>
#include <utility>                    // for swap
#include <vector>                     // for vector

namespace simbi {
    struct HydroBase : public Managed<global::managed_memory> {
        // Initializer members
        std::vector<std::vector<real>> state;
        real gamma;
        real cfl;
        std::string coord_system;
        sig_bool inFailureState;
        real hllc_z;
        luint nx, ny, nz, nzones;
        std::vector<real> x1, x2, x3;
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
        ndarray<bool> object_pos;

        // Common members
        real dt, t_interval, time_constant, hubble_param;
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dlogx2, dx2, dlogx3, dx3, invdx1, invdx2, invdx3;
        bool linspace, mesh_motion;
        bool half_sphere, all_outer_bounds;
        bool homolog, hasCrashed, wasInterrupted, using_fourvelocity;
        luint active_zones, idx_active, radius;
        luint xag, yag, zag, init_chkpt_idx, chkpt_idx;
        ndarray<simbi::BoundaryCondition> bcs;
        ndarray<int> troubled_cells;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> bfield;
        std::vector<bool> object_cells;

        luint xblockdim, yblockdim, zblockdim, sx, sy, sz;
        luint xblockspace, yblockspace, zblockspace;
        luint shBlockSpace, shBlockBytes;
        luint nxv, nyv, nzv, nv;
        ExecutionPolicy<> fullP, activeP;

        //=========================== GPU Threads Per Dimension
        std::string readGpuEnvVar(std::string const& key) const
        {
            char* val = std::getenv(key.c_str());
            if (val) {
                return std::string(val);
            }
            return std::string("1");
        }

        auto get_xblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUXBLOCK_SIZE"))
            );
        }

        auto get_yblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUYBLOCK_SIZE"))
            );
        }

        auto get_zblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUZBLOCK_SIZE"))
            );
        }

        void define_tinterval(real dlogt, real chkpt_interval)
        {
            real round_place = 1.0 / chkpt_interval;
            t_interval       = dlogt != 0
                                   ? tstart * std::pow(10.0, dlogt)
                                   : floor(tstart * round_place + 0.5) / round_place +
                                   chkpt_interval;
        }

        void define_chkpt_idx(int chkpt_idx)
        {
            init_chkpt_idx = chkpt_idx + (chkpt_idx > 0);
        }

        void deallocate_state()
        {
            state  = std::vector<std::vector<real>>();
            bfield = std::vector<std::vector<real>>();
        }

        void print_shared_mem()
        {
            if constexpr (global::on_sm) {
                printf(
                    "Requested shared memory: %.2f kB\n",
                    static_cast<real>(shBlockBytes / 1024)
                );
            }
        }

        template <typename P>
        void compute_bytes_and_strides(int dim)
        {
            xblockdim = xag > gpu_block_dimx ? gpu_block_dimx : xag;
            yblockdim = yag > gpu_block_dimy ? gpu_block_dimy : yag;
            zblockdim = zag > gpu_block_dimz ? gpu_block_dimz : zag;
            if constexpr (global::BuildPlatform == global::Platform::GPU) {
                if (xblockdim * yblockdim * zblockdim < global::WARP_SIZE) {
                    if (nz > 1) {
                        xblockdim = 4;
                        yblockdim = 4;
                        zblockdim = 4;
                    }
                    else if (ny > 1) {
                        xblockdim = 16;
                        yblockdim = 16;
                        zblockdim = 1;
                    }
                    else {
                        xblockdim = 128;
                        yblockdim = 1;
                        zblockdim = 1;
                    }
                }
            }
            step = (time_order == "rk1") ? 1.0 : 0.5;
            sx   = (global::on_sm) ? xblockdim + 2 * radius : nx;
            sy = (dim < 2) ? 1 : (global::on_sm) ? yblockdim + 2 * radius : ny;
            sz = (dim < 3) ? 1 : (global::on_sm) ? zblockdim + 2 * radius : nz;
            xblockspace  = xblockdim + 2 * radius;
            yblockspace  = (dim < 2) ? 1 : yblockdim + 2 * radius;
            zblockspace  = (dim < 3) ? 1 : zblockdim + 2 * radius;
            shBlockSpace = xblockspace * yblockspace * zblockspace;
            shBlockBytes = shBlockSpace * sizeof(P) * global::on_sm;

            fullP = simbi::ExecutionPolicy(
                {nx, ny, nz},
                {xblockdim, yblockdim, zblockdim}
            );
            activeP = simbi::ExecutionPolicy(
                {xag, yag, zag},
                {xblockdim, yblockdim, zblockdim},
                shBlockBytes
            );
        }

      protected:
        HydroBase() = default;

        ~HydroBase() = default;

        HydroBase(
            std::vector<std::vector<real>> state,
            const InitialConditions& init_conditions
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
              x1(init_conditions.x1),
              x2(init_conditions.x2),
              x3(init_conditions.x3),
              gpu_block_dimx(get_xblock_dims()),
              gpu_block_dimy(get_yblock_dims()),
              gpu_block_dimz(get_zblock_dims()),
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
              object_pos(std::move(init_conditions.object_cells)),
              using_fourvelocity(
                  global::VelocityType == global::Velocity::FourVelocity
              )
        {
            initialize(init_conditions);
            if (std::getenv("USE_OMP")) {
                global::use_omp = true;
                if (char* omp_tnum = std::getenv("OMP_NUM_THREADS")) {
                    omp_set_num_threads(std::stoi(omp_tnum));
                }
            }
        }

        void initialize(const InitialConditions& init_conditions)
        {
            const bool pcm = spatial_order == "pcm";
            radius         = pcm ? 1 : 2;
            // Define simulation params
            xag = nx - 2 * radius;
            yag = (ny == 1) ? 1 : ny - 2 * radius;
            zag = (nz == 1) ? 1 : nz - 2 * radius;
            nxv = xag + 1;
            nyv = yag + 1;
            nzv = zag + 1;

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

            if (zag > 1) {
                checkpoint_zones = zag;
            }
            else if (yag > 1) {
                checkpoint_zones = yag;
            }
            else {
                checkpoint_zones = xag;
            }

            define_tinterval(dlogt, chkpt_interval);
            define_chkpt_idx(init_conditions.chkpt_idx);
        }
    };

}   // namespace simbi
#endif