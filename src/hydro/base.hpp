/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
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
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */
#ifndef BASE_HPP
#define BASE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use...
#include "common/enums.hpp"    // for Cellspacing, BoundaryCondition (...
#include "common/helpers.hip.hpp"     // for geometry_map, solver_map
#include "common/hydro_structs.hpp"   // for InitialConditions, DataWriteMembers
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

std::unordered_map<std::string, simbi::Cellspacing> const str2cell = {
  {"log", simbi::Cellspacing::LOGSPACE},
  {"linear", simbi::Cellspacing::LINSPACE}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};

std::unordered_map<simbi::Cellspacing, std::string> const cell2str = {
  {simbi::Cellspacing::LOGSPACE, "log"},
  {simbi::Cellspacing::LINSPACE, "linear"}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};

namespace simbi {
    struct HydroBase : public Managed<global::managed_memory> {
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
        real dt, t, tend, t_interval, chkpt_interval, plm_theta, time_constant,
            hubble_param;
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dlogx2, dx2, dlogx3, dx3, dlogt, tstart,
            engine_duration, invdx1, invdx2, invdx3;
        bool first_order, linspace, mesh_motion, adaptive_mesh_motion,
            half_sphere, quirk_smoothing, constant_sources, all_outer_bounds,
            changing_volume;
        bool den_source_all_zeros, mom1_source_all_zeros, mom2_source_all_zeros,
            mom3_source_all_zeros, energy_source_all_zeros;
        bool mag1_source_all_zeros, mag2_source_all_zeros,
            mag3_source_all_zeros;
        bool zero_gravity1, zero_gravity2, zero_gravity3;
        luint active_zones, idx_active, total_zones, n, init_chkpt_idx, radius;
        luint xactive_grid, yactive_grid, zactive_grid;
        std::vector<std::string> boundary_conditions;
        simbi::Solver sim_solver;
        ndarray<simbi::BoundaryCondition> bcs;
        ndarray<int> troubled_cells;
        ndarray<real> sourceG1, sourceG2, sourceG3;
        ndarray<real> sourceB1, sourceB2, sourceB3;
        ndarray<real> density_source, m1_source, m2_source, m3_source,
            energy_source;
        simbi::Geometry geometry;
        simbi::Cellspacing x1_cell_spacing, x2_cell_spacing, x3_cell_spacing;
        luint blockSize, checkpoint_zones;
        std::vector<std::vector<real>> sources;
        std::vector<bool> object_cells;
        std::string data_directory;
        std::vector<std::vector<real>> boundary_sources;
        ndarray<bool> object_pos;

        //=========================== GPU Threads Per Dimension
        std::string readGpuEnvVar(std::string const& key) const
        {
            char* val = std::getenv(key.c_str());
            if (val) {
                return std::string(val);
            }
            return std::string("1");
        }

        const auto get_xblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUXBLOCK_SIZE"))
            );
        }

        const auto get_yblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUYBLOCK_SIZE"))
            );
        }

        const auto get_zblock_dims() const
        {
            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUZBLOCK_SIZE"))
            );
        }

        void define_tinterval(
            real t,
            real dlogt,
            real chkpt_interval,
            real chkpt_idx
        )
        {
            real round_place = 1 / chkpt_interval;
            t_interval       = dlogt != 0
                                   ? tstart * std::pow(10, dlogt)
                                   : floor(tstart * round_place + 0.5) / round_place +
                                   chkpt_interval;
        }

        void define_chkpt_idx(int chkpt_idx)
        {
            init_chkpt_idx = chkpt_idx + (chkpt_idx > 0);
        }

      protected:
        HydroBase() {}

        ~HydroBase() {}

        HydroBase(
            std::vector<std::vector<real>> state,
            const InitialConditions& init_conditions
        )
            : state(state),
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
              t(init_conditions.tstart),
              tend(init_conditions.tend),
              chkpt_interval(init_conditions.chkpt_interval),
              plm_theta(init_conditions.plm_theta),
              dlogt(init_conditions.dlogt),
              tstart(init_conditions.tstart),
              engine_duration(init_conditions.engine_duration),
              first_order(init_conditions.first_order),
              quirk_smoothing(init_conditions.quirk_smoothing),
              constant_sources(init_conditions.constant_sources),
              total_zones(nx * ny * nz),
              boundary_conditions(init_conditions.boundary_conditions),
              sim_solver(helpers::solver_map.at(init_conditions.solver)),
              geometry(helpers::geometry_map.at(init_conditions.coord_system)),
              x1_cell_spacing(str2cell.at(init_conditions.x1_cell_spacing)),
              x2_cell_spacing(str2cell.at(init_conditions.x2_cell_spacing)),
              x3_cell_spacing(str2cell.at(init_conditions.x3_cell_spacing)),
              data_directory(init_conditions.data_directory),
              boundary_sources(init_conditions.boundary_sources),
              object_pos(init_conditions.object_cells)
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
            // Define simulation params
            this->xactive_grid =
                (init_conditions.first_order) ? nx - 2 : nx - 4;
            this->yactive_grid = (ny == 1)                       ? 1
                                 : (init_conditions.first_order) ? ny - 2
                                                                 : ny - 4;
            this->zactive_grid = (nz == 1)                       ? 1
                                 : (init_conditions.first_order) ? nz - 2
                                                                 : nz - 4;
            this->idx_active   = (init_conditions.first_order) ? 1 : 2;
            this->active_zones = xactive_grid * yactive_grid * zactive_grid;
            this->x1min        = x1[0];
            this->x1max        = x1[xactive_grid - 1];
            if (x1_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                this->dlogx1 = std::log10(x1[xactive_grid - 1] / x1[0]) /
                               (xactive_grid - 1);
            }
            else {
                this->dx1 = (x1[xactive_grid - 1] - x1[0]) / (xactive_grid - 1);
            }
            this->invdx1 = 1 / dx1;
            // Define the source terms
            this->density_source = init_conditions.sources[0];
            this->m1_source      = init_conditions.sources[1];
            this->sourceG1       = init_conditions.gsources[0];
            if (init_conditions.regime == "rmhd") {
                this->sourceB1 = init_conditions.bsources[0];
            };
            if ((ny > 1) && (nz > 1)) {   // 3D check
                this->x2min         = x2[0];
                this->x2max         = x2[yactive_grid - 1];
                this->x3min         = x3[0];
                this->x3max         = x3[zactive_grid - 1];
                this->dx3           = (x3max - x3min) / (zactive_grid - 1);
                this->dx2           = (x2max - x2min) / (yactive_grid - 1);
                this->m2_source     = init_conditions.sources[2];
                this->m3_source     = init_conditions.sources[3];
                this->energy_source = init_conditions.sources[4];
                this->sourceG2      = init_conditions.gsources[1];
                this->sourceG3      = init_conditions.gsources[2];
                if (init_conditions.regime == "rmhd") {
                    this->sourceB2 = init_conditions.bsources[1];
                    this->sourceB3 = init_conditions.bsources[2];
                };
                if (x2_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                    this->dlogx2 = std::log10(x2[yactive_grid - 1] / x2[0]) /
                                   (yactive_grid - 1);
                }
                else {
                    this->dx2 =
                        (x2[yactive_grid - 1] - x2[0]) / (yactive_grid - 1);
                    this->invdx2 = 1 / dx2;
                }
                if (x3_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                    this->dlogx3 = std::log10(x3[zactive_grid - 1] / x3[0]) /
                                   (zactive_grid - 1);
                }
                else {
                    this->dx3 =
                        (x3[zactive_grid - 1] - x3[0]) / (zactive_grid - 1);
                    this->invdx3 = 1 / dx3;
                }
            }
            else if ((ny > 1) && (nz == 1)) {   // 2D Check
                this->x2min         = x2[0];
                this->x2max         = x2[yactive_grid - 1];
                this->m2_source     = init_conditions.sources[2];
                this->energy_source = init_conditions.sources[3];
                this->sourceG2      = init_conditions.gsources[1];
                if (init_conditions.regime == "rmhd") {
                    this->sourceB2 = init_conditions.bsources[1];
                };

                if (x2_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                    this->dlogx2 = std::log10(x2[yactive_grid - 1] / x2[0]) /
                                   (yactive_grid - 1);
                }
                else {
                    this->dx2 =
                        (x2[yactive_grid - 1] - x2[0]) / (yactive_grid - 1);
                    this->invdx2 = 1 / dx2;
                }
            }
            else {
                this->energy_source = init_conditions.sources[2];
            }
            this->zero_gravity1 =
                std::all_of(sourceG1.begin(), sourceG1.end(), [](real i) {
                    return i == 0;
                });
            this->zero_gravity2 =
                std::all_of(sourceG2.begin(), sourceG2.end(), [](real i) {
                    return i == 0;
                });
            this->zero_gravity3 =
                std::all_of(sourceG3.begin(), sourceG3.end(), [](real i) {
                    return i == 0;
                });
            this->mag1_source_all_zeros =
                std::all_of(sourceB1.begin(), sourceB1.end(), [](real i) {
                    return i == 0;
                });
            this->mag2_source_all_zeros =
                std::all_of(sourceB2.begin(), sourceB2.end(), [](real i) {
                    return i == 0;
                });
            this->mag3_source_all_zeros =
                std::all_of(sourceB3.begin(), sourceB3.end(), [](real i) {
                    return i == 0;
                });
            this->mom1_source_all_zeros =
                std::all_of(m1_source.begin(), m1_source.end(), [](real i) {
                    return i == 0;
                });
            this->mom2_source_all_zeros =
                std::all_of(m2_source.begin(), m2_source.end(), [](real i) {
                    return i == 0;
                });
            this->mom3_source_all_zeros =
                std::all_of(m3_source.begin(), sourceB3.end(), [](real i) {
                    return i == 0;
                });
            this->den_source_all_zeros = std::all_of(
                density_source.begin(),
                density_source.end(),
                [](real i) { return i == 0; }
            );
            this->mom1_source_all_zeros =
                std::all_of(m1_source.begin(), m1_source.end(), [](real i) {
                    return i == 0;
                });
            this->energy_source_all_zeros = std::all_of(
                energy_source.begin(),
                energy_source.end(),
                [](real i) { return i == 0; }
            );

            if (nz > 1) {
                this->checkpoint_zones = zactive_grid;
            }
            else if (ny > 1) {
                this->checkpoint_zones = yactive_grid;
            }
            else {
                this->checkpoint_zones = xactive_grid;
            }

            define_tinterval(
                t,
                dlogt,
                chkpt_interval,
                init_conditions.chkpt_idx
            );
            define_chkpt_idx(init_conditions.chkpt_idx);
        }
    };

}   // namespace simbi
#endif