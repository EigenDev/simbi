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
        sig_bool inFailureState;
        real hllc_z;
        luint nx, ny, nz, nzones;
        std::vector<real> x1, x2, x3;
        luint gpu_block_dimx, gpu_block_dimy, gpu_block_dimz, global_iter;

        // Common members
        DataWriteMembers setup;
        real dt, t, tend, t_interval, chkpt_interval, plm_theta, time_constant,
            hubble_param;
        real x1min, x1max, x2min, x2max, x3min, x3max, step;
        real dlogx1, dx1, dlogx2, dx2, dlogx3, dx3, dlogt, tstart,
            engine_duration, invdx1, invdx2, invdx3;
        std::string spatial_order, time_order;
        bool use_pcm, use_rk1, linspace, mesh_motion, adaptive_mesh_motion;
        bool half_sphere, quirk_smoothing, constant_sources, all_outer_bounds;
        bool homolog, hasCrashed, wasInterrupted;
        bool null_den, null_mom1, null_mom2, null_mom3, null_nrg;
        bool null_mag1, null_mag2, null_mag3;
        bool nullg1, nullg2, nullg3;
        luint active_zones, idx_active, total_zones, init_chkpt_idx, radius;
        luint xag, yag, zag;
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
        std::vector<std::vector<real>> sources, bfield;
        std::vector<bool> object_cells;
        std::string data_directory;
        std::vector<std::vector<real>> boundary_sources;
        ndarray<bool> object_pos;
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
            radius = (spatial_order == "pcm") ? 1 : 2;
            step   = (time_order == "rk1") ? 1.0 : 0.5;
            sx     = (global::on_sm) ? xblockdim + 2 * radius : nx;
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

        void set_output_params(int dim, std::string regime)
        {
            setup.x1max           = x1[nxv - 1];
            setup.x1min           = x1[0];
            setup.x1              = x1;
            setup.nx              = nx;
            setup.ny              = ny;
            setup.nz              = nz;
            setup.xactive_zones   = xag;
            setup.yactive_zones   = yag;
            setup.zactive_zones   = zag;
            setup.x1_cell_spacing = cell2str.at(x1_cell_spacing);
            setup.x2_cell_spacing = cell2str.at(x2_cell_spacing);
            setup.x3_cell_spacing = cell2str.at(x3_cell_spacing);
            setup.ad_gamma        = gamma;
            setup.spatial_order   = spatial_order;
            setup.time_order      = time_order;
            setup.coord_system    = coord_system;
            setup.using_fourvelocity =
                (global::VelocityType == global::Velocity::FourVelocity);
            setup.regime              = regime;
            setup.mesh_motion         = mesh_motion;
            setup.boundary_conditions = boundary_conditions;
            setup.dimensions          = dim;
            if (dim > 1) {
                setup.x2max = x2[nyv - 1];
                setup.x2min = x2[0];
                setup.x2    = x2;
                if (dim > 2) {
                    setup.x3max = x3[nzv - 1];
                    setup.x3min = x3[0];
                    setup.x3    = x3;
                }
            }
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
              boundary_sources(init_conditions.boundary_sources),
              object_pos(std::move(init_conditions.object_cells))
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
            xag = (init_conditions.spatial_order == "pcm") ? nx - 2 : nx - 4;
            yag = (ny == 1)                                  ? 1
                  : (init_conditions.spatial_order == "pcm") ? ny - 2
                                                             : ny - 4;
            zag = (nz == 1)                                  ? 1
                  : (init_conditions.spatial_order == "pcm") ? nz - 2
                                                             : nz - 4;
            nxv = xag + 1;
            nyv = yag + 1;
            nzv = zag + 1;
            nv  = nxv * nyv * nzv;
            idx_active   = (init_conditions.spatial_order == "pcm") ? 1 : 2;
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

            // Define the source terms
            density_source = std::move(init_conditions.sources[0]);
            m1_source      = std::move(init_conditions.sources[1]);
            sourceG1       = std::move(init_conditions.gsources[0]);
            bfield         = std::move(init_conditions.bfield);
            if (init_conditions.regime == "rmhd") {
                sourceB1 = std::move(init_conditions.bsources[0]);
            };

            if (ny > 1) {   // 2D check
                x2min     = x2[0];
                x2max     = x2[nyv - 1];
                m2_source = std::move(init_conditions.sources[2]);
                sourceG2  = std::move(init_conditions.gsources[1]);
                if (init_conditions.regime == "rmhd") {
                    sourceB2 = std::move(init_conditions.bsources[1]);
                };

                if (x2_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                    dlogx2 = std::log10(x2[nyv - 1] / x2[0]) / (nyv - 1);
                }
                else {
                    dx2    = (x2[nyv - 1] - x2[0]) / (nyv - 1);
                    invdx2 = 1.0 / dx2;
                }

                if (nz > 1) {   // 3D Check
                    x3min         = x3[0];
                    x3max         = x3[nzv - 1];
                    m3_source     = std::move(init_conditions.sources[3]);
                    energy_source = std::move(init_conditions.sources[4]);
                    sourceG3      = std::move(init_conditions.gsources[2]);
                    if (init_conditions.regime == "rmhd") {
                        sourceB3 = std::move(init_conditions.bsources[2]);
                    };
                    if (x3_cell_spacing == simbi::Cellspacing::LOGSPACE) {
                        dlogx3 = std::log10(x3[nzv - 1] / x3[0]) / (nzv - 1);
                    }
                    else {
                        dx3    = (x3[nzv - 1] - x3[0]) / (nzv - 1);
                        invdx3 = 1.0 / dx3;
                    }
                }
                else {
                    energy_source = std::move(init_conditions.sources[3]);
                }
            }
            else {
                energy_source = std::move(init_conditions.sources[2]);
            }

            nullg1 = std::all_of(sourceG1.begin(), sourceG1.end(), [](real i) {
                return i == real(0);
            });
            nullg2 = std::all_of(sourceG2.begin(), sourceG2.end(), [](real i) {
                return i == real(0);
            });
            nullg3 = std::all_of(sourceG3.begin(), sourceG3.end(), [](real i) {
                return i == real(0);
            });
            null_mag1 =
                std::all_of(sourceB1.begin(), sourceB1.end(), [](real i) {
                    return i == real(0);
                });
            null_mag2 =
                std::all_of(sourceB2.begin(), sourceB2.end(), [](real i) {
                    return i == real(0);
                });
            null_mag3 =
                std::all_of(sourceB3.begin(), sourceB3.end(), [](real i) {
                    return i == real(0);
                });

            null_den = std::all_of(
                density_source.begin(),
                density_source.end(),
                [](real i) { return i == real(0); }
            );
            null_mom1 =
                std::all_of(m1_source.begin(), m1_source.end(), [](real i) {
                    return i == real(0);
                });
            null_mom2 =
                std::all_of(m2_source.begin(), m2_source.end(), [](real i) {
                    return i == real(0);
                });
            null_mom3 =
                std::all_of(m3_source.begin(), m3_source.end(), [](real i) {
                    return i == real(0);
                });
            null_nrg = std::all_of(
                energy_source.begin(),
                energy_source.end(),
                [](real i) { return i == real(0); }
            );

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