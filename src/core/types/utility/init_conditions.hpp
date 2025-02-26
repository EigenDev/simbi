#ifndef INIT_CONDITIONS_HPP
#define INIT_CONDITIONS_HPP

#include "build_options.hpp"
#include <string>
#include <vector>

struct InitialConditions {
    real time, checkpoint_interval, dlogt;
    real plm_theta, engine_duration, gamma, cfl, tend;
    luint nx, ny, nz, checkpoint_idx;
    bool quirk_smoothing, homologous, mesh_motion;
    std::vector<std::vector<real>> sources, gsources, osources, bfield;
    std::vector<bool> object_cells;
    std::string data_directory, coord_system, solver;
    std::string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing, regime;
    std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;
    std::string spatial_order, time_order;
    std::vector<std::string> boundary_conditions;
    std::pair<real, real> x1bounds;
    std::pair<real, real> x2bounds;
    std::pair<real, real> x3bounds;

    std::tuple<lint, lint, lint> active_zones() const
    {
        const auto nghosts = 2 * (1 + (spatial_order == "plm"));
        return std::make_tuple(nx - nghosts, ny - nghosts, nz - nghosts);
    }
};

#endif