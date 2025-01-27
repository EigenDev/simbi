#ifndef INIT_CONDITIONS_HPP
#define INIT_CONDITIONS_HPP

#include "build_options.hpp"
#include <string>
#include <vector>

struct InitialConditions {
    real tstart, chkpt_interval, dlogt;
    real plm_theta, engine_duration, gamma, cfl, tend;
    luint nx, ny, nz, chkpt_idx;
    bool quirk_smoothing, constant_sources;
    std::vector<std::vector<real>> sources, gsources, osources, bfield;
    std::vector<bool> object_cells;
    std::string data_directory, coord_system, solver;
    std::string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing, regime;
    std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;
    std::string spatial_order, time_order;
    std::vector<std::string> boundary_conditions;
    std::vector<real> x1, x2, x3;
};

#endif