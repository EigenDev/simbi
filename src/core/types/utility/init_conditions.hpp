/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            init_conditions.hpp
 *  * @brief           a struct to hold initial conditions for the simulation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef INIT_CONDITIONS_HPP
#define INIT_CONDITIONS_HPP

#include "build_options.hpp"
#include "core/types/containers/vector.hpp"
#include "physics/hydro/schemes/ib/bodies/immersed_boundary.hpp"
#include <string>
#include <vector>

struct InitialConditions {
    real time, checkpoint_interval, dlogt;
    real plm_theta, gamma, cfl, tend;
    luint nx, ny, nz, checkpoint_idx;
    bool quirk_smoothing, homologous, mesh_motion;
    std::vector<std::vector<real>> bfield;
    std::string data_directory, coord_system, solver;
    std::string x1_spacing, x2_spacing, x3_spacing, regime;
    std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;
    std::string spatial_order, temporal_order;
    std::vector<std::string> boundary_conditions;
    std::pair<real, real> x1bounds;
    std::pair<real, real> x2bounds;
    std::pair<real, real> x3bounds;

    using PropertyValue = std::variant<
        real,               // for scalar properties
        std::vector<real>   // Python will enforce the dimensionality of the
                            // vector
        >;

    std::vector<std::pair<
        simbi::ib::BodyType,
        std::unordered_map<std::string, PropertyValue>   // Can store both
                                                         // scalars and vectors
        >>
        immersed_bodies;

    std::tuple<lint, lint, lint> active_zones() const
    {
        const auto nghosts = 2 * (1 + (spatial_order == "plm"));
        return std::make_tuple(nx - nghosts, ny - nghosts, nz - nghosts);
    }
};

#endif