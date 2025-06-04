/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            init_conditions_visitor.hpp
 * @brief
 * @details
 *
 * @version         0.8.0
 * @date            2025-06-02
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-06-02      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */

#ifndef INIT_CONDITIONS_VISITOR_HPP
#define INIT_CONDITIONS_VISITOR_HPP

#include "build_options.hpp"   // for real, luint types
#include "config_dict.hpp"     // for ConfigDict
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace simbi {

    /**
     * @brief Visitor interface for InitialConditions
     *
     * This visitor defines the interface for processing InitialConditions
     * fields. Concrete visitors implement this interface to populate, validate,
     * or process the InitialConditions in different ways.
     */
    class InitialConditionsVisitor
    {
      public:
        // Time-related fields
        virtual void visit_time_parameters(
            real& time,
            real& tend,
            real& dlogt,
            real& checkpoint_interval
        ) = 0;

        // Resolution fields
        virtual void visit_resolution(luint& nx, luint& ny, luint& nz) = 0;

        // Physics parameters
        virtual void visit_physics_parameters(
            real& gamma,
            real& cfl,
            real& sound_speed_squared,
            real& viscosity,
            real& shakura_sunyaev_alpha
        ) = 0;

        // Boolean flags
        virtual void visit_flags(
            bool& quirk_smoothing,
            bool& fleischmann,
            bool& homologous,
            bool& mesh_motion,
            bool& isothermal
        ) = 0;

        // Bounds
        virtual void visit_bounds(
            std::pair<real, real>& x1bounds,
            std::pair<real, real>& x2bounds,
            std::pair<real, real>& x3bounds
        ) = 0;

        // Coordinate system settings
        virtual void visit_coordinates(
            std::string& coord_system,
            std::string& x1_spacing,
            std::string& x2_spacing,
            std::string& x3_spacing
        ) = 0;

        // Solver settings
        virtual void visit_solver_settings(
            std::string& solver,
            std::string& spatial_order,
            std::string& temporal_order,
            std::string& regime,
            real& plm_theta
        ) = 0;

        // Boundary conditions
        virtual void visit_boundary_conditions(
            std::vector<std::string>& boundary_conditions
        ) = 0;

        // Source expressions (ConfigDict objects)
        virtual void visit_source_expressions(
            ConfigDict& bx1_inner_expressions,
            ConfigDict& bx1_outer_expressions,
            ConfigDict& bx2_inner_expressions,
            ConfigDict& bx2_outer_expressions,
            ConfigDict& bx3_inner_expressions,
            ConfigDict& bx3_outer_expressions,
            ConfigDict& hydro_source_expressions,
            ConfigDict& gravity_source_expressions,
            ConfigDict& local_sound_speed_expressions
        ) = 0;

        // Immersed bodies
        virtual void
        visit_immersed_bodies(std::vector<ConfigDict>& immersed_bodies) = 0;

        // Magnetic field
        virtual void
        visit_magnetic_field(std::vector<std::vector<real>>& bfield) = 0;

        // Output settings
        virtual void visit_output_settings(
            std::string& data_directory,
            luint& checkpoint_index
        ) = 0;

        // Other computed properties
        virtual void visit_computed_properties(
            luint& dimensionality,
            bool& is_mhd,
            bool& is_relativistic,
            luint& nvars
        ) = 0;

        // Allow for proper polymorphic destruction
        virtual ~InitialConditionsVisitor() = default;
    };

}   // namespace simbi

#endif   // INIT_CONDITIONS_VISITOR_HPP
