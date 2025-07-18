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

#include "config.hpp"        // for real, std::int64_t types
#include "config_dict.hpp"   // for config_dict_t
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace simbi {

    /**
     * @brief Visitor interface for initial_conditions_t
     *
     * This visitor defines the interface for processing initial_conditions_t
     * fields. Concrete visitors implement this interface to populate, validate,
     * or process the initial_conditions_t in different ways.
     */
    class initial_conditions_tVisitor
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
        virtual void visit_resolution(
            std::int64_t& nx,
            std::int64_t& ny,
            std::int64_t& nz
        ) = 0;

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
            std::string& reconstruction,
            std::string& timestepping,
            std::string& regime,
            real& plm_theta
        ) = 0;

        // Boundary conditions
        virtual void visit_boundary_conditions(
            std::vector<std::string>& boundary_conditions
        ) = 0;

        // Source expressions (config_dict_t objects)
        virtual void visit_source_expressions(
            config_dict_t& bx1_inner_expressions,
            config_dict_t& bx1_outer_expressions,
            config_dict_t& bx2_inner_expressions,
            config_dict_t& bx2_outer_expressions,
            config_dict_t& bx3_inner_expressions,
            config_dict_t& bx3_outer_expressions,
            config_dict_t& hydro_source_expressions,
            config_dict_t& gravity_source_expressions,
            config_dict_t& local_sound_speed_expressions
        ) = 0;

        // Immersed bodies
        virtual void
        visit_immersed_bodies(std::vector<config_dict_t>& immersed_bodies) = 0;

        // Output settings
        virtual void visit_output_settings(
            std::string& data_directory,
            std::int64_t& checkpoint_index
        ) = 0;

        // Other computed properties
        virtual void visit_computed_properties(
            std::uint64_t& dimensionality,
            bool& is_mhd,
            bool& is_relativistic,
            std::uint64_t& nvars,
            std::uint64_t& halo_radius
        ) = 0;

        // Allow for proper polymorphic destruction
        virtual ~initial_conditions_tVisitor() = default;
    };

}   // namespace simbi

#endif   // INIT_CONDITIONS_VISITOR_HPP
