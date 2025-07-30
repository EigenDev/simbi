/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            config_dict_visitor.hpp
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

#ifndef CONFIG_DICT_VISITOR_HPP
#define CONFIG_DICT_VISITOR_HPP

#include "config.hpp"
#include "config_dict.hpp"
#include "init_conditions_visitor.hpp"
#include "utility/enums.hpp"
#include <cmath>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simbi {
    /**
     * @brief Visitor that populates initial_conditions_t from a config_dict_t
     */
    class config_dict_tVisitor : public initial_conditions_tVisitor
    {
      private:
        const config_dict_t& dict;

      public:
        explicit config_dict_tVisitor(const config_dict_t& config)
            : dict(config)
        {
        }

        // Time-related fields
        void visit_time_parameters(
            real& time,
            real& tend,
            real& dlogt,
            real& checkpoint_interval
        ) override
        {
            if (dict.contains("start_time")) {
                time = dict.at("start_time").get<real>();
            }
            if (dict.contains("end_time")) {
                tend = dict.at("end_time").get<real>();
            }
            if (dict.contains("dlogt")) {
                dlogt = dict.at("dlogt").get<real>();
            }
            if (dict.contains("checkpoint_interval")) {
                checkpoint_interval =
                    dict.at("checkpoint_interval").get<real>();
            }
        }

        // Resolution fields
        void visit_resolution(
            std::int64_t& nx,
            std::int64_t& ny,
            std::int64_t& nz
        ) override
        {
            const auto nghosts =
                2 *
                (1 + (dict.at("reconstruction").get<std::string>() == "plm"));
            if (dict.contains("resolution")) {
                const auto& res = dict.at("resolution");
                if (res.is_array_of_ints()) {
                    auto res_vec = res.get<std::vector<std::int64_t>>();
                    nx           = res_vec.size() > 0 ? res_vec[0] : 1;
                    ny           = res_vec.size() > 1 ? res_vec[1] : 1;
                    nz           = res_vec.size() > 2 ? res_vec[2] : 1;
                }
                else if (res.is_int()) {
                    nx = res.get<std::int64_t>();
                    ny = 1;
                    nz = 1;
                }
            }
            nx += nghosts;
            ny += nghosts * (ny > 1 || dict.at("is_mhd").get<bool>());
            nz += nghosts * (nz > 1 || dict.at("is_mhd").get<bool>());
        }

        void visit_physics_parameters(
            real& gamma,
            real& cfl,
            real& sound_speed_squared,
            real& viscosity,
            real& shakura_sunyaev_alpha
        ) override
        {
            gamma = dict.at("adiabatic_index").get<real>();
            cfl   = dict.at("cfl_number").get<real>();
            sound_speed_squared =
                std::pow(dict.at("ambient_sound_speed").get<real>(), 2);
            viscosity = dict.at("viscosity").get<real>();
            shakura_sunyaev_alpha =
                dict.at("shakura_sunyaev_alpha").get<real>();
        }

        void visit_flags(
            bool& quirk_smoothing,
            bool& fleischmann,
            bool& homologous,
            bool& mesh_motion,
            bool& isothermal
        ) override
        {
            quirk_smoothing = dict.at("use_quirk_smoothing").get<bool>();
            fleischmann     = dict.at("use_fleischmann_limiter").get<bool>();
            homologous      = dict.at("is_homologous").get<bool>();
            mesh_motion     = dict.at("mesh_motion").get<bool>();
            isothermal      = dict.at("isothermal").get<bool>();
        }

        void visit_bounds(
            std::pair<real, real>& x1bounds,
            std::pair<real, real>& x2bounds,
            std::pair<real, real>& x3bounds
        ) override
        {
            x1bounds = dict.at("x1bounds").get<std::pair<real, real>>();
            x2bounds = dict.at("x2bounds").get<std::pair<real, real>>();
            x3bounds = dict.at("x3bounds").get<std::pair<real, real>>();
        }

        void visit_coordinates(
            std::string& coord_system,
            std::string& x1_spacing,
            std::string& x2_spacing,
            std::string& x3_spacing
        ) override
        {
            coord_system = dict.at("coord_system").get<std::string>();
            x1_spacing   = dict.at("x1_spacing").get<std::string>();
            x2_spacing   = dict.at("x2_spacing").get<std::string>();
            x3_spacing   = dict.at("x3_spacing").get<std::string>();
        }

        void visit_solver_settings(
            std::string& solver,
            std::string& reconstruction,
            std::string& timestepping,
            std::string& regime,
            real& plm_theta

        ) override
        {
            solver         = dict.at("solver").get<std::string>();
            reconstruction = dict.at("reconstruction").get<std::string>();
            timestepping   = dict.at("timestepping").get<std::string>();
            regime         = dict.at("regime").get<std::string>();
            plm_theta      = dict.at("plm_theta").get<real>();
        }

        void visit_boundary_conditions(
            std::vector<std::string>& boundary_conditions
        ) override
        {
            if (dict.contains("boundary_conditions")) {
                boundary_conditions = dict.at("boundary_conditions")
                                          .get<std::vector<std::string>>();
            }
            else {
                boundary_conditions.clear();
            }
        }

        void visit_source_expressions(
            config_dict_t& bx1_inner_expressions,
            config_dict_t& bx1_outer_expressions,
            config_dict_t& bx2_inner_expressions,
            config_dict_t& bx2_outer_expressions,
            config_dict_t& bx3_inner_expressions,
            config_dict_t& bx3_outer_expressions,
            config_dict_t& hydro_source_expressions,
            config_dict_t& gravity_source_expressions,
            config_dict_t& local_sound_speed_expressions
        ) override
        {
            if (dict.at("bx1_inner_expressions").is_dict()) {
                bx1_inner_expressions =
                    dict.at("bx1_inner_expressions").get<config_dict_t>();
            }
            if (dict.at("bx1_outer_expressions").is_dict()) {
                bx1_outer_expressions =
                    dict.at("bx1_outer_expressions").get<config_dict_t>();
            }
            if (dict.at("bx2_inner_expressions").is_dict()) {
                bx2_inner_expressions =
                    dict.at("bx2_inner_expressions").get<config_dict_t>();
            }
            if (dict.at("bx2_outer_expressions").is_dict()) {
                bx2_outer_expressions =
                    dict.at("bx2_outer_expressions").get<config_dict_t>();
            }
            if (dict.at("bx3_inner_expressions").is_dict()) {
                bx3_inner_expressions =
                    dict.at("bx3_inner_expressions").get<config_dict_t>();
            }
            if (dict.at("bx3_outer_expressions").is_dict()) {
                bx3_outer_expressions =
                    dict.at("bx3_outer_expressions").get<config_dict_t>();
            }
            if (dict.at("hydro_source_expressions").is_dict()) {
                hydro_source_expressions =
                    dict.at("hydro_source_expressions").get<config_dict_t>();
            }
            if (dict.at("gravity_source_expressions").is_dict()) {
                gravity_source_expressions =
                    dict.at("gravity_source_expressions").get<config_dict_t>();
            }
            if (dict.at("local_sound_speed_expressions").is_dict()) {
                local_sound_speed_expressions =
                    dict.at("local_sound_speed_expressions")
                        .get<config_dict_t>();
            }
        }

        void visit_immersed_bodies(
            std::vector<config_dict_t>& immersed_bodies
        ) override
        {
            if (dict.contains("immersed_bodies") &&
                dict.at("immersed_bodies").is_list()) {
                const auto& body_list =
                    dict.at("immersed_bodies").get<std::list<config_dict_t>>();

                for (const auto& body_dict : body_list) {
                    // extract body type from the dict!
                    if (!body_dict.contains("capability") ||
                        !body_dict.at("capability").is_body_cap()) {
                        continue;   // Skip invalid body entries
                    }

                    // create property map
                    simbi::config_dict_t props;

                    // add the required properties
                    add_vector_property(body_dict, "position", props);
                    add_vector_property(body_dict, "velocity", props);
                    add_scalar_property(body_dict, "mass", props);
                    add_scalar_property(body_dict, "radius", props);
                    add_boolean_property(body_dict, "two_way_coupling", props);
                    add_body_property(body_dict, "capability", props);

                    // add specifics/extra properties
                    // this is a dictionary of properties that are specific to
                    // the body type
                    if (body_dict.contains("specifics") &&
                        body_dict.at("specifics").is_dict()) {
                        const auto& specifics =
                            body_dict.at("specifics")
                                .template get<simbi::config_dict_t>();
                        for (const auto& [key, value] : specifics) {
                            add_property(key, value, props);
                        }
                    }

                    // add other properties (not in specifics)
                    for (const auto& [key, value] : body_dict) {
                        if (key != "capability" && key != "position" &&
                            key != "velocity" && key != "mass" &&
                            key != "radius" && key != "specifics") {
                            add_property(key, value, props);
                        }
                    }

                    immersed_bodies.push_back(props);
                }
            }
            else {
                immersed_bodies.clear();
            }
        }

        void visit_output_settings(
            std::string& data_directory,
            std::uint64_t& checkpoint_index
        ) override
        {
            data_directory   = dict.at("data_directory").get<std::string>();
            checkpoint_index = dict.at("checkpoint_index").get<std::uint64_t>();
        }

        void visit_computed_properties(
            std::uint64_t& dimensionality,
            bool& is_mhd,
            bool& is_relativistic,
            std::uint64_t& nvars,
            std::uint64_t& halo_radius
        ) override
        {
            dimensionality  = dict.at("dimensionality").get<std::int64_t>();
            is_mhd          = dict.at("is_mhd").get<bool>();
            is_relativistic = dict.at("is_relativistic").get<bool>();
            nvars           = dict.at("nvars").get<std::int64_t>();
            halo_radius =
                1 + (dict.at("reconstruction").get<std::string>() == "plm");
        }

        static void add_property(
            const std::string& name,
            const simbi::config_value_t& value,
            simbi::config_dict_t& props
        )
        {
            if (value.is_real_number()) {
                props[name] = static_cast<real>(value.get<double>());
            }
            else if (value.is_array_of_floats()) {
                props[name] = value.get<std::vector<double>>();
            }
            else if (value.is_bool()) {
                props[name] = value.get<bool>();
            }
        }

        static void add_vector_property(
            const simbi::config_dict_t& dict,
            const std::string& name,
            simbi::config_dict_t& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_array_of_floats()) {
                props[name] = dict.at(name).get<std::vector<double>>();
            }
        }

        static void add_scalar_property(
            const simbi::config_dict_t& dict,
            const std::string& name,
            simbi::config_dict_t& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_real_number()) {
                props[name] = static_cast<real>(dict.at(name).get<double>());
            }
        }
        static void add_boolean_property(
            const simbi::config_dict_t& dict,
            const std::string& name,
            simbi::config_dict_t& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_bool()) {
                props[name] = dict.at(name).get<bool>();
            }
        }

        static void add_body_property(
            const simbi::config_dict_t& dict,
            const std::string& name,
            simbi::config_dict_t& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_body_cap()) {
                props[name] = dict.at(name).get<simbi::BodyCapability>();
            }
        }
    };

    /**
     * @brief Visitor that applies default values to initial_conditions_t
     */
    class DefaultsVisitor : public initial_conditions_tVisitor
    {
      public:
        // Time-related fields
        void visit_time_parameters(
            real& time,
            real& tend,
            real& dlogt,
            real& checkpoint_interval
        ) override
        {
            time                = 0.0;
            tend                = 1.0;
            dlogt               = 0.0;
            checkpoint_interval = 0.1;
        }

        // Resolution fields
        void visit_resolution(
            std::int64_t& nx,
            std::int64_t& ny,
            std::int64_t& nz
        ) override
        {
            nx = 100;   // Default resolution
            ny = 1;
            nz = 1;
        }

        void visit_physics_parameters(
            real& gamma,
            real& cfl,
            real& sound_speed_squared,
            real& viscosity,
            real& shakura_sunyaev_alpha
        ) override
        {
            gamma                 = 1.4;
            cfl                   = 0.5;
            sound_speed_squared   = 1.0;
            viscosity             = 0.0;
            shakura_sunyaev_alpha = 0.0;
        }

        void visit_flags(
            bool& quirk_smoothing,
            bool& fleischmann,
            bool& homologous,
            bool& mesh_motion,
            bool& isothermal
        ) override
        {
            quirk_smoothing = false;
            fleischmann     = false;
            homologous      = false;
            mesh_motion     = false;
            isothermal      = false;
        }

        void visit_bounds(
            std::pair<real, real>& x1bounds,
            std::pair<real, real>& x2bounds,
            std::pair<real, real>& x3bounds
        ) override
        {
            x1bounds = {0.0, 1.0};
            x2bounds = {0.0, 1.0};
            x3bounds = {0.0, 1.0};
        }

        void visit_coordinates(
            std::string& coord_system,
            std::string& x1_spacing,
            std::string& x2_spacing,
            std::string& x3_spacing
        ) override
        {
            coord_system = "cartesian";
            x1_spacing   = "linear";
            x2_spacing   = "linear";
            x3_spacing   = "linear";
        }

        void visit_solver_settings(
            std::string& solver,
            std::string& reconstruction,
            std::string& timestepping,
            std::string& regime,
            real& plm_theta
        ) override
        {
            solver         = "hlle";
            reconstruction = "plm";
            timestepping   = "rk2";
            regime         = "classical";
            plm_theta      = 1.5;   // Default PLM theta
        }

        void visit_boundary_conditions(
            std::vector<std::string>& boundary_conditions
        ) override
        {
            boundary_conditions.clear();   // No default boundary conditions
        }

        void visit_source_expressions(
            config_dict_t& bx1_inner_expressions,
            config_dict_t& bx1_outer_expressions,
            config_dict_t& bx2_inner_expressions,
            config_dict_t& bx2_outer_expressions,
            config_dict_t& bx3_inner_expressions,
            config_dict_t& bx3_outer_expressions,
            config_dict_t& hydro_source_expressions,
            config_dict_t& gravity_source_expressions,
            config_dict_t& local_sound_speed_expressions
        ) override
        {
            bx1_inner_expressions.clear();
            bx1_outer_expressions.clear();
            bx2_inner_expressions.clear();
            bx2_outer_expressions.clear();
            bx3_inner_expressions.clear();
            bx3_outer_expressions.clear();
            hydro_source_expressions.clear();
            gravity_source_expressions.clear();
            local_sound_speed_expressions.clear();
        }

        void visit_immersed_bodies(
            std::vector<config_dict_t>& immersed_bodies
        ) override
        {
            immersed_bodies.clear();   // No default immersed bodies
        }

        void visit_output_settings(
            std::string& data_directory,
            std::uint64_t& checkpoint_index
        ) override
        {
            data_directory   = "output";
            checkpoint_index = 0;
        }

        void visit_computed_properties(
            std::uint64_t& dimensionality,
            bool& is_mhd,
            bool& is_relativistic,
            std::uint64_t& nvars,
            std::uint64_t& halo_radius
        ) override
        {
            dimensionality  = 3;   // Default to 3D
            is_mhd          = false;
            is_relativistic = false;
            nvars           = 5;   // Default number of variables
            halo_radius     = 1;   // Default halo radius
        }
    };

    /**
     * @brief Visitor that validates initial_conditions_t values
     */
    class ValidationVisitor : public initial_conditions_tVisitor
    {
      public:
        // Time-related fields
        void visit_time_parameters(
            real& time,
            real& tend,
            real& dlogt,
            real& checkpoint_interval
        ) override
        {
            if (tend <= time) {
                throw std::runtime_error(
                    "End time must be greater than start time"
                );
            }
            if (checkpoint_interval <= 0.0) {
                throw std::runtime_error(
                    "Checkpostd::int64_t interval must be positive"
                );
            }
            if (dlogt < 0.0) {
                throw std::runtime_error("dlogt must be non-negative");
            }
        }

        // Resolution fields
        void visit_resolution(
            std::int64_t& nx,
            std::int64_t& ny,
            std::int64_t& nz
        ) override
        {
            if (nx == 0 || ny == 0 || nz == 0) {
                throw std::runtime_error("Resolution cannot be zero");
            }
        }

        void visit_physics_parameters(
            real& gamma,
            real& cfl,
            real& sound_speed_squared,
            real& viscosity,
            real& shakura_sunyaev_alpha
        ) override
        {
            if (gamma < 1.0) {
                throw std::runtime_error("Gamma must be greater than 1.0");
            }
            if (cfl <= 0.0 || cfl > 1.0) {
                throw std::runtime_error("CFL must be in (0, 1]");
            }
            if (shakura_sunyaev_alpha < 0.0) {
                throw std::runtime_error(
                    "Shakura-Sunyaev alpha must be non-negative"
                );
            }
            if (viscosity < 0.0) {
                throw std::runtime_error("Viscosity must be non-negative");
            }
            if (sound_speed_squared < 0.0) {
                throw std::runtime_error(
                    "Sound speed squared must be positive"
                );
            }
        }

        void visit_flags(bool&, bool&, bool&, bool&, bool&) override
        {
            // No specific validation for flags, but could be added if needed
        }

        void visit_bounds(
            std::pair<real, real>& x1bounds,
            std::pair<real, real>& x2bounds,
            std::pair<real, real>& x3bounds
        ) override
        {
            if (x1bounds.first >= x1bounds.second) {
                throw std::runtime_error("X1 bounds are invalid");
            }
            if (x2bounds.first >= x2bounds.second) {
                throw std::runtime_error("X2 bounds are invalid");
            }
            if (x3bounds.first >= x3bounds.second) {
                throw std::runtime_error("X3 bounds are invalid");
            }
        }

        void visit_coordinates(
            std::string& coord_system,
            std::string& x1_spacing,
            std::string& x2_spacing,
            std::string& x3_spacing
        ) override
        {
            // Validate coordinate system and spacing types if needed
            if (coord_system.empty()) {
                throw std::runtime_error("Coordinate system cannot be empty");
            }
            if (!(x1_spacing == "linear") && !(x1_spacing == "log")) {
                throw std::runtime_error(
                    "X1 spacing must be 'linear' or 'log'"
                );
            }
            if (!(x2_spacing == "linear") && !(x2_spacing == "log")) {
                throw std::runtime_error(
                    "X2 spacing must be 'linear' or 'log'"
                );
            }
            if (!(x3_spacing == "linear") && !(x3_spacing == "log")) {
                throw std::runtime_error(
                    "X3 spacing must be 'linear' or 'log'"
                );
            }
        }

        void visit_solver_settings(
            std::string& solver,
            std::string& reconstruction,
            std::string& timestepping,
            std::string& regime,
            real& plm_theta
        ) override
        {
            if (solver.empty()) {
                throw std::runtime_error("Solver cannot be empty");
            }
            if (reconstruction.empty()) {
                throw std::runtime_error("Spatial order cannot be empty");
            }
            if (timestepping.empty()) {
                throw std::runtime_error("Temporal order cannot be empty");
            }
            if (regime.empty()) {
                throw std::runtime_error("Regime cannot be empty");
            }
            if (plm_theta < 0.0 || plm_theta > 2.0) {
                throw std::runtime_error("PLM theta must be in [0, 2]");
            }
        }

        void visit_boundary_conditions(
            std::vector<std::string>& boundary_conditions
        ) override
        {
            if (boundary_conditions.empty()) {
                throw std::runtime_error("Boundary conditions cannot be empty");
            }
        }

        void visit_source_expressions(
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&,
            config_dict_t&
        ) override
        {
            // Validate expressions if needed
        }

        void visit_immersed_bodies(std::vector<config_dict_t>&) override
        {
            // Validate immersed bodies if needed
        }

        void visit_output_settings(
            std::string& data_directory,
            std::uint64_t&
        ) override
        {
            if (data_directory.empty()) {
                throw std::runtime_error("Data directory cannot be empty");
            }
        }

        void visit_computed_properties(
            std::uint64_t& dimensionality,
            bool&,
            bool&,
            std::uint64_t& nvars,
            std::uint64_t& halo_radius
        ) override
        {
            if (dimensionality < 1 || dimensionality > 3) {
                throw std::runtime_error("Dimensionality must be 1, 2, or 3");
            }
            if (nvars == 0) {
                throw std::runtime_error("Number of variables cannot be zero");
            }
            if (halo_radius < 0) {
                throw std::runtime_error("Halo radius cannot be negative");
            }
        }
    };

}   // namespace simbi

#endif   // CONFIG_DICT_VISITOR_HPP
