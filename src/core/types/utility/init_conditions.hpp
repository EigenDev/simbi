/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            init_conditions.hpp
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

#ifndef INIT_CONDITIONS_HPP
#define INIT_CONDITIONS_HPP

#include "build_options.hpp"
#include "config_dict_visitor.hpp"
#include "init_conditions_visitor.hpp"
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace simbi {

    struct InitialConditions {
        // Existing fields
        real time;
        real checkpoint_interval;
        real dlogt;
        real viscosity;
        real plm_theta;
        real gamma;
        real cfl;
        real tend;
        real sound_speed_squared;
        real shakura_sunyaev_alpha;

        luint nx;
        luint ny;
        luint nz;
        luint checkpoint_index;
        luint dimensionality;
        luint nvars;

        bool quirk_smoothing;
        bool homologous;
        bool mesh_motion;
        bool isothermal;
        bool is_mhd;
        bool is_relativistic;

        std::vector<std::vector<real>> bfield;

        std::string data_directory;
        std::string coord_system;
        std::string solver;
        std::string x1_spacing;
        std::string x2_spacing;
        std::string x3_spacing;
        std::string regime;
        std::string spatial_order;
        std::string temporal_order;

        std::vector<std::string> boundary_conditions;

        std::pair<real, real> x1bounds;
        std::pair<real, real> x2bounds;
        std::pair<real, real> x3bounds;

        ConfigDict config;
        std::vector<ConfigDict> immersed_bodies;

        ConfigDict bx1_outer_expressions;
        ConfigDict bx1_inner_expressions;
        ConfigDict bx2_outer_expressions;
        ConfigDict bx2_inner_expressions;
        ConfigDict bx3_outer_expressions;
        ConfigDict bx3_inner_expressions;
        ConfigDict hydro_source_expressions;
        ConfigDict gravity_source_expressions;
        ConfigDict local_sound_speed_expressions;

        // gpu-related fields
        bool enable_peer_access{true};
        bool managed_memory{false};

        // New method to accept a visitor
        template <typename Visitor>
        void accept(Visitor& visitor)
        {
            // Call visitor methods for each field group
            visitor
                .visit_time_parameters(time, tend, dlogt, checkpoint_interval);
            visitor.visit_resolution(nx, ny, nz);
            visitor.visit_physics_parameters(
                gamma,
                cfl,
                sound_speed_squared,
                viscosity,
                shakura_sunyaev_alpha
            );
            visitor.visit_flags(
                quirk_smoothing,
                homologous,
                mesh_motion,
                isothermal
            );
            visitor.visit_bounds(x1bounds, x2bounds, x3bounds);
            visitor.visit_coordinates(
                coord_system,
                x1_spacing,
                x2_spacing,
                x3_spacing
            );
            visitor.visit_solver_settings(
                solver,
                spatial_order,
                temporal_order,
                regime,
                plm_theta
            );
            visitor.visit_boundary_conditions(boundary_conditions);
            visitor.visit_source_expressions(
                bx1_inner_expressions,
                bx1_outer_expressions,
                bx2_inner_expressions,
                bx2_outer_expressions,
                bx3_inner_expressions,
                bx3_outer_expressions,
                hydro_source_expressions,
                gravity_source_expressions,
                local_sound_speed_expressions
            );
            visitor.visit_immersed_bodies(immersed_bodies);
            visitor.visit_magnetic_field(bfield);
            visitor.visit_output_settings(data_directory, checkpoint_index);
            visitor.visit_computed_properties(
                dimensionality,
                is_mhd,
                is_relativistic,
                nvars
            );
        }

        // Factory method using visitor pattern
        static InitialConditions create(const ConfigDict& config)
        {
            InitialConditions result{};

            // First apply defaults
            DefaultsVisitor defaults_visitor;
            result.accept(defaults_visitor);

            // Then populate from config
            ConfigDictVisitor config_visitor(config);
            result.accept(config_visitor);

            // Validate the configuration
            ValidationVisitor validation_visitor;
            result.accept(validation_visitor);

            // Store the original config for reference
            result.config = config;

            return result;
        }

        // Existing methods...
        std::tuple<size_type, size_type, size_type> active_zones() const
        {
            const auto nghosts = 2 * (1 + (spatial_order == "plm"));
            return std::make_tuple(
                nx - nghosts,
                std::max<lint>(ny - nghosts, 1),
                std::max<lint>(nz - nghosts, 1)
            );
        }

        bool contains(const std::string& key) const
        {
            return config.find(key) != config.end();
        }

        const ConfigValue& at(const std::string& key) const
        {
            static const ConfigValue empty_value;
            auto it = config.find(key);
            return (it != config.end()) ? it->second : empty_value;
        }

        template <typename T>
        T get(const std::string& key) const
        {
            if (!contains(key)) {
                throw std::runtime_error("Key not found: " + key);
            }
            try {
                return at(key).get<T>();
            }
            catch (const std::bad_cast&) {
                throw std::runtime_error("Type mismatch for key: " + key);
            }
        }

        ConfigDict get_dict(const std::string& key) const
        {
            if (!contains(key) || !at(key).is_dict()) {
                return {};   // Return empty dict
            }
            return at(key).get<ConfigDict>();
        }

        ConfigValue get_nested(const std::string& nested_key) const
        {
            std::istringstream ss(nested_key);
            std::string key;
            std::vector<std::string> keys;

            while (std::getline(ss, key, '.')) {
                keys.push_back(key);
            }

            if (keys.empty()) {
                return {};
            }

            const simbi::ConfigDict* current_dict = &config;
            for (size_t i = 0; i < keys.size() - 1; ++i) {
                auto it = current_dict->find(keys[i]);
                if (it == current_dict->end() || !it->second.is_dict()) {
                    return {};   // Key not found or not a dictionary
                }
                current_dict = &std::get<simbi::ConfigDict>(it->second.value);
            }

            auto it = current_dict->find(keys.back());
            if (it == current_dict->end()) {
                return {};   // Final key not found
            }

            return it->second;
        }
    };

}   // namespace simbi

#endif   // INIT_CONDITIONS_HPP
