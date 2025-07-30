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

#include "config.hpp"
#include "config_dict.hpp"
#include "config_dict_visitor.hpp"
#include "system/io/exceptions.hpp"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

namespace simbi {

    struct initial_conditions_t {
        // existing fields
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

        std::int64_t nx;
        std::int64_t ny;
        std::int64_t nz;
        std::uint64_t checkpoint_index;
        std::uint64_t dimensionality;
        std::uint64_t nvars;
        std::uint64_t halo_radius;

        bool quirk_smoothing;
        bool fleischmann_limiter;
        bool homologous;
        bool mesh_motion;
        bool isothermal;
        bool is_mhd;
        bool is_relativistic;

        std::string data_directory;
        std::string coord_system;
        std::string solver;
        std::string x1_spacing;
        std::string x2_spacing;
        std::string x3_spacing;
        std::string regime;
        std::string reconstruct;
        std::string timestepping;

        std::vector<std::string> boundary_conditions;

        std::pair<real, real> x1bounds;
        std::pair<real, real> x2bounds;
        std::pair<real, real> x3bounds;

        config_dict_t config;
        std::vector<config_dict_t> immersed_bodies;

        config_dict_t bx1_outer_expressions;
        config_dict_t bx1_inner_expressions;
        config_dict_t bx2_outer_expressions;
        config_dict_t bx2_inner_expressions;
        config_dict_t bx3_outer_expressions;
        config_dict_t bx3_inner_expressions;
        config_dict_t hydro_source_expressions;
        config_dict_t gravity_source_expressions;
        config_dict_t local_sound_speed_expressions;

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
                fleischmann_limiter,
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
                reconstruct,
                timestepping,
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
            visitor.visit_output_settings(data_directory, checkpoint_index);
            visitor.visit_computed_properties(
                dimensionality,
                is_mhd,
                is_relativistic,
                nvars,
                halo_radius
            );
        }

        // Factory method using visitor pattern
        static initial_conditions_t create(const config_dict_t& config)
        {
            initial_conditions_t result{};

            // First apply defaults
            DefaultsVisitor defaults_visitor;
            result.accept(defaults_visitor);

            // Then populate from config
            config_dict_tVisitor config_visitor(config);
            result.accept(config_visitor);

            // Validate the configuration
            ValidationVisitor validation_visitor;
            result.accept(validation_visitor);

            // Store the original config for reference
            result.config = config;

            return result;
        }

        std::tuple<std::int64_t, std::int64_t, std::int64_t>
        active_zones() const
        {
            const auto nghosts = 2 * halo_radius;
            return std::make_tuple(
                nx - nghosts,
                std::max<std::int64_t>(ny - nghosts, 1),
                std::max<std::int64_t>(nz - nghosts, 1)
            );
        }

        template <std::uint64_t Dims>
        iarray<Dims> get_active_shape() const
        {
            if constexpr (Dims == 1) {
                return {static_cast<std::int64_t>(nx - 2 * halo_radius)};
            }
            else if constexpr (Dims == 2) {
                return {
                  std::max<std::int64_t>(ny - 2 * halo_radius, 1),
                  static_cast<std::int64_t>(nx - 2 * halo_radius)
                };
            }
            else if constexpr (Dims == 3) {
                return {
                  std::max<std::int64_t>(nz - 2 * halo_radius, 1),
                  std::max<std::int64_t>(ny - 2 * halo_radius, 1),
                  static_cast<std::int64_t>(nx - 2 * halo_radius)
                };
            }
            else {
                static_assert(Dims <= 3, "Unsupported dimension");
            }
        };

        template <std::uint64_t Dims>
        iarray<Dims> get_full_shape() const
        {
            if constexpr (Dims == 1) {
                return {nx};
            }
            else if constexpr (Dims == 2) {
                return {ny, nx};
            }
            else if constexpr (Dims == 3) {
                return {nz, ny, nx};
            }
            else {
                static_assert(Dims <= 3, "Unsupported dimension");
            }
        }

        template <std::uint64_t Dims>
        vector_t<real, Dims> get_bounds() const
        {
            if constexpr (Dims == 1) {
                return {x1bounds.first, x1bounds.second};
            }
            else if constexpr (Dims == 2) {
                return {
                  x2bounds.first,
                  x1bounds.first,
                  x2bounds.second,
                  x1bounds.second
                };
            }
            else if constexpr (Dims == 3) {
                return {
                  x3bounds.first,
                  x2bounds.first,
                  x1bounds.first,
                  x3bounds.second,
                  x2bounds.second,
                  x1bounds.second
                };
            }
            else {
                static_assert(Dims <= 3, "Unsupported dimension");
            }
        }

        auto checkpoint_zones() const
        {
            auto [active_nx, active_ny, active_nz] = active_zones();
            if (active_nz > 1) {
                return static_cast<std::uint64_t>(active_nz);
            }
            else if (active_ny > 1) {
                return static_cast<std::uint64_t>(active_ny);
            }
            return static_cast<std::uint64_t>(active_nx);
        }

        bool contains(const std::string& key) const
        {
            return config.find(key) != config.end();
        }

        const config_value_t& at(const std::string& key) const
        {
            static const config_value_t empty_value;
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

        config_dict_t get_dict(const std::string& key) const
        {
            if (!contains(key) || !at(key).is_dict()) {
                return {};   // Return empty dict
            }
            return at(key).get<config_dict_t>();
        }

        config_value_t get_nested(const std::string& nested_key) const
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

            const simbi::config_dict_t* current_dict = &config;
            for (std::uint64_t ii = 0; ii < keys.size() - 1; ++ii) {
                auto it = current_dict->find(keys[ii]);
                if (it == current_dict->end() || !it->second.is_dict()) {
                    return {};   // Key not found or not a dictionary
                }
                current_dict =
                    &std::get<simbi::config_dict_t>(it->second.value);
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
