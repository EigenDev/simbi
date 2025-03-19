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

#include "build_options.hpp"   // for real
#include "config_dict.hpp"     // for ConfigDict
#include "core/types/containers/vector.hpp"
#include "core/types/utility/enums.hpp"
#include "enums.hpp"
#include "physics/hydro/schemes/ib/bodies/immersed_boundary.hpp"
#include <cinttypes>
#include <unordered_map>

struct InitialConditions {
    real time, checkpoint_interval, dlogt;
    real plm_theta, gamma, cfl, tend, sound_speed_squared;
    luint nx, ny, nz, checkpoint_idx;
    bool quirk_smoothing, homologous, mesh_motion, isothermal;
    std::vector<std::vector<real>> bfield;
    std::string data_directory, coord_system, solver;
    std::string x1_spacing, x2_spacing, x3_spacing, regime;
    std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;
    std::string spatial_order, temporal_order;
    std::vector<std::string> boundary_conditions;
    std::pair<real, real> x1bounds;
    std::pair<real, real> x2bounds;
    std::pair<real, real> x3bounds;
    bool enable_peer_access{true}, managed_memory{false};
    simbi::ConfigDict config;

    using PropertyValue = std::variant<
        real,                // for scalar properties
        std::vector<real>,   // Python will enforce the dimensionality of the
                             // vector
        bool                 // for boolean properties
        >;
    using PropertyMap = std::unordered_map<std::string, PropertyValue>;

    std::vector<std::pair<simbi::BodyType, PropertyMap>> immersed_bodies;

    std::tuple<lint, lint, lint> active_zones() const
    {
        const auto nghosts = 2 * (1 + (spatial_order == "plm"));
        return std::make_tuple(nx - nghosts, ny - nghosts, nz - nghosts);
    }

    // Check if a key exists
    bool contains(const std::string& key) const
    {
        return config.find(key) != config.end();
    }

    // Access a value (with optional default)
    const simbi::ConfigValue& at(const std::string& key) const
    {
        static const simbi::ConfigValue empty_value;
        auto it = config.find(key);
        return (it != config.end()) ? it->second : empty_value;
    }

    // Accessor with default value
    template <typename T>
    T get(const std::string& key, const T& default_value) const
    {
        if (!contains(key)) {
            return default_value;
        }
        try {
            return at(key).get<T>();
        }
        catch (...) {
            return default_value;
        }
    }

    // Nested dictionary access
    simbi::ConfigDict get_dict(const std::string& key) const
    {
        if (!contains(key) || !at(key).is_dict()) {
            return {};   // Return empty dict
        }
        return at(key).get<simbi::ConfigDict>();
    }

    // Helper for nested key access (e.g.
    // "gravitational_system.binary_config.semi_major")
    simbi::ConfigValue get_nested(const std::string& nested_key) const
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

    // Function to populate the immersed_bodies from config
    // This allows backward compatibility while we transition to new config
    // system
    void populate_immersed_bodies()
    {
        if (!contains("immersed_bodies") || !at("immersed_bodies").is_dict()) {
            return;
        }

        const auto& bodies_dict =
            at("immersed_bodies").get<simbi::ConfigDict>();
        for (const auto& [body_name, body_config] : bodies_dict) {
            if (!body_config.is_dict()) {
                continue;
            }

            const auto& body_dict = body_config.get<simbi::ConfigDict>();

            // Extract body properties
            auto it_type = body_dict.find("body_type");
            if (it_type == body_dict.end() || !it_type->second.is_string()) {
                continue;   // Skip if no valid type
            }

            const std::string type_str = it_type->second.get<std::string>();
            simbi::BodyType body_type;

            if (type_str == "gravitational") {
                body_type = simbi::BodyType::GRAVITATIONAL;
            }
            else if (type_str == "gravitational_sink") {
                body_type = simbi::BodyType::GRAVITATIONAL_SINK;
            }
            else {
                continue;   // Skip if type is not recognized
            }

            // Extract common properties
            PropertyMap props;

            // Add each property to the map
            for (const auto& [prop_name, prop_value] : body_dict) {
                if (prop_name == "body_type") {
                    continue;   // Already handled
                }

                // Convert the ConfigValue to PropertyValue
                if (prop_value.is_double()) {
                    props[prop_name] =
                        static_cast<real>(prop_value.get<double>());
                }
                else if (prop_value.is_array()) {
                    props[prop_name] = prop_value.get<std::vector<double>>();
                }
                else if (prop_value.is_bool()) {
                    props[prop_name] = prop_value.get<bool>();
                }
                else {
                    throw std::runtime_error(
                        "Unsupported property type for " + prop_name
                    );
                }
            }

            // Add to the immersed_bodies list
            immersed_bodies.push_back(std::make_pair(body_type, props));
        }
    }
};

#endif
