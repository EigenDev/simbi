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
#include <cinttypes>
#include <memory>
#include <unordered_map>
#include <utility>

struct InitialConditions {
    real time, checkpoint_interval, dlogt;
    real plm_theta, gamma, cfl, tend, sound_speed_squared;
    luint nx, ny, nz, checkpoint_index;
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
    std::vector<std::pair<simbi::BodyType, simbi::ConfigDict>> immersed_bodies;

    // user-defined expressions to be evaluated
    simbi::ConfigDict bx1_outer_expressions, bx1_inner_expressions;
    simbi::ConfigDict bx2_outer_expressions, bx2_inner_expressions;
    simbi::ConfigDict bx3_outer_expressions, bx3_inner_expressions;
    simbi::ConfigDict hydro_source_expressions;
    simbi::ConfigDict gravity_source_expressions;

    std::tuple<size_type, size_type, size_type> active_zones() const
    {
        const auto nghosts = 2 * (1 + (spatial_order == "plm"));
        return std::make_tuple(
            nx - nghosts,
            std::max<lint>(ny - nghosts, 1),
            std::max<lint>(nz - nghosts, 1)
        );
    }

    // Basic dictionary access methods
    bool contains(const std::string& key) const
    {
        return config.find(key) != config.end();
    }

    const simbi::ConfigValue& at(const std::string& key) const
    {
        static const simbi::ConfigValue empty_value;
        auto it = config.find(key);
        return (it != config.end()) ? it->second : empty_value;
    }

    template <typename T>
    T get(const std::string& key, const T& default_value) const
    {
        if (!contains(key)) {
            return default_value;
        }
        try {
            return at(key).template get<T>();
        }
        catch (...) {
            return default_value;
        }
    }

    simbi::ConfigDict get_dict(const std::string& key) const
    {
        if (!contains(key) || !at(key).is_dict()) {
            return {};   // Return empty dict
        }
        return at(key).template get<simbi::ConfigDict>();
    }

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

    // Builder class for InitialConditions
    class Builder
    {
      public:
        static InitialConditions
        from_config(const simbi::ConfigDict& config_dict)
        {
            InitialConditions init;
            // Store the full config for reference
            init.config = config_dict;

            // Build basic properties
            build_basic_properties(init);

            // Build boundary conditions
            build_boundary_conditions(init);

            // Build bounds
            build_bounds(init);

            // Build mesh properties
            build_mesh_properties(init);

            // Build physics properties
            build_physics_properties(init);

            // Build source libraries
            build_source_libraries(init);

            // Build immersed bodies (if present)
            build_immersed_bodies(init);

            // Build source expressions
            build_source_expressions(init);

            return init;
        }

      private:
        // Helper methods to build different parts of InitialConditions
        static void build_basic_properties(InitialConditions& init)
        {
            // Time settings
            init.time = init.get<real>("tstart", 0.0);
            init.tend = init.get<real>("tend", 1.0);
            init.checkpoint_interval =
                init.get<real>("checkpoint_interval", 0.1);
            init.dlogt            = init.get<real>("dlogt", 0.0);
            init.checkpoint_index = init.get<luint>("checkpoint_index", 0);

            // Solver settings
            init.solver        = init.get<std::string>("solver", "hllc");
            init.spatial_order = init.get<std::string>("spatial_order", "plm");
            init.temporal_order =
                init.get<std::string>("temporal_order", "rk2");
            init.plm_theta       = init.get<real>("plm_theta", 1.5);
            init.quirk_smoothing = init.get<bool>("quirk_smoothing", false);
            init.regime          = init.get<std::string>("regime", "classical");
            init.cfl             = init.get<real>("cfl", 0.3);

            // I/O settings
            init.data_directory =
                init.get<std::string>("data_directory", "data/");
        }

        static void build_boundary_conditions(InitialConditions& init)
        {
            if (init.contains("boundary_conditions")) {
                if (init.at("boundary_conditions").is_array_of_strings()) {
                    init.boundary_conditions =
                        init.at("boundary_conditions")
                            .get<std::vector<std::string>>();
                }
                else if (init.at("boundary_conditions").is_string()) {
                    // Single boundary condition for all boundaries
                    std::string bc = init.at("boundary_conditions")
                                         .template get<std::string>();
                    // Create vector with appropriate size based on
                    // dimensionality
                    int ndims = 1;
                    if (init.contains("dimensionality")) {
                        ndims = init.get<int>("dimensionality", 1);
                    }
                    else if (init.contains("resolution")) {
                        auto res = init.at("resolution")
                                       .template get<std::vector<real>>();
                        ndims = res.size();
                    }
                    init.boundary_conditions.resize(2 * ndims, bc);
                }
            }
            else {
                // default to outflow for all boundaries
                int ndims = 1;
                if (init.contains("dimensionality")) {
                    ndims = init.get<int>("dimensionality", 1);
                }
                else if (init.contains("resolution")) {
                    auto res =
                        init.at("resolution").template get<std::vector<real>>();
                    ndims = res.size();
                }
                init.boundary_conditions.resize(2 * ndims, "outflow");
            }
        }

        static void build_bounds(InitialConditions& init)
        {
            if (init.contains("bounds") &&
                init.at("bounds").is_nested_array_of_floats()) {
                auto bounds =
                    init.at("bounds")
                        .template get<std::vector<std::vector<real>>>();
                if (bounds.size() >= 1 && bounds[0].size() >= 2) {
                    init.x1bounds = std::make_pair(bounds[0][0], bounds[0][1]);
                }
                if (bounds.size() >= 2 && bounds[1].size() >= 2) {
                    init.x2bounds = std::make_pair(bounds[1][0], bounds[1][1]);
                }
                if (bounds.size() >= 3 && bounds[2].size() >= 2) {
                    init.x3bounds = std::make_pair(bounds[2][0], bounds[2][1]);
                }
            }
            else {
                // get individual x1bounds, x2bounds, x3bounds
                if (init.contains("x1bounds") &&
                    init.at("x1bounds").is_pair()) {
                    init.x1bounds = init.at("x1bounds")
                                        .template get<std::pair<real, real>>();
                }
                if (init.contains("x2bounds") &&
                    init.at("x2bounds").is_pair()) {
                    init.x2bounds = init.at("x2bounds")
                                        .template get<std::pair<real, real>>();
                }
                if (init.contains("x3bounds") &&
                    init.at("x3bounds").is_pair()) {
                    init.x3bounds = init.at("x3bounds")
                                        .template get<std::pair<real, real>>();
                }
            }
        }

        static void build_mesh_properties(InitialConditions& init)
        {
            // Resolution
            if (init.contains("resolution") &&
                init.at("resolution").is_array()) {
                auto res =
                    init.at("resolution").template get<std::vector<int>>();
                if (res.size() >= 1) {
                    init.nx = res[0];
                }
                if (res.size() >= 2) {
                    init.ny = res[1];
                }
                if (res.size() >= 3) {
                    init.nz = res[2];
                }
            }
            else {
                // Try individual nx, ny, nz
                // python should take care of this, though
                // so maybe I'll remove this later
                // TODO: rethink this part
                init.nx = init.get<luint>("nx", 100);
                init.ny = init.get<luint>("ny", 1);
                init.nz = init.get<luint>("nz", 1);
            }

            // Coordinate system
            init.coord_system =
                init.get<std::string>("coord_system", "cartesian");

            // Grid spacing
            init.x1_spacing = init.get<std::string>("x1_spacing", "linear");
            init.x2_spacing = init.get<std::string>("x2_spacing", "linear");
            init.x3_spacing = init.get<std::string>("x3_spacing", "linear");

            // Mesh motion
            init.mesh_motion = init.get<bool>("mesh_motion", false);
            init.homologous  = init.get<bool>("is_homologous", false);
        }

        static void build_physics_properties(InitialConditions& init)
        {
            // Equation of state
            init.gamma      = init.get<real>("adiabatic_index", 5.0 / 3.0);
            init.isothermal = init.get<bool>("isothermal", false);

            real sound_speed = init.get<real>("sound_speed", 1.0);
            if (sound_speed != 0.0) {
                init.sound_speed_squared = sound_speed * sound_speed;
            }

            // Magnetic field (if present)
            if (init.contains("bfield") &&
                init.at("bfield").is_nested_array_of_floats()) {
                init.bfield =
                    init.at("bfield")
                        .template get<std::vector<std::vector<real>>>();
            }
        }

        static void build_source_libraries(InitialConditions& init)
        {
            init.hydro_source_lib =
                init.get<std::string>("hydro_source_lib", "");
            init.gravity_source_lib =
                init.get<std::string>("gravity_source_lib", "");
            init.boundary_source_lib =
                init.get<std::string>("boundary_source_lib", "");
        }

        static void build_source_expressions(InitialConditions& init)
        {
            // Load expressions for boundary conditions
            if (init.contains("bx1_inner_expressions")) {
                init.bx1_inner_expressions =
                    init.get_dict("bx1_inner_expressions");
            }
            if (init.contains("bx1_outer_expressions")) {
                init.bx1_outer_expressions =
                    init.get_dict("bx1_outer_expressions");
            }
            if (init.contains("bx2_inner_expressions")) {
                init.bx2_inner_expressions =
                    init.get_dict("bx2_inner_expressions");
            }
            if (init.contains("bx2_outer_expressions")) {
                init.bx2_outer_expressions =
                    init.get_dict("bx2_outer_expressions");
            }
            if (init.contains("bx3_inner_expressions")) {
                init.bx3_inner_expressions =
                    init.get_dict("bx3_inner_expressions");
            }
            if (init.contains("bx3_outer_expressions")) {
                init.bx3_outer_expressions =
                    init.get_dict("bx3_outer_expressions");
            }

            // Load expressions for hydro sources
            if (init.contains("hydro_source_expressions")) {
                init.hydro_source_expressions =
                    init.get_dict("hydro_source_expressions");
            }

            // Load expressions for gravity sources
            if (init.contains("gravity_source_expressions")) {
                init.gravity_source_expressions =
                    init.get_dict("gravity_source_expressions");
            }
        }

        static void build_immersed_bodies(InitialConditions& init)
        {
            // Clear existing bodies
            init.immersed_bodies.clear();

            // Check if bodies are provided in list format
            if (init.contains("bodies") && init.at("bodies").is_list()) {
                const auto& bodies_list =
                    init.at("bodies")
                        .template get<std::list<simbi::ConfigDict>>();

                for (const auto& body_dict : bodies_list) {
                    // extract body type from the dict!
                    if (!body_dict.contains("body_type") ||
                        !body_dict.at("body_type").is_string()) {
                        continue;   // Skip invalid body entries
                    }
                    const std::string type_str =
                        body_dict.at("body_type").template get<std::string>();
                    simbi::BodyType body_type = string_to_body_type(type_str);
                    // create property map
                    simbi::ConfigDict props;

                    // add the required properties
                    add_vector_property(body_dict, "position", props);
                    add_vector_property(body_dict, "velocity", props);
                    add_scalar_property(body_dict, "mass", props);
                    add_scalar_property(body_dict, "radius", props);

                    // add specifics/extra properties
                    // this is a dictionary of properties that are specific to
                    // the body type
                    if (body_dict.contains("specifics") &&
                        body_dict.at("specifics").is_dict()) {
                        const auto& specifics =
                            body_dict.at("specifics")
                                .template get<simbi::ConfigDict>();
                        for (const auto& [key, value] : specifics) {
                            add_property(key, value, props);
                        }
                    }

                    // Add other properties (not in specifics)
                    for (const auto& [key, value] : body_dict) {
                        if (key != "body_type" && key != "position" &&
                            key != "velocity" && key != "mass" &&
                            key != "radius" && key != "specifics") {
                            add_property(key, value, props);
                        }
                    }

                    // Add to immersed_bodies
                    init.immersed_bodies.push_back(
                        std::make_pair(body_type, props)
                    );
                }
            }
            // Check for old-style immersed_bodies dictionary
            else if (init.contains("immersed_bodies") &&
                     init.at("immersed_bodies").is_dict()) {
                const auto& bodies_dict =
                    init.at("immersed_bodies")
                        .template get<simbi::ConfigDict>();

                for (const auto& [body_name, body_config] : bodies_dict) {
                    if (!body_config.is_dict()) {
                        continue;
                    }

                    const auto& body_dict =
                        body_config.get<simbi::ConfigDict>();

                    // Extract body type
                    auto it_type = body_dict.find("body_type");
                    if (it_type == body_dict.end() ||
                        !it_type->second.is_string()) {
                        continue;   // Skip if no valid type
                    }

                    const std::string type_str =
                        it_type->second.get<std::string>();
                    simbi::BodyType body_type = string_to_body_type(type_str);

                    // Create property map
                    simbi::ConfigDict props;

                    // Add each property to the map
                    for (const auto& [prop_name, prop_value] : body_dict) {
                        if (prop_name == "body_type") {
                            continue;   // Already handled
                        }

                        add_property(prop_name, prop_value, props);
                    }

                    // Add to immersed_bodies
                    init.immersed_bodies.push_back(
                        std::make_pair(body_type, props)
                    );
                }
            }
        }

        // helpers
        static simbi::BodyType string_to_body_type(const std::string& type_str)
        {
            std::string type_upper = type_str;
            std::transform(
                type_upper.begin(),
                type_upper.end(),
                type_upper.begin(),
                ::toupper
            );

            if (type_upper == "GRAVITATIONAL") {
                return simbi::BodyType::GRAVITATIONAL;
            }
            else if (type_upper == "ELASTIC") {
                return simbi::BodyType::ELASTIC;
            }
            else if (type_upper == "RIGID") {
                return simbi::BodyType::RIGID;
            }
            else if (type_upper == "VISCOUS") {
                return simbi::BodyType::VISCOUS;
            }
            else if (type_upper == "SINK") {
                return simbi::BodyType::SINK;
            }
            else if (type_upper == "SOURCE") {
                return simbi::BodyType::SOURCE;
            }
            else if (type_upper == "GRAVITATIONAL_SINK") {
                return simbi::BodyType::GRAVITATIONAL_SINK;
            }
            else {
                // Default to gravitational
                return simbi::BodyType::GRAVITATIONAL;
            }
        }

        static void add_property(
            const std::string& name,
            const simbi::ConfigValue& value,
            simbi::ConfigDict& props
        )
        {
            if (value.is_real_number()) {
                props[name] = static_cast<real>(value.get<double>());
            }
            else if (value.is_array()) {
                props[name] = value.get<std::vector<double>>();
            }
            else if (value.is_bool()) {
                props[name] = value.get<bool>();
            }
        }

        static void add_vector_property(
            const simbi::ConfigDict& dict,
            const std::string& name,
            simbi::ConfigDict& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_array()) {
                props[name] = dict.at(name).template get<std::vector<double>>();
            }
        }

        static void add_scalar_property(
            const simbi::ConfigDict& dict,
            const std::string& name,
            simbi::ConfigDict& props
        )
        {
            if (dict.contains(name) && dict.at(name).is_real_number()) {
                props[name] =
                    static_cast<real>(dict.at(name).template get<double>());
            }
        }
    };

    // static factory method to create InitialConditions from ConfigDict
    static InitialConditions create(const simbi::ConfigDict& config)
    {
        return Builder::from_config(config);
    }
};

#endif
