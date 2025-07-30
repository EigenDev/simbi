/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body_factory.hpp
 * @brief           factory functions for creating body collections from init
 * @details         replaces the complex component_generator system with
 *                  simple functional approach
 *
 * @version         0.8.0
 * @date            2025-07-15
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
 */
#ifndef SIMBI_BODY_FACTORY_HPP
#define SIMBI_BODY_FACTORY_HPP

// #include "body_delta.hpp"
// #include "compute/functional/fp.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "utility/config_dict.hpp"
#include "utility/init_conditions.hpp"
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace simbi::ibsystem {

    // ========================================================================
    // core data structures
    // ========================================================================

    template <typename T, std::uint64_t Dims>
    struct body_state_t {
        vector_t<T, Dims> position;
        vector_t<T, Dims> velocity;
        vector_t<T, Dims> force;
        T mass;
        T radius;
        std::uint64_t id;

        // default constructor
        body_state_t()
            : position{}, velocity{}, force{}, mass{0}, radius{0}, id{0}
        {
        }

        body_state_t(
            const vector_t<T, Dims>& pos,
            const vector_t<T, Dims>& vel,
            T m,
            T r,
            std::uint64_t body_id
        )
            : position{pos},
              velocity{vel},
              force{},
              mass{m},
              radius{r},
              id{body_id}
        {
        }
    };

    template <typename T>
    struct body_properties_t {
        std::unordered_map<std::string, T> scalars;
        std::unordered_map<std::string, bool> flags;

        // helper to check if body has a capability
        bool has_capability(const std::string& cap) const
        {
            auto it = flags.find(cap);
            return it != flags.end() && it->second;
        }

        // helper to get scalar property with default
        T get_scalar(const std::string& key, T default_val = T{0}) const
        {
            auto it = scalars.find(key);
            return it != scalars.end() ? it->second : default_val;
        }
    };

    template <typename T>
    struct binary_system_config_t {
        T semi_major;
        T mass_ratio;
        T eccentricity;
        T orbital_period;
        bool prescribed_motion;
        bool is_circular_orbit;
        std::pair<std::uint64_t, std::uint64_t> body_indices;
    };

    template <typename T, std::uint64_t Dims>
    struct body_collection_t {
        vector_t<body_state_t<T, Dims>> states;
        vector_t<body_properties_t<T>> properties;
        std::string reference_frame{"inertial"};
        std::optional<binary_system_config_t<T>> binary_config;

        // helper functions
        std::size_t size() const { return states.size(); }
        bool empty() const { return states.empty(); }

        // get bodies with specific capability
        vector_t<std::uint64_t>
        bodies_with_capability(const std::string& cap) const
        {
            return fp::range(size()) | fp::filter([&](std::uint64_t i) {
                       return properties[i].has_capability(cap);
                   }) |
                   fp::collect<vector_t<std::uint64_t>>;
        }
    };

    // ========================================================================
    // factory functions
    // ========================================================================

    namespace detail {
        // simple id generator
        inline std::uint64_t next_body_id()
        {
            static std::uint64_t counter = 0;
            return ++counter;
        }

        // extract reference frame from init conditions
        inline std::string
        extract_reference_frame(const initial_conditions_t& init)
        {
            if (init.contains("body_system")) {
                const auto& sys_props = init.get_dict("body_system");
                if (sys_props.contains("reference_frame")) {
                    return sys_props.at("reference_frame")
                        .template get<std::string>();
                }
            }
            return "inertial";
        }

        // create body from individual body config dict
        template <typename T, std::uint64_t Dims>
        auto create_body_from_dict(const config_dict_t& props)
            -> std::pair<body_state_t<T, Dims>, body_properties_t<T>>
        {

            // required properties
            auto position = config::read_vec<T, Dims>(props, "position");
            auto velocity = config::read_vec<T, Dims>(props, "velocity");
            auto mass     = config::read<T>(props, "mass");
            auto radius   = config::read<T>(props, "radius");

            auto state = body_state_t<T, Dims>{
              position,
              velocity,
              mass,
              radius,
              next_body_id()
            };

            auto body_props = body_properties_t<T>{};

            // gravitational properties
            if (auto soft = config::try_read<T>(props, "softening_length")) {
                body_props.scalars["softening_length"] = *soft;
                body_props.flags["gravitational"]      = true;
            }

            // accretion properties
            if (auto eff = config::try_read<T>(props, "accretion_efficiency")) {
                body_props.scalars["accretion_efficiency"] = *eff;
                body_props.scalars["accretion_radius"] =
                    config::try_read<T>(props, "accretion_radius")
                        .value_or(radius);
                body_props.scalars["total_accreted_mass"] =
                    config::try_read<T>(props, "total_accreted_mass")
                        .value_or(T{0});
                body_props.flags["accretion"] = true;
                body_props.flags["is_accretor"] =
                    config::try_read<bool>(props, "is_an_accretor")
                        .value_or(false);
            }

            // rigid body properties
            if (auto inertia = config::try_read<T>(props, "inertia")) {
                body_props.scalars["inertia"] = *inertia;
                body_props.flags["apply_no_slip"] =
                    config::try_read<bool>(props, "apply_no_slip")
                        .value_or(false);
                body_props.flags["rigid"] = true;
            }

            // two-way coupling flag
            body_props.flags["two_way_coupling"] =
                config::try_read<bool>(props, "two_way_coupling")
                    .value_or(false);

            return {state, body_props};
        }

        // binary system helpers
        template <typename T, std::uint64_t Dims>
        auto calculate_binary_positions(T semi_major, T mass_ratio)
            -> std::pair<vector_t<T, Dims>, vector_t<T, Dims>>
        {
            T a1 = semi_major / (T{1} + mass_ratio);
            T a2 = semi_major - a1;

            if constexpr (Dims == 2) {
                return {
                  vector_t<T, Dims>{a1, T{0}},
                  vector_t<T, Dims>{-a2, T{0}}
                };
            }
            else {
                return {
                  vector_t<T, Dims>{a1, T{0}, T{0}},
                  vector_t<T, Dims>{-a2, T{0}, T{0}}
                };
            }
        }

        template <typename T, std::uint64_t Dims>
        auto
        calculate_binary_velocities(T semi_major, T total_mass, T mass_ratio)
            -> std::pair<vector_t<T, Dims>, vector_t<T, Dims>>
        {

            const T separation = semi_major;
            const T phi_dot =
                std::sqrt(total_mass / (separation * separation * separation));
            const T a1 = separation / (T{1} + mass_ratio);
            const T a2 = separation - a1;

            if constexpr (Dims == 2) {
                return {
                  vector_t<T, Dims>{T{0}, phi_dot * a2},
                  vector_t<T, Dims>{T{0}, -phi_dot * a1}
                };
            }
            else {
                return {
                  vector_t<T, Dims>{T{0}, phi_dot * a2, T{0}},
                  vector_t<T, Dims>{T{0}, -phi_dot * a1, T{0}}
                };
            }
        }

        template <typename T, std::uint64_t Dims>
        auto create_binary_component(
            const config_dict_t& component,
            const vector_t<T, Dims>& position,
            const vector_t<T, Dims>& velocity
        ) -> std::pair<body_state_t<T, Dims>, body_properties_t<T>>
        {

            auto mass   = config::read<T>(component, "mass");
            auto radius = config::read<T>(component, "radius");

            auto state = body_state_t<T, Dims>{
              position,
              velocity,
              mass,
              radius,
              next_body_id()
            };

            auto props = body_properties_t<T>{};

            // binary components are typically gravitational
            if (auto soft =
                    config::try_read<T>(component, "softening_length")) {
                props.scalars["softening_length"] = *soft;
                props.flags["gravitational"]      = true;
            }

            // check for accretion
            if (config::try_read<bool>(component, "is_an_accretor")
                    .value_or(false)) {
                props.scalars["accretion_efficiency"] =
                    config::read<T>(component, "accretion_efficiency");
                props.scalars["accretion_radius"] =
                    config::read<T>(component, "accretion_radius");
                props.scalars["total_accreted_mass"] =
                    config::try_read<T>(component, "total_accreted_mass")
                        .value_or(T{0});
                props.flags["accretion"]   = true;
                props.flags["is_accretor"] = true;
            }

            props.flags["two_way_coupling"] =
                config::try_read<bool>(component, "two_way_coupling")
                    .value_or(false);

            return {state, props};
        }

        template <typename T, std::uint64_t Dims>
        auto create_binary_system(
            body_collection_t<T, Dims> collection,
            const config_dict_t& sys_props
        ) -> body_collection_t<T, Dims>
        {

            const auto& binary_props =
                sys_props.at("binary_config").template get<config_dict_t>();

            // extract binary parameters
            auto semi_major =
                config::try_read<real>(binary_props, "semi_major");
            auto mass_ratio =
                config::try_read<real>(binary_props, "mass_ratio");
            auto eccentricity =
                config::try_read<real>(binary_props, "eccentricity");
            auto total_mass =
                config::try_read<real>(binary_props, "total_mass");
            auto prescribed_motion =
                config::try_read<bool>(sys_props, "prescribed_motion")
                    .value_or(true);

            // calculate orbital period
            auto orbital_period =
                T{2} * std::numbers::pi_v<T> *
                std::sqrt((semi_major * semi_major * semi_major) / total_mass);

            bool is_circular = (eccentricity < T{1e-10});

            // get binary components
            auto components = binary_props.at("components")
                                  .template get<std::list<config_dict_t>>();
            if (components.size() != 2) {
                throw std::runtime_error(
                    "binary system must have exactly 2 components"
                );
            }

            // calculate positions and velocities
            auto [pos1, pos2] =
                calculate_binary_positions<T, Dims>(semi_major, mass_ratio);
            auto [vel1, vel2] = calculate_binary_velocities<T, Dims>(
                semi_major,
                total_mass,
                mass_ratio
            );

            // create components
            auto comp_it = components.begin();
            auto [state1, props1] =
                create_binary_component<T, Dims>(*comp_it++, pos1, vel1);
            auto [state2, props2] =
                create_binary_component<T, Dims>(*comp_it, pos2, vel2);

            // store body indices for binary config
            auto body1_idx = collection.size();
            auto body2_idx = collection.size() + 1;

            // add to collection
            collection.states.push_back(state1);
            collection.states.push_back(state2);
            collection.properties.push_back(props1);
            collection.properties.push_back(props2);

            // set binary configuration
            collection.binary_config = binary_system_config_t<T>{
              .semi_major        = semi_major,
              .mass_ratio        = mass_ratio,
              .eccentricity      = eccentricity,
              .orbital_period    = orbital_period,
              .prescribed_motion = prescribed_motion,
              .is_circular_orbit = is_circular,
              .body_indices      = {body1_idx, body2_idx}
            };

            return collection;
        }
    }   // namespace detail

    // ========================================================================
    // main factory function
    // ========================================================================

    template <typename T, std::uint64_t Dims>
    auto create_body_collection_from_init(const initial_conditions_t& init)
        -> std::optional<body_collection_t<T, Dims>>
    {

        // no bodies needed
        if (!init.contains("body_system") && init.immersed_bodies.empty()) {
            return std::nullopt;
        }

        auto collection            = body_collection_t<T, Dims>{};
        collection.reference_frame = detail::extract_reference_frame(init);

        // handle individual bodies
        if (!init.immersed_bodies.empty()) {
            auto individual_bodies =
                init.immersed_bodies |
                fp::map(detail::create_body_from_dict<T, Dims>) | fp::collect;

            for (const auto& [state, props] : individual_bodies) {
                collection.states.push_back(state);
                collection.properties.push_back(props);
            }
        }

        // handle system definitions (binary, etc.)
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");

            if (sys_props.contains("system_type")) {
                auto system_type =
                    sys_props.at("system_type").template get<std::string>();

                if (system_type == "binary" &&
                    sys_props.contains("binary_config")) {
                    collection = detail::create_binary_system<T, Dims>(
                        std::move(collection),
                        sys_props
                    );
                }
            }
        }

        return collection;
    }

    // convenience alias for real precision
    template <std::uint64_t Dims>
    using body_collection = body_collection_t<real, Dims>;

}   // namespace simbi::ibsystem

#endif   // SIMBI_BODY_FACTORY_HPP
