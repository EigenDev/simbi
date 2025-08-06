#ifndef BODY_FACTORY_HPP
#define BODY_FACTORY_HPP

#include "body.hpp"
#include "collection.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "utility/config_dict.hpp"
#include "utility/init_conditions.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace simbi::body::factory {
    using namespace simbi::config;

    // ========================================================================
    // capability detection from config
    // ========================================================================

    namespace detail {

        // check if config has gravitational properties
        bool has_gravitational_config(const config_dict_t& props);

        // check if config has accretion properties
        bool has_accretion_config(const config_dict_t& props);

        // check if config has rigid properties
        bool has_rigid_config(const config_dict_t& props);

        // determine body type from config
        std::string determine_body_type(const config_dict_t& props);

        // binary orbital mechanics
        template <std::uint64_t Dims>
        auto calculate_binary_positions(real semi_major, real mass_ratio)
            -> std::pair<vector_t<real, Dims>, vector_t<real, Dims>>
        {

            real a1 = semi_major / (real{1} + mass_ratio);
            real a2 = semi_major - a1;

            if constexpr (Dims == 2) {
                return {
                  vector_t<real, Dims>{a1, real{0}},
                  vector_t<real, Dims>{-a2, real{0}}
                };
            }
            else if constexpr (Dims == 3) {
                return {
                  vector_t<real, Dims>{a1, real{0}, real{0}},
                  vector_t<real, Dims>{-a2, real{0}, real{0}}
                };
            }
            else {
                throw std::runtime_error(
                    "calculate_binary_positions only supports 2D and 3D"
                );
            }
        }

        template <std::uint64_t Dims>
        auto calculate_binary_velocities(
            real semi_major,
            real total_mass,
            real mass_ratio
        ) -> std::pair<vector_t<real, Dims>, vector_t<real, Dims>>
        {

            const real phi_dot =
                std::sqrt(total_mass / (semi_major * semi_major * semi_major));
            const real a1 = semi_major / (real{1} + mass_ratio);
            const real a2 = semi_major - a1;

            if constexpr (Dims == 2) {
                return {
                  vector_t<real, Dims>{real{0}, phi_dot * a2},
                  vector_t<real, Dims>{real{0}, -phi_dot * a1}
                };
            }
            else if constexpr (Dims == 3) {
                return {
                  vector_t<real, Dims>{real{0}, phi_dot * a2, real{0}},
                  vector_t<real, Dims>{real{0}, -phi_dot * a1, real{0}}
                };
            }
            else {
                throw std::runtime_error(
                    "calculate_binary_velocities only supports 2D and 3D"
                );
            }
        }
    }   // namespace detail

    // ========================================================================
    // body creation functions - compile-time dispatch
    // ========================================================================

    template <std::uint64_t Dims>
    auto create_body_from_config(std::uint64_t idx, const config_dict_t& props)
        -> body_variant_t<Dims>
    {
        // extract basic properties
        auto position = try_read_vec<real, Dims>(props, "position").value();
        auto velocity = try_read_vec<real, Dims>(props, "velocity").value();
        auto mass     = try_read<real>(props, "mass").value();
        auto radius   = try_read<real>(props, "radius").value();
        bool two_way =
            try_read<bool>(props, "two_way_coupling").unwrap_or(false);

        // determine body type and create appropriate variant
        auto body_type = detail::determine_body_type(props);

        if (body_type == "black_hole") {
            auto softening = try_read<real>(props, "softening_length").value();
            auto accr_eff =
                try_read<real>(props, "accretion_efficiency").value();
            auto accr_radius =
                try_read<real>(props, "accretion_radius").unwrap_or(radius);
            auto total_accreted =
                try_read<real>(props, "total_accreted_mass").unwrap_or(real{0});

            return make_black_hole<Dims>(
                idx,
                position,
                velocity,
                mass,
                radius,
                softening,
                accr_eff,
                accr_radius,
                real{0},
                total_accreted,
                two_way
            );
        }
        else if (body_type == "planet") {
            // auto softening = try_read<real>(props,
            // "softening_length")
            //                      .unwrap_or(real{0});
            auto inertia = try_read<real>(props, "inertia").value();
            bool no_slip =
                try_read<bool>(props, "apply_no_slip").unwrap_or(true);

            return make_planet<Dims>(
                idx,
                position,
                velocity,
                mass,
                radius,
                inertia,
                no_slip,
                two_way
            );
        }
        else if (body_type == "gravitational") {
            auto softening = try_read<real>(props, "softening_length").value();

            return make_gravitational_body<Dims>(
                idx,
                position,
                velocity,
                mass,
                radius,
                softening,
                two_way
            );
        }
        else if (body_type == "rigid_sphere") {
            auto inertia = try_read<real>(props, "inertia").value();
            bool no_slip =
                try_read<bool>(props, "apply_no_slip").unwrap_or(true);

            return make_rigid_sphere<Dims>(
                idx,
                position,
                velocity,
                mass,
                radius,
                inertia,
                no_slip,
                two_way
            );
        }
        else {
            throw std::runtime_error("unknown body type: " + body_type);
        }
    }

    // ========================================================================
    // collection creation from config
    // ========================================================================

    template <std::uint64_t Dims>
    auto
    create_collection_from_individual_bodies(const initial_conditions_t& init)
    {
        auto collection   = make_body_collection<Dims>();
        std::uint64_t idx = 0;
        for (const auto& body_config : init.immersed_bodies) {
            auto body  = create_body_from_config<Dims>(idx, body_config);
            collection = std::move(collection).add(body);
            ++idx;
        }

        return collection;
    }

    template <std::uint64_t Dims>
    auto create_binary_system_from_config(const config_dict_t& sys_props)
    {
        using real = real;

        if (!sys_props.contains("binary_config")) {
            throw std::runtime_error("binary_config section missing");
        }

        const auto& binary_config =
            sys_props.at("binary_config").template get<config_dict_t>();
        auto binary_params = binary_parameters_t::from_config(binary_config);

        // get prescribed motion setting
        if (sys_props.contains("prescribed_motion")) {
            binary_params.prescribed_motion =
                sys_props.at("prescribed_motion").template get<bool>();
        }

        // get component configurations
        auto components = binary_config.at("components")
                              .template get<std::list<config_dict_t>>();
        if (components.size() != 2) {
            throw std::runtime_error(
                "binary system must have exactly 2 components"
            );
        }

        // calculate orbital positions and velocities
        auto [pos1, pos2] = detail::calculate_binary_positions<Dims>(
            binary_params.semi_major,
            binary_params.mass_ratio
        );
        auto [vel1, vel2] = detail::calculate_binary_velocities<Dims>(
            binary_params.semi_major,
            binary_params.total_mass,
            binary_params.mass_ratio
        );

        // create components with calculated kinematics
        auto comp_it = components.begin();

        // first component
        auto config1       = *comp_it++;
        auto pos_override1 = try_read_vec<real, Dims>(config1, "position");
        auto vel_override1 = try_read_vec<real, Dims>(config1, "velocity");

        if (!pos_override1.has_value() ||
            std::all_of(
                pos_override1->begin(),
                pos_override1->end(),
                [](real v) { return v == real{0}; }
            )) {
            config1["position"] = pos1;
        }
        if (!vel_override1.has_value() ||
            std::all_of(
                vel_override1->begin(),
                vel_override1->end(),
                [](real v) { return v == real{0}; }
            )) {
            config1["velocity"] = vel1;
        }

        // second component
        auto config2       = *comp_it;
        auto pos_override2 = try_read_vec<real, Dims>(config2, "position");
        auto vel_override2 = try_read_vec<real, Dims>(config2, "velocity");

        if (!pos_override2.has_value() ||
            std::all_of(
                pos_override2->begin(),
                pos_override2->end(),
                [](real v) { return v == real{0}; }
            )) {
            config2["position"] = pos2;
        }
        if (!vel_override2.has_value() ||
            std::all_of(
                vel_override2->begin(),
                vel_override2->end(),
                [](real v) { return v == real{0}; }
            )) {
            config2["velocity"] = vel2;
        }

        // create bodies and collection
        auto body1 = create_body_from_config<Dims>(0, config1);
        auto body2 = create_body_from_config<Dims>(1, config2);

        return make_body_collection<Dims>()
            .add(body1)
            .add(body2)
            .with_name("binary_system")
            .with_system_config(binary_params);
    }

    // ========================================================================
    // main factory function
    // ========================================================================

    template <std::uint64_t Dims>
    auto create_body_collection_from_init(const initial_conditions_t& init)
        -> std::optional<body_collection_t<Dims>>
    {
        // no bodies needed
        if (!init.contains("body_system") && init.immersed_bodies.empty()) {
            return std::nullopt;
        }

        // handle individual bodies
        if (!init.immersed_bodies.empty() && !init.contains("body_system")) {
            return create_collection_from_individual_bodies<Dims>(init);
        }

        // handle system definitions
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");

            if (sys_props.contains("system_type")) {
                auto system_type =
                    sys_props.at("system_type").template get<std::string>();

                if (system_type == "binary") {
                    return create_binary_system_from_config<Dims>(sys_props);
                }
                else {
                    throw std::runtime_error(
                        "unsupported system type: " + system_type
                    );
                }
            }
        }

        // fallback - create empty collection
        return make_body_collection<Dims>();
    }

    template <std::uint64_t Dims>
    auto create_default_collection_from_init(const initial_conditions_t& init)
    {
        return create_body_collection_from_init<Dims, 2>(init);
    }

}   // namespace simbi::body::factory

#endif
