// =============================================================================
// factory.hpp
//
// factory functions for creating immersed boundary bodies.
// provides functions to construct body objects and collections from
// configuration data (`config_dict_t` and blueprints). it handles dispatching
// to the correct body type based on specified capabilities and calculating
// orbital kinematics for binary systems.
//
// usage:
//   auto collection = create_body_collection<3>(bodies_bp, grav_sys_bp);
// =============================================================================
#pragma once

#include "body.hpp"
#include "build_config.hpp"
#include "collection.hpp"
#include "containers/vector.hpp"
#include "ecs/blueprints.hpp"
#include "utility/config_dict.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simbi::body::factory {
    using namespace simbi::config;

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
        template <std::uint64_t Rank>
        auto calculate_binary_positions(real semi_major, real mass_ratio)
            -> std::pair<vector_t<real, Rank>, vector_t<real, Rank>>
        {

            real a1 = semi_major / (real{1} + mass_ratio);
            real a2 = semi_major - a1;

            if constexpr (Rank == 2) {
                return {vector_t<real, Rank>{a1, real{0}}, vector_t<real, Rank>{-a2, real{0}}};
            }
            else if constexpr (Rank == 3) {
                return {
                    vector_t<real, Rank>{a1, real{0}, real{0}},
                    vector_t<real, Rank>{-a2, real{0}, real{0}}
                };
            }
            else {
                throw std::runtime_error("calculate_binary_positions only supports 2D and 3D");
            }
        }

        template <std::uint64_t Rank>
        auto calculate_binary_velocities(real semi_major, real total_mass, real mass_ratio)
            -> std::pair<vector_t<real, Rank>, vector_t<real, Rank>>
        {

            const real phi_dot = std::sqrt(total_mass / (semi_major * semi_major * semi_major));
            const real a1      = semi_major / (real{1} + mass_ratio);
            const real a2      = semi_major - a1;

            if constexpr (Rank == 2) {
                return {
                    vector_t<real, Rank>{real{0}, phi_dot * a2},
                    vector_t<real, Rank>{real{0}, -phi_dot * a1}
                };
            }
            else if constexpr (Rank == 3) {
                return {
                    vector_t<real, Rank>{real{0}, phi_dot * a2, real{0}},
                    vector_t<real, Rank>{real{0}, -phi_dot * a1, real{0}}
                };
            }
            else {
                throw std::runtime_error("calculate_binary_velocities only supports 2D and 3D");
            }
        }
    } // namespace detail

    template <std::uint64_t Rank>
    auto create_body_from_config(std::uint64_t idx, const config_dict_t& props)
        -> body_variant_t<Rank>
    {
        // extract basic properties
        auto position = try_read_vec<real, Rank>(props, "position").value();
        auto velocity = try_read_vec<real, Rank>(props, "velocity").value();
        auto mass     = try_read<real>(props, "mass").value();
        auto radius   = try_read<real>(props, "radius").value();
        bool two_way  = try_read<bool>(props, "two_way_coupling").unwrap_or(false);

        // determine body type and create appropriate variant
        auto body_type = detail::determine_body_type(props);

        if (body_type == "black_hole") {
            auto grav           = props.at("gravitational").template get<config_dict_t>();
            auto accr           = props.at("accretion").template get<config_dict_t>();
            auto softening      = try_read<real>(grav, "softening_length").value();
            auto sink_rate      = try_read<real>(accr, "sink_rate").value();
            auto sink_delta     = try_read<real>(accr, "sink_delta").value();
            auto accr_radius    = try_read<real>(accr, "accretion_radius").unwrap_or(radius);
            auto total_accreted = try_read<real>(accr, "total_accreted_mass").unwrap_or(real{0});

            return make_black_hole<Rank>(
                idx,
                position,
                velocity,
                mass,
                radius,
                softening,
                sink_rate,
                sink_delta,
                accr_radius,
                real{0}, // accretion rate
                total_accreted,
                two_way
            );
        }
        else if (body_type == "planet") {
            // auto grav = props.at("gravitational").template
            // get<config_dict_t>();
            auto rigid   = props.at("rigid").template get<config_dict_t>();
            auto inertia = try_read<real>(rigid, "inertia").value();
            bool no_slip = try_read<bool>(rigid, "apply_no_slip").unwrap_or(true);

            return make_planet<
                Rank>(idx, position, velocity, mass, radius, inertia, no_slip, two_way);
        }
        else if (body_type == "gravitational") {
            auto grav      = props.at("gravitational").template get<config_dict_t>();
            auto softening = try_read<real>(grav, "softening_length").value();

            return make_gravitational_body<
                Rank>(idx, position, velocity, mass, radius, softening, two_way);
        }
        else if (body_type == "rigid_sphere") {
            auto rigid   = props.at("rigid").template get<config_dict_t>();
            auto inertia = try_read<real>(rigid, "inertia").value();
            bool no_slip = try_read<bool>(rigid, "apply_no_slip").unwrap_or(true);

            return make_rigid_sphere<
                Rank>(idx, position, velocity, mass, radius, inertia, no_slip, two_way);
        }
        else {
            throw std::runtime_error("unknown body type: " + body_type);
        }
    }

    template <std::uint64_t Rank>
    auto create_collection_from_bodies(const std::vector<config_dict_t>& body_configs)
    {
        auto          collection = make_body_collection<Rank>();
        std::uint64_t idx        = 0;
        for (const auto& body_config : body_configs) {
            auto body  = create_body_from_config<Rank>(idx, body_config);
            collection = std::move(collection).add(body);
            ++idx;
        }

        return collection;
    }

    template <std::uint64_t Rank>
    auto create_binary_system_from_blueprint(const ecs::binary_system_blueprint_t& bp)
    {
        // create binary_parameters_t from blueprint
        binary_parameters_t binary_params{
            .total_mass        = bp.total_mass,
            .semi_major        = bp.semi_major,
            .eccentricity      = bp.eccentricity,
            .mass_ratio        = bp.mass_ratio,
            .orbital_period    = bp.orbital_period,
            .is_circular_orbit = bp.is_circular_orbit,
            .prescribed_motion = bp.prescribed_motion
        };

        // get components from blueprint
        const auto& components      = bp.components;
        const auto& reference_frame = bp.reference_frame;

        // calculate orbital positions and velocities
        auto [pos1, pos2] = detail::calculate_binary_positions<Rank>(
            binary_params.semi_major,
            binary_params.mass_ratio
        );
        auto [vel1, vel2] = [reference_frame, binary_params]() {
            if (reference_frame != "inertial") {
                return std::make_pair(vector_t<real, Rank>{}, vector_t<real, Rank>{});
            }
            return detail::calculate_binary_velocities<Rank>(
                binary_params.semi_major,
                binary_params.total_mass,
                binary_params.mass_ratio
            );
        }();

        // create components with calculated kinematics
        // first component
        auto config1       = components[0];
        auto pos_override1 = try_read_vec<real, Rank>(config1, "position");
        auto vel_override1 = try_read_vec<real, Rank>(config1, "velocity");

        if (!pos_override1.has_value() ||
            std::all_of(pos_override1->begin(), pos_override1->end(), [](real v) {
                return v == real{0};
            })) {
            config1["position"] = pos1;
        }
        if (!vel_override1.has_value() ||
            std::all_of(vel_override1->begin(), vel_override1->end(), [](real v) {
                return v == real{0};
            })) {
            config1["velocity"] = vel1;
        }

        // second component
        auto config2       = components[1];
        auto pos_override2 = try_read_vec<real, Rank>(config2, "position");
        auto vel_override2 = try_read_vec<real, Rank>(config2, "velocity");

        if (!pos_override2.has_value() ||
            std::all_of(pos_override2->begin(), pos_override2->end(), [](real v) {
                return v == real{0};
            })) {
            config2["position"] = pos2;
        }
        if (!vel_override2.has_value() ||
            std::all_of(vel_override2->begin(), vel_override2->end(), [](real v) {
                return v == real{0};
            })) {
            config2["velocity"] = vel2;
        }

        // create bodies and collection
        auto body1 = create_body_from_config<Rank>(0, config1);
        auto body2 = create_body_from_config<Rank>(1, config2);

        return make_body_collection<Rank>()
            .add(body1)
            .add(body2)
            .with_name("binary_system")
            .with_reference_frame(reference_frame)
            .with_system_config(binary_params);
    }

    template <std::uint64_t Rank>
    auto create_body_collection(
        const ecs::bodies_blueprint_t&                              bodies_bp,
        const std::optional<ecs::gravitational_system_blueprint_t>& gravitational_system_bp
    ) -> std::optional<body_collection_t<Rank>>
    {
        // prioritize gravitational system over individual bodies
        if (gravitational_system_bp.has_value()) {
            const auto& grav_sys = *gravitational_system_bp;

            if (grav_sys.system_type == "binary") {
                if (!grav_sys.binary.has_value()) {
                    throw std::runtime_error(
                        "binary system type specified but binary blueprint "
                        "missing"
                    );
                }
                return create_binary_system_from_blueprint<Rank>(*grav_sys.binary);
            }
            else {
                throw std::runtime_error(
                    "unsupported gravitational system type: " + grav_sys.system_type
                );
            }
        }

        // fall back to individual bodies
        if (!bodies_bp.body_configs.empty()) {
            return create_collection_from_bodies<Rank>(bodies_bp.body_configs);
        }

        return std::nullopt;
    }

} // namespace simbi::body::factory
