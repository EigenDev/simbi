/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            component_generator.hpp
 * @brief           Component generator for the IB scheme
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-11
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
 * 2025-05-11      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */
#ifndef COMPONENT_GENERATOR_HPP
#define COMPONENT_GENERATOR_HPP

#include "binary.hpp"
#include "body.hpp"
#include "capability.hpp"
#include "component_body_system.hpp"
#include "compute/functional/fp.hpp"
#include "config.hpp"
#include "core/utility/config_dict.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "core/utility/smart_ptr.hpp"
#include "data/containers/vector.hpp"
#include "system/mesh/mesh_config.hpp"
#include "system_config.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <list>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

using namespace simbi::ibsystem::body_functions::binary;
using namespace simbi::config;

namespace simbi::ibsystem {

    template <typename T>
    struct BinaryParameters {
        T total_mass;
        T semi_major;
        T eccentricity;
        T mass_ratio;
        T orbital_period;
        bool is_circular_orbit;
    };

    template <typename T, std::uint64_t Dims>
    bool
    should_be_accretor(const config_dict_t& props, const Body<T, Dims>& body)
    {
        return fp::any_of(
            props,
            property_equals<bool>("is_an_accretor", true),
            [&body](const auto&) {
                return has_capability(
                    body.capabilities(),
                    BodyCapability::GRAVITATIONAL
                );
            },
            [](const auto& p) {
                return has_property_of_type<T>("accretion_radius")(p) &&
                       has_property_of_type<T>("accretion_efficiency")(p) &&
                       has_property_of_type<T>("total_accreted_mass")(p) &&
                       has_property_of_type<T>("softening_length")(p);
            }
        );
    }

    template <typename T = real>
    bool should_be_binary_system(const config_dict_t& props)
    {
        return fp::any_of(
            props,
            property_equals<std::string>("body_system", "binary"),
            [](const auto& p) {
                return has_property_of_type<T>("semi_major")(p) &&
                       has_property_of_type<T>("total_mass")(p) &&
                       has_property_of_type<T>("eccentricity")(p) &&
                       has_property_of_type<T>("mass_ratio")(p);
            }
        );
    }

    inline std::string extract_reference_frame(const initial_conditions_t& init)
    {
        std::string reference_frame = "inertial";
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");
            if (sys_props.contains("reference_frame")) {
                reference_frame =
                    sys_props.at("reference_frame").template get<std::string>();
            }
        }
        return reference_frame;
    }

    template <typename T = real>
    BinaryParameters<T>
    extract_binary_parameters(const config_dict_t& binary_props)
    {
        BinaryParameters<T> params;
        if (!should_be_binary_system(binary_props)) {
            throw std::runtime_error(
                "Binary parameters not found in the configuration"
            );
        }
        auto total_mass = try_read<real>(binary_props, "total_mass").value();
        auto semi_major = try_read<real>(binary_props, "semi_major").value();
        auto ecc        = try_read<real>(binary_props, "eccentricity").value();
        auto mass_ratio = try_read<real>(binary_props, "mass_ratio").value();
        auto t          = 2.0 * std::numbers::pi *
                 std::sqrt(std::pow(semi_major, 3.0) / total_mass);

        params.total_mass        = total_mass;
        params.semi_major        = semi_major;
        params.eccentricity      = ecc;
        params.mass_ratio        = mass_ratio;
        params.orbital_period    = t;
        params.is_circular_orbit = goes_to_zero(params.eccentricity);

        return params;
    }

    template <typename T, std::uint64_t Dims>
    Body<T, Dims>
    maybe_add_accretion(Body<T, Dims> body, const config_dict_t& props)
    {

        if (!should_be_accretor(props, body)) {
            return body;
        }

        const T efficiency =
            config::try_read<real>(props, "accretion_efficiency").value();
        const T accr_radius =
            config::try_read<real>(props, "accretion_radius").value();
        const T tot_accr_mass =
            config::try_read<real>(props, "total_accreted_mass").value();

        return body.with_accretion(efficiency, accr_radius, tot_accr_mass);
    }

    template <typename T, std::uint64_t Dims>
    Body<T, Dims>
    maybe_add_rigid(Body<T, Dims> body, const config_dict_t& props)
    {
        const auto inertia = config::try_read<real>(props, "inertia");
        if (!inertia.has_value()) {
            throw std::runtime_error("Rotation vector not found in properties");
        }
        const auto apply_no_slip =
            config::try_read<bool>(props, "apply_no_slip");
        if (!apply_no_slip.has_value()) {
            throw std::runtime_error(
                "apply_no_slip property not found in properties"
            );
        }

        return body.with_rigid(inertia.value(), apply_no_slip.value());
    }

    template <typename T, std::uint64_t Dims>
    Body<T, Dims> create_individual_body(const config_dict_t& props)
    {
        // basic body properties
        const auto caps = config::try_read<BodyCapability>(props, "capability");
        const auto pos_vec = config::try_read_vec<T, Dims>(props, "position");
        const auto vel_vec = config::try_read_vec<T, Dims>(props, "velocity");
        const auto mass    = config::try_read<real>(props, "mass");
        const auto radius  = config::try_read<real>(props, "radius");
        const auto two_way = config::try_read<bool>(props, "two_way_coupling");

        if (!pos_vec.has_value() || !vel_vec.has_value() || !mass.has_value() ||
            !radius.has_value()) {
            throw std::runtime_error(
                "Missing required properties for body creation"
            );
        }

        if (two_way) {
            throw std::runtime_error("Two-way coupling not yet implemented");
        }

        // create body
        Body<T, Dims> body(pos_vec, vel_vec, mass, radius, two_way);
        // add gravitational body capability if properties exist
        if (has_capability(caps.value(), BodyCapability::GRAVITATIONAL)) {
            const auto soft = config::try_read<real>(props, "softening_length");
            if (soft.has_value()) {
                body = body.with_gravitational(soft.value());
            }
        }
        if (has_capability(caps.value(), BodyCapability::ACCRETION)) {
            // add accretion capability if needed
            body = maybe_add_accretion(body, props);
        }

        if (has_capability(caps.value(), BodyCapability::RIGID)) {
            // add rigid body capability if needed
            body = maybe_add_rigid(body, props);
        }

        return body;
    }

    template <typename T, std::uint64_t Dims>
    void configure_from_individual_bodies(
        ComponentBodySystem<T, Dims>& system,
        const initial_conditions_t& init
    )
    {
        for (const auto& props : init.immersed_bodies) {
            system = system.add_body(create_individual_body<T, Dims>(props));
        }
    }

    template <typename T, std::int64_t Dims>
    vector_t<T, Dims> extract_position_vectors(
        const config_dict_t& component,
        const BinaryParameters<T>& params,
        std::int64_t component_index
    )
    {
        const auto pos = config::try_read_vec<T, Dims>(component, "position");
        if (!pos.has_value()) {
            throw std::runtime_error(
                "Position vector not found in component properties"
            );
        }
        const auto pos_vec = pos.value();
        // if pos is all zeros, calculate from binary parameters
        if (fp::all_of(pos_vec, [](real val) { return val == 0.0; })) {
            auto [pos1, pos2] =
                initial_positions<Dims>(params.semi_major, params.mass_ratio);

            return (component_index == 1) ? pos1 : pos2;
        }

        return pos_vec;
    }

    template <typename T, std::int64_t Dims>
    vector_t<T, Dims> extract_velocity_vectors(
        const config_dict_t& component,
        const BinaryParameters<T>& params,
        std::int64_t component_index
    )
    {
        const auto v = config::try_read_vec<T, Dims>(component, "velocity");
        if (!v.has_value()) {
            throw std::runtime_error(
                "Velocity vector not found in component properties"
            );
        }
        const auto vel_vec = v.value();
        // if vel is all zeros, calculate from binary parameters
        if (fp::all_of(vel_vec, [](real val) { return val == 0.0; })) {
            auto [vel1, vel2] = initial_velocities<Dims>(
                params.semi_major,
                params.total_mass,
                params.mass_ratio,
                params.is_circular_orbit
            );

            return (component_index == 1) ? vel1 : vel2;
        }

        return vel_vec;
    }

    template <typename T, std::uint64_t Dims>
    size_t add_binary_component(
        ComponentBodySystem<T, Dims>& system,
        const config_dict_t& component,
        const BinaryParameters<T>& params,
        std::int64_t component_index
    )
    {
        // extract position and velocity
        auto positions = extract_position_vectors<T, Dims>(
            component,
            params,
            component_index
        );

        auto velocities = extract_velocity_vectors<T, Dims>(
            component,
            params,
            component_index
        );

        // get basic body properties
        const T mass   = try_read<real>(component, "mass").value();
        const T radius = try_read<real>(component, "radius").value();
        const bool two_way =
            try_read<bool>(component, "two_way_coupling").value();

        if (mass < 0.0 || radius < 0.0) {
            throw std::runtime_error("Mass and radius must be positive values");
        }

        if (two_way) {
            throw std::runtime_error("Two-way coupling not yet implemented");
        }

        // Create body
        Body<T, Dims> body(positions, velocities, mass, radius, two_way);

        // add gravitational capability
        body = body.with_gravitational(
            try_read<real>(component, "softening_length").value()
        );

        // add accretion if specified
        bool is_acrretor = try_read<bool>(component, "is_an_accretor").value();
        if (is_acrretor) {
            body = body.with_accretion(
                try_read<real>(component, "accretion_efficiency").value(),
                try_read<real>(component, "accretion_radius").value(),
                try_read<real>(component, "total_accreted_mass").value()
            );
        }

        system = system.add_body(body);
        return system.size() - 1;
    }

    template <typename T, std::uint64_t Dims>
    void configure_binary_system_parameters(
        ComponentBodySystem<T, Dims>& system,
        const BinaryParameters<T>& params,
        bool prescribed_motion,
        size_t body1_idx,
        size_t body2_idx
    )
    {
        system = system.template with_system_config<BinarySystemConfig<T>>(
            params.semi_major,
            params.mass_ratio,
            params.eccentricity,
            params.orbital_period,
            params.is_circular_orbit,
            prescribed_motion,
            body1_idx,
            body2_idx
        );
    }

    template <typename T, std::uint64_t Dims>
    void configure_binary_system(
        ComponentBodySystem<T, Dims>& system,
        const config_dict_t& sys_props
    )
    {
        if constexpr (Dims >= 2) {
            const auto& binary_props =
                sys_props.at("binary_config").template get<config_dict_t>();

            // extract binary system parameters
            BinaryParameters<T> params =
                extract_binary_parameters(binary_props);

            bool prescribed_motion =
                sys_props.contains("prescribed_motion")
                    ? sys_props.at("prescribed_motion").template get<bool>()
                    : true;

            // get binary components
            auto binary_components =
                binary_props.at("components").get<std::list<config_dict_t>>();

            if (binary_components.size() != 2) {
                throw std::runtime_error(
                    "Binary system must have exactly 2 components"
                );
            }

            // process the components
            size_t body1_idx = add_binary_component(
                system,
                binary_components.front(),
                params,
                1
            );
            size_t body2_idx = add_binary_component(
                system,
                binary_components.back(),
                params,
                2
            );

            // configure the binary system
            configure_binary_system_parameters(
                system,
                params,
                prescribed_motion,
                body1_idx,
                body2_idx
            );
        }
    }

    template <typename T, std::uint64_t Dims>
    void configure_from_system_definition(
        ComponentBodySystem<T, Dims>& system,
        const initial_conditions_t& init
    )
    {
        const auto& sys_props = init.get_dict("body_system");

        // Validate system type
        if (sys_props.contains("system_type")) {
            const auto& system_type =
                sys_props.at("system_type").template get<std::string>();
            if (system_type != "binary") {
                throw std::runtime_error(
                    "Only binary systems are supported at this time"
                );
            }

            // Configure binary system
            if (system_type == "binary" &&
                sys_props.contains("binary_config")) {
                configure_binary_system(system, sys_props);
            }
        }
    }

    template <typename T, std::uint64_t Dims>
    util::smart_ptr<ComponentBodySystem<T, Dims>>
    create_body_system_from_config(
        const mesh_config_t<Dims>& mesh,
        const initial_conditions_t& init
    )
    {
        if (!init.contains("body_system") && init.immersed_bodies.empty()) {
            return nullptr;
        }

        std::string reference_frame = extract_reference_frame(init);

        // create initial empty system
        auto system = util::make_unique<ComponentBodySystem<T, Dims>>(
            mesh,
            std::move(reference_frame)
        );

        // either configure from body_system or from individual bodies
        if (init.contains("body_system")) {
            configure_from_system_definition(*system, init);
        }
        else if (!init.immersed_bodies.empty()) {
            configure_from_individual_bodies(*system, init);
        }

        return system;
    }

}   // namespace simbi::ibsystem

#endif
