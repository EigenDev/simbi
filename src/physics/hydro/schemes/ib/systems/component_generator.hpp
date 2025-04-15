#ifndef COMPONENT_GENERATOR_HPP
#define COMPONENT_GENERATOR_HPP

#include "build_options.hpp"
#include "core/types/utility/smart_ptr.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include "util/tools/helpers.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    util::smart_ptr<ComponentBodySystem<T, Dims>>
    create_body_system_from_config(
        const Mesh<Dims>& mesh,
        const InitialConditions& init,
        T gamma
    )
    {
        auto system = util::make_unique<ComponentBodySystem<T, Dims>>(mesh);
        // check if body system configuration exists
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");

            // process system configuration :^]
            if (sys_props.contains("system_type")) {
                const auto& system_type =
                    sys_props.at("system_type").get<std::string>();

                // Handle binary system configuration
                if (system_type == "binary" &&
                    sys_props.contains("binary_config")) {
                    if constexpr (Dims >= 2) {
                        const auto& binary_props =
                            sys_props.at("binary_config").get<ConfigDict>();

                        // Extract binary parameters
                        real total_mass =
                            binary_props.contains("total_mass")
                                ? binary_props.at("total_mass").get<real>()
                                : 1.0;

                        real semi_major =
                            binary_props.contains("semi_major")
                                ? binary_props.at("semi_major").get<real>()
                                : 1.0;

                        real eccentricity =
                            binary_props.contains("eccentricity")
                                ? binary_props.at("eccentricity").get<real>()
                                : 0.0;

                        real mass_ratio =
                            binary_props.contains("mass_ratio")
                                ? binary_props.at("mass_ratio").get<real>()
                                : 1.0;

                        bool prescribed_motion =
                            sys_props.contains("prescribed_motion")
                                ? sys_props.at("prescribed_motion").get<bool>()
                                : true;

                        // Get binary components
                        auto binary_components =
                            binary_props.at("components")
                                .get<std::list<ConfigDict>>();

                        if (binary_components.size() != 2) {
                            throw std::runtime_error(
                                "Binary system must have exactly 2 components"
                            );
                        }

                        // Calculate individual masses based on total mass and
                        // mass ratio
                        T m1 = total_mass / (1.0 + mass_ratio);
                        T m2 = total_mass - m1;

                        // Calculate positions based on semi-major axis
                        // For a circular orbit initially aligned with x-axis
                        T r1 = semi_major * mass_ratio / (1.0 + mass_ratio);
                        T r2 = semi_major - r1;

                        // Initial positions
                        spatial_vector_t<T, Dims> pos1, pos2;
                        pos1[0] = -r1;
                        pos2[0] = r2;

                        // For dimensions > 1, set y to 0
                        if constexpr (Dims > 1) {
                            pos1[1] = 0;
                            pos2[1] = 0;
                        }

                        // For dimensions > 2, set z to 0
                        if constexpr (Dims > 2) {
                            pos1[2] = 0;
                            pos2[2] = 0;
                        }

                        // Calculate orbital velocity - for circular orbit
                        T orbital_velocity = std::sqrt(total_mass / semi_major);

                        // Initial velocities (perpendicular to position)
                        spatial_vector_t<T, Dims> vel1, vel2;

                        if constexpr (Dims > 1) {
                            vel1[0] = 0;
                            vel1[1] = orbital_velocity * mass_ratio /
                                      (1.0 + mass_ratio);

                            vel2[0] = 0;
                            vel2[1] =
                                -orbital_velocity * 1.0 / (1.0 + mass_ratio);

                            if constexpr (Dims > 2) {
                                vel1[2] = 0;
                                vel2[2] = 0;
                            }
                        }
                        else {
                            // Can't have orbital motion in 1D
                            vel1[0] = 0;
                            vel2[0] = 0;
                        }

                        // Process the first component
                        auto& comp1 = binary_components.front();
                        T radius1   = comp1.at("radius").get<T>();

                        size_t body1_idx = system->add_body(
                            BodyType::GRAVITATIONAL,
                            pos1,
                            vel1,
                            m1,
                            radius1
                        );

                        // Add capabilities to first body
                        system->add_gravitational_capability(
                            body1_idx,
                            comp1.at("softening_length").get<T>(),
                            comp1.at("two_way_coupling").get<bool>()
                        );

                        // Add accretion if specified
                        if (comp1.at("is_an_accretor").get<bool>()) {
                            system->add_accretion_capability(
                                body1_idx,
                                comp1.at("accretion_efficiency").get<T>(),
                                comp1.at("accretion_radius").get<T>()
                            );
                        }

                        // Process the second component
                        auto& comp2 = binary_components.back();
                        T radius2   = comp2.at("radius").get<T>();

                        size_t body2_idx = system->add_body(
                            BodyType::GRAVITATIONAL,
                            pos2,
                            vel2,
                            m2,
                            radius2
                        );

                        // Add capabilities to second body
                        system->add_gravitational_capability(
                            body2_idx,
                            comp2.at("softening_length").get<T>(),
                            comp2.at("two_way_coupling").get<bool>()
                        );

                        // Add accretion if specified
                        if (comp2.at("is_an_accretor").get<bool>()) {
                            system->add_accretion_capability(
                                body2_idx,
                                comp2.at("accretion_efficiency").get<T>(),
                                comp2.at("accretion_radius").get<T>()
                            );
                        }

                        const auto orbital_period =
                            2.0 * M_PI *
                            std::sqrt(
                                semi_major * semi_major * semi_major /
                                total_mass
                            );

                        bool is_circular_orbit = goes_to_zero(eccentricity);
                        // Store orbital parameters for later use
                        system
                            ->template set_system_config<BinarySystemConfig<T>>(
                                semi_major,
                                mass_ratio,
                                eccentricity,
                                orbital_period,
                                is_circular_orbit,
                                prescribed_motion,
                                body1_idx,
                                body2_idx
                            );
                    }
                }
            }
        }
        // handle individually specified bodies
        else if (!init.immersed_bodies.empty()) {
            for (const auto& [body_type, props] : init.immersed_bodies) {
                // common properties
                const auto& position =
                    props.at("position").get<std::vector<real>>();
                const auto& velocity =
                    props.at("velocity").get<std::vector<real>>();
                const real mass   = props.at("mass").get<real>();
                const real radius = props.at("radius").get<real>();

                // position and velocity
                spatial_vector_t<T, Dims> pos_vec, vel_vec;
                for (size_type i = 0; i < Dims && i < position.size(); i++) {
                    pos_vec[i] = position[i];
                }

                for (size_type i = 0; i < Dims && i < velocity.size(); i++) {
                    vel_vec[i] = velocity[i];
                }

                // add basic body
                size_t body_idx =
                    system->add_body(body_type, pos_vec, vel_vec, mass, radius);

                // add gravitational capability if properties exist
                if (props.contains("softening_length") ||
                    props.contains("two_way_coupling")) {
                    T softening = props.contains("softening_length")
                                      ? props.at("softening_length").get<T>()
                                      : T(0.01);

                    bool two_way =
                        props.contains("two_way_coupling")
                            ? props.at("two_way_coupling").get<bool>()
                            : false;

                    system->add_gravitational_capability(
                        body_idx,
                        softening,
                        two_way
                    );
                }

                // add accretion capability if properties exist
                if (props.contains("accretion_efficiency") ||
                    props.contains("accretion_radius") ||
                    props.contains("is_an_accretor")) {

                    bool is_accretor =
                        props.contains("is_an_accretor")
                            ? props.at("is_an_accretor").get<bool>()
                            : false;

                    if (is_accretor ||
                        body_type == BodyType::GRAVITATIONAL_SINK) {
                        T efficiency =
                            props.contains("accretion_efficiency")
                                ? props.at("accretion_efficiency").get<T>()
                                : T(0.01);

                        T accr_radius =
                            props.contains("accretion_radius")
                                ? props.at("accretion_radius").get<T>()
                                : radius;

                        system->add_accretion_capability(
                            body_idx,
                            efficiency,
                            accr_radius
                        );
                    }
                }

                // TODO: add additional capabilities as needed
            }
        }

        return system;
    }

}   // namespace simbi::ibsystem

#endif
