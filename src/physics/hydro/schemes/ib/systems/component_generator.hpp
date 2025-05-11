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

#include "build_options.hpp"
#include "core/types/utility/smart_ptr.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"
#include "util/tools/helpers.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    util::smart_ptr<ComponentBodySystem<T, Dims>>
    create_body_system_from_config(
        const Mesh<Dims>& mesh,
        const InitialConditions& init
    )
    {
        if (!init.contains("body_system")) {
            if (init.immersed_bodies.empty()) {
                return nullptr;
            }
        }

        std::string reference_frame = "inertial";
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");
            if (sys_props.contains("system_type")) {
                const auto& system_type =
                    sys_props.at("system_type").template get<std::string>();
                if (system_type != "binary") {
                    throw std::runtime_error(
                        "Only binary systems are supported at this time"
                    );
                }
            }
            if (sys_props.contains("reference_frame")) {
                reference_frame =
                    sys_props.at("reference_frame").template get<std::string>();
            }
        }

        // create initial empty system
        auto system = util::make_unique<ComponentBodySystem<T, Dims>>(
            mesh,
            std::move(reference_frame)
        );

        // check if body system configuration exists
        if (init.contains("body_system")) {
            const auto& sys_props = init.get_dict("body_system");

            // process system configuration
            if (sys_props.contains("system_type")) {
                const auto& system_type =
                    sys_props.at("system_type").template get<std::string>();

                // handle binary system configuration
                if (system_type == "binary" &&
                    sys_props.contains("binary_config")) {
                    if constexpr (Dims >= 2) {
                        const auto& binary_props =
                            sys_props.at("binary_config")
                                .template get<ConfigDict>();

                        // Extract binary parameters
                        real total_mass = binary_props.contains("total_mass")
                                              ? binary_props.at("total_mass")
                                                    .template get<real>()
                                              : 1.0;

                        real semi_major = binary_props.contains("semi_major")
                                              ? binary_props.at("semi_major")
                                                    .template get<real>()
                                              : 1.0;

                        real eccentricity =
                            binary_props.contains("eccentricity")
                                ? binary_props.at("eccentricity")
                                      .template get<real>()
                                : 0.0;

                        real mass_ratio = binary_props.contains("mass_ratio")
                                              ? binary_props.at("mass_ratio")
                                                    .template get<real>()
                                              : 1.0;

                        bool prescribed_motion =
                            sys_props.contains("prescribed_motion")
                                ? sys_props.at("prescribed_motion")
                                      .template get<bool>()
                                : true;

                        // get binary components
                        auto binary_components =
                            binary_props.at("components")
                                .get<std::list<ConfigDict>>();

                        if (binary_components.size() != 2) {
                            throw std::runtime_error(
                                "Binary system must have exactly 2 components"
                            );
                        }

                        // calculate individual masses based on total mass and
                        // mass ratio
                        T m1 = total_mass / (1.0 + mass_ratio);
                        T m2 = total_mass - m1;

                        // calculate positions based on semi-major axis
                        T r1 = semi_major * mass_ratio / (1.0 + mass_ratio);
                        T r2 = semi_major - r1;

                        // initial positions
                        spatial_vector_t<T, Dims> pos1, pos2;
                        pos1[0] = -r1;
                        pos2[0] = r2;

                        // for dimensions > 1, set y to 0
                        if constexpr (Dims > 1) {
                            pos1[1] = 0;
                            pos2[1] = 0;
                        }

                        // for dimensions > 2, set z to 0
                        if constexpr (Dims > 2) {
                            pos1[2] = 0;
                            pos2[2] = 0;
                        }

                        // calculate orbital velocity - for circular orbit
                        T orbital_velocity = std::sqrt(total_mass / semi_major);

                        // initial velocities (perpendicular to position)
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
                            // cant't have orbital motion in 1D
                            vel1[0] = 0;
                            vel2[0] = 0;
                        }

                        // process the first component
                        auto& comp1 = binary_components.front();
                        T radius1   = comp1.at("radius").template get<T>();

                        // create body with functional approach
                        Body<T, Dims> body1(
                            BodyType::GRAVITATIONAL,
                            pos1,
                            vel1,
                            m1,
                            radius1
                        );

                        // add capabilities using functional approach
                        body1 = body1.with_gravitational(
                            comp1.at("softening_length").template get<T>(),
                            comp1.at("two_way_coupling").template get<bool>()
                        );

                        // add accretion if specified
                        if (comp1.at("is_an_accretor").template get<bool>()) {
                            body1 = body1.with_accretion(
                                comp1.at("accretion_efficiency")
                                    .template get<T>(),
                                comp1.at("accretion_radius").template get<T>()
                            );
                        }

                        // add body to system
                        *system          = system->add_body(body1);
                        size_t body1_idx = system->size() - 1;

                        // process the second component
                        auto& comp2 = binary_components.back();
                        T radius2   = comp2.at("radius").template get<T>();

                        // create body with functional approach
                        Body<T, Dims> body2(
                            BodyType::GRAVITATIONAL,
                            pos2,
                            vel2,
                            m2,
                            radius2
                        );

                        // add capabilities using functional approach
                        body2 = body2.with_gravitational(
                            comp2.at("softening_length").template get<T>(),
                            comp2.at("two_way_coupling").template get<bool>()
                        );

                        // add accretion if specified
                        if (comp2.at("is_an_accretor").template get<bool>()) {
                            body2 = body2.with_accretion(
                                comp2.at("accretion_efficiency")
                                    .template get<T>(),
                                comp2.at("accretion_radius").template get<T>()
                            );
                        }

                        // add body to system
                        *system          = system->add_body(body2);
                        size_t body2_idx = system->size() - 1;

                        const auto orbital_period =
                            2.0 * M_PI *
                            std::sqrt(
                                semi_major * semi_major * semi_major /
                                total_mass
                            );

                        bool is_circular_orbit = goes_to_zero(eccentricity);

                        // set system config using functional approach
                        *system = system->template with_system_config<
                            BinarySystemConfig<T>>(
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
                    props.at("position").template get<std::vector<real>>();
                const auto& velocity =
                    props.at("velocity").template get<std::vector<real>>();
                const real mass   = props.at("mass").template get<real>();
                const real radius = props.at("radius").template get<real>();

                // position and velocity
                spatial_vector_t<T, Dims> pos_vec, vel_vec;
                for (size_type i = 0; i < Dims && i < position.size(); i++) {
                    pos_vec[i] = position[i];
                }

                for (size_type i = 0; i < Dims && i < velocity.size(); i++) {
                    vel_vec[i] = velocity[i];
                }

                // create body with functional approach
                Body<T, Dims> body(body_type, pos_vec, vel_vec, mass, radius);

                // add gravitational capability if properties exist
                if (props.contains("softening_length") ||
                    props.contains("two_way_coupling")) {
                    T softening =
                        props.contains("softening_length")
                            ? props.at("softening_length").template get<T>()
                            : T(0.01);

                    bool two_way =
                        props.contains("two_way_coupling")
                            ? props.at("two_way_coupling").template get<bool>()
                            : false;

                    body = body.with_gravitational(softening, two_way);
                }

                // add accretion capability if properties exist
                if (props.contains("accretion_efficiency") ||
                    props.contains("accretion_radius") ||
                    props.contains("is_an_accretor")) {

                    bool is_accretor =
                        props.contains("is_an_accretor")
                            ? props.at("is_an_accretor").template get<bool>()
                            : false;

                    if (is_accretor ||
                        body_type == BodyType::GRAVITATIONAL_SINK) {
                        T efficiency = props.contains("accretion_efficiency")
                                           ? props.at("accretion_efficiency")
                                                 .template get<T>()
                                           : T(0.01);

                        T accr_radius =
                            props.contains("accretion_radius")
                                ? props.at("accretion_radius").template get<T>()
                                : radius;

                        body = body.with_accretion(efficiency, accr_radius);
                    }
                }

                // add body to system using functional approach
                *system = system->add_body(body);
            }
        }

        return system;
    }

}   // namespace simbi::ibsystem

#endif
