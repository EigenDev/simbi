/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            binary.hpp
 * @brief
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-19
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
 * 2025-05-19      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */

#ifndef BINARY_HPP
#define BINARY_HPP

#include "config.hpp"
#include "core/containers/vector.hpp"
#include "physics/hydro/schemes/ib/systems/body.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"

namespace simbi::ibsystem::body_functions {
    namespace binary {
        // Binary system parameters
        template <size_type Dims, typename T>
        std::tuple<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_positions(T semi_major, T mass_ratio)
            requires(Dims >= 2)
        {
            T a1 = semi_major / (T(1) + mass_ratio);
            T a2 = semi_major - a1;

            if constexpr (Dims == 2) {
                spatial_vector_t<T, Dims> r1{a1, 0.0};
                spatial_vector_t<T, Dims> r2{-a2, 0.0};
                return {r1, r2};
            }
            else {
                spatial_vector_t<T, Dims> r1{a1, 0.0, 0.0};
                spatial_vector_t<T, Dims> r2{-a2, 0.0, 0.0};
                return {r1, r2};
            }
        }

        template <size_type Dims, typename T>
        std::tuple<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_velocities(
            T semi_major,
            T total_mass,
            T mass_ratio,
            bool circular_orbit
        )
            requires(Dims >= 2)
        {
            if (circular_orbit) {
                const T separation = semi_major;
                const T mu         = total_mass;
                const T phi_dot    = std::sqrt(mu / std::pow(separation, T(3)));
                T a1               = separation / (T(1) + mass_ratio);
                T a2               = separation - a1;

                if constexpr (Dims == 2) {
                    spatial_vector_t<T, Dims> v1{0.0, phi_dot * a2};
                    spatial_vector_t<T, Dims> v2{0.0, -phi_dot * a1};
                    return {v1, v2};
                }
                else {
                    spatial_vector_t<T, Dims> v1{0.0, phi_dot * a2, 0.0};
                    spatial_vector_t<T, Dims> v2{0.0, -phi_dot * a1, 0.0};
                    return {v1, v2};
                }
            }
            else {
                // TODO: implement eccentric orbits
                if constexpr (platform::is_gpu) {
                    printf("Non-circular orbits not yet implemented\n");
                    return {
                      spatial_vector_t<T, Dims>(),
                      spatial_vector_t<T, Dims>()
                    };
                }
                else {
                    throw std::runtime_error(
                        "Non-circular orbits not yet implemented"
                    );
                }
            }
        }

        template <typename T, size_type Dims>
        array_t<Body<T, Dims>, 2> calculate_binary_motion(
            const ibsystem::ComponentBodySystem<T, Dims>& system,
            T dt
        )
        {
            // get binary config
            auto config = system.template get_system_config<
                ibsystem::BinarySystemConfig<T>>();
            if (!config || !config->prescribed_motion) {
                const auto& bodies = system.bodies();
                return {bodies[0], bodies[1]};
            }

            // Extract parameters
            const auto [body1_idx, body2_idx] = config->body_indices;
            // const T orbital_period            = config->orbital_period;
            const T semi_major = config->semi_major;
            // const T ecc                       = config->eccentricity;

            // Calculate masses
            const auto& bodies = system.bodies();
            const auto body1   = bodies[body1_idx];
            const auto body2   = bodies[body2_idx];
            const T m1         = body1.mass;
            const T m2         = body2.mass;
            const T total_mass = m1 + m2;
            // const T mass_ratio = m2 / m1;

            // Load or calculate positions and velocities
            const auto& pos1 = body1.position;
            const auto& pos2 = body2.position;
            const auto& vel1 = body1.velocity;
            const auto& vel2 = body2.velocity;

            T phi_dot = std::sqrt(total_mass / std::pow(semi_major, T(3)));
            T phi     = phi_dot * dt;

            // Update positions and velocities
            return {
              body1.update_position(vecops::rotate(pos1, phi))
                  .update_velocity(vecops::rotate(vel1, phi)),
              body2.update_position(vecops::rotate(pos2, phi))
                  .update_velocity(vecops::rotate(vel2, phi))
            };
        }

        // Calculate orbital timestep
        template <typename T, size_type Dims>
        T calculate_binary_timestep(
            const ibsystem::ComponentBodySystem<T, Dims>& system,
            T cfl
        )
        {
            auto config = system.template get_system_config<
                ibsystem::BinarySystemConfig<T>>();
            if (!config) {
                return std::numeric_limits<T>::infinity();
            }

            // for binary systems, a sensible timestep is a fraction of the
            // orbital period
            return config->orbital_period * cfl / 100.0;
        }

    }   // namespace binary
}   // namespace simbi::ibsystem::body_functions

#endif
