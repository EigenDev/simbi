#ifndef BODY_SYSTEM_OPERATIONS_HPP
#define BODY_SYSTEM_OPERATIONS_HPP

#include "build_options.hpp"
#include "component_body_system.hpp"
#include "physics/hydro/schemes/ib/bodies/policies/fluid_interaction_functions.hpp"
#include "physics/hydro/types/context.hpp"

using namespace simbi::body_functions::gravitational;
using namespace simbi::body_functions::accretion;

namespace simbi::ibsystem::functions {
    template <typename T, size_type Dims>
    T get_system_timestep(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        T cfl
    )
    {
        T orbital_dt = std::numeric_limits<T>::infinity();

        if (system.size() < 2) {
            return orbital_dt;
        }

        for (size_type idx = 0; idx < system.size(); ++idx) {
            auto maybe_body = system.get_body(idx);
            if (!maybe_body.has_value()) {
                continue;
            }

            const auto& body = maybe_body.value();

            // Skip if any values are invalid
            if (body.mass <= 0) {
                continue;
            }

            const auto& pos       = body.position;
            const auto& vel       = body.velocity;
            const auto& force     = body.force;
            const auto& mass      = body.mass;
            const auto total_mass = system.total_mass();

            const auto accel = force / mass;

            // Calculate relevant timescales
            const auto r_mag = pos.norm();
            const auto v_mag = vel.norm();
            const auto a_mag = accel.norm();

            if (v_mag > 0) {
                orbital_dt = std::min(orbital_dt, r_mag / v_mag);
            }

            if (a_mag > 0) {
                orbital_dt = std::min(orbital_dt, std::sqrt(r_mag / a_mag));
                orbital_dt = std::min(orbital_dt, v_mag / a_mag);
            }

            if (total_mass > 0 && r_mag > 0) {
                const auto period =
                    2.0 * M_PI * std::sqrt(std::pow(r_mag, 3) / (total_mass));
                orbital_dt = std::min(orbital_dt, period / 100.0);
            }
        }

        return orbital_dt * cfl;
    }

    template <typename T, size_type Dims>
    ComponentBodySystem<T, Dims> update_body_system(
        const ComponentBodySystem<T, Dims>& system,
        const T time,
        const T dt
    )
    {
        ComponentBodySystem<T, Dims> updated_system = system;

        // Update the body system
        if constexpr (Dims >= 2) {
            if (system.is_binary()) {
                // use binary system update logic that returns new bodies
                auto updated_bodies =
                    body_functions::binary::calculate_binary_motion(
                        system,
                        time
                    );

                // apply updates to the system
                for (size_t i = 0; i < updated_bodies.size(); i++) {
                    updated_system =
                        updated_system.update_body(i, updated_bodies[i]);
                }
            }
        }

        return updated_system;
    }

    // Apply forces with pure functional approach
    template <typename T, size_type Dims, typename Primitive>
    DEV Primitive::counterpart_t apply_forces_to_fluid(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        const Primitive& prim,
        const auto& cell,
        std::tuple<size_type, size_type, size_type> coords,
        const HydroContext& context,
        const T dt
    )
    {
        using conserved_t = typename Primitive::counterpart_t;
        conserved_t result;

        // apply gravitational forces from all bodies
        for (size_type body_idx = 0; body_idx < system.size(); ++body_idx) {
            const auto maybe_body = system.get_body(body_idx);
            if (!maybe_body.has_value()) {
                continue;
            }

            const auto& body = maybe_body.value();
            printf("body mass: %f\n", body.mass);
            printf("body pos: %f\n", body.position[0]);

            // if (body.has_capability(ibsystem::BodyCapability::GRAVITATIONAL))
            // {
            //     result += apply_gravitational_force(
            //         body.position,
            //         body.mass,
            //         body.softening_length(),
            //         body.two_way_coupling(),
            //         prim,
            //         cell,
            //         context,
            //         dt
            //     );
            // }
        }

        // apply accretion effects from all bodies
        // for (size_type body_idx = 0; body_idx < system.size(); ++body_idx) {
        //     auto maybe_body = system.get_body(body_idx);
        //     if (!maybe_body.has_value()) {
        //         continue;
        //     }

        //     const auto& body = maybe_body.value();

        //     if (body.has_capability(ibsystem::BodyCapability::ACCRETION)) {
        //         result += apply_accretion_effect(
        //             body.position,
        //             body.velocity,
        //             body.mass,
        //             body.accretion_efficiency(),
        //             body.accretion_radius(),
        //             prim,
        //             cell,
        //             context,
        //             dt
        //         );
        //     }
        // }

        return result;
    }
}   // namespace simbi::ibsystem::functions
#endif
