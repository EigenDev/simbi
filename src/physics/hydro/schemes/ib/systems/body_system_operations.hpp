/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body_system_operations.hpp
 * @brief           Body system operations for the IB scheme
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
#ifndef BODY_SYSTEM_OPERATIONS_HPP
#define BODY_SYSTEM_OPERATIONS_HPP

#include "component_body_system.hpp"   // for ComponentBodySystem
#include "config.hpp"                  // for real, size_type, Dims
#include "physics/hydro/schemes/ib/delta/collector.hpp"
#include "physics/hydro/schemes/ib/policies/interaction_functions.hpp"   // for apply_gravitational_force, apply_accretion_effect
#include "physics/hydro/schemes/ib/processing/lazy.hpp"   // for LazyCapabilityView
#include "physics/hydro/types/context.hpp"                // for HydroContext

using namespace simbi::ibsystem::body_functions::gravitational;
using namespace simbi::ibsystem::body_functions::accretion;
using namespace simbi::ibsystem::body_functions::rigid;

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
    ComponentBodySystem<T, Dims>
    update_body_system(ComponentBodySystem<T, Dims>&& system, const T dt)
    {
        using system_t = ComponentBodySystem<T, Dims>;
        // update the body system
        if constexpr (Dims >= 2) {
            if (system.is_binary() && system.inertial()) {
                // use binary system update logic that returns new bodies
                auto updated_bodies =
                    body_functions::binary::calculate_binary_motion(system, dt);

                // apply updates to the system
                for (size_t ii = 0; ii < updated_bodies.size(); ii++) {
                    system = system_t::update_body_in(
                        std::move(system),
                        ii,
                        std::move(updated_bodies[ii])
                    );
                }
            }
        }

        return std::move(system);
    }

    template <typename T, size_type Dims, typename Primitive>
    DEV Primitive::counterpart_t apply_forces_to_fluid(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        const Primitive& prim,
        const auto& mesh_cell,
        std::tuple<size_type, size_type, size_type> coords,
        const HydroContext& context,
        const T dt,
        GridBodyDeltaCollector<T, Dims>& collector
    )
    {
        using conserved_t = typename Primitive::counterpart_t;
        conserved_t fluid_state{};

        {
            LazyCapabilityView<T, Dims> gravitational_bodies(
                system,
                BodyCapability::GRAVITATIONAL
            );

            for (const auto& [body_idx, body] : gravitational_bodies) {
                // apply gravitational forces
                auto [fluid_change, body_delta] = apply_gravitational_force(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );

                fluid_state += fluid_change;

                collector.record_delta(
                    coords,
                    body_idx,
                    body_delta.force_delta,
                    0.0,
                    0.0,
                    0.0
                );
            }
        }

        {
            LazyCapabilityView<T, Dims> accretor_bodies(
                system,
                BodyCapability::ACCRETION
            );

            for (const auto& [body_idx, body] : accretor_bodies) {
                // Apply accretion effects
                auto [fluid_change, body_delta] = apply_simple_accretion(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );

                fluid_state += fluid_change;

                collector.record_delta(
                    coords,
                    body_idx,
                    body_delta.force_delta,
                    body_delta.mass_delta,
                    body_delta.accreted_mass_delta,
                    body_delta.accretion_rate_delta
                );
            }
        }

        {
            LazyCapabilityView<T, Dims> rigid_bodies(
                system,
                BodyCapability::RIGID
            );

            for (const auto& [body_idx, body] : rigid_bodies) {
                // apply rigid body forces
                auto [fluid_change, body_delta] = apply_rigid_body_ibm(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );

                fluid_state += fluid_change;

                // collector
                //     .record_delta(coords, body_idx, body_delta.force_delta);
            }
        }
        return fluid_state;
    }
}   // namespace simbi::ibsystem::functions
#endif
