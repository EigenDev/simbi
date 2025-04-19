#ifndef BODY_SYSTEM_OPERATIONS_HPP
#define BODY_SYSTEM_OPERATIONS_HPP

#include "build_options.hpp"
#include "component_body_system.hpp"
#include "core/types/containers/vector.hpp"
#include "physics/hydro/schemes/ib/bodies/policies/fluid_interaction_functions.hpp"
#include "physics/hydro/types/context.hpp"

using namespace simbi::body_functions::gravitational;
using namespace simbi::body_functions::accretion;

namespace simbi::ibsystem::functions {
    //---------------------------------------------------------------------------//
    //  CHARACTERISTIC TIME OF BODY SYSTEM
    // ---------------------------------------------------------------------------//
    template <typename T, size_type Dims>
    T get_system_timestep(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        T cfl
    )
    {
        // get the orbital timestep just for now
        T orbital_dt = std::numeric_limits<T>::infinity();

        // if there is only one body, we can't calculate a timestep
        // so we return infinity
        if (system.size() < 2) {
            return orbital_dt;
        }

        for (size_type idx = 0; idx < system.size(); ++idx) {
            // Get the body properties
            const auto& pos   = system.positions()[idx];
            const auto& vel   = system.velocities()[idx];
            const auto& force = system.forces()[idx];
            const auto& mass  = system.masses()[idx];
            // const auto& radius    = system.radii()[idx];
            const auto total_mass = system.total_mass();

            // Skip if any values are invalid
            if (mass <= 0) {
                continue;
            }

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

            // Approximate orbital period
            if (total_mass > 0 && r_mag > 0) {
                const auto period =
                    2.0 * M_PI * std::sqrt(std::pow(r_mag, 3) / (total_mass));
                orbital_dt = std::min(orbital_dt, period / 100.0);
            }
        }

        return orbital_dt * cfl;
    }

    //---------------------------------------------------------------------------//
    // ADVANCE BODY SYSTEM
    // ---------------------------------------------------------------------------//
    template <typename T, size_type Dims>
    void update_body_system(
        ibsystem::ComponentBodySystem<T, Dims>& system,
        const T time,
        const T dt
    )
    {
        // Update the body system
        // [default to binary system update for now]
        if constexpr (Dims >= 2) {
            if (system.is_binary()) {
                body_functions::binary::update_binary_prescribed_motion(
                    system,
                    time
                );
            }
        }
        // TODO: implement other body system types
    }

    //---------------------------------------------------------------------------//
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
        using conserved_t = Primitive::counterpart_t;
        conserved_t result;

        // Loop over all bodies and apply their respective forces
        for (size_type body_idx = 0; body_idx < system.size(); ++body_idx) {
            // Get the body properties
            const auto position = system.position_at(body_idx);
            // const auto& velocity = system.velocities()[body_idx];
            // const auto& force = system.forces()[body_idx];
            const auto mass = system.mass_at(body_idx);
            // const auto& radius = system.radius_at(body_idx);

            if (system.has_capability(
                    body_idx,
                    ibsystem::BodyCapability::GRAVITATIONAL
                )) {
                result += apply_gravitational_force(
                    position,
                    mass,
                    system.softening_length(body_idx),
                    system.two_way_coupling(body_idx),
                    prim,
                    cell,
                    context,
                    dt
                );
            }
        }

        // apply accretion if a body is capable of it
        for (size_type body_idx = 0; body_idx < system.size(); ++body_idx) {
            if (system.has_capability(
                    body_idx,
                    ibsystem::BodyCapability::ACCRETION
                )) {
                // Get the body properties
                const auto position = system.position_at(body_idx);
                const auto velocity = system.velocity_at(body_idx);
                // const auto& force    = system.forces()[body_idx];
                const auto mass = system.mass_at(body_idx);
                // const auto& radius   = system.radii()[body_idx];
                const auto accr_eff = system.accretion_efficiency(body_idx);
                const auto accretion_radius = system.accretion_radius(body_idx);
                // const auto& total_accreted_mass =
                //     system.total_accreted_mass(body_idx);
                //

                result += apply_accretion_effect(
                    position,
                    velocity,
                    mass,
                    // total_accreted_mass,
                    accr_eff,
                    accretion_radius,
                    // total_accreted_mass,
                    prim,
                    cell,
                    context,
                    dt
                );
            }
        }

        return result;
    }

}   // namespace simbi::ibsystem::functions
#endif
