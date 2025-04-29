#ifndef BODY_SYSTEM_OPERATIONS_HPP
#define BODY_SYSTEM_OPERATIONS_HPP

#include "build_options.hpp"           // for real, size_type, Dims
#include "component_body_system.hpp"   // for ComponentBodySystem
#include "physics/hydro/schemes/ib/policies/body_delta.hpp"   // for BodyDelta
#include "physics/hydro/schemes/ib/policies/interaction_functions.hpp"   // for apply_gravitational_force, apply_accretion_effect
#include "physics/hydro/schemes/ib/processing/lazy.hpp"   // for LazyCapabilityView
#include "physics/hydro/types/context.hpp"                // for HydroContext

using namespace simbi::ibsystem::body_functions::gravitational;
using namespace simbi::ibsystem::body_functions::accretion;

namespace simbi::ibsystem {
    // small enough to be returned by value from GPU functions
    template <typename T, size_type Dims>
    struct BodyDeltaBuffer {
        // max 10 bodies x 2 interaction types
        static constexpr size_t MAX_DELTAS = 20;

        // actual deltas
        array_t<BodyDelta<T, Dims>, MAX_DELTAS> deltas;
        // number of valid deltas
        size_t count = 0;

        // add a delta to the buffer
        DUAL void collect(const BodyDelta<T, Dims>& delta)
        {
            if (count < MAX_DELTAS) {
                deltas[count++] = delta;
            }
        }
    };

    template <typename T, size_type Dims>
    class BodyDeltaCombiner : public Managed<global::managed_memory>
    {
      private:
        static constexpr size_t MAX_BODIES = 10;
        // one entry per body - indexed directly by body_idx
        array_t<BodyDelta<T, Dims>, MAX_BODIES> combined_deltas;
        array_t<bool, MAX_BODIES> has_delta;

      public:
        BodyDeltaCombiner()
        {
            // initialize has_delta to all false
            for (size_t ii = 0; ii < MAX_BODIES; ii++) {
                has_delta[ii] = false;
            }
        }

        // add all deltas from a buffer
        DEV void add_buffer(const BodyDeltaBuffer<T, Dims>& buffer)
        {
            for (size_t ii = 0; ii < buffer.count; ii++) {
                const auto& delta = buffer.deltas[ii];
                size_t body_idx   = delta.body_idx;

                if (body_idx >= MAX_BODIES) {
                    continue;   // safety check
                }

                if (has_delta[body_idx]) {
                    // combine with existing delta
                    combined_deltas[body_idx] =
                        combined_deltas[body_idx].combine(delta);
                }
                else {
                    // first delta for this body
                    combined_deltas[body_idx] = delta;
                    has_delta[body_idx]       = true;
                }
            }
        }

        // apply all deltas to system
        ComponentBodySystem<T, Dims>
        apply_to(ComponentBodySystem<T, Dims>&& system)
        {
            for (size_t body_idx = 0; body_idx < MAX_BODIES; body_idx++) {
                if (!has_delta[body_idx]) {
                    continue;
                }

                auto maybe_body = system.get_body(body_idx);
                if (!maybe_body.has_value()) {
                    continue;
                }

                auto body         = maybe_body.value();
                const auto& delta = combined_deltas[body_idx];

                // apply force delta
                auto new_force = body.force + delta.force_delta;
                body = std::move(body).with_force(std::move(new_force));

                // apply accreted mass
                if (delta.accreted_mass_delta > 0) {
                    body = std::move(body)
                               .add_accreted_mass(delta.accreted_mass_delta)
                               .with_accretion_rate(delta.accretion_rate_delta);
                }

                // update system with move semantics
                system = ComponentBodySystem<T, Dims>::update_body_in(
                    std::move(system),
                    body_idx,
                    std::move(body)
                );
            }

            // reset for next use
            for (size_t ii = 0; ii < MAX_BODIES; ii++) {
                has_delta[ii] = false;
            }

            return std::move(system);
        }
    };

}   // namespace simbi::ibsystem

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
        ComponentBodySystem<T, Dims>&& system,
        const T time,
        const T dt
    )
    {
        using system_t = ComponentBodySystem<T, Dims>;
        // update the body system
        if constexpr (Dims >= 2) {
            if (system.is_binary() && system.inertial()) {
                // use binary system update logic that returns new bodies
                auto updated_bodies =
                    body_functions::binary::calculate_binary_motion(
                        system,
                        time
                    );

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
    DEV std::pair<typename Primitive::counterpart_t, BodyDeltaBuffer<T, Dims>>
    apply_forces_to_fluid(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        const Primitive& prim,
        const auto& mesh_cell,
        std::tuple<size_type, size_type, size_type> coords,
        const HydroContext& context,
        const T dt
    )
    {
        using conserved_t = typename Primitive::counterpart_t;
        conserved_t fluid_state{};
        BodyDeltaBuffer<T, Dims> accumulator{};

        {
            LazyCapabilityView<T, Dims> gravitational_bodies(
                system,
                BodyCapability::GRAVITATIONAL
            );

            for (const auto& [body_idx, body] : gravitational_bodies) {
                // Apply gravitational forces
                auto [fluid_change, body_delta] = apply_gravitational_force(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );

                fluid_state += fluid_change;

                accumulator.collect(std::move(body_delta));
            }
        }

        {
            LazyCapabilityView<T, Dims> accretor_bodies(
                system,
                BodyCapability::ACCRETION
            );

            for (const auto& [body_idx, body] : accretor_bodies) {
                // Apply accretion effects
                auto [fluid_change, body_delta] = apply_accretion_effect(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );

                fluid_state += fluid_change;

                accumulator.collect(std::move(body_delta));
            }
        }
        return {std::move(fluid_state), std::move(accumulator)};
    }
}   // namespace simbi::ibsystem::functions
#endif
