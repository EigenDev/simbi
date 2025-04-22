#ifndef BODY_SYSTEM_OPERATIONS_HPP
#define BODY_SYSTEM_OPERATIONS_HPP

#include "build_options.hpp"
#include "component_body_system.hpp"
#include "core/types/utility/atomic_acc.hpp"
#include "physics/hydro/schemes/ib/policies/body_delta.hpp"
#include "physics/hydro/schemes/ib/policies/interaction_functions.hpp"
#include "physics/hydro/schemes/ib/processing/lazy.hpp"
#include "physics/hydro/types/context.hpp"

using namespace simbi::ibsystem::body_functions::gravitational;
using namespace simbi::ibsystem::body_functions::accretion;

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class BodyDeltaAccumulator : public Managed<global::managed_memory>
    {
      private:
        using atomic_acc_t = AtomicAccumulator<T, Dims>;
        atomic_acc_t force_x_deltas_;
        atomic_acc_t force_y_deltas_;
        atomic_acc_t force_z_deltas_;
        atomic_acc_t mass_deltas_;
        atomic_acc_t accreted_mass_deltas_;
        atomic_acc_t accretion_rate_deltas_;

      public:
        BodyDeltaAccumulator(size_t body_count)
            : force_x_deltas_(body_count),
              force_y_deltas_(body_count * (Dims > 1)),
              force_z_deltas_(body_count * (Dims > 2)),
              mass_deltas_(body_count),
              accreted_mass_deltas_(body_count),
              accretion_rate_deltas_(body_count)
        {
        }

        DUAL void accumulate(const BodyDelta<T, Dims>& delta)
        {
            const size_t idx = delta.body_idx;
            force_x_deltas_.add(idx, delta.force_delta[0]);
            if constexpr (Dims > 1) {
                force_y_deltas_.add(idx, delta.force_delta[1]);
            }
            if constexpr (Dims > 2) {
                force_z_deltas_.add(idx, delta.force_delta[2]);
            }
            mass_deltas_.add(idx, delta.mass_delta);
            accreted_mass_deltas_.add(idx, delta.accreted_mass_delta);
            accretion_rate_deltas_.add(idx, delta.accretion_rate_delta);
        }

        ComponentBodySystem<T, Dims>
        apply_deltas(const ComponentBodySystem<T, Dims>& system)
        {
            ComponentBodySystem<T, Dims> new_system = system;

            for (size_type ii = 0; ii < system.size(); ii++) {
                auto maybe_body = system.get_body(ii);
                if (!maybe_body.has_value()) {
                    continue;
                }

                auto body = maybe_body.value();

                // create new force vector
                spatial_vector_t<T, Dims> new_force = body.force;
                new_force[0] += force_x_deltas_[ii];
                if constexpr (Dims > 1) {
                    new_force[1] += force_y_deltas_[ii];
                }
                if constexpr (Dims > 2) {
                    new_force[2] += force_z_deltas_[ii];
                }

                body = body.with_force(new_force);

                // add accreted mass if relevant
                if (accreted_mass_deltas_[ii] > 0) {
                    body = body.add_accreted_mass(accreted_mass_deltas_[ii]);
                }

                // update the body in the system
                new_system = new_system.update_body(ii, body);
            }

            return new_system;
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
        const ComponentBodySystem<T, Dims>& system,
        const T time,
        const T dt
    )
    {
        ComponentBodySystem<T, Dims> updated_system = system;

        // update the body system
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

    template <typename T, size_type Dims, typename Primitive>
    DEV Primitive::counterpart_t apply_forces_to_fluid(
        const ibsystem::ComponentBodySystem<T, Dims>& system,
        BodyDeltaAccumulator<T, Dims>& accumulator,
        const Primitive& prim,
        const auto& mesh_cell,
        std::tuple<size_type, size_type, size_type> coords,
        const HydroContext& context,
        const T dt
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

                accumulator.accumulate(body_delta);
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

                accumulator.accumulate(body_delta);
            }
        }
        return fluid_state;
    }
}   // namespace simbi::ibsystem::functions
#endif
