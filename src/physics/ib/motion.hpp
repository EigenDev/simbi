// =============================================================================
// motion.hpp
//
// evolves the positions and velocities of immersed bodies.
// provides the `evolve_bodies` function, which is responsible for updating
// the kinematic state of bodies in the simulation, particularly for orbital
// systems like binaries where bodies follow a prescribed or computed path.
//
// usage:
//   // called inside the main time integration loop
//   body::evolve_bodies(sim);
// =============================================================================
#pragma once

#include "collection.hpp"
#include "containers/vector.hpp"

#include <cmath>
#include <cstddef>

namespace simbi::body {

    // compute advanced body variants without mutating the collection.
    // returns the rotated body variant array representing t^{n+1} state
    template <typename SimState>
    auto compute_advanced_bodies(const SimState& state)
        -> vector_t<body_variant_t<SimState::rank>, 2>
    {
        constexpr auto Rank   = SimState::rank;
        const auto&    bodies = state.bodies();

        // default: return current positions (no advancement)
        vector_t<body_variant_t<Rank>, 2> result;
        for (std::size_t ii = 0; ii < bodies.size(); ++ii) {
            result[ii] = bodies[ii];
        }

        if constexpr (Rank < 2) {
            return result;
        }
        else {
            if (bodies.name() != "binary_system") {
                return result;
            }
            if (bodies.reference_frame() == "corotating" ||
                bodies.reference_frame() == "stationary") {
                return result;
            }

            const auto binary_params = bodies.binary_params();
            const auto total_mass    = binary_params.total_mass;
            const auto a             = binary_params.semi_major;
            const auto omega         = std::sqrt(total_mass / (a * a * a));
            const auto dt            = state.metadata().global_dt;

            auto advanced =
                bodies |
                collection_ops::map_bodies([omega, dt](const auto& body) -> body_variant_t<Rank> {
                    auto pos = vecops::rotate(body.position, omega * dt);
                    auto vel = vecops::rotate(body.velocity, omega * dt);
                    return at_position(with_velocity(body, vel), pos);
                });

            for (std::size_t ii = 0; ii < bodies.size(); ++ii) {
                result[ii] = advanced[ii];
            }
            return result;
        }
    }

    // apply the pre-computed advance: sets bodies to t^{n+1} state
    template <typename SimState>
    void evolve_bodies(SimState& state)
    {
        if constexpr (SimState::rank < 2) {
            return;
        }
        else {
            auto& bodies = state.bodies();
            if (bodies.has_snapshot()) {
                // subcycle interpolation was active: finalize to t^{n+1}
                bodies.finalize_advance();
            }
            else {
                // no subcycling: advance directly (legacy path)
                constexpr auto Rank = SimState::rank;
                if (bodies.name() != "binary_system") {
                    return;
                }
                if (bodies.reference_frame() == "corotating" ||
                    bodies.reference_frame() == "stationary") {
                    return;
                }

                const auto binary_params = bodies.binary_params();
                const auto total_mass    = binary_params.total_mass;
                const auto a             = binary_params.semi_major;
                const auto omega         = std::sqrt(total_mass / (a * a * a));
                const auto dt            = state.metadata().global_dt;

                auto new_coll = make_body_collection<Rank>();
                if (bodies.binary_params_) {
                    new_coll = std::move(new_coll).with_system_config(bodies.binary_params());
                }
                if (!bodies.system_name_.empty()) {
                    new_coll = std::move(new_coll).with_name(bodies.system_name_);
                }

                auto updated =
                    bodies | collection_ops::map_bodies(
                                 [omega, dt](const auto& body) -> body_variant_t<Rank> {
                                     auto pos = vecops::rotate(body.position, omega * dt);
                                     auto vel = vecops::rotate(body.velocity, omega * dt);
                                     return at_position(with_velocity(body, vel), pos);
                                 }
                             );

                for (std::size_t ii = 0; ii < bodies.size(); ++ii) {
                    new_coll = std::move(new_coll).add(updated[ii]);
                }
                bodies = std::move(new_coll);
            }
        }
    }
} // namespace simbi::body
