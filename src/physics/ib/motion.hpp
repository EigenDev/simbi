#ifndef BODY_MOTION_HPP
#define BODY_MOTION_HPP

#include "collection.hpp"
#include "containers/vector.hpp"

#include <cmath>
#include <cstddef>

namespace simbi::body {
    template <typename SimState>
    void evolve_bodies(SimState& state)
    {
        if constexpr (SimState::rank < 2) {
            return;
        }
        else {
            constexpr auto Rank = SimState::rank;
            auto& bodies        = state.bodies();
            if (bodies.name() != "binary_system") {
                return;
            }

            if (bodies.reference_frame() == "corotating") {
                return;   // nothing to do, already in rotating frame
            }

            if (bodies.reference_frame() == "stationary") {
                return;   // nothing to do, static frame
            }

            const auto binary_params = bodies.binary_params();
            const auto total_mass    = binary_params.total_mass;
            const auto a             = binary_params.semi_major;
            const auto omega         = std::sqrt(total_mass / (a * a * a));
            const auto dt            = state.metadata().global_dt;

            // the new collection to hold updated bodies
            auto new_coll = make_body_collection<Rank>();

            if (bodies.binary_params_) {
                new_coll = std::move(new_coll).with_system_config(
                    bodies.binary_params()
                );
            }

            if (!bodies.system_name_.empty()) {
                new_coll = std::move(new_coll).with_name(bodies.system_name_);
            }

            auto updated_body_variants =
                bodies |
                collection_ops::map_bodies(
                    [omega, dt](const auto& body) -> body_variant_t<Rank> {
                        auto pos = vecops::rotate(body.position, omega * dt);
                        auto vel = vecops::rotate(body.velocity, omega * dt);
                        return at_position(with_velocity(body, vel), pos);
                    }
                );

            for (std::size_t ii = 0; ii < bodies.size(); ++ii) {
                new_coll = std::move(new_coll).add(updated_body_variants[ii]);
            }

            bodies = std::move(new_coll);
        }
    }
}   // namespace simbi::body
#endif
