#ifndef BODY_MOTION_HPP
#define BODY_MOTION_HPP

#include "collection.hpp"
#include "containers/vector.hpp"

#include <cmath>
#include <cstddef>

namespace simbi::body {
    template <typename HydroState>
    void evolve_bodies(HydroState& state)
    {
        if constexpr (HydroState::dimensions < 2) {
            return;
        }
        else {
            constexpr auto Dims = HydroState::dimensions;
            if (!state.bodies || state.bodies->empty() ||
                state.bodies->name() != "binary_system") {
                return;
            }

            const auto binary_params = state.bodies->binary_params();
            const auto total_mass    = binary_params.total_mass;
            const auto a             = binary_params.semi_major;
            const auto omega         = std::sqrt(total_mass / (a * a * a));
            const auto dt            = state.metadata.dt;

            // the new collection to hold updated bodies
            auto new_coll = make_body_collection<Dims>();

            if (state.bodies->binary_params_) {
                new_coll = std::move(new_coll).with_system_config(
                    state.bodies->binary_params()
                );
            }

            if (!state.bodies->system_name_.empty()) {
                new_coll =
                    std::move(new_coll).with_name(state.bodies->system_name_);
            }

            auto updated_body_variants =
                *state.bodies |
                collection_ops::map_bodies(
                    [omega, dt](const auto& body) -> body_variant_t<Dims> {
                        auto pos = vecops::rotate(body.position, omega * dt);
                        auto vel = vecops::rotate(body.velocity, omega * dt);
                        return at_position(with_velocity(body, vel), pos);
                    }
                );

            for (std::size_t i = 0; i < state.bodies->size(); ++i) {
                new_coll = std::move(new_coll).add(updated_body_variants[i]);
            }

            state.bodies = std::move(new_coll);
        }
    }
}   // namespace simbi::body
#endif
