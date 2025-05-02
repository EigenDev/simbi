#ifndef COLLECTOR_HPP
#define COLLECTOR_HPP

#include "body_delta.hpp"
#include "build_options.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/utility/managed.hpp"   // for Managed
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"   // for ComponentBodySystem
#include "util/parallel/exec_policy.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class GridBodyDeltaCollector : public Managed<global::managed_memory>
    {
      private:
        // one delta per cell in the grid
        ndarray<BodyDelta<T, Dims>, Dims> cell_deltas;

        // reduced deltas per body (final output)
        ndarray<BodyDelta<T, Dims>, 1> body_deltas;

        // tracks which bodies were modified
        ndarray<unsigned int, 1> body_modified;

        size_type max_bodies_;

        ExecutionPolicy<> body_policy_{};
        ExecutionPolicy<> cell_policy_{};

      public:
        GridBodyDeltaCollector(
            const collapsable<Dims>& grid_shape,
            size_type max_bodies
        )
            : max_bodies_(max_bodies)
        {
            // init with one delta per cell
            size_type size                      = 1;
            array_t<size_type, 3> cpolicy_shape = {1, 1, 1};
            for (size_type dim = 0; dim < Dims; ++dim) {
                size *= grid_shape[dim];
                cpolicy_shape[dim] = grid_shape[dim];
            }
            cell_deltas.resize(size);
            cell_deltas.reshape(grid_shape);

            cell_policy_ = ExecutionPolicy<>(cpolicy_shape, {1, 1, 1});
            body_policy_ = ExecutionPolicy<>({max_bodies, 1, 1}, {1, 1, 1});

            // prep body deltas array (one per possible body)
            body_deltas.resize(max_bodies_);

            // init body modified flags (0 = not modified)
            body_modified.resize(max_bodies_);
            body_modified.fill(0);

            // ensure device has initial state
            cell_deltas.sync_to_device();
            body_deltas.sync_to_device();
            body_modified.sync_to_device();
        }

        // thread-safe recording - each cell writes to its own slot
        DUAL void record_delta(
            std::tuple<size_type, size_type, size_type> cell_idx,
            size_t body_idx,
            const spatial_vector_t<T, Dims>& force,
            T mass           = 0,
            T accreted_mass  = 0,
            T accretion_rate = 0
        )
        {
            if (body_idx >= max_bodies_) {
                return;
            }

            // access the delta for this specific cell
            auto& delta = cell_deltas.at(
                std::get<0>(cell_idx),
                std::get<1>(cell_idx),
                std::get<2>(cell_idx)
            );

            // // set the delta values
            delta.body_idx             = body_idx;
            delta.force_delta          = force;
            delta.mass_delta           = mass;
            delta.accreted_mass_delta  = accreted_mass;
            delta.accretion_rate_delta = accretion_rate;

            // mark that this body was modified
            // body_modified[body_idx] = 1;
        }

        // parallel reduction of deltas on device
        void reduce_deltas()
        {
            // reset the body deltas array
            body_deltas.transform(
                [] DEV(auto&) {
                    return BodyDelta<T, Dims>{
                      std::numeric_limits<size_t>::max(),   // invalid body_idx
                      spatial_vector_t<T, Dims>{},          // zero force
                      0,
                      0,
                      0   // zero mass changes
                    };
                },
                body_policy_
            );

            // for each delta in the grid, atomically accumulate to body_deltas
            cell_deltas.transform_with_indices(
                [this] DEV(const auto& delta, size_type idx) {
                    if (delta.accretion_rate_delta != 0.0) {
                        printf("accr rate: %f\n", delta.accretion_rate_delta);
                    }
                    // skip invalid deltas
                    if (delta.body_idx == std::numeric_limits<size_t>::max()) {
                        return delta;
                    }

                    // init body delta if it's the first contribution
                    if (body_deltas[delta.body_idx].body_idx ==
                        std::numeric_limits<size_t>::max()) {
                        body_deltas[delta.body_idx].body_idx = delta.body_idx;
                    }

                    if constexpr (global::on_gpu) {
                        // accumulate force components with atomics
                        for (size_type dim = 0; dim < Dims; ++dim) {
                            gpu::api::atomicAdd(
                                &body_deltas[delta.body_idx].force_delta[dim],
                                delta.force_delta[dim]
                            );
                        }

                        // accumulate scalar values with atomics
                        gpu::api::atomicAdd(
                            &body_deltas[delta.body_idx].mass_delta,
                            delta.mass_delta
                        );
                        gpu::api::atomicAdd(
                            &body_deltas[delta.body_idx].accreted_mass_delta,
                            delta.accreted_mass_delta
                        );
                        gpu::api::atomicAdd(
                            &body_deltas[delta.body_idx].accretion_rate_delta,
                            delta.accretion_rate_delta
                        );
                    }
                    else {
                    // atomic ops via openmp
#pragma omp atomic
                        body_deltas[delta.body_idx].mass_delta +=
                            delta.mass_delta;
                        body_deltas[delta.body_idx].accreted_mass_delta +=
                            delta.accreted_mass_delta;
                        body_deltas[delta.body_idx].accretion_rate_delta +=
                            delta.accretion_rate_delta;
                    }

                    return delta;
                },
                cell_policy_
            );
        }

        // apply all reduced deltas to body system
        ComponentBodySystem<T, Dims>
        apply_to(ComponentBodySystem<T, Dims>&& system)
        {
            // perform parallel reduction on device
            reduce_deltas();

            // apply the deltas from the already-reduced array
            for (size_t body_idx = 0; body_idx < max_bodies_; body_idx++) {
                // skip bodies that weren't modified
                // if (body_modified[body_idx] == 0) {
                //     continue;
                // }

                auto maybe_body = system.get_body(body_idx);
                if (!maybe_body.has_value()) {
                    continue;
                }

                auto body         = maybe_body.value();
                const auto& delta = body_deltas[body_idx];

                // Apply deltas immutably
                if (!delta.force_delta.is_zero()) {
                    body = std::move(body).with_force(
                        body.force + delta.force_delta
                    );
                }

                if (delta.mass_delta != 0) {
                    body = std::move(body).add_mass(delta.mass_delta);
                }

                if (delta.accreted_mass_delta > 0) {
                    printf("accretion active\n");
                    body = std::move(body)
                               .add_accreted_mass(delta.accreted_mass_delta)
                               .with_accretion_rate(delta.accretion_rate_delta);
                }

                system = ComponentBodySystem<T, Dims>::update_body_in(
                    std::move(system),
                    body_idx,
                    std::move(body)
                );
            }

            // reset the body_modified array for next use
            // body_modified.fill(0);
            // body_modified.sync_to_device();

            return std::move(system);
        }
    };   // namespace simbi::ibsystem

}   // namespace simbi::ibsystem

#endif
