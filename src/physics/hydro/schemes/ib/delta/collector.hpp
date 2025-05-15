/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            collector.hpp
 * @brief           A collector for body deltas in a grid-based system
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
#ifndef COLLECTOR_HPP
#define COLLECTOR_HPP

#include "body_delta.hpp"
#include "build_options.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/utility/managed.hpp"   // for Managed
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"   // for ComponentBodySystem
#include "util/parallel/exec_policy.hpp"
#include "util/parallel/parallel_for.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class GridBodyDeltaCollector : public Managed<global::managed_memory>
    {
      private:
        // one delta per cell in the grid
        ndarray<BodyDelta<T, Dims>, 4> cell_deltas;

        // reduced deltas per body (final output)
        ndarray<BodyDelta<T, Dims>, 1> body_deltas;

        // tracks which bodies were modified
        ndarray<unsigned int, 1> body_modified;

        size_type max_bodies_;
        array_t<size_type, 3> cell_shape_;

        ExecutionPolicy<> spatial_policy_{};

      public:
        GridBodyDeltaCollector(
            const collapsable<Dims>& grid_shape,
            size_type max_bodies
        )
            : max_bodies_(max_bodies)
        {
            // init with one delta per cell
            size_type size                   = 1;
            array_t<size_type, 3> cell_shape = {1, 1, 1};
            for (size_type dim = 0; dim < Dims; ++dim) {
                size *= grid_shape[dim];
                cell_shape[dim] = grid_shape[dim];
            }
            cell_shape_ = cell_shape;

            cell_deltas.resize(size * max_bodies);
            cell_deltas.reshape(
                {max_bodies, cell_shape[2], cell_shape[1], cell_shape[0]}
            );

            // Create execution policy for spatial dimensions only
            spatial_policy_ = ExecutionPolicy<>(cell_shape, {8, 8, 8});

            // prep body deltas array (one per possible body)
            body_deltas.resize(max_bodies_);

            // init body modified flags (0 = not modified)
            body_modified.resize(max_bodies_);
            body_modified.fill(0);

            // Initialize cell deltas for all bodies in parallel
            // We'll use a flattened 3D approach to handle the 4D array
            spatial_policy_.optimize_batch_size();

            // Initialize cell_deltas with default values
            for (size_type idx = 0; idx < size; ++idx) {
                // Convert flat index to 3D spatial coordinates
                const size_type x = idx % cell_shape[0];
                const size_type y = (idx / cell_shape[0]) % cell_shape[1];
                const size_type z = idx / (cell_shape[0] * cell_shape[1]);

                // Initialize for all bodies at this spatial location
                for (size_type body_idx = 0; body_idx < max_bodies_;
                     body_idx++) {
                    cell_deltas.at(x, y, z, body_idx) = BodyDelta<T, Dims>{
                      body_idx,
                      spatial_vector_t<T, Dims>{},   // zero force
                      0,                             // zero mass
                      0,                             // zero accreted mass
                      0                              // zero accretion rate
                    };
                }
            }

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
                std::get<2>(cell_idx),
                body_idx
            );

            // combine the delta values
            delta.force_delta          = force;
            delta.mass_delta           = mass;
            delta.accretion_rate_delta = accretion_rate;
            delta.accreted_mass_delta  = accreted_mass;
        }

        // parallel reduction of deltas on device
        void reduce_deltas()
        {
            // reset the body deltas
            for (size_type body_idx = 0; body_idx < max_bodies_; body_idx++) {
                body_deltas[body_idx] = BodyDelta<T, Dims>{
                  body_idx,                      // valid body_idx from start
                  spatial_vector_t<T, Dims>{},   // zero force
                  0,
                  0,
                  0   // zero other values
                };
            }
            body_deltas.sync_to_device();

            // create a policy for parallelizing over bodies
            ExecutionPolicy<> body_policy({max_bodies_, 1, 1}, {128, 1, 1});

            // Parallelize over bodies - each thread handles one body
            parallel_for(body_policy, [this] DEV(luint body_idx) {
                if (body_idx >= max_bodies_) {
                    return;
                }

                // local accumulation variables
                spatial_vector_t<T, Dims> force_sum{};
                T mass_sum = 0, accreted_mass_sum = 0, accretion_rate_sum = 0;

                // process all cells for this body
                for (size_type z = 0; z < cell_shape_[2]; z++) {
                    for (size_type y = 0; y < cell_shape_[1]; y++) {
                        for (size_type x = 0; x < cell_shape_[0]; x++) {
                            const auto& delta =
                                cell_deltas.at(x, y, z, body_idx);

                            // skip invalid body indices
                            if (delta.body_idx >= max_bodies_) {
                                continue;
                            }

                            // acc without atomics (thread-local)
                            force_sum += delta.force_delta;
                            mass_sum += delta.mass_delta;
                            accreted_mass_sum += delta.accreted_mass_delta;
                            accretion_rate_sum += delta.accretion_rate_delta;
                        }
                    }
                }

                // Write final sums to global memory once
                body_deltas[body_idx].force_delta          = force_sum;
                body_deltas[body_idx].mass_delta           = mass_sum;
                body_deltas[body_idx].accreted_mass_delta  = accreted_mass_sum;
                body_deltas[body_idx].accretion_rate_delta = accretion_rate_sum;
            });
        }

        // apply all reduced deltas to body system
        ComponentBodySystem<T, Dims>
        apply_to(ComponentBodySystem<T, Dims>&& system)
        {
            // perform parallel reduction on device
            reduce_deltas();

            // apply the deltas from the already-reduced array
            for (size_t body_idx = 0; body_idx < max_bodies_; body_idx++) {
                auto maybe_body = system.get_body(body_idx);
                if (!maybe_body.has_value()) {
                    continue;
                }

                auto body         = maybe_body.value();
                const auto& delta = body_deltas[body_idx];

                // apply deltas immutably
                if (!delta.force_delta.is_zero()) {
                    body = std::move(body).with_force(
                        body.force + delta.force_delta
                    );
                }

                if (delta.mass_delta != 0) {
                    body = std::move(body).add_mass(delta.mass_delta);
                }

                if (delta.accreted_mass_delta != 0.0) {
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

            return std::move(system);
        }
    };

}   // namespace simbi::ibsystem

#endif
