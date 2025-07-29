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
#include "component_body_system.hpp"   // for ComponentBodySystem
#include "compute/field.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "core/utility/helpers.hpp"   // for unravel_idx
#include "core/utility/managed.hpp"   // for Managed
#include "system/adapter/device_adapter_api.hpp"
#include "util/parallel/exec_policy.hpp"
#include "util/parallel/parallel_for.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <tuple>

using namespace simbi::helpers;

namespace simbi::ibsystem {
    template <typename T, std::uint64_t Dims>
    class GridBodyDeltaCollector : public managed_t<global::managed_memory>
    {
      private:
        // one delta per cell in the grid
        field_t<BodyDelta<T, Dims>, Dims> cell_deltas;
        // reduced deltas per body (final output)
        field_t<BodyDelta<T, Dims>, Dims> body_deltas;

        std::uint64_t max_bodies_;
        vector_t<std::uint64_t, 3> cell_shape_;

        using ulist = std::initializer_list<std::uint64_t>;
        using ilist = std::initializer_list<std::int64_t>;

      public:
        GridBodyDeltaCollector(
            const ilist& grid_shape,
            std::uint64_t max_bodies
        )
            : max_bodies_(max_bodies)
        {
            // init with one delta per cell
            std::uint64_t size                    = 1;
            vector_t<std::uint64_t, 3> cell_shape = {1, 1, 1};
            for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                if (auto dim = *(std::begin(grid_shape) + 1) > 1) {
                    const auto s = dim;
                    size *= s;
                    cell_shape[dim] = s;
                }
            }
            cell_shape_ = cell_shape;

            cell_deltas.resize(size * max_bodies);
            cell_deltas.reshape(
                {max_bodies, cell_shape[2], cell_shape[1], cell_shape[0]}
            );

            // prep body deltas array (one per possible body)
            body_deltas.resize(max_bodies_);

            // init cell_deltas with default values
            for (std::uint64_t idx = 0; idx < size; ++idx) {
                // convert flat index to 3D spatial coordinates
                const std::uint64_t x = idx % cell_shape[0];
                const std::uint64_t y = (idx / cell_shape[0]) % cell_shape[1];
                const std::uint64_t z = idx / (cell_shape[0] * cell_shape[1]);

                // init for all bodies at this spatial location
                for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                     body_idx++) {
                    cell_deltas.at(x, y, z, body_idx) =
                        BodyDelta<T, Dims>{body_idx};
                }
            }

            // ensure device has initial state
            cell_deltas.sync_to_device();
            body_deltas.sync_to_device();
        }

        // thread-safe recording - each cell writes to its own slot
        DEV void record_delta(
            std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> cell_idx,
            std::uint64_t body_idx,
            const vector_t<T, Dims>& force,
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
            for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                 body_idx++) {
                body_deltas[body_idx] = BodyDelta<T, Dims>{
                  body_idx,   // valid body_idx from start
                };
            }

            // create a policy for parallelizing over bodies
            ExecutionPolicy<> body_policy({max_bodies_, 1, 1}, {128, 1, 1});

            // Parallelize over bodies - each thread handles one body
            parallel_for(body_policy, [this] DEV(std::uint64_t body_idx) {
                if (body_idx >= max_bodies_) {
                    return;
                }

                // local accumulation variables
                vector_t<T, Dims> force_sum{};
                T mass_sum = 0, accreted_mass_sum = 0, accretion_rate_sum = 0;

                // process all cells for this body
                for (std::uint64_t z = 0; z < cell_shape_[2]; z++) {
                    for (std::uint64_t y = 0; y < cell_shape_[1]; y++) {
                        for (std::uint64_t x = 0; x < cell_shape_[0]; x++) {
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

        // since gpus hate not having work,
        // we do a reduction for each thread block
        // in each cell and then finally reduce on the host
        void reduce_deltas_gpu()
        {
            for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                 body_idx++) {
                body_deltas[body_idx] = BodyDelta<T, Dims>{
                  body_idx,   // valid body_idx from start
                };
            }
            body_deltas.sync_to_device();

            // allocate intermediate reduction buffer (one entry per block per
            // body) Calculate grid dimensions for a suitable block size
            constexpr std::uint64_t BLOCK_SIZE = 256;
            const std::uint64_t total_cells =
                cell_shape_[0] * cell_shape_[1] * cell_shape_[2];
            const std::uint64_t num_blocks =
                (total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // allocate intermediate results buffer
            field_t<BodyDelta<T, Dims>, Dims> block_results;
            block_results.resize(num_blocks * max_bodies_);
            block_results.reshape({num_blocks, max_bodies_});

            // init with zeros
            for (std::uint64_t block_idx = 0; block_idx < num_blocks;
                 block_idx++) {
                for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                     body_idx++) {
                    block_results.at(block_idx, body_idx) = BodyDelta<T, Dims>{
                      body_idx,   // valid body_idx from start
                    };
                }
            }
            block_results.sync_to_device();

            // first phase: ech thread block processes a chunk of cells
            // and produces one partial result per body
            ExecutionPolicy<> block_policy(
                {total_cells, 1, 1},
                {BLOCK_SIZE, 1, 1}
            );
            block_policy.shared_mem_bytes =
                max_bodies_ * sizeof(BodyDelta<T, Dims>);
            auto* block_results_ptr = block_results.data();

            parallel_for(
                block_policy,
                [this, block_results_ptr, total_cells] DEV(std::uint64_t) {
                    // block index
                    const std::uint64_t block_idx  = get_block_id();
                    const std::uint64_t thread_idx = get_thread_id();

                    // set up shared memory for block-level reduction
                    extern SHARED char shared_mem[];
                    BodyDelta<T, Dims>* shared_deltas =
                        reinterpret_cast<BodyDelta<T, Dims>*>(shared_mem);

                    // init shared memory
                    if (thread_idx < max_bodies_) {
                        shared_deltas[thread_idx] =
                            BodyDelta<T, Dims>{thread_idx};
                    }
                    gpu::api::synchronize();

                    // each thread processes multiple cells based on its global
                    // index
                    const std::uint64_t cells_per_block = BLOCK_SIZE;
                    const std::uint64_t start_cell =
                        block_idx * cells_per_block;
                    const std::uint64_t end_cell =
                        std::min(start_cell + cells_per_block, total_cells);

                    // process cells assigned to this thread
                    for (std::uint64_t cell_idx = start_cell + thread_idx;
                         cell_idx < end_cell;
                         cell_idx += get_threads_per_block()) {

                        // covert linear index to 3D coordinates
                        std::uint64_t x = 0, y = 0, z = 0;
                        const auto pos =
                            unravel_idx<Dims>(cell_idx, cell_shape_);
                        x = pos[0];
                        if constexpr (Dims > 1) {
                            y = pos[1];
                        }
                        if constexpr (Dims > 2) {
                            z = pos[2];
                        }

                        // process all bodies for this cell
                        for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                             body_idx++) {
                            const auto& delta =
                                cell_deltas.at(x, y, z, body_idx);

                            // skip invalid body indices
                            if (delta.body_idx >= max_bodies_) {
                                continue;
                            }

                            // use atomic operations for thread-safe
                            // accumulation in shared memory
                            for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                                gpu::api::atomic_add(
                                    &shared_deltas[body_idx].force_delta[dim],
                                    delta.force_delta[dim]
                                );
                            }
                            gpu::api::atomic_add(
                                &shared_deltas[body_idx].mass_delta,
                                delta.mass_delta
                            );
                            gpu::api::atomic_add(
                                &shared_deltas[body_idx].accreted_mass_delta,
                                delta.accreted_mass_delta
                            );
                            gpu::api::atomic_add(
                                &shared_deltas[body_idx].accretion_rate_delta,
                                delta.accretion_rate_delta
                            );
                        }
                    }

                    // ensure all threads have finished processing
                    gpu::api::synchronize();

                    // first thread in block writes results to global memory
                    if (thread_idx < max_bodies_) {
                        const auto res_idx =
                            block_idx * max_bodies_ + thread_idx;
                        block_results_ptr[res_idx] = shared_deltas[thread_idx];
                    }
                }
            );

            // second phase: Reduce block results on the host
            // transfer block results to host (relatively small data)
            block_results.sync_to_host();

            // final reduction on host
            for (std::uint64_t body_idx = 0; body_idx < max_bodies_;
                 body_idx++) {
                vector_t<T, Dims> force_sum{};
                T mass_sum = 0, accreted_mass_sum = 0, accretion_rate_sum = 0;

                // acc from all blocks
                for (std::uint64_t block_idx = 0; block_idx < num_blocks;
                     block_idx++) {
                    const auto& block_delta =
                        block_results.at(body_idx, block_idx);
                    force_sum += block_delta.force_delta;
                    mass_sum += block_delta.mass_delta;
                    accreted_mass_sum += block_delta.accreted_mass_delta;
                    accretion_rate_sum += block_delta.accretion_rate_delta;
                }

                // store final results
                body_deltas[body_idx].force_delta          = force_sum;
                body_deltas[body_idx].mass_delta           = mass_sum;
                body_deltas[body_idx].accreted_mass_delta  = accreted_mass_sum;
                body_deltas[body_idx].accretion_rate_delta = accretion_rate_sum;
            }

            // sync the final results back to device if needed
            body_deltas.sync_to_device();
        }

        ComponentBodySystem<T, Dims>
        apply_to(ComponentBodySystem<T, Dims>&& system)
        {
            // Use appropriate reduction strategy based on platform
            if constexpr (platform::is_gpu) {
                reduce_deltas_gpu();
            }
            else {
                reduce_deltas();
            }

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
