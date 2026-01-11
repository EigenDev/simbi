// =============================================================================
// diagnostics.hpp
//
// body diagnostics for immersed boundary simulations.
// accumulates forces, torques, and mass changes on cpu (thread_local) or
// gpu (block-level). fully vendor-agnostic via xpu abstraction layer.
//
// design:
//   - cpu: thread_local accumulators (one per openmp thread)
//   - gpu: block-level accumulators in unified memory (one per cuda block)
//   - zero raw vendor calls - all through xpu memory spaces
//   - device-accessible via cached raw pointer
//
// usage:
//   auto diag = create_diagnostics<2>(grid_size);
//   diag->accumulate_delta(delta);  // device code
//   auto totals = diag->consolidate();  // host code
//   diag->reset();
// =============================================================================

#ifndef BODY_DIAGNOSTICS_HPP
#define BODY_DIAGNOSTICS_HPP

#include "body_delta.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "platform.hpp"
#include "xpu/device/atomic.hpp"
#include "xpu/device/grid.hpp"
#include "xpu/mem/block.hpp"
#include "xpu/mem/managed.hpp"
#include "xpu/xpu.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace simbi::body {
    // =============================================================================
    // cpu implementation - thread_local accumulators
    // =============================================================================

    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    class cpu_diagnostics_t
    {
        mutable std::vector<vector_t<body_delta_t<Rank>, MaxBodies>*> registered_accumulators;
        mutable std::mutex                                            registration_mutex;

        thread_local static vector_t<body_delta_t<Rank>, MaxBodies> thread_data;
        thread_local static bool                                    registered;

      public:
        cpu_diagnostics_t() = default;

        void accumulate_delta(const body_delta_t<Rank>& delta)
        {
            if (!registered) {
                std::lock_guard lock(registration_mutex);
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    thread_data[ii].idx = ii;
                }
                registered_accumulators.push_back(&thread_data);
                registered = true;
            }

            thread_data[delta.idx] += delta;
        }

        vector_t<body_delta_t<Rank>, MaxBodies> consolidate()
        {
            std::lock_guard lock(registration_mutex);

            vector_t<body_delta_t<Rank>, MaxBodies> result{};
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                result[ii].idx = ii;
            }

            for (auto* acc : registered_accumulators) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    result[ii] += (*acc)[ii];
                }
            }

            return result;
        }

        void reset()
        {
            std::lock_guard lock(registration_mutex);

            for (auto* acc : registered_accumulators) {
                *acc = {};
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    (*acc)[ii].idx = ii;
                }
            }
        }

        void restore_deltas(const std::vector<body_delta_t<Rank>>&)
        {
            // no-op for cpu - no persistent state
        }
    };

    template <std::uint64_t Rank, std::uint64_t MaxBodies>
    thread_local vector_t<body_delta_t<Rank>, MaxBodies>
        cpu_diagnostics_t<Rank, MaxBodies>::thread_data{};

    template <std::uint64_t Rank, std::uint64_t MaxBodies>
    thread_local bool cpu_diagnostics_t<Rank, MaxBodies>::registered = false;

    // =============================================================================
    // gpu implementation - block-level accumulators via xpu unified memory
    // =============================================================================

    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    class gpu_diagnostics_t : public managed_t
    {
        xpu::mem::memory_block_t<xpu::unified_memory_t> block_accumulators;
        body_delta_t<Rank>*                             data_ptr_;
        std::int64_t                                    num_blocks;

      public:
        explicit gpu_diagnostics_t(int grid_size)
            : block_accumulators(
                  xpu::mem::make_block<body_delta_t<Rank>, xpu::unified_memory_t>(
                      grid_size * MaxBodies
                  )
              ),
              data_ptr_(block_accumulators.template as<body_delta_t<Rank>>()), num_blocks(grid_size)
        {
            // initialize via unified memory (accessible from host)
            for (std::int64_t block = 0; block < num_blocks; ++block) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    data_ptr_[block * MaxBodies + ii]     = {};
                    data_ptr_[block * MaxBodies + ii].idx = ii;
                }
            }
        }

        DEV void accumulate_delta(const body_delta_t<Rank>& delta)
        {
            const std::int64_t block_id = xpu::device::get_block_id();

            if (block_id < num_blocks) {
                const std::int64_t offset = block_id * MaxBodies + delta.idx;

                xpu::device::atomic_add(&data_ptr_[offset].force_delta[0], delta.force_delta[0]);
                xpu::device::atomic_add(&data_ptr_[offset].force_delta[1], delta.force_delta[1]);
                if constexpr (Rank == 3) {
                    xpu::device::atomic_add(
                        &data_ptr_[offset].force_delta[2],
                        delta.force_delta[2]
                    );
                }

                xpu::device::atomic_add(&data_ptr_[offset].torque_delta[0], delta.torque_delta[0]);
                xpu::device::atomic_add(&data_ptr_[offset].torque_delta[1], delta.torque_delta[1]);
                if constexpr (Rank == 3) {
                    xpu::device::atomic_add(
                        &data_ptr_[offset].torque_delta[2],
                        delta.torque_delta[2]
                    );
                }

                xpu::device::atomic_add(&data_ptr_[offset].mass_delta, delta.mass_delta);
            }
        }

        vector_t<body_delta_t<Rank>, MaxBodies> consolidate()
        {
            xpu::synchronize();

            vector_t<body_delta_t<Rank>, MaxBodies> result{};
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                result[ii].idx = ii;
            }

            // access data_ptr_ directly after sync
            for (std::int64_t block = 0; block < num_blocks; ++block) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    const auto& delta = data_ptr_[block * MaxBodies + ii];
                    result[ii] += delta;
                }
            }

            return result;
        }

        void reset()
        {
            // synchronize via xpu
            xpu::synchronize();

            // clear unified memory buffer via cached pointer
            for (std::int64_t block = 0; block < num_blocks; ++block) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    data_ptr_[block * MaxBodies + ii]     = {};
                    data_ptr_[block * MaxBodies + ii].idx = ii;
                }
            }
        }

        void restore_deltas(const std::vector<body_delta_t<Rank>>&)
        {
            // no-op for gpu - no persistent state
        }
    };

    // =============================================================================
    // factory function - compile-time dispatch based on platform
    // =============================================================================

    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    auto create_diagnostics_accumulator(int grid_size = 1024)
    {
        if constexpr (platform::is_gpu) {
            return std::make_unique<gpu_diagnostics_t<Rank, MaxBodies>>(grid_size);
        }
        else {
            (void) grid_size;
            return std::make_unique<cpu_diagnostics_t<Rank, MaxBodies>>();
        }
    }

} // namespace simbi::body

#endif
