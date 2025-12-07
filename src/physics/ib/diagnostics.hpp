#ifndef BODY_DIAGNOSTICS_HPP
#define BODY_DIAGNOSTICS_HPP

#include "body_delta.hpp"
#include "containers/vector.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace simbi::body {
    /**
     * we track the body diagnostics since no bodies in my code are
     * alive yet. we simply track the would-be modifications to
     * them to then later be serialized to disk.
     * this is useful for debugging and for post-processing
     * simulations where we want to know how much force was applied
     * to each body, how much torque was applied, how much mass
     * was accreted, and the instantaneous accretion rate.
     *
     *
     * srp: get the body diagnostics
     */
    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    class body_diagnostics_t
    {
      public:
        // platform-agnostic interface
        virtual ~body_diagnostics_t()                                              = default;
        virtual void accumulate_delta(const body_delta_t<Rank>& delta)             = 0;
        virtual vector_t<body_delta_t<Rank>, MaxBodies> consolidate()              = 0;
        virtual void                                    reset()                    = 0;
        virtual void restore_deltas(const std::vector<body_delta_t<Rank>>& deltas) = 0;
    };

    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    class cpu_diagnostics_t : public body_diagnostics_t<Rank, MaxBodies>
    {
        mutable std::vector<vector_t<body_delta_t<Rank>, MaxBodies>*> registered_accumulators;
        mutable std::mutex                                            registration_mutex;

        // checkpoint-level accumulator (persists between timesteps)
        vector_t<body_delta_t<Rank>, MaxBodies> checkpoint_accumulator{};

        thread_local static vector_t<body_delta_t<Rank>, MaxBodies> thread_data;
        thread_local static bool                                    registered;

      public:
        cpu_diagnostics_t()
        {
            // initialize checkpoint accumulator
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                checkpoint_accumulator[ii].idx = ii;
            }
        }

        void accumulate_delta(const body_delta_t<Rank>& delta) override
        {
            // register this thread's data on first use
            if (!registered) {
                std::lock_guard lock(registration_mutex);
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    thread_data[ii].idx = ii;
                }
                registered_accumulators.push_back(&thread_data);
                registered = true;
            }

            // spatial accumulation within timestep
            thread_data[delta.idx] += delta;
        }

        vector_t<body_delta_t<Rank>, MaxBodies> consolidate() override
        {
            std::lock_guard lock(registration_mutex);

            // sum across threads to get timestep totals
            vector_t<body_delta_t<Rank>, MaxBodies> timestep_total{};
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                timestep_total[ii].idx = ii;
            }

            for (auto* acc : registered_accumulators) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    timestep_total[ii] += (*acc)[ii];
                }
            }

            // update checkpoint accumulator
            // force/torque: use latest timestep values
            // mass: accumulate across timesteps
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                checkpoint_accumulator[ii].force_delta  = timestep_total[ii].force_delta;
                checkpoint_accumulator[ii].torque_delta = timestep_total[ii].torque_delta;
                checkpoint_accumulator[ii].mass_delta += timestep_total[ii].mass_delta;
            }

            return checkpoint_accumulator;
        }

        void reset() override
        {
            std::lock_guard lock(registration_mutex);

            // reset per-timestep thread accumulators
            for (auto* acc : registered_accumulators) {
                *acc = {};
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    (*acc)[ii].idx = ii;
                }
            }

            // reset checkpoint accumulator (called after writing diagnostics)
            for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                checkpoint_accumulator[ii]     = {};
                checkpoint_accumulator[ii].idx = ii;
            }
        }

        void restore_deltas(const std::vector<body_delta_t<Rank>>& deltas) override
        {
            std::lock_guard lock(registration_mutex);

            // restore checkpoint accumulator from loaded deltas
            // this ensures accretion rate continuity after checkpoint reload
            for (std::size_t ii = 0; ii < deltas.size() && ii < MaxBodies; ++ii) {
                checkpoint_accumulator[ii] = deltas[ii];
            }
        }
    };

    template <std::uint64_t Rank, std::uint64_t MaxBodies>
    thread_local vector_t<body_delta_t<Rank>, MaxBodies>
        cpu_diagnostics_t<Rank, MaxBodies>::thread_data{};

    template <std::uint64_t Rank, std::uint64_t MaxBodies>
    thread_local bool cpu_diagnostics_t<Rank, MaxBodies>::registered = false;

    // [TODO]: impl the gpu diagnostics accumulator

    template <std::uint64_t Rank>
    auto create_diagnostics_accumulator() -> std::unique_ptr<body_diagnostics_t<Rank>>
    {
        return std::make_unique<cpu_diagnostics_t<Rank>>();
    }
} // namespace simbi::body

#endif
