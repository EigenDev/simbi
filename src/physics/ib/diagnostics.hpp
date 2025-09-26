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
    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    class body_diagnostics_t
    {
      public:
        // platform-agnostic interface
        virtual ~body_diagnostics_t() = default;
        virtual void accumulate_delta(const body_delta_t<Dims>& delta) = 0;
        virtual vector_t<body_delta_t<Dims>, MaxBodies> consolidate()  = 0;
        virtual void reset()                                           = 0;
    };

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    class cpu_diagnostics_t : public body_diagnostics_t<Dims, MaxBodies>
    {
        mutable std::vector<vector_t<body_delta_t<Dims>, MaxBodies>*>
            registered_accumulators;
        mutable std::mutex registration_mutex;

        thread_local static vector_t<body_delta_t<Dims>, MaxBodies> thread_data;
        thread_local static bool registered;

      public:
        void accumulate_delta(const body_delta_t<Dims>& delta) override
        {
            // register this thread's data on first use
            if (!registered) {
                std::lock_guard lock(registration_mutex);
                registered_accumulators.push_back(&thread_data);
                registered = true;
            }

            thread_data[delta.idx] += delta;
        }

        vector_t<body_delta_t<Dims>, MaxBodies> consolidate() override
        {
            std::lock_guard lock(registration_mutex);
            vector_t<body_delta_t<Dims>, MaxBodies> total{};

            for (auto* acc : registered_accumulators) {
                for (std::uint64_t ii = 0; ii < MaxBodies; ++ii) {
                    total[ii] += (*acc)[ii];
                }
            }
            return total;
        }

        void reset() override
        {
            std::lock_guard lock(registration_mutex);
            for (auto* acc : registered_accumulators) {
                *acc = {};   // reset each thread's accumulator
            }
        }
    };

    template <std::uint64_t Dims, std::uint64_t MaxBodies>
    thread_local vector_t<body_delta_t<Dims>, MaxBodies>
        cpu_diagnostics_t<Dims, MaxBodies>::thread_data{};

    template <std::uint64_t Dims, std::uint64_t MaxBodies>
    thread_local bool cpu_diagnostics_t<Dims, MaxBodies>::registered = false;

    // [TODO]: impl the gpu diagnostics accumulator

    template <std::uint64_t Dims>
    auto create_diagnostics_accumulator()
        -> std::unique_ptr<body_diagnostics_t<Dims>>
    {
        return std::make_unique<cpu_diagnostics_t<Dims>>();
    }
}   // namespace simbi::body

#endif
