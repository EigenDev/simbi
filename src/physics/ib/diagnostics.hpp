#ifndef BODY_DIAGNOSTICS_HPP
#define BODY_DIAGNOSTICS_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "functional/monad/reader.hpp"

#include <atomic>
#include <cstdint>

namespace simbi::body {
    /**
     * we track the body diagnostics since no bodies in my code are
     * alive yet. we simply track the would-be modifications to
     * them to then later be serialized to disk.
     * this is useful for debugging and for post-processing
     * simulations where we want to know how much force was applied
     * to each body, how much torque was applied, how much mass
     * was accreted, and how much accretion rate was applied.
     *
     *
     * srp: get the body diagnostics
     */
    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    struct body_diagnostics_t {
        vector_t<std::atomic<real>, MaxBodies> force_1{0};
        vector_t<std::atomic<real>, MaxBodies> force_2{0};
        vector_t<std::atomic<real>, MaxBodies> force_3{0};
        vector_t<std::atomic<real>, MaxBodies> torque_1{0};
        vector_t<std::atomic<real>, MaxBodies> torque_2{0};
        vector_t<std::atomic<real>, MaxBodies> torque_3{0};
        vector_t<std::atomic<real>, MaxBodies> total_mass{0};
        vector_t<std::atomic<real>, MaxBodies> accreted_mass{0};
        vector_t<std::atomic<real>, MaxBodies> accretion_rate{0};

        void accumulate_delta(const auto& body_delta)
        {
            force_1[body_delta.idx].fetch_add(
                body_delta.force_delta[0],
                std::memory_order_relaxed
            );
            if constexpr (Dims > 1) {
                force_2[body_delta.idx].fetch_add(
                    body_delta.force_delta[1],
                    std::memory_order_relaxed
                );
            }
            if constexpr (Dims > 2) {
                force_3[body_delta.idx].fetch_add(
                    body_delta.force_delta[2],
                    std::memory_order_relaxed
                );
            }

            // toeque vector is always 3D
            // where it is 0 for 1D, it exists
            // along the z-index in 2D, and
            // works the usual way for 3D
            torque_1[body_delta.idx].fetch_add(
                body_delta.torque_delta[0],
                std::memory_order_relaxed
            );
            torque_2[body_delta.idx].fetch_add(
                body_delta.torque_delta[1],
                std::memory_order_relaxed
            );
            torque_3[body_delta.idx].fetch_add(
                body_delta.torque_delta[2],
                std::memory_order_relaxed
            );

            total_mass[body_delta.idx].fetch_add(
                body_delta.mass_delta,
                std::memory_order_relaxed
            );

            accretion_rate[body_delta.idx].fetch_add(
                body_delta.accretion_rate_delta,
                std::memory_order_relaxed
            );
        }
    };
}   // namespace simbi::body

namespace simbi {
    // diagnostics reader for body diagnostics
    template <std::uint64_t Dims>
    using diagnostics_reader_t = reader_t<body::body_diagnostics_t<Dims>>;
}   // namespace simbi

#endif
