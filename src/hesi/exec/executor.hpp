#ifndef HET_EXECUTOR_HPP
#define HET_EXECUTOR_HPP

#include "detail/launcher.hpp"
#include "grid/domain.hpp"
#include "hesi/core/types.hpp"
#include "policy.hpp"
#include "stream.hpp"
#include "token.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace simbi::het::exec {

    // the unified execution context
    // holds a non-owning reference to a stream
    // streams are owned by partitions; executors are short-lived handles
    struct executor_t {
        const stream_t& stream_;

        explicit executor_t(const stream_t& s) : stream_(s) {}

        // fire and forget (void return)
        template <typename Functor>
        void dispatch(const launch_policy_t& policy, Functor&& f) const
        {
            detail::launch(stream_, policy, std::forward<Functor>(f));
        }

        template <std::uint64_t Rank>
        dim3_t get_hint(const std::string& key, grid::domain_t<Rank> dom) const
        {
            // future: implement a hint storage system
            // for now, return default tile sizes
            if (key == "cpu_tile") {
                if constexpr (Rank == 1) {
                    return dim3_t{64, 1, 1};
                }
                else if constexpr (Rank == 2) {
                    // sometimes we are doing a 2D problem,
                    // but the domain is such that one dimension is small
                    // (e.g., a thin slice). In that case, use a different tile
                    // size. Heuristic: if one dimension is less than 16, use a
                    // taller tile. This helps improve occupancy.
                    if (dom.shape()[0] < 16) {
                        return dim3_t{64, 1, 1};
                    }
                    return dim3_t{16, 16, 1};
                }
                else if constexpr (Rank == 3) {
                    // quasi-1D problems require different tiling
                    if (dom.shape()[0] < 16 && dom.shape()[1] < 16) {
                        return dim3_t{64, 1, 1};
                    }
                    else if (dom.shape()[0] < 16) {
                        return dim3_t{16, 16, 1};
                    }
                    return dim3_t{8, 8, 8};
                }
            }
            return dim3_t{1, 1, 1};   // fallback
        }

        // chained dispatch (token in -> token out)
        // waits for dependency, runs kernel, records new event
        template <typename Functor>
        token_t then(
            const token_t& dep,
            const launch_policy_t& policy,
            Functor&& f
        ) const
        {

            // enforce dependency
            if (dep) {
                dep.wait(stream_);
            }

            // execute
            detail::launch(stream_, policy, std::forward<Functor>(f));

            // record completion
            auto new_token = token_t::create(stream_.backend());
            new_token.event_->record(stream_);

            return new_token;
        }

        // syntactic sugar: start a chain (no input dep) -> token out
        template <typename Functor>
        token_t submit(const launch_policy_t& policy, Functor&& f) const
        {
            detail::launch(stream_, policy, std::forward<Functor>(f));

            auto new_token = token_t::create(stream_.backend());
            new_token.event_->record(stream_);

            return new_token;
        }

        // helper for 1d linear
        template <typename Functor>
        token_t then(const token_t& dep, std::size_t n, Functor&& f) const
        {
            return then(
                dep,
                launch_policy_t::linear(n),
                std::forward<Functor>(f)
            );
        }

        const stream_t& stream() const { return stream_; }
        backend_type_t backend() const { return stream_.backend(); }
    };

}   // namespace simbi::het::exec

#endif   // HETERO_EXECUTOR_HPP
