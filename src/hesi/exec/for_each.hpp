#ifndef HET_EXEC_FOR_EACH_HPP
#define HET_EXEC_FOR_EACH_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "hesi/backend/parallel_for.hpp"
#include "hesi/core/types.hpp"
#include "hesi/cuda/parallel_for.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/exec/policy.hpp"
#include "hesi/exec/token.hpp"

#include <cstdint>

namespace simbi::het::exec {

    // tag types for execution policy
    struct default_t {
    };
    struct cpu_serial_t {
    };
    struct openmp_t {
        dim3_t tile_size;
    };
    struct gpu_t {
    };

    // helper: make launch policy from domain shape
    template <std::uint64_t R>
    launch_policy_t make_launch_policy(const iarray<R>& shape)
    {
        if constexpr (R == 1) {
            return launch_policy_t::linear(shape[0]);
        }
        else if constexpr (R == 2) {
            return launch_policy_t::surface(shape[1], shape[0]);
        }
        else if constexpr (R == 3) {
            return launch_policy_t::volume(shape[2], shape[1], shape[0]);
        }
    }

    // cpu serial
    template <std::uint64_t Rank, typename F>
    auto parallel_for(
        cpu_serial_t /*policy*/,
        const grid::domain_t<Rank>& domain,
        F&& f
    )
    {
        auto policy = make_launch_policy(domain.shape());
        backend::parallel_for(
            backend_type_t::cpu,
            nullptr,   // no stream for cpu
            policy,
            domain,
            std::forward<F>(f)
        );
    }

    // openmp
    template <std::uint64_t Rank, typename F>
    auto
    parallel_for(openmp_t policy_tag, const grid::domain_t<Rank>& domain, F&& f)
    {
        auto policy = launch_policy_t(
            {1, 1, 1},   // grid doesn't matter for cpu
            policy_tag.tile_size
        );

        backend::parallel_for(
            backend_type_t::cpu,
            nullptr,
            policy,
            domain,
            std::forward<F>(f)
        );
    }

    // gpu
    template <std::uint64_t Rank, typename F>
    token_t parallel_for(
        gpu_t /*policy*/,
        executor_t& exec,
        const grid::domain_t<Rank>& domain,
        F&& f
    )
    {
        if (domain.empty()) {
            return token_t::immediate(exec.backend());
        }

        auto policy = make_launch_policy(domain.shape());

        backend::parallel_for(
            exec.backend(),
            exec.stream().native(),
            policy,
            domain,
            std::forward<F>(f)
        );

        // record completion
        auto token = token_t::create(exec.backend());
        token.record(exec.stream());

        return token;
    }

    // default (runtime dispatch)
    template <std::uint64_t Rank, typename F>
    token_t parallel_for(
        default_t /*policy*/,
        executor_t& exec,
        const grid::domain_t<Rank>& domain,
        F&& f
    )
    {
        if (exec.backend() == backend_type_t::cpu) {
            // check if openmp available
            if (global::use_omp) {
                auto tile = exec.get_hint<Rank>("cpu_tile", domain);
                parallel_for(openmp_t{tile}, domain, std::forward<F>(f));
            }
            else {
                parallel_for(cpu_serial_t{}, domain, std::forward<F>(f));
            }

            return token_t::immediate(backend_type_t::cpu);
        }
        else {
            // gpu path
            return parallel_for(gpu_t{}, exec, domain, std::forward<F>(f));
        }
    }

}   // namespace simbi::het::exec

#endif
