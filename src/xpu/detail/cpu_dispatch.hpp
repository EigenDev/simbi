// =============================================================================
// cpu_dispatch.hpp
//
// cpu execution space dispatch implementation for parallel domain iteration.
// uses openmp for parallelization with grid-stride iteration pattern for
// consistency with cuda implementation.
//
// design principles:
//   - openmp parallel for with collapse directives
//   - row-major iteration order (matches cuda grid-stride)
//   - minimal overhead for small domains
//   - scales to arbitrary rank via template specialization
//
// usage:
//   // internal use only - called from executor_t::dispatch()
//   cpu_dispatch(domain, [](auto idx) { /* work */ });
// =============================================================================

#pragma once

#include "../domain.hpp"

#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xpu {

    // =============================================================================
    // cpu dispatch implementations by rank
    // =============================================================================

    // 1d dispatch - simple parallel for
    template <typename Func>
    void cpu_dispatch_1d(const domain_t<1>& domain, Func&& func)
    {
        const auto start = domain.start[0];
        const auto end   = domain.end[0];

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t ii = start; ii < end; ++ii) {
            typename domain_t<1>::coord_t coord{ii};
            func(coord);
        }
    }

    // 2d dispatch - collapsed parallel loops
    template <typename Func>
    void cpu_dispatch_2d(const domain_t<2>& domain, Func&& func)
    {
        const auto start_y = domain.start[0];
        const auto end_y   = domain.end[0];
        const auto start_x = domain.start[1];
        const auto end_x   = domain.end[1];

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (std::int64_t jj = start_y; jj < end_y; ++jj) {
            for (std::int64_t ii = start_x; ii < end_x; ++ii) {
                typename domain_t<2>::coord_t coord{jj, ii};
                func(coord);
            }
        }
    }

    // 3d dispatch - triple collapsed loops
    template <typename Func>
    void cpu_dispatch_3d(const domain_t<3>& domain, Func&& func)
    {
        const auto start_z = domain.start[0];
        const auto end_z   = domain.end[0];
        const auto start_y = domain.start[1];
        const auto end_y   = domain.end[1];
        const auto start_x = domain.start[2];
        const auto end_x   = domain.end[2];

#if defined(_OPENMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (std::int64_t kk = start_z; kk < end_z; ++kk) {
            for (std::int64_t jj = start_y; jj < end_y; ++jj) {
                for (std::int64_t ii = start_x; ii < end_x; ++ii) {
                    typename domain_t<3>::coord_t coord{kk, jj, ii};
                    func(coord);
                }
            }
        }
    }

    // =============================================================================
    // generic dispatch - delegates to rank-specific implementation
    // =============================================================================

    template <std::uint64_t Rank, typename Func>
    void cpu_dispatch(const domain_t<Rank>& domain, Func&& func)
    {
        if constexpr (Rank == 1) {
            cpu_dispatch_1d(domain, std::forward<Func>(func));
        }
        else if constexpr (Rank == 2) {
            cpu_dispatch_2d(domain, std::forward<Func>(func));
        }
        else if constexpr (Rank == 3) {
            cpu_dispatch_3d(domain, std::forward<Func>(func));
        }
        else {
            // fallback for higher ranks - linearized iteration
            const auto total_size = domain.size();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t linear = 0; linear < total_size; ++linear) {
                auto coord = domain.linear_to_coord(linear);
                func(coord);
            }
        }
    }

    // =============================================================================
    // cpu reduction implementations
    // =============================================================================

    // generic cpu reduce with map-reduce pattern
    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    T cpu_reduce(
        const domain_t<Rank>& domain,
        T                     init_value,
        MapFunc&&             map_func,
        ReduceOp&&            reduce_op
    )
    {
        const auto total_size = domain.size();
        T          result     = init_value;

        // use custom reduction for c++20
        // openmp user-defined reductions require declaring reduction operator
        // simpler approach: manual reduction with critical section

#if defined(_OPENMP)
// parallel map phase with thread-local accumulation
#pragma omp parallel
        {
            T thread_local_result = init_value;

#pragma omp for schedule(static)
            for (std::int64_t linear = 0; linear < total_size; ++linear) {
                auto coord          = domain.linear_to_coord(linear);
                T    mapped_value   = map_func(coord);
                thread_local_result = reduce_op(thread_local_result, mapped_value);
            }

// combine thread results with critical section
#pragma omp critical
            {
                result = reduce_op(result, thread_local_result);
            }
        }
#else
        // serial fallback
        for (std::int64_t linear = 0; linear < total_size; ++linear) {
            auto coord        = domain.linear_to_coord(linear);
            T    mapped_value = map_func(coord);
            result            = reduce_op(result, mapped_value);
        }
#endif

        return result;
    }

} // namespace xpu
