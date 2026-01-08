#ifndef HET_BACKEND_CPU_REDUCE_HPP
#define HET_BACKEND_CPU_REDUCE_HPP

#include "compat.hpp"
#include <cstdint>

namespace simbi::het::backend::cpu {

    // reduce contiguous range
    template <typename T, typename BinaryOp>
    void reduce_range(
        const T* input,
        T* output,
        std::uint64_t n,
        T init,
        BinaryOp op,
        T /*identity*/,
        bool use_openmp
    )
    {
        T acc = init;

        if (use_openmp && n > 1000) {   // threshold for openmp overhead
            if (global::use_omp) {
// parallel reduction
#pragma omp parallel for reduction(+ : acc)
                for (std::uint64_t ii = 0; ii < n; ++ii) {
                    acc = op(acc, input[ii]);
                }
            }
            else {   // fallback to serial
                for (std::uint64_t ii = 0; ii < n; ++ii) {
                    acc = op(acc, input[ii]);
                }
            }
        }
        else {
            // serial for small arrays
            for (std::uint64_t ii = 0; ii < n; ++ii) {
                acc = op(acc, input[ii]);
            }
        }

        *output = acc;
    }

    // transform-reduce over domain
    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    void transform_reduce(
        const Computation& comp,
        T* result,
        T init,
        TransformOp transform,
        BinaryOp op,
        T /*identity*/,
        bool use_openmp
    )
    {
        auto domain     = comp.domain();
        std::uint64_t n = domain.size();
        T acc           = init;

        if (use_openmp && n > 1000) {
            if (global::use_omp) {
// parallel reduction over domain
// note: need to be careful with custom reduction ops
// openmp only understands +, *, min, max out of the box

// for now, use critical section (correct but not optimal)
#pragma omp parallel
                {
                    T local_acc = init;

#pragma omp for
                    for (std::uint64_t ii = 0; ii < n; ++ii) {
                        auto coord = domain.linear_to_coord(ii);
                        local_acc  = op(local_acc, transform(comp(coord)));
                    }

#pragma omp critical
                    {
                        acc = op(acc, local_acc);
                    }
                }
            }
            else {
                // fallback to serial
                for (std::uint64_t ii = 0; ii < n; ++ii) {
                    auto coord = domain.linear_to_coord(ii);
                    acc        = op(acc, transform(comp(coord)));
                }
            }
        }
        else {
            // serial for small domains
            for (std::uint64_t ii = 0; ii < n; ++ii) {
                auto coord = domain.linear_to_coord(ii);
                acc        = op(acc, transform(comp(coord)));
            }
        }

        *result = acc;
    }

}   // namespace simbi::het::backend::cpu

#endif
