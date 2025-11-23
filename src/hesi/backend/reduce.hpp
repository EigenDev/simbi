#ifndef HET_BACKEND_REDUCE_HPP
#define HET_BACKEND_REDUCE_HPP

#include "compat.hpp"
#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"

#include <cstdint>
#include <stdexcept>

// forward declarations
namespace simbi::het::backend::cpu {
    template <typename T, typename BinaryOp>
    void reduce_range(const T*, T*, std::uint64_t, T, BinaryOp, T, bool);

    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    void
    transform_reduce(const Computation&, T*, T, TransformOp, BinaryOp, T, bool);
}   // namespace simbi::het::backend::cpu

#ifdef CUDA_ENABLED
namespace simbi::het::backend::cuda {
    template <typename T, typename BinaryOp>
    void reduce_range(void*, const T*, T*, std::uint64_t, T, BinaryOp, T);

    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    void transform_reduce(
        void*,
        const Computation&,
        T*,
        T,
        TransformOp,
        BinaryOp,
        T
    );
}   // namespace simbi::het::backend::cuda
#endif

namespace simbi::het::backend {
    // reduce_range dispatcher
    template <typename T, typename BinaryOp>
    void reduce_range(
        backend_type_t backend,
        stream_handle_t stream,
        const T* input,
        T* output,
        std::uint64_t n,
        T init,
        BinaryOp op,
        T identity
    )
    {
        switch (backend) {
            case backend_type_t::cpu:
                (void) stream;   // unused
                cpu::reduce_range(
                    input,
                    output,
                    n,
                    init,
                    op,
                    identity,
                    global::use_omp
                );
                break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: {
                auto cuda_stream = static_cast<cudaStream_t>(stream);
                cuda::reduce_range(
                    cuda_stream,
                    input,
                    output,
                    n,
                    init,
                    op,
                    identity
                );
                break;
            }
#endif

            default:
                throw std::runtime_error(
                    "unsupported backend for reduce_range"
                );
        }
    }

    // transform_reduce dispatcher
    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    void transform_reduce(
        backend_type_t backend,
        stream_handle_t stream,
        const Computation& comp,
        T* result,
        T init,
        TransformOp transform,
        BinaryOp op,
        T identity
    )
    {
        switch (backend) {
            case backend_type_t::cpu:
                (void) stream;   // unused
                cpu::transform_reduce(
                    comp,
                    result,
                    init,
                    transform,
                    op,
                    identity,
                    global::use_omp
                );
                break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: {
                auto cuda_stream = static_cast<cudaStream_t>(stream);
                cuda::transform_reduce(
                    cuda_stream,
                    comp,
                    result,
                    init,
                    transform,
                    op,
                    identity
                );
                break;
            }
#endif

            default:
                throw std::runtime_error(
                    "unsupported backend for transform_reduce"
                );
        }
    }

}   // namespace simbi::het::backend

// include implementations
#include "hesi/cpu/reduce.hpp"

#ifdef CUDA_ENABLED
#include "hesi/cuda/reduce.cuh"
#endif

#endif
