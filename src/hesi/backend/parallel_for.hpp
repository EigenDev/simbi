#ifndef HET_BACKEND_PARALLEL_FOR_HPP
#define HET_BACKEND_PARALLEL_FOR_HPP

#include "compat.hpp"
#include "grid/domain.hpp"
#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/policy.hpp"

#include <cstdint>
#include <stdexcept>

// forward declare backend implementations
namespace simbi::het::backend::cpu {
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f,
        bool use_openmp
    );
}

#ifdef CUDA_ENABLED
namespace simbi::het::backend::cuda {
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        void* stream,   // cudaStream_t as void*
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f
    );
}
#endif

#ifdef HIP_ENABLED
namespace simbi::het::backend::hip {
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        void* stream,   // hipStream_t as void*
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f
    );
}
#endif

namespace simbi::het::backend {

    // main dispatcher (header-only template)
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        backend_type_t backend,
        stream_handle_t stream,
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f
    )
    {
        switch (backend) {
            case backend_type_t::cpu: {
                (void) stream;   // unused
                cpu::parallel_for(
                    policy,
                    domain,
                    std::forward<Functor>(f),
                    global::use_omp
                );
                break;
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                cuda::parallel_for(
                    stream,
                    policy,
                    domain,
                    std::forward<Functor>(f)
                );
                break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                hip::parallel_for(
                    stream,
                    policy,
                    domain,
                    std::forward<Functor>(f)
                );
                break;
#endif

            default:
                throw std::runtime_error(
                    "unsupported backend for parallel_for"
                );
        }
    }

}   // namespace simbi::het::backend

// include implementations (order matters!)
#include "hesi/cpu/parallel_for.hpp"

#ifdef CUDA_ENABLED
#include "hesi/cuda/parallel_for.hpp"
#endif

#ifdef HIP_ENABLED
#include "hesi/hip/parallel_for.hpp"
#endif

#endif
