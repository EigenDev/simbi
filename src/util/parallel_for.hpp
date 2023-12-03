#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "build_options.hpp"   // for global::BuildPlatform, GPU_LAMBDA, Platform ...
#include "thread_pool.hpp"        // for (anonymous), ThreadPool, get_nthreads
#include "util/exec_policy.hpp"   // for ExecutionPolicy
#include "util/launch.hpp"        // for launch
#if GPU_CODE
    #include "util/range.hpp"   // for range
#endif

namespace simbi {
    template <typename index_type, typename F>
    void parallel_for(index_type first, index_type last, F function)
    {
        ExecutionPolicy p(last - first);
        parallel_for(p, first, last, function);
    }

    template <typename index_type,
              typename F,
              global::Platform P = global::BuildPlatform>
    void parallel_for(const ExecutionPolicy<>& p,
                      index_type first,
                      index_type last,
                      F function)
    {
        simbi::launch(p, [=] GPU_LAMBDA() {
#if GPU_CODE
            for (auto idx : range(first, last, globalThreadCount()))
                function(idx);
#else	
				if (global::use_omp) {
    #pragma omp parallel for schedule(static) 
					for(auto idx = first; idx < last; idx++) function(idx);
				} else {
					// singleton instance of thread pool. lazy-evaluated
    				static auto &thread_pool = simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
					thread_pool.parallel_for(first, last, function);
				}
#endif
        });
    }
}   // namespace simbi

#endif
