#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "util/launch.hpp"
#include "util/range.hpp"
#include "device_api.hpp"
#include "thread_pool.hpp"

namespace simbi 
{
	template <typename index_type, typename F>
	void parallel_for(index_type first, index_type last, F function)  {
		ExecutionPolicy p(last - first);
		parallel_for(p, first, last, function);
	}

	template <typename index_type, typename F, Platform P = BuildPlatform>
	void parallel_for(const ExecutionPolicy<> &p, index_type first, index_type last, F function) {
		simbi::launch(p, [=] GPU_LAMBDA () {
			#if GPU_CODE
				for (auto idx : range(first, last, globalThreadCount()))  function(idx);
			#else	
				thread_pool.parallel_for(first, last, function);
				// #pragma omp parallel for schedule(static) 
				// for(auto idx = first; idx < last; idx++) function(idx);
			#endif
			
		});
			
	}
}

#endif
