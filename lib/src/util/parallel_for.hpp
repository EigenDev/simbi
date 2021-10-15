#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "util/launch.hpp"
#include "util/range.hpp"
#include "device_api.hpp"
#include <omp.h>
namespace simbi 
{
	// class ExecutionPolicy; // forward decl

	template <typename index_type, typename F>
	void parallel_for(index_type first, index_type last, F function)  {
		ExecutionPolicy p(last - first);
		parallel_for(p, first, last, function);
	}

	template <typename index_type, typename F>
	void parallel_for(const ExecutionPolicy<> &p, index_type first, index_type last, F function) {
		simbi::launch(p, [=] GPU_LAMBDA () {
			
			if constexpr(BuildPlatform ==Platform::GPU)
			{
				for (auto idx : range(first, last, globalThreadXCount()))  function(idx);
			} else {
				#pragma omp parallel for schedule(static) 
				for(auto idx = first; idx < last; idx++) function(idx);
			}
			
		});
			
	}
}

#endif
