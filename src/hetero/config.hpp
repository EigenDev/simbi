#ifndef HETERO_CONFIG_HPP
#define HETERO_CONFIG_HPP

#include "core/backend_traits.hpp"

namespace simbi::hetero {

#ifdef CUDA_ENABLED
    using default_backend_t = cuda_backend_t;
#elif defined(HIP_ENABLED)
    using default_backend_t = hip_backend_t;
#elif defined(SYCL_ENABLED)
    using default_backend_t = sycl_backend_t;
#elif defined(METAL_ENABLED)
    using default_backend_t = metal_backend_t;
#else
    using default_backend_t = cpu_backend_t;
#endif

}   // namespace simbi::hetero

#endif
