// =============================================================================
// xpu.hpp
//
// main header for xpu execution framework.
// provides compile-time execution space abstraction for heterogeneous computing.
// includes all core xpu components and convenient aliases.
//
// usage:
//   #include <xpu/xpu.hpp>
//   using namespace simbi::xpu;
//
//   parallel_for<cpu_space>(range, kernel);
//   executor_t<default_space> exec;
// =============================================================================

#pragma once

// core c++20 concepts and framework
#include "core/device_concepts.hpp"
#include "core/execution_concepts.hpp"
#include "core/memory_concepts.hpp"
#include "domain_partition.hpp"
#include "execution_space.hpp"
#include "grid/domain.hpp"

// temporary backward compatibility includes
#include "cpu_space.hpp"
#include "cuda_space.hpp"

// async execution framework
#include "executor.hpp"
#include "executor_arena.hpp"
#include "token.hpp"

// memory management - order matters for complete types
#include "buffer_ops.hpp"
#include "device_memory.hpp"
#include "host_memory.hpp"
#include "memory_space.hpp"
#include "shared_buffer.hpp"
#include "unified_memory.hpp"
#include "view.hpp"

// communication layer (multi-device coordination)
#include "comm/comm.hpp"

namespace simbi::xpu {

    // =============================================================================
    // default space selection (after all includes)
    // =============================================================================

// execution space selection
#ifdef XPU_CUDA_AVAILABLE
    using default_space = cuda_space;
#else
    using default_space = cpu_space;
#endif

// memory space selection
#ifdef XPU_CUDA_AVAILABLE
    using default_memory_space = unified_memory;
#else
    using default_memory_space = host_memory;
#endif

    // =============================================================================
    // convenience aliases
    // =============================================================================

    // execution spaces
    using cpu  = cpu_space;
    using cuda = cuda_space;

    // memory spaces
    using host    = host_memory;
    using device  = device_memory;
    using unified = unified_memory;

    // default spaces based on compilation flags
    using default_exec = default_space;
    using default_mem  = default_memory_space;

    // async execution
    using cpu_executor     = executor_t<cpu_space>;
    using cuda_executor    = executor_t<cuda_space>;
    using default_executor = executor_t<default_space>;

    using cpu_token     = token_t<cpu_space>;
    using cuda_token    = token_t<cuda_space>;
    using default_token = token_t<default_space>;

    // =============================================================================
    // version information
    // =============================================================================

    constexpr int version_major = 0;
    constexpr int version_minor = 1;
    constexpr int version_patch = 0;

    constexpr std::string_view version_string()
    {
        return "0.1.0";
    }

    // =============================================================================
    // compile-time feature detection
    // =============================================================================

    constexpr bool cuda_enabled()
    {
#ifdef XPU_CUDA_AVAILABLE
        return true;
#else
        return false;
#endif
    }

} // namespace simbi::xpu
