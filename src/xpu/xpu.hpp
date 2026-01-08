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
#include "device/domain_partition.hpp"
#include "execution/execution_space.hpp"
#include "grid/domain.hpp"

// execution framework
#include "execution/cpu_space.hpp"
#include "execution/cuda_space.hpp"
#include "execution/executor.hpp"
#include "execution/executor_arena.hpp"
#include "execution/token.hpp"

// memory management - order matters for complete types
#include "mem/device_memory.hpp"
#include "mem/host_memory.hpp"
#include "mem/memory_config.hpp"
#include "mem/memory_space.hpp"
#include "mem/ops.hpp"
#include "mem/unified_memory.hpp"
#include "mem/view.hpp"

// new hesi-style memory system
#include "mem/mem.hpp"

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

// memory space selection based on build configuration
#ifdef XPU_CUDA_AVAILABLE
    using default_memory_space = sim_memory_space;
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
    // new memory system integration
    // =============================================================================

    // import mem namespace for convenience

    // aliases for the new memory system
    template <typename T>
    using shared_handle_t = mem::shared_handle_t<T>;

    template <typename MemorySpace>
    using memory_block_t = mem::memory_block_t<MemorySpace>;

    // convenience aliases for configured memory space
    using sim_block_t = mem::memory_block_t<sim_memory_space>;

    // factory functions
    template <typename T, typename... Args>
    auto make_shared_handle(Args&&... args)
    {
        return mem::shared_handle_t<T>::make(std::forward<Args>(args)...);
    }

    template <typename T, typename MemorySpace = sim_memory_space>
    auto make_memory_block(std::size_t count)
    {
        return mem::make_block<T, MemorySpace>(count);
    }

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
