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
#include "core/types.hpp"
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
#include "mem/block.hpp"
#include "mem/device_memory.hpp"
#include "mem/handle.hpp"
#include "mem/host_memory.hpp"
#include "mem/memory_config.hpp"
#include "mem/memory_space.hpp"
#include "mem/ops.hpp"
#include "mem/unified_memory.hpp"
#include "mem/view.hpp"

//
#include "mem/mem.hpp"

// communication layer (multi-device coordination)
#include "comm/comm.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace simbi::xpu {

    // =============================================================================
    // default space selection (after all includes)
    // =============================================================================

// execution space selection
#ifdef XPU_CUDA_AVAILABLE
    using default_space = exec::cuda_space;
#else
    using default_space = exec::cpu_space;
#endif

// memory space selection based on build configuration
#ifdef XPU_CUDA_AVAILABLE
    using default_memory_space = mem::sim_memory_space;
#else
    using default_memory_space = mem::host_memory;
#endif

    // =============================================================================
    // convenience aliases
    // =============================================================================

    // execution spaces
    using cpu_space_t  = exec::cpu_space;
    using cuda_space_t = exec::cuda_space;

    // memory spaces
    using host_memory_t    = mem::host_memory_t;
    using device_memory_t  = mem::device_memory_t;
    using unified_memory_t = mem::unified_memory_t;

    // default spaces based on compilation flags
    using default_exec_t = default_space;
    using default_mem_t  = default_memory_space;

    // async execution
    using cpu_executor_t     = exec::executor_t<cpu_space_t>;
    using cuda_executor_t    = exec::executor_t<cuda_space_t>;
    using default_executor_t = exec::executor_t<default_space>;

    using cpu_token     = exec::token_t<cpu_space_t>;
    using cuda_token    = exec::token_t<cuda_space_t>;
    using default_token = exec::token_t<default_space>;

    using core::execution_space_c;

    template <typename ExecSpace>
    using executor_t = exec::executor_t<ExecSpace>;

    template <execution_space_c ExecSpace>
    using token_t = exec::token_t<ExecSpace>;

    // =============================================================================
    // memory system integration
    // =============================================================================

    // aliases for the memory system
    template <typename T>
    using shared_handle_t = mem::shared_handle_t<T>;

    template <typename MemorySpace>
    using memory_block_t = mem::memory_block_t<MemorySpace>;

    // convenience aliases for configured memory space
    using sim_block_t = mem::memory_block_t<mem::sim_memory_space>;

    // factory functions
    template <typename T, typename... Args>
    auto make_shared_handle(Args&&... args)
    {
        return mem::shared_handle_t<T>::make(std::forward<Args>(args)...);
    }

    template <typename T, typename MemorySpace = mem::sim_memory_space>
    auto make_memory_block(std::size_t count)
    {
        return mem::make_block<T, MemorySpace>(count);
    }

    template <typename T, std::uint64_t Rank>
    using view_t = mem::view_t<T, Rank>;

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

    inline void synchronize()
    {
#ifdef XPU_CUDA_AVAILABLE
        cudaDeviceSynchronize();
#elif defined(XPU_HIP_AVAILABLE)
        hipDeviceSynchronize();
#else
// no-op for CPU
#endif
    }

} // namespace simbi::xpu
