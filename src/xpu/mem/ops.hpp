// =============================================================================
// ops.hpp
//
// simple memory operations for xpu framework.
// one job: fill, zero, copy operations. no complex transfers, no async.
// follows hesi pattern of minimal, focused components.
//
// usage:
//   zero_memory(block);
//   fill_memory(block, value);
//   copy_memory(src_block, dst_block);
// =============================================================================

#pragma once

#include "block.hpp"
#include "xpu/mem/device_memory.hpp"
#include "xpu/mem/host_memory.hpp"
#include "xpu/mem/unified_memory.hpp"

#include <algorithm>
#include <cstring>

namespace simbi::xpu::mem {

    // =============================================================================
    // zero operations
    // =============================================================================

    template <typename MemorySpace>
    void zero_memory(memory_block_t<MemorySpace>& block)
    {
        if (block.empty()) {
            return;
        }

        if constexpr (std::is_same_v<MemorySpace, host_memory_t> ||
                      std::is_same_v<MemorySpace, unified_memory_t>) {
            // host-accessible: direct memset
            std::memset(block.data(), 0, block.size());
        }
        else if constexpr (std::is_same_v<MemorySpace, device_memory>) {
            // device memory: use device memset
            device_memory_t::memset(block.data(), 0, block.size());
        }
    }

    // =============================================================================
    // fill operations
    // =============================================================================

    template <typename T, typename MemorySpace>
    void fill_memory(memory_block_t<MemorySpace>& block, const T& value)
    {
        if (block.empty()) {
            return;
        }

        const auto count     = block.size() / sizeof(T);
        auto*      typed_ptr = block.template as<T>();

        if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                      std::is_same_v<MemorySpace, unified_memory>) {
            // host-accessible: direct fill
            std::fill_n(typed_ptr, count, value);
        }
        else if constexpr (std::is_same_v<MemorySpace, device_memory>) {
            // device memory: stage through host
            auto  temp_block = memory_block_t<host_memory>(block.size());
            auto* temp_ptr   = temp_block.template as<T>();

            std::fill_n(temp_ptr, count, value);
            device_memory_t::memcpy_from_host(block.data(), temp_block.data(), block.size());
        }
    }

    // =============================================================================
    // copy operations - same memory space only
    // =============================================================================

    template <typename T, typename MemorySpace>
    void copy_memory(const memory_block_t<MemorySpace>& src, memory_block_t<MemorySpace>& dst)
    {
        if (src.empty() || dst.empty()) {
            return;
        }

        const auto copy_size = std::min(src.size(), dst.size());

        if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                      std::is_same_v<MemorySpace, unified_memory>) {
            // host-accessible: direct copy
            std::memcpy(dst.data(), src.data(), copy_size);
        }
        else if constexpr (std::is_same_v<MemorySpace, device_memory>) {
            // device memory: device-to-device copy
            device_memory_t::memcpy_device_to_device(dst.data(), src.data(), copy_size);
        }
    }

} // namespace simbi::xpu::mem
