// =============================================================================
// ops.hpp
//
// generic memory operations for different memory spaces.
// provides functions like `zero_memory`, `fill_memory`, and `copy_memory` that
// are templated on the memory space, allowing them to work correctly for
// host, device, or unified memory by dispatching to the appropriate backend
// (e.g., `memset` vs. `cudaMemset`).
//
// usage:
//   zero_memory(my_device_block);
//   fill_memory(my_host_block, 42);
// =============================================================================
#pragma once

#include "block.hpp"
#include "xpu/mem/device_memory.hpp"
#include "xpu/mem/host_memory.hpp"
#include "xpu/mem/unified_memory.hpp"

#include <algorithm>
#include <cstring>

namespace simbi::xpu::mem {

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
        else if constexpr (std::is_same_v<MemorySpace, device_memory_t>) {
            // device memory: use device memset
            device_memory_t::memset(block.data(), 0, block.size());
        }
    }

    template <typename T, typename MemorySpace>
    void fill_memory(memory_block_t<MemorySpace>& block, const T& value)
    {
        if (block.empty()) {
            return;
        }

        const auto count     = block.size() / sizeof(T);
        auto*      typed_ptr = block.template as<T>();

        if constexpr (std::is_same_v<MemorySpace, host_memory_t> ||
                      std::is_same_v<MemorySpace, unified_memory_t>) {
            // host-accessible: direct fill
            std::fill_n(typed_ptr, count, value);
        }
        else if constexpr (std::is_same_v<MemorySpace, device_memory_t>) {
            // device memory: stage through host
            auto  temp_block = memory_block_t<host_memory_t>(block.size());
            auto* temp_ptr   = temp_block.template as<T>();

            std::fill_n(temp_ptr, count, value);
            device_memory_t::memcpy_from_host(block.data(), temp_block.data(), block.size());
        }
    }

    template <typename T, typename MemorySpace>
    void copy_memory(const memory_block_t<MemorySpace>& src, memory_block_t<MemorySpace>& dst)
    {
        if (src.empty() || dst.empty()) {
            return;
        }

        const auto copy_size = std::min(src.size(), dst.size());

        if constexpr (std::is_same_v<MemorySpace, host_memory_t> ||
                      std::is_same_v<MemorySpace, unified_memory_t>) {
            // host-accessible: direct copy
            std::memcpy(dst.data(), src.data(), copy_size);
        }
        else if constexpr (std::is_same_v<MemorySpace, device_memory_t>) {
            // device memory: device-to-device copy
            device_memory_t::memcpy_device_to_device(dst.data(), src.data(), copy_size);
        }
    }

} // namespace simbi::xpu::mem
