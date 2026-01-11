// =============================================================================
// transfer.hpp
//
// explicit memory transfer operations between memory spaces.
// one job: sync data between host, device, and unified memory.
// follows hesi pattern of minimal, focused components.
//
// usage:
//   sync_host_to_device(host_block, device_block);
//   sync_device_to_host(device_block, host_block);
//   sync_unified_to_host(unified_block, host_block);
// =============================================================================

#pragma once

#include "block.hpp"
#include "xpu/mem/device_memory.hpp"
#include "xpu/mem/host_memory.hpp"
#include "xpu/mem/unified_memory.hpp"

#include <cstring>
#include <type_traits>

namespace simbi::xpu::mem {

    // =============================================================================
    // host to device transfers
    // =============================================================================

    template <typename T>
    void sync_host_to_device(
        const memory_block_t<host_memory_t>& src,
        memory_block_t<device_memory_t>&     dst,
        std::size_t                          count
    )
    {
        device_memory_t::memcpy_from_host(
            dst.template as<T>(),
            src.template as<T>(),
            count * sizeof(T)
        );
    }

    // =============================================================================
    // device to host transfers
    // =============================================================================

    template <typename T>
    void sync_device_to_host(
        const memory_block_t<device_memory_t>& src,
        memory_block_t<host_memory_t>&         dst,
        std::size_t                            count
    )
    {
        device_memory_t::memcpy_to_host(
            dst.template as<T>(),
            src.template as<T>(),
            count * sizeof(T)
        );
    }

    // =============================================================================
    // device to device transfers
    // =============================================================================

    template <typename T>
    void sync_device_to_device(
        const memory_block_t<device_memory_t>& src,
        memory_block_t<device_memory_t>&       dst,
        std::size_t                            count
    )
    {
        device_memory_t::memcpy_device_to_device(
            dst.template as<T>(),
            src.template as<T>(),
            count * sizeof(T)
        );
    }

    // =============================================================================
    // unified memory transfers - direct copy (already unified)
    // =============================================================================

    template <typename T>
    void sync_unified_to_unified(
        const memory_block_t<unified_memory_t>& src,
        memory_block_t<unified_memory_t>&       dst,
        std::size_t                             count
    )
    {
        std::memcpy(dst.template as<T>(), src.template as<T>(), count * sizeof(T));
    }

    template <typename T>
    void sync_unified_to_host(
        const memory_block_t<unified_memory_t>& src,
        memory_block_t<host_memory_t>&          dst,
        std::size_t                             count
    )
    {
        std::memcpy(dst.template as<T>(), src.template as<T>(), count * sizeof(T));
    }

    template <typename T>
    void sync_host_to_unified(
        const memory_block_t<host_memory_t>& src,
        memory_block_t<unified_memory_t>&    dst,
        std::size_t                          count
    )
    {
        std::memcpy(dst.template as<T>(), src.template as<T>(), count * sizeof(T));
    }

    // =============================================================================
    // convenience wrappers - auto-detect count from block size
    // =============================================================================

    template <typename T, typename SrcSpace, typename DstSpace>
    void sync_memory(const memory_block_t<SrcSpace>& src, memory_block_t<DstSpace>& dst)
    {
        if (src.empty() || dst.empty()) {
            return;
        }

        const auto count = std::min(src.size(), dst.size()) / sizeof(T);

        if constexpr (std::is_same_v<SrcSpace, host_memory_t> &&
                      std::is_same_v<DstSpace, device_memory_t>) {
            sync_host_to_device<T>(src, dst, count);
        }
        else if constexpr (std::is_same_v<SrcSpace, device_memory_t> &&
                           std::is_same_v<DstSpace, host_memory_t>) {
            sync_device_to_host<T>(src, dst, count);
        }
        else if constexpr (std::is_same_v<SrcSpace, device_memory_t> &&
                           std::is_same_v<DstSpace, device_memory_t>) {
            sync_device_to_device<T>(src, dst, count);
        }
        else if constexpr (std::is_same_v<SrcSpace, unified_memory_t> &&
                           std::is_same_v<DstSpace, unified_memory_t>) {
            sync_unified_to_unified<T>(src, dst, count);
        }
        else if constexpr (std::is_same_v<SrcSpace, unified_memory_t> &&
                           std::is_same_v<DstSpace, host_memory_t>) {
            sync_unified_to_host<T>(src, dst, count);
        }
        else if constexpr (std::is_same_v<SrcSpace, host_memory_t> &&
                           std::is_same_v<DstSpace, unified_memory_t>) {
            sync_host_to_unified<T>(src, dst, count);
        }
        else {
            static_assert(false, "Unsupported memory space combination");
        }
    }

} // namespace simbi::xpu::mem
