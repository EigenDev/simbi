// =============================================================================
// types.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include <cstdint>

namespace simbi::xpu::core {
    // -------------------------------------------------------------------------
    // enums
    // -------------------------------------------------------------------------

    enum class backend_type_t : std::uint8_t {
        cpu,
        cuda,
        hip,
        sycl,
        metal
    };

    enum class memory_type_t : std::uint8_t {
        host_visible, // standard ram
        pinned,       // page-locked ram (dma accessible)
        device_local, // vram (fastest, gpu only)
        managed       // unified virtual memory
    };

    enum class access_mode_t : std::uint8_t {
        read_only,
        write_only,
        read_write,
        atomic
    };

    enum class memory_direction_t : std::uint8_t {
        host_to_device,
        device_to_host,
        device_to_device,
        host_to_host
    };

    // -------------------------------------------------------------------------
    // structs
    // -------------------------------------------------------------------------

    // represents a physical location in the system
    struct locality_t
    {
        backend_type_t backend;
        std::int64_t   device_id;

        constexpr bool operator==(const locality_t& other) const
        {
            return backend == other.backend && device_id == other.device_id;
        }

        constexpr bool operator!=(const locality_t& other) const
        {
            return !(*this == other);
        }

        // helper for default host locality
        static constexpr locality_t host()
        {
            return {backend_type_t::cpu, 0};
        }
    };

    // template <std::unsigned_integral T = std::uint64_t>
    struct dim3_t
    {
        std::uint32_t x, y, z;
        constexpr dim3_t(std::uint32_t x = 1, std::uint32_t y = 1, std::uint32_t z = 1)
            : x(x), y(y), z(z)
        {
        }
        constexpr std::uint64_t volume() const
        {
            return static_cast<std::uint64_t>(x) * y * z;
        }
    };

    // for multi-dimensional indexing in functional kernels
    struct index_space_t
    {
        std::int64_t ii, jj, kk; // local index
        std::int64_t gi, gj, gk; // global index
    };

} // namespace simbi::xpu::core



