/**
 * config.hpp
 * configuration for parallel execution layer
 */

#ifndef SIMBI_CORE_PARALLEL_CONFIG_HPP
#define SIMBI_CORE_PARALLEL_CONFIG_HPP

#include "config.hpp"
#include "core/types/alias.hpp"
#include "system/adapter/device_adapter_api.hpp"
#include "system/adapter/device_types.hpp"
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace simbi::parallel {

    // hardware platform detection (re-exported from comp config layer)
    namespace platform {
        // cpu/gpu platform flags
        inline constexpr bool is_gpu = build::platform::is_gpu;
        inline constexpr bool is_cpu = build::platform::is_cpu;

        // specific backend flags
        inline constexpr bool is_cuda = build::platform::is_cuda;
        inline constexpr bool is_hip  = build::platform::is_hip;
    }   // namespace platform

    // execution constants
    inline constexpr std::uint64_t default_threads_per_block = 256;
    inline constexpr std::uint64_t default_tile_size_1d      = 256;
    inline constexpr std::uint64_t default_tile_size_2d      = 16;
    inline constexpr std::uint64_t default_tile_size_3d      = 8;

    // memory configuration
    enum class memory_type_t {
        device,    // regular device memory
        managed,   // unified managed memory
        host       // pinned host memory
    };

    // execution modes
    enum class execution_mode_t {
        sync,   // synchronous execution
        async   // asynchronous execution
    };

    // execution schedule type
    enum class schedule_type_t {
        automatic,   // let the system decide
        statique,    // equal chunks
        dynamic,     // work stealing
        guided       // decreasing chunk sizes
    };

    // get optimal number of threads for CPU execution
    inline std::uint64_t get_optimal_thread_count()
    {
        return std::thread::hardware_concurrency();
    }

    // get number of available devices
    inline std::uint64_t get_device_count()
    {
        std::int64_t count = 0;
        gpu::api::get_device_count(&count);
        return static_cast<std::uint64_t>(count);
    }

    // check if system supports unified memory
    inline bool supports_unified_memory()
    {
        // return true for CUDA, false for CPU and others
        return platform::is_cuda;
    }

    inline std::uint64_t get_block_dims(const std::string& key)
    {
        if (const char* val = std::getenv(key.c_str())) {
            return static_cast<std::uint64_t>(std::stoi(val));
        }
        return 1;
    }

    inline adapter::types::dim3
    get_default_block_size(std::uint64_t effective_dim)
    {
        if (effective_dim == 1) {
            const auto block_x = get_block_dims("BLOCK_X");
            return {block_x, 1, 1};
        }
        else if (effective_dim == 2) {
            const auto block_x = get_block_dims("BLOCK_X");
            const auto block_y = get_block_dims("BLOCK_Y");
            return {block_x, block_y, 1};
        }
        else if (effective_dim == 3) {
            const auto block_x = get_block_dims("BLOCK_X");
            const auto block_y = get_block_dims("BLOCK_Y");
            const auto block_z = get_block_dims("BLOCK_Z");
            return {block_x, block_y, block_z};
        }
        else {
            throw std::invalid_argument(
                "Invalid effective dimension: " + std::to_string(effective_dim)
            );
        }
    }

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_CONFIG_HPP
