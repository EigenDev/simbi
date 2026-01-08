/*
 * compat.hpp
 *
 * This header provides a compatibility layer for different build
 * configurations and platforms. It defines compile-time constants,
 * type definitions, and macros based on the build options specified
 * during the Meson build process.
 *
 * The goal is to centralize configuration-dependent code and provide
 * a clear interface for platform-specific features.
 *
 * Created by: Marcus DuPont
 * Date: 2025-11-23
 * Copyright (c) 2025 Marcus DuPont
 * All rights reserved.
 */
#ifndef COMPAT_HPP
#define COMPAT_HPP

#include "build_options.hpp" // include the Meson-generated configuration

#include <cstdint> // for fixed-width integer types
#include <cstdlib>
#include <type_traits> // for std::conditional_t and other type traits

namespace simbi {
    namespace build {

        /**
         * platform - compile-time platform detection
         *
         * usage:
         *   if constexpr (platform::is_gpu) {
         *       // gpu-specific code
         *   }
         */
        namespace platform {
            inline constexpr bool is_gpu =
#if GPU_ENABLED
                true;
#else
                false;
#endif

            inline constexpr bool is_cuda =
#if defined(GPU_ENABLED) && defined(CUDA_ENABLED)
                true;
#else
                false;
#endif

            inline constexpr bool is_hip =
#if defined(GPU_ENABLED) && defined(HIP_ENABLED)
                true;
#else
                false;
#endif

            inline constexpr bool is_cpu = !is_gpu;

            enum class type {
                cpu,
                cuda,
                hip,
                unknown
            };

            // determine the platform at compile-time
            inline constexpr type current = is_cuda  ? type::cuda
                                            : is_hip ? type::hip
                                            : is_cpu ? type::cpu
                                                     : type::unknown;
        } // namespace platform

        /**
         * features - compile-time feature flags
         *
         * usage:
         *   if constexpr (features::column_major) {
         *       // column-major specific code
         *   }
         */
        namespace features {
            // direct boolean constants are clearer than function calls
            inline constexpr bool column_major =
#if COLUMN_MAJOR
                true;
#else
                false;
#endif

            inline constexpr bool four_velocity =
#if FOUR_VELOCITY
                true;
#else
                false;
#endif

            inline constexpr bool debug =
#ifdef DEBUG
                true;
#else
                false;
#endif

            inline constexpr bool bounds_checking =
#if BOUNDS_CHECKING
                true;
#else
                false;
#endif
        } // namespace features

        /**
         * types - compile-time type definitions based on configuration
         *
         * usage:
         *   types::real value = 3.14;
         *   types::size_t size = 42;
         */
        namespace types {
            using real =
#if FLOAT_PRECISION
                float;
#else
                double;
#endif
        } // namespace types

        /**
         * constants - hardware/algorithm constants based on configuration
         *
         * usage:
         *   for (std::int64_t i = 0; i < constants::warp_size; ++i) { ... }
         */
        namespace constants {
            // hardware-specific constants
            inline constexpr std::uint64_t warp_size = (platform::is_cuda)  ? 32
                                                       : (platform::is_hip) ? 64
                                                                            : 1;

            // numerical constants
            inline constexpr types::real epsilon =
                std::is_same_v<types::real, float> ? 1e-6f : 1e-12;

            // algorithm parameters
            inline constexpr std::int64_t max_iterations = 1000;
        } // namespace constants
    } // namespace build

    /**
     * Backward compatibility layer
     *
     * These definitions maintain compatibility with existing code
     * but should be considered deprecated. New code should use the
     * simbi::build namespace directly. [TODO: remove in future versions]
     */
    namespace global {
        // platform options
        enum class Platform : std::int64_t {
            CPU = 0,
            GPU = 1
        };

        enum class Runtime : std::int64_t {
            CUDA = 0,
            ROCM = 1,
            CPU  = 2
        };

        enum class Velocity : std::int64_t {
            Beta         = 0,
            FourVelocity = 1
        };

        // openmp flag - runtime configurable
        inline bool use_omp = false;

        // convert from new platform detection to old enum
        constexpr Platform BuildPlatform = build::platform::is_gpu ? Platform::GPU : Platform::CPU;

        constexpr Runtime BuildRuntime = build::platform::is_cuda  ? Runtime::CUDA
                                         : build::platform::is_hip ? Runtime::ROCM
                                                                   : Runtime::CPU;

        // convert from new warp size constant
        constexpr std::uint64_t WARP_SIZE = build::constants::warp_size;

        // convert from new velocity feature
        constexpr Velocity VelocityType =
            build::features::four_velocity ? Velocity::FourVelocity : Velocity::Beta;

        // other feature flag conversions
        constexpr bool debug_mode      = build::features::debug;
        constexpr bool bounds_checking = build::features::bounds_checking;
        constexpr bool col_major       = build::features::column_major;

        // epsilon conversion
        constexpr build::types::real epsilon = build::constants::epsilon;

        // four velocity flag - this is redundant and should be removed
        constexpr bool using_four_velocity = build::features::four_velocity;

        // these shortcuts are actually useful - keep them
        constexpr bool on_gpu = build::platform::is_gpu;
    } // namespace global

    // Expose real in global namespace for backward compatibility
    // but this is bad practice - [TODO] consider removing in future versions
    using real = build::types::real;
} // namespace simbi

using namespace simbi::build::types;
using namespace simbi::build::constants;
using namespace simbi::build::features;
using namespace simbi::build;
/**
 * Macro definitions based on platform
 *
 * These macros provide decorators for code that needs to run on different
 * platforms. However, macros should be minimized in modern C++ - consider
 * alternatives where possible.
 */
#if GPU_ENABLED
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
#define DEV    __device__
#define KERNEL __global__
#define DUAL   __host__ __device__

#if CUDA_ENABLED
#include <cuda_runtime.h>
#elif HIP_ENABLED
#include <hip/hip_runtime.h>
#endif

#else
// fallbacks for other GPU platforms - should never reach here
#define DEV
#define KERNEL
#define DUAL
#endif
#else
// CPU mode
#define DEV
#define KERNEL
#define DUAL
#endif

#endif // CONFIG_HPP
