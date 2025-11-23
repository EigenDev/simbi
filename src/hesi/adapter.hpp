// =============================================================================
// SIMBI Heterogeneous Execution Adapter
//
// Unified interface for CPU and GPU execution with zero preprocessor
// visible to user code. All backend dispatch is handled internally.
//
// Usage:
//   #include "het/adapter.hpp"
//
//   het::stream_t stream(het::backend_type_t::cuda, 0);
//   het::executor_t exec(std::move(stream));
//
//   auto token = het::exec::parallel_for(
//       het::exec::gpu_t{}, exec, domain,
//       [=] DUAL(auto coord) { field(coord) = compute(coord); }
//   );
//   token.synchronize();
// =============================================================================
#ifndef HET_ADAPTER_HPP
#define HET_ADAPTER_HPP

// -----------------------------------------------------------------------------
// Core Types (backend-agnostic)
// -----------------------------------------------------------------------------
#include "hesi/config.hpp"        // default_backend_t
#include "hesi/core/traits.hpp"   // backend_traits<T>
#include "hesi/core/types.hpp"    // backend_type_t, locality_t, memory_type_t

// -----------------------------------------------------------------------------
// Memory Subsystem
// -----------------------------------------------------------------------------
#include "hesi/mem/block.hpp"   // block_t (RAII memory)
#include "hesi/mem/ops.hpp"     // copy, fill, prefetch
#include "hesi/mem/rc.hpp"      // handle_t<T> (reference counting)
#include "hesi/mem/view.hpp"    // view_t<T, Rank> (strided views)

// -----------------------------------------------------------------------------
// Execution Subsystem
// -----------------------------------------------------------------------------
#include "hesi/exec/event.hpp"      // event_t (completion markers)
#include "hesi/exec/executor.hpp"   // executor_t (main execution interface)
#include "hesi/exec/policy.hpp"     // launch_policy_t (grid/block config)
#include "hesi/exec/stream.hpp"     // stream_t (execution context)
#include "hesi/exec/token.hpp"      // token_t (async synchronization)

// -----------------------------------------------------------------------------
// High-Level Operations
// -----------------------------------------------------------------------------
#include "hesi/backend/transfer.hpp"   // backend::enable_peer_access
#include "hesi/exec/for_each.hpp"      // parallel_for
#include "hesi/exec/reduce.hpp"        // reduce, transform_reduce

// -----------------------------------------------------------------------------
// Communication (Multi-GPU/Multi-Node)
// -----------------------------------------------------------------------------
#include "hesi/comm/communicator.hpp"   // communicator_t (MPI/halo exchange)

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
// -----------------------------------------------------------------------------
// Public API Namespace
// -----------------------------------------------------------------------------
namespace simbi::het {

    // -------------------------------------------------------------------------
    // Type Aliases (user-facing names)
    // -------------------------------------------------------------------------

    // execution resources
    using stream_t   = exec::stream_t;
    using event_t    = exec::event_t;
    using executor_t = exec::executor_t;
    using token_t    = exec::token_t;
    using policy_t   = exec::launch_policy_t;

    // memory resources
    using block_t = mem::block_t;

    // views (multi-dimensional, strided)
    template <typename T, std::uint64_t Rank = 1>
    using view_t = mem::view_t<T, Rank>;

    // smart pointers (reference counted)
    template <typename T>
    using shared_handle_t = mem::handle_t<T>;

    // communication
    using communicator_t = comm::communicator_t;

    // -------------------------------------------------------------------------
    // Execution Policy Tags
    // -------------------------------------------------------------------------
    namespace policy {
        using cpu_serial = exec::cpu_serial_t;
        using openmp     = exec::openmp_t;
        using gpu        = exec::gpu_t;
        using automatic  = exec::default_t;   // runtime dispatch
    }   // namespace policy

    // -------------------------------------------------------------------------
    // Memory Operations (convenience wrappers)
    // -------------------------------------------------------------------------
    namespace memory {
        using mem::copy;
        using mem::copy_async;
        using mem::copy_raw;
        using mem::copy_raw_async;
        using mem::fill;
        using mem::fill_async;
    }   // namespace memory

    // -------------------------------------------------------------------------
    // Execution Operations
    // -------------------------------------------------------------------------
    namespace execution {
        using exec::parallel_for;
        using exec::reduce;
        using exec::transform_reduce;
    }   // namespace execution

    // -------------------------------------------------------------------------
    // Runtime Information
    // -------------------------------------------------------------------------
    namespace info {
        // backend name at compile time
        static constexpr auto backend_name =
            backend_traits<default_backend_t>::name;

        // is this a gpu backend?
        static constexpr bool is_gpu = is_gpu_backend_v<default_backend_t>;

        // warp/wavefront size
        static constexpr int warp_size =
            backend_traits<default_backend_t>::warp_size;

        // query available devices at runtime
        inline int device_count()
        {
#if defined(CUDA_ENABLED)
            int count;
            cudaGetDeviceCount(&count);
            return count;
#elif defined(HIP_ENABLED)
            int count;
            hipGetDeviceCount(&count);
            return count;
#else
            return 1;   // cpu
#endif
        }

        // query device properties
        struct device_info_t {
            std::string name;
            std::size_t total_memory;
            int compute_capability_major;
            int compute_capability_minor;
            int multiprocessor_count;
        };

        inline device_info_t query_device(int device_id)
        {
            device_info_t info;

#if defined(CUDA_ENABLED)
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_id);
            info.name                     = prop.name;
            info.total_memory             = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count     = prop.multiProcessorCount;
#elif defined(HIP_ENABLED)
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, device_id);
            info.name                     = prop.name;
            info.total_memory             = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count     = prop.multiProcessorCount;
#else
            (void) device_id;
            info.name                     = "CPU";
            info.total_memory             = 0;
            info.compute_capability_major = 0;
            info.compute_capability_minor = 0;
            info.multiprocessor_count     = 1;
#endif

            return info;
        }
    }   // namespace info

    // -------------------------------------------------------------------------
    // Global Configuration
    // -------------------------------------------------------------------------
    namespace config {
        // set default device for thread
        inline void set_device(int device_id)
        {
#if defined(CUDA_ENABLED)
            cudaSetDevice(device_id);
#elif defined(HIP_ENABLED)
            hipSetDevice(device_id);
#else
            (void) device_id;
#endif
        }

        // enable peer access between devices
        inline void enable_peer_access(int from_device, int to_device)
        {
            locality_t from_loc(backend_type_t::cuda, from_device);
            locality_t to_loc(backend_type_t::cuda, to_device);
            backend::enable_peer_access(from_loc, to_loc);
        }

        // check if peer access is available
        inline bool can_access_peer(int from_device, int to_device)
        {
            locality_t from_loc(backend_type_t::cuda, from_device);
            locality_t to_loc(backend_type_t::cuda, to_device);
            return backend::can_access_peer(from_loc, to_loc);
        }
    }   // namespace config

    // -------------------------------------------------------------------------
    // Convenience Factories
    // -------------------------------------------------------------------------

    // create stream on specific device
    inline stream_t make_stream(backend_type_t backend, int device_id = 0)
    {
        return stream_t(backend, device_id);
    }

    // create stream on default device
    inline stream_t make_default_stream()
    {
#if defined(CUDA_ENABLED)
        return stream_t(backend_type_t::cuda, 0);
#elif defined(HIP_ENABLED)
        return stream_t(backend_type_t::hip, 0);
#else
        return stream_t(backend_type_t::cpu, 0);
#endif
    }

    // allocate block on device
    template <typename T>
    block_t allocate(
        std::size_t count,
        backend_type_t backend = backend_type_t::cuda,
        int device_id          = 0,
        memory_type_t mem_type = memory_type_t::device_local
    )
    {
        locality_t loc(backend, device_id);
        return block_t(count * sizeof(T), loc, mem_type);
    }

}   // namespace simbi::het

// -----------------------------------------------------------------------------
// Backward Compatibility (optional - remove if not needed)
// -----------------------------------------------------------------------------
namespace simbi {
    // allow simbi::executor_t as alias
    using het::block_t;
    using het::executor_t;
    using het::stream_t;
    using het::token_t;
}   // namespace simbi

#endif   // HET_ADAPTER_HPP
