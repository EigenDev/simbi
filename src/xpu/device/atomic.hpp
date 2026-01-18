// =============================================================================
// atomic.hpp
//
// vendor-agnostic atomic operations for device code.
// provides unified interface across cuda, hip, sycl, and cpu.
//
// design:
//   - compile-time dispatch via platform detection
//   - zero overhead - resolves to native vendor atomics
//   - type-safe templates for float, double, int, uint64_t
//   - cpu fallback for host-only builds
//
// usage:
//   DEV void kernel() {
//       xpu::atomic_add(&shared_counter, 1.0f);
//       xpu::atomic_max(&max_value, local_val);
//   }
// =============================================================================
#pragma once

#include "decorators.hpp"

#include <cstdint>

namespace simbi::xpu::device {

    // =============================================================================
    // atomic add
    // =============================================================================

    template <typename T>
    DEV inline T atomic_add(T* addr, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicAdd(addr, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicAdd(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        // sycl atomic_ref approach
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        return atomic_val.fetch_add(val);
#else
        // cpu fallback - should not be called from device
        // but needed for compilation
        *addr += val;
        return *addr;
#endif
    }

    // =============================================================================
    // atomic sub
    // =============================================================================

    template <typename T>
    DEV inline T atomic_sub(T* addr, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicSub(addr, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicSub(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        return atomic_val.fetch_sub(val);
#else
        *addr -= val;
        return *addr;
#endif
    }

    // =============================================================================
    // atomic min
    // =============================================================================

    template <typename T>
    DEV inline T atomic_min(T* addr, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicMin(addr, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicMin(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        T old = atomic_val.load();
        while (val < old && !atomic_val.compare_exchange_weak(old, val)) {
        }
        return old;
#else
        T old = *addr;
        *addr = val < old ? val : old;
        return old;
#endif
    }

    // =============================================================================
    // atomic max
    // =============================================================================

    template <typename T>
    DEV inline T atomic_max(T* addr, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicMax(addr, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicMax(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        T old = atomic_val.load();
        while (val > old && !atomic_val.compare_exchange_weak(old, val)) {
        }
        return old;
#else
        T old = *addr;
        *addr = val > old ? val : old;
        return old;
#endif
    }

    // =============================================================================
    // atomic exchange
    // =============================================================================

    template <typename T>
    DEV inline T atomic_exch(T* addr, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicExch(addr, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicExch(addr, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        return atomic_val.exchange(val);
#else
        T old = *addr;
        *addr = val;
        return old;
#endif
    }

    // =============================================================================
    // atomic compare-and-swap
    // =============================================================================

    template <typename T>
    DEV inline T atomic_cas(T* addr, T compare, T val)
    {
#if defined(__CUDA_ARCH__)
        return ::atomicCAS(addr, compare, val);
#elif defined(__HIP_DEVICE_COMPILE__)
        return ::atomicCAS(addr, compare, val);
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_val(
            *addr
        );
        atomic_val.compare_exchange_strong(compare, val);
        return compare;
#else
        T old = *addr;
        if (old == compare) {
            *addr = val;
        }
        return old;
#endif
    }

} // namespace simbi::xpu::device



