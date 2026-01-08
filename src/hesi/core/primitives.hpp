#ifndef HET_PRIMITIVES_HPP
#define HET_PRIMITIVES_HPP

#include "compat.hpp"
#include "hesi/device/context.hpp" // for grid::idx() and api::

#include <cstddef>
#include <cstdint>

// forward declarations for shuffle specializations
namespace simbi::body {
    template <std::uint64_t Rank>
    struct weighted_sums_t;
}

namespace simbi::het {

    /**
     * @brief   a c++-style, object-oriented wrapper for atomic operations.
     * it refers to a memory location, but does not own it.
     */
    template <typename T>
    class atomic_ref_t
    {
      private:
        T* ptr_;

      public:
        DEV explicit atomic_ref_t(T* ptr) : ptr_(ptr) {}

        DEV T fetch_add(T val)
        {
            return api::atomic_add(ptr_, val);
        }

        DEV T fetch_min(T val)
        {
            return api::atomic_min(ptr_, val);
        }

        // [TODO]: can add fetch_max, fetch_and, fetch_or, etc. here
    };

    /**
     * @brief   a zero-cost, type-safe wrapper for dynamic shared memory.
     * this object has no size; its address is the start of the array.
     */
    template <typename T>
    struct shared_memory_t
    {
        // provides s_mem[idx] syntax
        DEV T& operator[](std::size_t idx)
        {
            // "rehydrate" the 'this' pointer to the start of the array
            return reinterpret_cast<T*>(this)[idx];
        }

        DEV const T& operator[](std::size_t idx) const
        {
            return reinterpret_cast<const T*>(this)[idx];
        }
    };

    /**
     * @brief   an abstraction for a "warp" or "sub-group" (e.g., 32 threads).
     * this is the key to all fast, register-level data exchange.
     */
    class sub_group_t
    {
      private:
        // get the platform-specific hardware warp size
        static constexpr std::uint64_t group_size = build::constants::warp_size;

        // mask for all active threads in this warp
        // (assumes 32)
        static constexpr std::uint64_t active_mask = 0xffffffff;
        // [todo]: this should be (1ull << group_size) - 1 if warp_size is 64

      public:
        /**
         * @brief   this thread's id within the sub-group (e.g., 0-31).
         */
        DEV std::uint64_t rank() const
        {
            return dgrid::ctx().thread_id() % group_size;
        }

        /**
         * @brief   the id of this sub-group within the block.
         */
        DEV std::uint64_t id() const
        {
            return dgrid::ctx().thread_id() / group_size;
        }

        /**
         * @brief   the number of threads in this sub-group (e.g., 32 or 64).
         */
        DEV std::uint64_t size() const
        {
            return group_size;
        }

        /**
         * @brief   is this thread the leader (lane 0) of the sub-group?
         */
        DEV bool is_leader() const
        {
            return rank() == 0;
        }

        /**
         * @brief   get a value from another thread in this sub-group.
         */
        template <typename T>
        DEV T broadcast(T var, int root_lane) const
        {
#if defined(__CUDA_ARCH__)
            return __shfl_sync(active_mask, var, root_lane, group_size);
#elif defined(__HIP_DEVICE_COMPILE__)
            return __shfl(var, root_lane, group_size);
#else
            (void) root_lane;
            return var; // cpu fallback
#endif
        }

        /**
         * @brief   get a value from a thread 'offset' lanes *below* this one.
         */
        template <typename T>
        DEV T shuffle_down(T var, std::uint64_t offset) const
        {
#if defined(__CUDA_ARCH__)
            return __shfl_down_sync(active_mask, var, offset, group_size);
#elif defined(__HIP_DEVICE_COMPILE__)
            return __shfl_down(var, offset, group_size);
#else
            (void) offset;
            return var; // cpu fallback
#endif
        }

        /**
         * @brief   get a value from a thread 'offset' lanes *above* this one.
         */
        template <typename T>
        DEV T shuffle_up(T var, std::uint32_t offset) const
        {
#if defined(__CUDA_ARCH__)
            return __shfl_up_sync(active_mask, var, offset, group_size);
#elif defined(__HIP_DEVICE_COMPILE__)
            return __shfl_up(var, offset, group_size);
#else
            (void) offset;
            return var; // cpu fallback
#endif
        }

        // specialization for weighted_sums_t: shuffle each member independently
        template <std::uint64_t Rank>
        DEV body::weighted_sums_t<Rank>
            shuffle_down(body::weighted_sums_t<Rank> var, std::uint64_t offset) const
        {
            body::weighted_sums_t<Rank> result;
            result.weighted_density = shuffle_down(var.weighted_density, offset);
            result.weighted_cs      = shuffle_down(var.weighted_cs, offset);
            result.sum_weight       = shuffle_down(var.sum_weight, offset);
            result.sum_mass         = shuffle_down(var.sum_mass, offset);

            // shuffle vector components individually
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                result.weighted_v_vec[ii] = shuffle_down(var.weighted_v_vec[ii], offset);
            }

            return result;
        }
    };

    /**
     * @brief   an abstraction for a full thread block (e.g., 256 threads).
     */
    class block_group_t
    {
      public:
        /**
         * @brief   this thread's id within the block (e.g., 0-255).
         */
        DEV std::uint32_t rank() const
        {
            return dgrid::ctx().thread_id();
        }

        /**
         * @brief   the total number of threads in this block.
         */
        DEV std::uint32_t size() const
        {
            return dgrid::ctx().threads_per_block();
        }

        /**
         * @brief   is this the leader (thread 0) of the entire block?
         */
        DEV bool is_leader() const
        {
            return rank() == 0;
        }

        /**
         * @brief   get this thread's sub_group_t object.
         */
        DEV sub_group_t get_sub_group() const
        {
            return sub_group_t{};
        }

        /**
         * @brief   get the total number of sub-groups in this block.
         */
        DEV std::uint32_t num_sub_groups() const
        {
            return (size() + sub_group_t{}.size() - 1) / sub_group_t{}.size();
        }

        /**
         * @brief   synchronize all threads in this block.
         */
        DEV void sync() const
        {
            api::sync_threads();
        }
    };

    /**
     * @brief   get the sub_group_t for the currently executing thread.
     */
    DEV inline sub_group_t this_sub_group()
    {
        return sub_group_t{};
    }

    /**
     * @brief   get the block_group_t for the currently executing thread.
     */
    DEV inline block_group_t this_block()
    {
        return block_group_t{};
    }

    /**
     * @brief   performs a fast, parallel reduction *within* a sub-group.
     * uses shuffle_down for register-to-register communication.
     * only the leader (lane 0) will have the correct final result.
     */
    template <typename T, typename Reducer>
    DEV T reduce(const sub_group_t& group, T val, Reducer op)
    {
        // this is a standard log-step reduction using shuffles
        for (std::uint32_t offset = group.size() / 2; offset > 0; offset /= 2) {
            val = op(val, group.shuffle_down(val, offset));
        }
        return val;
    }

} // namespace simbi::het

#endif // HETERO_PRIMITIVES_HPP
