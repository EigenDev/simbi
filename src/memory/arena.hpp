#ifndef ARENA_HPP
#define ARENA_HPP

#include "adapter/device_adapter_api.hpp"
#include "memory/device.hpp"
#include "smart_ptr.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace simbi::mem {
    /**
     * arena_t - lightweight bucketed memory pool
     *
     * Features:
     * - O(1) allocation/deallocation via power-of-2 buckets
     * - Thread-safe with per-bucket locking (minimal contention)
     * - Automatic cleanup via shared_ptr custom deleters
     * - Handles 1 to 2^30 elements efficiently
     * - Zero fragmentation with bucketed approach
     * - Some statistics included as well
     */
    template <typename T>
    class arena_t : public std::enable_shared_from_this<arena_t<T>>
    {
        static constexpr int MAX_BUCKETS = 31;   // handles up to 2^30 elements

        // per-bucket storage and synchronization
        std::array<std::vector<std::unique_ptr<T[]>>, MAX_BUCKETS> buckets_;
        mutable std::array<std::mutex, MAX_BUCKETS> bucket_mutexes_;

        // statistics (optional, maybe it can be disabled for performance)
        // [TODO]: consider using atomic types for stats
        mutable std::mutex stats_mutex_;
        std::size_t total_allocated_bytes_ = 0;
        std::size_t active_allocations_    = 0;

        // location
        device_id_t device_;

      public:
        // factory method - arena must be managed by shared_ptr for safe
        // deleters
        static std::shared_ptr<arena_t>
        create(device_id_t device = device_id_t::cpu_device())
        {
            return std::shared_ptr<arena_t>(new arena_t{device});
        }

        /**
         * get - allocate memory from appropriate bucket
         * returns shared_ptr with custom deleter that returns memory to
         arena
         */
        mem::shared_ptr<T> get(std::size_t count)
        {
            if (count == 0) {
                throw std::invalid_argument(
                    "arena_t::get: count cannot be zero"
                );
            }

            const int bucket              = bucket_for(count);
            const std::size_t actual_size = bucket_size(bucket);

            // try to reuse from bucket first
            {
                std::lock_guard lock(bucket_mutexes_[bucket]);
                auto& bucket_pool = buckets_[bucket];

                if (!bucket_pool.empty()) {
                    auto buffer = std::move(bucket_pool.back());
                    bucket_pool.pop_back();

                    update_stats(actual_size * sizeof(T), +1);

                    // create shared_ptr with custom deleter that returns to
                    // arena
                    auto self = this->shared_from_this();
                    return mem::shared_ptr<T>{
                      buffer.release(),
                      [self, bucket](T* ptr) {
                          self->return_to_bucket(bucket, ptr);
                      }
                    };
                }
            }

            // allocate new buffer
            auto buffer = allocate_on_device(actual_size);
            update_stats(actual_size * sizeof(T), +1);

            // return with custom deleter
            auto self = this->shared_from_this();
            return mem::shared_ptr<T>{buffer.release(), [self, bucket](T* ptr) {
                                          self->return_to_bucket(bucket, ptr);
                                      }};
        }

        /**
         * get_zeroed - allocate zero-initialized memory
         */
        mem::shared_ptr<T> get_zeroed(std::size_t count)
        {
            auto buffer                   = get(count);
            const std::size_t actual_size = bucket_size(bucket_for(count));
            std::fill_n(buffer.get(), actual_size, T{});
            return buffer;
        }

        /**
         * clear - return all pooled memory to system
         * useful for memory pressure situations
         */
        void clear()
        {
            for (int i = 0; i < MAX_BUCKETS; ++i) {
                std::lock_guard lock(bucket_mutexes_[i]);
                buckets_[i].clear();
            }
        }

        // statistics
        std::size_t total_allocated_bytes() const
        {
            std::lock_guard lock(stats_mutex_);
            return total_allocated_bytes_;
        }

        std::size_t active_allocations() const
        {
            std::lock_guard lock(stats_mutex_);
            return active_allocations_;
        }

        std::size_t pooled_buffers() const
        {
            std::size_t total = 0;
            for (int i = 0; i < MAX_BUCKETS; ++i) {
                std::lock_guard lock(bucket_mutexes_[i]);
                total += buckets_[i].size();
            }
            return total;
        }

        device_id_t device() const noexcept { return device_; }

      private:
        // private constructor - use create() factory
        arena_t() = default;

        explicit arena_t(device_id_t device) : device_(device) {}

        auto allocate_on_device(std::size_t count)
        {
            T* raw_ptr = nullptr;

            if (device_.type == device_type_t::gpu) {
                gpu::api::set_device(device_.device_id);
                gpu::api::malloc(
                    reinterpret_cast<void**>(&raw_ptr),
                    count * sizeof(T)
                );
            }
            else {
                raw_ptr = new T[count];
            }

            return mem::unique_ptr{raw_ptr, device_deleter{device_}};
        }

        struct device_deleter {
            device_id_t device;

            device_deleter() = default;
            device_deleter(device_id_t dev) : device(dev) {}

            void operator()(T* ptr) const noexcept
            {
                if (!ptr) {
                    return;
                }

                if (device.type == device_type_t::gpu) {
                    gpu::api::set_device(device.device_id);
                    gpu::api::free(ptr);
                }
                else {
                    delete[] ptr;
                }
            }
        };

        /**
         * bucket_for - determine bucket index for given element count
         * uses bit manipulation for O(1) calculation
         */
        static constexpr int bucket_for(std::size_t count) noexcept
        {
            if (count <= 1) {
                return 0;
            }

            // find highest bit position, handle potential overflow
            const int bit_pos = 64 - __builtin_clzl(count - 1);
            return std::min(bit_pos, MAX_BUCKETS - 1);
        }

        /**
         * bucket_size - get actual allocation size for bucket
         */
        static constexpr std::size_t bucket_size(int bucket) noexcept
        {
            return std::size_t{1} << bucket;
        }

        /**
         * return_to_bucket - called by shared_ptr custom deleter
         * thread-safe return of memory to appropriate bucket
         */
        void return_to_bucket(int bucket, T* ptr) noexcept
        {
            if (!ptr) {
                return;
            }

            try {
                const std::size_t size = bucket_size(bucket);

                {
                    std::lock_guard lock(bucket_mutexes_[bucket]);
                    buckets_[bucket].emplace_back(ptr);
                }

                update_stats(size * sizeof(T), -1);
            }
            catch (...) {
                // fallback: just delete if bucket return fails
                delete[] ptr;
                update_stats(bucket_size(bucket) * sizeof(T), -1);
            }
        }

        /**
         * update_stats - thread-safe statistics tracking
         */
        void update_stats(std::size_t bytes, int allocation_delta) noexcept
        {
            try {
                std::lock_guard lock(stats_mutex_);
                if (allocation_delta > 0) {
                    total_allocated_bytes_ += bytes;
                    active_allocations_ += allocation_delta;
                }
                else {
                    total_allocated_bytes_ =
                        (total_allocated_bytes_ >= bytes)
                            ? total_allocated_bytes_ - bytes
                            : 0;
                    active_allocations_ =
                        (static_cast<int>(active_allocations_) >=
                         -allocation_delta)
                            ? active_allocations_ + allocation_delta
                            : 0;
                }
            }
            catch (...) {
                // stats are non-critical, don't propagate exceptions
            }
        }
    };

    // global arena factory for convenience
    template <typename T>
    std::shared_ptr<arena_t<T>>& global_arena()
    {
        static auto instance = arena_t<T>::create();
        return instance;
    }

    template <typename T>
    std::shared_ptr<arena_t<T>>& get_arena_for_device(device_id_t device)
    {
        // separate arenas per device
        static std::map<device_id_t, std::shared_ptr<arena_t<T>>> device_arenas;
        static std::mutex arena_map_mutex;

        std::lock_guard lock(arena_map_mutex);

        auto it = device_arenas.find(device);
        if (it == device_arenas.end()) {
            device_arenas[device] = arena_t<T>::create(device);
        }

        return device_arenas[device];
    }

}   // namespace simbi::mem

#endif
