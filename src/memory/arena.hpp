#ifndef ARENA_HPP
#define ARENA_HPP

#include <algorithm>
#include <array>
#include <cstddef>
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
     * - No statistics overhead for maximum performance
     */
    template <typename T>
    class arena_t : public std::enable_shared_from_this<arena_t<T>>
    {
        static constexpr int MAX_BUCKETS = 31;   // handles up to 2^30 elements

        // per-bucket storage and synchronization
        std::array<std::vector<std::unique_ptr<T[]>>, MAX_BUCKETS> buckets_;
        mutable std::array<std::mutex, MAX_BUCKETS> bucket_mutexes_;

      public:
        // factory method - arena must be managed by shared_ptr for safe
        // deleters
        static std::shared_ptr<arena_t> create()
        {
            return std::shared_ptr<arena_t>(new arena_t{});
        }

        /**
         * get - allocate memory from appropriate bucket
         * returns shared_ptr with custom deleter that returns memory to arena
         */
        std::shared_ptr<T[]> get(std::size_t count)
        {
            if (count == 0) [[unlikely]] {
                throw std::invalid_argument(
                    "arena_t::get: count cannot be zero"
                );
            }

            const int bucket = bucket_for(count);

            // try to reuse from bucket first
            {
                std::lock_guard lock(bucket_mutexes_[bucket]);
                auto& bucket_pool = buckets_[bucket];

                if (!bucket_pool.empty()) [[likely]] {
                    auto buffer = std::move(bucket_pool.back());
                    bucket_pool.pop_back();

                    // create shared_ptr with custom deleter
                    auto self = this->shared_from_this();
                    return {buffer.release(), [self, bucket](T* ptr) noexcept {
                                self->return_to_bucket(bucket, ptr);
                            }};
                }
            }

            // allocate new buffer
            const std::size_t actual_size = bucket_size(bucket);
            auto buffer                   = std::make_unique<T[]>(actual_size);

            // return with custom deleter
            auto self = this->shared_from_this();
            return {buffer.release(), [self, bucket](T* ptr) noexcept {
                        self->return_to_bucket(bucket, ptr);
                    }};
        }

        /**
         * get_zeroed - allocate zero-initialized memory
         */
        std::shared_ptr<T[]> get_zeroed(std::size_t count)
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
        void clear() noexcept
        {
            for (int i = 0; i < MAX_BUCKETS; ++i) {
                std::lock_guard lock(bucket_mutexes_[i]);
                buckets_[i].clear();
            }
        }

        /**
         * pooled_buffers - get count of currently pooled (reusable) buffers
         */
        std::size_t pooled_buffers() const noexcept
        {
            std::size_t total = 0;
            for (int i = 0; i < MAX_BUCKETS; ++i) {
                std::lock_guard lock(bucket_mutexes_[i]);
                total += buckets_[i].size();
            }
            return total;
        }

      private:
        // private constructor - use create() factory
        arena_t() = default;

        /**
         * bucket_for - determine bucket index for given element count
         * uses bit manipulation for O(1) calculation
         */
        static constexpr int bucket_for(std::size_t count) noexcept
        {
            if (count <= 1) [[likely]] {
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
            if (!ptr) [[unlikely]] {
                return;
            }

            try {
                std::lock_guard lock(bucket_mutexes_[bucket]);
                buckets_[bucket].emplace_back(ptr);
            }
            catch (...) {
                // fallback: just delete if bucket return fails
                // this ensures we never leak memory even if bucket is full
                delete[] ptr;
            }
        }
    };

    // template <typename T>
    // class arena_t : public std::enable_shared_from_this<arena_t<T>>
    // {
    //     static constexpr int MAX_BUCKETS = 31;   // handles up to 2^30
    //     elements

    //     // per-bucket storage and synchronization
    //     std::array<std::vector<std::unique_ptr<T[]>>, MAX_BUCKETS> buckets_;
    //     mutable std::array<std::mutex, MAX_BUCKETS> bucket_mutexes_;

    //     // statistics (optional, maybe it can be disabled for performance)
    //     // [TODO]: consider using atomic types for stats
    //     mutable std::mutex stats_mutex_;
    //     std::size_t total_allocated_bytes_ = 0;
    //     std::size_t active_allocations_    = 0;

    //   public:
    //     // factory method - arena must be managed by shared_ptr for safe
    //     // deleters
    //     static std::shared_ptr<arena_t> create()
    //     {
    //         return std::shared_ptr<arena_t>(new arena_t{});
    //     }

    //     /**
    //      * get - allocate memory from appropriate bucket
    //      * returns shared_ptr with custom deleter that returns memory to
    //      arena
    //      */
    //     std::shared_ptr<T[]> get(std::size_t count)
    //     {
    //         if (count == 0) {
    //             throw std::invalid_argument(
    //                 "arena_t::get: count cannot be zero"
    //             );
    //         }

    //         const int bucket              = bucket_for(count);
    //         const std::size_t actual_size = bucket_size(bucket);

    //         // try to reuse from bucket first
    //         {
    //             std::lock_guard lock(bucket_mutexes_[bucket]);
    //             auto& bucket_pool = buckets_[bucket];

    //             if (!bucket_pool.empty()) {
    //                 auto buffer = std::move(bucket_pool.back());
    //                 bucket_pool.pop_back();

    //                 update_stats(actual_size * sizeof(T), +1);

    //                 // create shared_ptr with custom deleter that returns to
    //                 // arena
    //                 auto self = this->shared_from_this();
    //                 return {buffer.release(), [self, bucket](T* ptr) {
    //                             self->return_to_bucket(bucket, ptr);
    //                         }};
    //             }
    //         }

    //         // allocate new buffer
    //         auto buffer = std::make_unique<T[]>(actual_size);
    //         update_stats(actual_size * sizeof(T), +1);

    //         // return with custom deleter
    //         auto self = this->shared_from_this();
    //         return {buffer.release(), [self, bucket](T* ptr) {
    //                     self->return_to_bucket(bucket, ptr);
    //                 }};
    //     }

    //     /**
    //      * get_zeroed - allocate zero-initialized memory
    //      */
    //     std::shared_ptr<T[]> get_zeroed(std::size_t count)
    //     {
    //         auto buffer                   = get(count);
    //         const std::size_t actual_size = bucket_size(bucket_for(count));
    //         std::fill_n(buffer.get(), actual_size, T{});
    //         return buffer;
    //     }

    //     /**
    //      * clear - return all pooled memory to system
    //      * useful for memory pressure situations
    //      */
    //     void clear()
    //     {
    //         for (int i = 0; i < MAX_BUCKETS; ++i) {
    //             std::lock_guard lock(bucket_mutexes_[i]);
    //             buckets_[i].clear();
    //         }
    //     }

    //     // statistics
    //     std::size_t total_allocated_bytes() const
    //     {
    //         std::lock_guard lock(stats_mutex_);
    //         return total_allocated_bytes_;
    //     }

    //     std::size_t active_allocations() const
    //     {
    //         std::lock_guard lock(stats_mutex_);
    //         return active_allocations_;
    //     }

    //     std::size_t pooled_buffers() const
    //     {
    //         std::size_t total = 0;
    //         for (int i = 0; i < MAX_BUCKETS; ++i) {
    //             std::lock_guard lock(bucket_mutexes_[i]);
    //             total += buckets_[i].size();
    //         }
    //         return total;
    //     }

    //   private:
    //     // private constructor - use create() factory
    //     arena_t() = default;

    //     /**
    //      * bucket_for - determine bucket index for given element count
    //      * uses bit manipulation for O(1) calculation
    //      */
    //     static constexpr int bucket_for(std::size_t count) noexcept
    //     {
    //         if (count <= 1) {
    //             return 0;
    //         }

    //         // find highest bit position, handle potential overflow
    //         const int bit_pos = 64 - __builtin_clzl(count - 1);
    //         return std::min(bit_pos, MAX_BUCKETS - 1);
    //     }

    //     /**
    //      * bucket_size - get actual allocation size for bucket
    //      */
    //     static constexpr std::size_t bucket_size(int bucket) noexcept
    //     {
    //         return std::size_t{1} << bucket;
    //     }

    //     /**
    //      * return_to_bucket - called by shared_ptr custom deleter
    //      * thread-safe return of memory to appropriate bucket
    //      */
    //     void return_to_bucket(int bucket, T* ptr) noexcept
    //     {
    //         if (!ptr) {
    //             return;
    //         }

    //         try {
    //             const std::size_t size = bucket_size(bucket);

    //             {
    //                 std::lock_guard lock(bucket_mutexes_[bucket]);
    //                 buckets_[bucket].emplace_back(ptr);
    //             }

    //             update_stats(size * sizeof(T), -1);
    //         }
    //         catch (...) {
    //             // fallback: just delete if bucket return fails
    //             delete[] ptr;
    //             update_stats(bucket_size(bucket) * sizeof(T), -1);
    //         }
    //     }

    //     /**
    //      * update_stats - thread-safe statistics tracking
    //      */
    //     void update_stats(std::size_t bytes, int allocation_delta) noexcept
    //     {
    //         try {
    //             std::lock_guard lock(stats_mutex_);
    //             if (allocation_delta > 0) {
    //                 total_allocated_bytes_ += bytes;
    //                 active_allocations_ += allocation_delta;
    //             }
    //             else {
    //                 total_allocated_bytes_ =
    //                     (total_allocated_bytes_ >= bytes)
    //                         ? total_allocated_bytes_ - bytes
    //                         : 0;
    //                 active_allocations_ =
    //                     (static_cast<int>(active_allocations_) >=
    //                      -allocation_delta)
    //                         ? active_allocations_ + allocation_delta
    //                         : 0;
    //             }
    //         }
    //         catch (...) {
    //             // stats are non-critical, don't propagate exceptions
    //         }
    //     }
    // };

    // global arena factory for convenience
    template <typename T>
    std::shared_ptr<arena_t<T>>& global_arena()
    {
        static auto instance = arena_t<T>::create();
        return instance;
    }

}   // namespace simbi::mem

#endif
