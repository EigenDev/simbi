#ifndef ARENA_HPP
#define ARENA_HPP

#include "compat.hpp"
#include "hetero/adapter.hpp"
#include "memory/device.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simbi::mem {

    struct memory_stats_t {
        std::size_t allocated;
        std::size_t peak_allocated;
        std::size_t arena_size;
    };

    // forward declaration
    template <typename T>
    class arena_t;

    template <typename T>
    std::shared_ptr<arena_t<T>> arena(device_t dev);

    // memory arena with bucketing for efficient reuse
    template <typename T>
    class arena_t : public std::enable_shared_from_this<arena_t<T>>
    {
      private:
        // 31 buckets handle sizes from 2^0 to 2^30 elements
        // 2^30 * sizeof(T) = max ~8GB for 8-byte types
        static constexpr int MAX_BUCKETS = 31;

        std::vector<hetero::memory> buckets[MAX_BUCKETS];
        std::mutex mutex;
        device_t dev_;

        explicit arena_t(device_t dev) : dev_(dev) {}

        static int bucket_for(std::size_t count)
        {
            if (count <= 1) {
                return 0;
            }
            const int bit_pos = 64 - __builtin_clzl(count - 1);
            return std::min(bit_pos, MAX_BUCKETS - 1);
        }

        static std::size_t bucket_size(int bucket)
        {
            return std::size_t{1} << bucket;
        }

      public:
        static std::shared_ptr<arena_t> create(device_t dev)
        {
            return std::shared_ptr<arena_t>(new arena_t(dev));
        }

        std::shared_ptr<T[]> get(std::size_t count)
        {
            if (count == 0) {
                return nullptr;
            }

            const int bucket              = bucket_for(count);
            const std::size_t actual_size = bucket_size(bucket);
            const std::size_t bytes       = actual_size * sizeof(T);

            // try to reuse from bucket
            {
                std::lock_guard lock(mutex);
                auto& pool = buckets[bucket];

                if (!pool.empty()) {
                    auto memory = std::move(pool.back());
                    pool.pop_back();

                    T* ptr    = static_cast<T*>(memory.data());
                    auto self = this->shared_from_this();

                    return std::shared_ptr<T[]>(
                        ptr,
                        [self, bucket, memory = std::move(memory)](T*) mutable {
                            std::lock_guard lock(self->mutex);
                            self->buckets[bucket].push_back(std::move(memory));
                        }
                    );
                }
            }

            // allocate new memory
            if (dev_.is_gpu) {
                hetero::device::set_device(dev_.device_id);
            }

            auto memory = [this, bytes]() {
                using alloc_type = hetero::device_memory::alloc_type;
                if (dev_.is_gpu) {
                    return hetero::device_memory(bytes, alloc_type::device);
                }
                else {
                    return hetero::device_memory(bytes, alloc_type::host);
                }
            }();

            T* ptr    = static_cast<T*>(memory.data());
            auto self = this->shared_from_this();

            return std::shared_ptr<T[]>(
                ptr,
                [self, bucket, memory = std::move(memory)](T*) mutable {
                    std::lock_guard lock(self->mutex);
                    self->buckets[bucket].push_back(std::move(memory));
                }
            );
        }

        void clear()
        {
            std::lock_guard lock(mutex);
            for (auto& bucket : buckets) {
                bucket.clear();
            }
        }

        device_t device() const { return dev_; }

        memory_stats_t stats() const
        {
            memory_stats_t stats{0, 0, 0};

            std::lock_guard lock(mutex);
            for (int bucket = 0; bucket < MAX_BUCKETS; ++bucket) {
                const std::size_t bucket_sz = bucket_size(bucket) * sizeof(T);
                const std::size_t count     = buckets[bucket].size();

                stats.allocated += count * bucket_sz;
                stats.peak_allocated += count * bucket_sz;   // simplistic
                stats.arena_size += count * bucket_sz;
            }

            return stats;
        }

        void release_unused()
        {
            std::lock_guard lock(mutex);
            for (auto& bucket : buckets) {
                bucket.clear();
            }
        }
    };

    // global arenas for each device and type
    template <typename T>
    std::shared_ptr<arena_t<T>> arena(device_t dev)
    {
        // map of arenas per device
        static std::unordered_map<device_t, std::shared_ptr<arena_t<T>>> arenas;
        static std::mutex arena_mutex;

        std::lock_guard<std::mutex> lock(arena_mutex);

        auto it = arenas.find(dev);
        if (it == arenas.end()) {
            arenas[dev] = arena_t<T>::create(dev);
        }

        return arenas[dev];
    }

    template <typename T>
    std::shared_ptr<arena_t<T>> cpu_arena()
    {
        return arena<T>(device_t::cpu());
    }

    template <typename T>
    std::shared_ptr<arena_t<T>> gpu_arena(int device_id)
    {
        return arena<T>(device_t::gpu(device_id));
    }

    template <typename T>
    std::shared_ptr<arena_t<T>> default_arena()
    {
        if constexpr (platform::is_gpu) {
            return gpu_arena<T>(0);
        }
        else {
            return cpu_arena<T>();
        }
    }

}   // namespace simbi::mem

#endif   // ARENA_HPP
