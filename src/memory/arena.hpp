#ifndef ARENA_HPP
#define ARENA_HPP

#include "device.hpp"
#include "memory/smart_ptr.hpp"
#include "memory_block.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simbi::mem {

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
        // reduced number of buckets (power of 2 sizes)
        static constexpr int MAX_BUCKETS = 31;   // handles up to 2^30 elements

        // storage for each bucket size
        struct bucket_entry_t {
            memory_block_t block;
            T* typed_ptr;

            bucket_entry_t(memory_block_t&& b)
                : block(std::move(b)), typed_ptr(static_cast<T*>(block.data))
            {
            }

            // no copy
            bucket_entry_t(const bucket_entry_t&)            = delete;
            bucket_entry_t& operator=(const bucket_entry_t&) = delete;

            // move only
            bucket_entry_t(bucket_entry_t&& other) noexcept
                : block(std::move(other.block)), typed_ptr(other.typed_ptr)
            {
                other.typed_ptr = nullptr;
            }

            bucket_entry_t& operator=(bucket_entry_t&& other) noexcept
            {
                if (this != &other) {
                    block           = std::move(other.block);
                    typed_ptr       = other.typed_ptr;
                    other.typed_ptr = nullptr;
                }
                return *this;
            }
        };

        std::vector<bucket_entry_t> buckets[MAX_BUCKETS];
        std::mutex mutex;
        device_t dev_;

        // private constructor - use factory function
        explicit arena_t(device_t dev) : dev_(dev) {}

        // calculate bucket index for count
        static int bucket_for(std::size_t count)
        {
            if (count <= 1) {
                return 0;
            }

            // find highest bit position
            const int bit_pos = 64 - __builtin_clzl(count - 1);
            return std::min(bit_pos, MAX_BUCKETS - 1);
        }

        static std::size_t bucket_size(int bucket)
        {
            return std::size_t{1} << bucket;
        }

      public:
        // factory method - arena must be managed by shared_ptr
        static std::shared_ptr<arena_t> create(device_t dev)
        {
            return std::shared_ptr<arena_t>(new arena_t(dev));
        }

        mem::shared_ptr<T> get(std::size_t count)
        {
            if (count == 0) {
                return nullptr;
            }

            const int bucket              = bucket_for(count);
            const std::size_t actual_size = bucket_size(bucket);
            const std::size_t bytes       = actual_size * sizeof(T);

            // try to reuse from bucket
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto& bucket_pool = buckets[bucket];

                if (!bucket_pool.empty()) {
                    T* ptr = bucket_pool.back().typed_ptr;
                    bucket_pool.pop_back();

                    // create shared_ptr with custom deleter that returns to
                    // arena
                    auto self = this->shared_from_this();
                    return mem::shared_ptr(ptr, [self, bucket, bytes](T* ptr) {
                        if (!ptr) {
                            return;
                        }

                        try {
                            std::lock_guard<std::mutex> lock(self->mutex);

                            // create memory block from pointer
                            memory_block_t block(ptr, bytes, self->dev_);

                            // add to bucket
                            self->buckets[bucket].emplace_back(
                                std::move(block)
                            );
                        }
                        catch (...) {
                            // fallback: just delete if bucket return fails
                            memory_block_t block(ptr, bytes, self->dev_);
                            // block destructor will free memory
                        }
                    });
                }
            }

            memory_block_t block = memory_block_t::allocate(bytes, dev_);
            T* typed_ptr         = static_cast<T*>(block.data);

            // detach block from RAII to prevent double-free
            block.data = nullptr;
            block.size = 0;

            // return with custom deleter
            auto self = this->shared_from_this();
            return mem::shared_ptr(typed_ptr, [self, bucket, bytes](T* ptr) {
                if (!ptr) {
                    return;
                }

                try {
                    std::lock_guard<std::mutex> lock(self->mutex);

                    // create memory block from pointer
                    memory_block_t block(ptr, bytes, self->dev_);

                    // add to bucket
                    self->buckets[bucket].emplace_back(std::move(block));
                }
                catch (...) {
                    // fallback: just delete if bucket return fails
                    memory_block_t block(ptr, bytes, self->dev_);
                    // block destructor will free memory
                }
            });
        }

        // clear all pooled memory
        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& bucket : buckets) {
                bucket.clear();
            }
        }

        device_t device() const { return dev_; }
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

}   // namespace simbi::mem

#endif   // ARENA_HPP
