#ifndef ARENA_HPP
#define ARENA_HPP

#include "compat.hpp"
#include "hetero/adapter.hpp"
#include "memory/device.hpp"
#include "memory/smart_ptr.hpp"

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
            hetero::memory memory_;
            device_t device;

            bucket_entry_t(hetero::memory&& mem, device_t dev)
                : memory_(std::move(mem)), device(dev)
            {
            }

            // no copy
            bucket_entry_t(const bucket_entry_t&)            = delete;
            bucket_entry_t& operator=(const bucket_entry_t&) = delete;

            // move only
            bucket_entry_t(bucket_entry_t&& other) noexcept
                : memory_(std::move(other.memory_)), device(other.device)
            {
            }

            bucket_entry_t& operator=(bucket_entry_t&& other) noexcept
            {
                if (this != &other) {
                    memory_ = std::move(other.memory_);
                    device  = other.device;
                }
                return *this;
            }
            ~bucket_entry_t() = default;

            void* data() const { return memory_.data(); }
            size_t size() const { return memory_.size(); }
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
                    // extract the memory from bucket entry
                    auto entry = std::move(bucket_pool.back());
                    bucket_pool.pop_back();

                    T* ptr = static_cast<T*>(entry.data());

                    // create shared_ptr with custom deleter that returns to
                    // arena
                    auto self = this->shared_from_this();
                    return mem::shared_ptr<T>(
                        ptr,
                        [self,
                         bucket,
                         entry = std::move(entry)](T* ptr) mutable {
                            if (!ptr) {
                                return;
                            }

                            try {
                                std::lock_guard<std::mutex> lock(self->mutex);

                                // return the entry back to the bucket
                                self->buckets[bucket].emplace_back(
                                    std::move(entry)
                                );
                            }
                            catch (...) {
                                // if bucket return fails, entry destructor will
                                // clean up
                            }
                        }
                    );
                }
            }

            // allocate new memory using the hetero adapter
            if (dev_.is_gpu) {
                hetero::device::set_device(dev_.device_id);
            }

            auto memory = [this, bytes]() {
                using alloc_type = hetero::device_memory::alloc_type;

                if (dev_.is_gpu) {
                    hetero::device::set_device(dev_.device_id);
                    return hetero::device_memory(bytes, alloc_type::device);
                }
                else {
                    return hetero::device_memory(bytes, alloc_type::host);
                }
            }();
            T* typed_ptr = static_cast<T*>(memory.data());

            // move memory ownership to the deleter
            auto self = this->shared_from_this();
            return mem::shared_ptr<T>(
                typed_ptr,
                [self, bucket, memory = std::move(memory)](T* ptr) mutable {
                    if (!ptr) {
                        return;
                    }

                    try {
                        std::lock_guard<std::mutex> lock(self->mutex);

                        // create bucket entry and add to pool
                        bucket_entry_t entry(std::move(memory), self->dev_);
                        self->buckets[bucket].emplace_back(std::move(entry));
                    }
                    catch (...) {
                        // if bucket return fails, memory destructor will clean
                        // up
                    }
                }
            );
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
