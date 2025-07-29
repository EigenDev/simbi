#ifndef BUFFER_POOL_HPP
#define BUFFER_POOL_HPP

#include "arena.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

namespace simbi::mem {

    using buffer_id_t = std::uint64_t;

    /**
     * buffer_pool_t - arena-backed buffer management
     *
     * SRP: manage buffer lifecycle with logical/physical separation
     * enables future buffer reuse optimizations
     */
    template <typename T>
    class buffer_pool_t
    {
        std::shared_ptr<arena_t<T>> arena_;
        std::map<buffer_id_t, std::shared_ptr<T[]>> buffers_;
        buffer_id_t next_id_ = 1;

      public:
        explicit buffer_pool_t(
            std::shared_ptr<arena_t<T>> arena = global_arena<T>()
        )
            : arena_(std::move(arena))
        {
        }

        // allocate new buffer
        buffer_id_t allocate(std::size_t count)
        {
            auto id      = next_id_++;
            buffers_[id] = arena_->get(count);
            return id;
        }

        // get buffer data
        T* get_data(buffer_id_t id)
        {
            auto it = buffers_.find(id);
            return it != buffers_.end() ? it->second.get() : nullptr;
        }

        const T* get_data(buffer_id_t id) const
        {
            auto it = buffers_.find(id);
            return it != buffers_.end() ? it->second.get() : nullptr;
        }

        // deallocate buffer (returns memory to arena)
        void deallocate(buffer_id_t id) { buffers_.erase(id); }

        // check if buffer exists
        bool exists(buffer_id_t id) const
        {
            return buffers_.find(id) != buffers_.end();
        }

        // get underlying arena for advanced usage
        std::shared_ptr<arena_t<T>> arena() const { return arena_; }

        // future: buffer aliasing for reuse optimization
        void alias(buffer_id_t dead_id, buffer_id_t new_id)
        {
            if (auto it = buffers_.find(dead_id); it != buffers_.end()) {
                buffers_[new_id] = std::move(it->second);
                buffers_.erase(it);
            }
        }
    };

    // global pool for convenience
    template <typename T>
    std::shared_ptr<buffer_pool_t<T>>& global_buffer_pool()
    {
        static auto instance = std::make_shared<buffer_pool_t<T>>();
        return instance;
    }

}   // namespace simbi::mem

#endif
