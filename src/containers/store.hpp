#ifndef CONTAINERS_STORE_HPP
#define CONTAINERS_STORE_HPP

#include "compat.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"
#include "hesi/mem/block.hpp"
#include "hesi/mem/transfer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace simbi {

    // -------------------------------------------------------------------------
    // dynamic array for heterogeneous memory
    // essentially std::vector but backed by hal block_t (managed memory)
    // -------------------------------------------------------------------------
    template <typename T>
    class store_t
    {
      private:
        het::mem::block_t storage_;
        std::uint64_t size_     = 0;
        std::uint64_t capacity_ = 0;

        // cached typed pointer to avoid casting on every access
        T* data_ = nullptr;

        // helper: expand storage if needed
        void ensure_capacity(std::uint64_t required)
        {
            if (required <= capacity_) {
                return;
            }

            // geometric growth (1.5x or 2x)
            std::uint64_t new_capacity = std::max(required, capacity_ * 2);
            if (capacity_ == 0) {
                new_capacity = std::max(required, std::uint64_t(16));
            }

            // allocate new block (managed memory by default for store_t)
            auto new_block = het::mem::block_t(
                new_capacity * sizeof(T),
                storage_.locality(),   // keep same locality
                het::memory_type_t::managed
            );

            // move data
            if (size_ > 0) {
                // synchronous copy for resize operations
                // usage of store_t implies host-side setup usually
                het::mem::copy(
                    new_block.data(),
                    storage_.locality(),
                    storage_.data(),
                    storage_.locality(),
                    size_ * sizeof(T)
                );
            }

            // swap and update
            storage_  = std::move(new_block);
            capacity_ = new_capacity;
            data_     = static_cast<T*>(storage_.data());
        }

      public:
        // ---------------------------------------------------------------------
        // construction
        // ---------------------------------------------------------------------

        // default: empty on host
        store_t() : storage_() {}   // default block is empty

        // sized: allocates on default gpu if available
        explicit store_t(std::uint64_t size, const T& val = T())
        {
            // default to gpu 0 if available, else cpu
            // auto backend = het::info::is_gpu ? het::backend_type_t::cuda
            //                                  : het::backend_type_t::cpu;

            // het::locality_t loc{backend, 0};

            resize(size, val);

            // if we just resized, storage_ is set. update locality if needed?
            // block_t handles its own locality.
        }

        // from host vector
        explicit store_t(const std::vector<T>& values)
        {
            // same locality logic
            auto backend = het::info::is_gpu ? het::backend_type_t::cuda
                                             : het::backend_type_t::cpu;
            het::locality_t loc{backend, 0};

            reserve(values.size());

            // initial copy
            // since storage is managed, we can just memcpy or use hal copy
            het::mem::copy(
                storage_.data(),
                loc,
                values.data(),
                het::locality_t::host(),
                values.size() * sizeof(T)
            );

            size_ = values.size();
        }

        // disable copy (block_t is unique), allow move
        store_t(const store_t&)            = delete;
        store_t& operator=(const store_t&) = delete;

        store_t(store_t&& other) noexcept
            : storage_(std::move(other.storage_)),
              size_(other.size_),
              capacity_(other.capacity_),
              data_(other.data_)
        {
            other.size_     = 0;
            other.capacity_ = 0;
            other.data_     = nullptr;
        }

        store_t& operator=(store_t&& other) noexcept
        {
            if (this != &other) {
                storage_  = std::move(other.storage_);
                size_     = other.size_;
                capacity_ = other.capacity_;
                data_     = other.data_;

                other.size_     = 0;
                other.capacity_ = 0;
                other.data_     = nullptr;
            }
            return *this;
        }

        // ---------------------------------------------------------------------
        // accessors
        // ---------------------------------------------------------------------

        DUAL T& operator[](std::uint64_t idx) { return data_[idx]; }
        DUAL const T& operator[](std::uint64_t idx) const { return data_[idx]; }

        DUAL T* data() { return data_; }
        DUAL const T* data() const { return data_; }

        DUAL std::uint64_t size() const { return size_; }
        DUAL std::uint64_t capacity() const { return capacity_; }
        DUAL bool empty() const { return size_ == 0; }

        // ---------------------------------------------------------------------
        // mutation
        // ---------------------------------------------------------------------

        void add(const T& value)
        {
            ensure_capacity(size_ + 1);
            // since it's managed memory, we can write from host directly
            data_[size_++] = value;
        }

        void reserve(std::uint64_t new_capacity)
        {
            ensure_capacity(new_capacity);
        }

        void clear() { size_ = 0; }

        void resize(std::uint64_t new_size, const T& value = T())
        {
            ensure_capacity(new_size);

            if (new_size > size_) {
                // fill new elements
                // manually loop on host since managed memory allows it
                for (std::uint64_t i = size_; i < new_size; ++i) {
                    data_[i] = value;
                }
            }
            size_ = new_size;
        }

        // ---------------------------------------------------------------------
        // optimization
        // ---------------------------------------------------------------------

        // prefetch to all available devices (for read-only broadcast data)
        void sync_to_all_devices()
        {
            // simple loop over detected devices
            // logic assumes managed memory prefetch support

            // [todo]: get real device count from HAL
            // int device_count = 0;
            // if (het::info::is_gpu) {
            //     // naive assumption or query device layer
            //     device_count = 1;   // placeholder
            // }

            // strictly speaking, prefetch requires a stream.
            // creating temporary streams here is expensive.
            // usually, the unified memory driver handles migration on demand.
            // explicit prefetch is an optimization.
            // we skip implementation unless we pass in an executor.
        }

        // prefetch using a specific stream (preferred)
        void prefetch(het::stream_t& stream, het::locality_t loc)
        {
            het::mem::prefetch_async(
                storage_.data(),
                loc,
                size_ * sizeof(T),
                stream
            );
        }
    };

}   // namespace simbi

#endif   // CONTAINERS_STORE_HPP
