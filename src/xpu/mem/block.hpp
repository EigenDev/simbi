// =============================================================================
// block.hpp
//
// simple raii memory ownership for xpu framework.
// one job: allocate and free memory. no coherency, no sharing, no complexity.
// follows hesi pattern of minimal, focused components.
//
// usage:
//   memory_block_t<device_memory> block(1024);
//   auto* ptr = block.template as<float>();
//   // automatically freed on destruction
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

namespace simbi::xpu::mem {

    // =============================================================================
    // memory block - simple raii ownership
    // =============================================================================

    template <typename MemorySpace>
    class memory_block_t
    {
      private:
        void*        ptr_       = nullptr;
        std::size_t  bytes_     = 0;
        std::int64_t device_id_ = 0;

      public:
        using memory_space_type = MemorySpace;

        // default constructor
        memory_block_t() = default;

        // allocate memory
        explicit memory_block_t(std::size_t bytes, std::int64_t device_id = 0)
            : bytes_(bytes), device_id_(device_id)
        {
            if (bytes > 0) {
                ptr_ = MemorySpace::allocate(bytes);
            }
        }

        // destructor - free memory
        ~memory_block_t()
        {
            if (ptr_) {
                MemorySpace::deallocate(ptr_, bytes_);
            }
        }

        // disable copy (unique ownership)
        memory_block_t(const memory_block_t&)            = delete;
        memory_block_t& operator=(const memory_block_t&) = delete;

        // enable move
        memory_block_t(memory_block_t&& other) noexcept
            : ptr_(other.ptr_), bytes_(other.bytes_), device_id_(other.device_id_)
        {
            other.ptr_       = nullptr;
            other.bytes_     = 0;
            other.device_id_ = 0;
        }

        memory_block_t& operator=(memory_block_t&& other) noexcept
        {
            if (this != &other) {
                if (ptr_) {
                    MemorySpace::deallocate(ptr_, bytes_);
                }
                ptr_       = other.ptr_;
                bytes_     = other.bytes_;
                device_id_ = other.device_id_;

                other.ptr_       = nullptr;
                other.bytes_     = 0;
                other.device_id_ = 0;
            }
            return *this;
        }

        // accessors
        void* data() const noexcept
        {
            return ptr_;
        }

        std::size_t size() const noexcept
        {
            return bytes_;
        }

        bool empty() const noexcept
        {
            return ptr_ == nullptr || bytes_ == 0;
        }

        explicit operator bool() const noexcept
        {
            return !empty();
        }

        // typed access
        template <typename T>
        T* as() const noexcept
        {
            return static_cast<T*>(ptr_);
        }

        // memory space queries
        static constexpr bool is_host_accessible()
        {
            return MemorySpace::is_host_accessible;
        }

        static constexpr bool is_device_accessible()
        {
            return MemorySpace::is_device_accessible;
        }

        static constexpr bool is_unified()
        {
            return MemorySpace::is_unified;
        }

        // device affinity
        std::int64_t device_id() const noexcept
        {
            return device_id_;
        }
    };

    // =============================================================================
    // factory function
    // =============================================================================

    template <typename T, typename MemorySpace>
    memory_block_t<MemorySpace> make_block(std::size_t count, std::int64_t device_id = 0)
    {
        return memory_block_t<MemorySpace>(count * sizeof(T), device_id);
    }

} // namespace simbi::xpu::mem
