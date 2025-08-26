#ifndef MEMORY_BLOCK_HPP
#define MEMORY_BLOCK_HPP

#include "adapter/device_adapter_api.hpp"
#include "memory/device.hpp"

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace simbi::mem {

    // raw memory allocation with device affinity
    struct memory_block_t {
        void* data       = nullptr;   // raw memory pointer
        std::size_t size = 0;         // size in bytes
        device_t dev;                 // owning device

        memory_block_t() = default;
        memory_block_t(void* data_ptr, std::size_t byte_size, device_t device)
            : data(data_ptr), size(byte_size), dev(device)
        {
        }

        static memory_block_t allocate(std::size_t bytes, device_t d)
        {
            if (bytes == 0) {
                return {nullptr, 0, d};
            }

            void* ptr = nullptr;

            // set device if GPU
            if (d.is_gpu) {
                gpu::api::set_device(d.device_id);
                gpu::api::malloc(&ptr, bytes);
            }
            else {
                // CPU allocation
                ptr = std::malloc(bytes);
            }

            if (!ptr && bytes > 0) {
                throw std::runtime_error("Failed to allocate memory");
            }

            return {ptr, bytes, d};
        }

        void free()
        {
            if (!data) {
                return;
            }

            if (dev.is_gpu) {
                gpu::api::set_device(dev.device_id);
                gpu::api::free(data);
            }
            else {
                std::free(data);
            }

            data = nullptr;
            size = 0;
        }

        ~memory_block_t() { free(); }

        memory_block_t(memory_block_t&& other) noexcept
            : data(other.data), size(other.size), dev(other.dev)
        {
            other.data = nullptr;
            other.size = 0;
        }

        memory_block_t& operator=(memory_block_t&& other) noexcept
        {
            if (this != &other) {
                free();
                data       = other.data;
                size       = other.size;
                dev        = other.dev;
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }

        // no copy
        memory_block_t(const memory_block_t&)            = delete;
        memory_block_t& operator=(const memory_block_t&) = delete;
    };

}   // namespace simbi::mem

#endif   // MEMORY_BLOCK_HPP
