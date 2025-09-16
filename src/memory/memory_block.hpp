#ifndef MEMORY_BLOCK_HPP
#define MEMORY_BLOCK_HPP

#include "hetero/adapter.hpp"
#include "memory/device.hpp"

#include <cstddef>
#include <utility>

namespace simbi::mem {

    class memory_block_t
    {
        hetero::memory memory_;
        device_t dev_;

      public:
        memory_block_t(hetero::memory&& mem, device_t device)
            : memory_(std::move(mem)), dev_(device), owns_memory_(true)
        {
        }

        static memory_block_t allocate(size_t bytes, device_t device)
        {
            if (device.is_gpu) {
                hetero::device::set_device(device.device_id);
            }
            auto mem = hetero::device::allocate(bytes);
            return memory_block_t(std::move(mem), device);
        }

        void* data() const { return memory_.data(); }
        size_t size() const { return memory_.size(); }
        const device_t& device() const { return dev_; }
    };

}   // namespace simbi::mem

#endif
