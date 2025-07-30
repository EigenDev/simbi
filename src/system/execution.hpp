#ifndef EXECUTION_HPP
#define EXECUTION_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "base/buffer.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace simbi {
    struct execution_context_t {
        device_id_t device_;
        std::vector<device_id_t> peer_devices_;
        adapter::stream_t<> stream_;

        execution_context_t(device_id_t device) : device_(device)
        {
            if (device_.type == device_type_t::gpu) {
                gpu::api::set_device(device_.device_id);
                gpu::api::stream_create(&stream_);
                setup_peer_devices();
            }
        }

        ~execution_context_t()
        {
            if (device_.type == device_type_t::gpu) {
                gpu::api::stream_destroy(stream_);
            }
        }

        // move only
        execution_context_t(execution_context_t&&)                 = default;
        execution_context_t& operator=(execution_context_t&&)      = default;
        execution_context_t(const execution_context_t&)            = delete;
        execution_context_t& operator=(const execution_context_t&) = delete;

        // pure transformations - return new buffers
        template <typename T>
        buffer_t<T> to_device(const buffer_t<T>& src) const
        {
            // already on target device?
            if (src.device().type == device_.type &&
                src.device().device_id == device_.device_id) {
                // create a new buffer on the same device with a copy of the
                // data
                buffer_t<T> new_buffer(src.size(), device_);
                if (device_.type == device_type_t::cpu) {
                    std::memcpy(
                        new_buffer.data(),
                        src.data(),
                        src.size() * sizeof(T)
                    );
                }
                else {
                    gpu::api::set_device(device_.device_id);
                    gpu::api::copy_device_to_device(
                        new_buffer.data(),
                        src.data(),
                        src.size() * sizeof(T)
                    );
                }
                return new_buffer;
            }

            // create buffer on target device
            buffer_t<T> target(src.size(), device_);

            if (src.device().type == device_type_t::cpu &&
                device_.type == device_type_t::gpu) {
                // host to device
                gpu::api::set_device(device_.device_id);
                gpu::api::copy_host_to_device(
                    target.data(),
                    src.data(),
                    src.size() * sizeof(T)
                );
            }
            else if (src.device().type == device_type_t::gpu &&
                     device_.type == device_type_t::gpu) {
                // device to device (peer copy)
                peer_copy(src, target);
            }

            return target;
        }

        template <typename T>
        buffer_t<T> to_host(const buffer_t<T>& src) const
        {
            if (src.device().type == device_type_t::cpu) {
                return std::move(src);   // already on host
            }

            // create host buffer
            buffer_t<T> host_buffer(src.size(), device_id_t::cpu_device());

            gpu::api::set_device(src.device().device_id);
            gpu::api::copy_device_to_host(
                host_buffer.data(),
                src.data(),
                src.size() * sizeof(T)
            );

            return host_buffer;
        }

        // multi-gpu operations
        template <typename T>
        buffer_t<T>
        to_peer_device(const buffer_t<T>& src, device_id_t target_device) const
        {
            if (src.device() == target_device) {
                return std::move(src);
            }

            buffer_t<T> target_buffer(src.size(), target_device);

            if (src.device().type == device_type_t::gpu &&
                target_device.type == device_type_t::gpu) {
                gpu::api::peer_copy_async(
                    target_buffer.data(),
                    target_device.device_id,
                    src.data(),
                    src.device().device_id,
                    src.size() * sizeof(T),
                    stream_
                );
                gpu::api::stream_synchronize(stream_);
            }

            return target_buffer;
        }

        // async versions for pipelining
        template <typename T>
        buffer_t<T> async_to_device(const buffer_t<T>& src) const
        {
            buffer_t<T> target(src.size(), device_);

            if (src.device().type == device_type_t::cpu &&
                device_.type == device_type_t::gpu) {
                gpu::api::async_copy_host_to_device(
                    target.data(),
                    src.data(),
                    src.size() * sizeof(T),
                    stream_
                );
            }

            return target;   // caller must synchronize stream
        }

        void synchronize() const
        {
            if (device_.type == device_type_t::gpu) {
                gpu::api::stream_synchronize(stream_);
            }
        }

      private:
        template <typename T>
        void peer_copy(const buffer_t<T>& src, buffer_t<T>& dst) const
        {
            gpu::api::peer_copy_async(
                dst.data(),
                dst.device().device_id,
                src.data(),
                src.device().device_id,
                src.size() * sizeof(T),
                stream_
            );
            gpu::api::stream_synchronize(stream_);
        }

        void setup_peer_devices()
        {
            auto num_devices = std::getenv("SIMBI_NUM_DEVICES")
                                   ? std::atoi(std::getenv("SIMBI_NUM_DEVICES"))
                                   : 1;

            for (int i = 0; i < num_devices; ++i) {
                if (i != device_.device_id) {
                    auto peer = device_id_t::gpu_device(i);
                    peer_devices_.push_back(peer);
                    gpu::api::enable_peer_access(i);
                }
            }
        }
    };
}   // namespace simbi

#endif
