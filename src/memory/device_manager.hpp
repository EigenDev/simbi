#ifndef DEVICE_POOL_HPP
#define DEVICE_POOL_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simbi::mem {
    /**
     * device_pool_t - manages available compute devices
     *
     * srp: device discovery, validation, and pool management
     * functional factory pattern
     */
    class device_pool_t
    {
        std::vector<device_id_t> devices_;

      public:
        // factory methods
        static device_pool_t from_env()
        {
            auto device_count = parse_device_count_from_env();
            return create_pool(device_count);
        }

        static device_pool_t single_cpu()
        {
            return device_pool_t{{device_id_t::cpu_device()}};
        }

        static device_pool_t single_gpu(int device_id = 0)
        {
            validate_gpu_exists(device_id);
            return device_pool_t{{device_id_t::gpu_device(device_id)}};
        }

        static device_pool_t multi_gpu(int count) { return create_pool(count); }

        // queries
        const std::vector<device_id_t>& devices() const { return devices_; }
        std::size_t size() const { return devices_.size(); }
        bool empty() const { return devices_.empty(); }
        bool is_multi_device() const { return devices_.size() > 1; }
        bool has_gpus() const
        {
            return std::any_of(
                devices_.begin(),
                devices_.end(),
                [](const auto& dev) { return dev.type == device_type_t::gpu; }
            );
        }

        // device access
        device_id_t primary_device() const
        {
            if (devices_.empty()) {
                throw std::runtime_error("No devices in pool");
            }
            return devices_[0];
        }

        device_id_t device_at(std::size_t index) const
        {
            if (index >= devices_.size()) {
                throw std::out_of_range("Device index out of range");
            }
            return devices_[index];
        }

        // iteration support
        auto begin() const { return devices_.begin(); }
        auto end() const { return devices_.end(); }

      private:
        explicit device_pool_t(std::vector<device_id_t> devices)
            : devices_(std::move(devices))
        {
            if (devices_.empty()) {
                throw std::runtime_error("Device pool cannot be empty");
            }
            validate_device_capabilities();
            setup_device_context();
        }

        static int parse_device_count_from_env()
        {
            const char* env_var = std::getenv("SIMBI_NUM_DEVICES");

            if (!env_var || std::strlen(env_var) == 0) {
                return 1;
            }

            std::string env_str = env_var;
            env_str.erase(0, env_str.find_first_not_of(" \t\n\r"));
            env_str.erase(env_str.find_last_not_of(" \t\n\r") + 1);

            if (env_str.empty()) {
                return 1;
            }

            try {
                int requested = std::stoi(env_str);
                if (requested <= 0) {
                    throw std::runtime_error(
                        "SIMBI_NUM_DEVICES must be positive, got: " +
                        std::to_string(requested)
                    );
                }
                return requested;
            }
            catch (const std::invalid_argument&) {
                throw std::runtime_error(
                    "SIMBI_NUM_DEVICES must be a valid integer, got: '" +
                    env_str + "'"
                );
            }
            catch (const std::out_of_range&) {
                throw std::runtime_error(
                    "SIMBI_NUM_DEVICES value out of range: '" + env_str + "'"
                );
            }
        }

        static device_pool_t create_pool(int requested_count)
        {
            std::int64_t available_gpus = 0;
            gpu::api::get_device_count(&available_gpus);

            if (available_gpus == 0) {
                // no gpus available - use cpu
                if (requested_count > 1) {
                    throw std::runtime_error(
                        "Requested " + std::to_string(requested_count) +
                        " devices but no GPUs available, only CPU supported"
                    );
                }
                return device_pool_t{{device_id_t::cpu_device()}};
            }

            if (requested_count > available_gpus) {
                throw std::runtime_error(
                    "Requested " + std::to_string(requested_count) +
                    " GPU devices, but only " + std::to_string(available_gpus) +
                    " available"
                );
            }

            // create gpu device pool
            std::vector<device_id_t> devices;
            devices.reserve(requested_count);

            for (int i = 0; i < requested_count; ++i) {
                devices.push_back(device_id_t::gpu_device(i));
            }

            return device_pool_t{std::move(devices)};
        }

        static void validate_gpu_exists(int device_id)
        {
            std::int64_t available_gpus = 0;
            gpu::api::get_device_count(&available_gpus);

            if (device_id >= available_gpus) {
                throw std::runtime_error(
                    "GPU device " + std::to_string(device_id) +
                    " does not exist, only " + std::to_string(available_gpus) +
                    " GPUs available"
                );
            }
        }

        void validate_device_capabilities() const
        {
            for (const auto& device : devices_) {
                if (device.type == device_type_t::gpu) {
                    adapter::device_properties_t<> props;
                    gpu::api::get_device_properties(&props, device.device_id);

                    // basic capability checks
                    constexpr size_t min_memory_gb = 1;
                    constexpr size_t min_memory_bytes =
                        min_memory_gb * 1024 * 1024 * 1024;

                    if (props.totalGlobalMem < min_memory_bytes) {
                        throw std::runtime_error(
                            "GPU device " + std::to_string(device.device_id) +
                            " has insufficient memory: " +
                            std::to_string(
                                props.totalGlobalMem / (1024 * 1024)
                            ) +
                            " MB available, minimum " +
                            std::to_string(min_memory_gb * 1024) +
                            " MB required"
                        );
                    }
                }
            }
        }

        void setup_device_context() const
        {
            if (devices_.size() <= 1 || !has_gpus()) {
                return;   // no peer access needed
            }

            // enable peer access between all gpu pairs
            for (const auto& device_i : devices_) {
                if (device_i.type != device_type_t::gpu) {
                    continue;
                }

                gpu::api::set_device(device_i.device_id);

                for (const auto& device_j : devices_) {
                    if (device_j.type != device_type_t::gpu ||
                        device_i.device_id == device_j.device_id) {
                        continue;
                    }

                    try {
                        gpu::api::enable_peer_access(device_j.device_id, 0);
                    }
                    catch (...) {
                        // peer access not supported - not fatal
                        // [TODO]: add logging here?
                    }
                }
            }
        }
    };

    // convenience functions
    inline device_pool_t default_device_pool()
    {
        return device_pool_t::from_env();
    }

    inline bool is_multi_device_enabled()
    {
        return default_device_pool().is_multi_device();
    }

}   // namespace simbi::mem

#endif   // DEVICE_POOL_HPP
