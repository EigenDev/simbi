#ifndef NEW_DEVICE_HPP
#define NEW_DEVICE_HPP

#include "adapter/device_adapter_api.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>

namespace simbi::mem {

    // simple device identifier
    struct device_t {
        std::int64_t index;   // global device index
        bool is_gpu;          // true for gpu, false for cpu
        // device id for gpus (only relevant when is_gpu=true)
        std::int64_t device_id;

        // equality comparison for containers
        auto operator<=>(const device_t&) const = default;

        // helper factory functions
        static device_t cpu() { return {0, false, 0}; }

        static device_t gpu(int id) { return {id + 1, true, id}; }
    };

    // track current device (thread local)
    namespace {
        thread_local device_t current_dev = device_t::cpu();
    }

    // get current device
    inline device_t current_device() { return current_dev; }

    // set current device
    inline void set_current_device(device_t dev)
    {
        current_dev = dev;

        // update hardware state if it's a GPU
        if (dev.is_gpu) {
            gpu::api::set_device(dev.device_id);
        }
    }

    // get count of available devices
    inline std::int64_t device_count()
    {
        std::int64_t count = 0;

        gpu::api::get_device_count(&count);

        if (char* user_count = std::getenv("SIMBI_NUM_DEVICES")) {
            try {
                int requested = std::stoi(user_count);
                if (requested < count) {
                    count = requested;
                }
            }
            catch (...) {
                // ignore invalid env var
            }
        }

        return static_cast<int>(count) + 1;
    }

    inline std::vector<device_t> all_devices()
    {
        std::vector<device_t> devices;

        devices.push_back(device_t::cpu());

        std::int64_t gpu_count = device_count() - 1;
        for (int ii = 0; ii < gpu_count; ii++) {
            devices.push_back(device_t::gpu(ii));
        }

        return devices;
    }

}   // namespace simbi::mem

// allow for hashing of device_t
namespace std {
    template <>
    struct hash<simbi::mem::device_t> {
        std::size_t operator()(const simbi::mem::device_t& loc) const
        {
            return std::hash<std::int64_t>{}(loc.index);
        }
    };
}   // namespace std
#endif   // DEVICE_HPP
