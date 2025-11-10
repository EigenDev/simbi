#ifndef MEM_DEVICE_HPP
#define MEM_DEVICE_HPP

#include "hetero/adapter.hpp"

#include <compare>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace simbi::mem {
    // where memory lives and kernels execute
    struct device_t {
        bool is_gpu;
        std::int64_t device_id;   // gpu id, or numa node for cpu

        std::strong_ordering operator<=>(const device_t&) const = default;

        // factories
        static device_t cpu(std::int64_t numa_node = 0)
        {
            return {false, numa_node};
        }

        static device_t gpu(std::int64_t id) { return {true, id}; }

        // hash support for unordered_map
        std::size_t hash() const
        {
            return std::hash<bool>{}(is_gpu) ^
                   (std::hash<std::int64_t>{}(device_id) << 1);
        }
    };

    // track current device (thread local)
    namespace {
        thread_local device_t current_dev = device_t::cpu();
    }

    inline device_t current_device() { return current_dev; }

    inline void set_current_device(device_t dev)
    {
        current_dev = dev;

        if (dev.is_gpu) {
            hetero::device::set_device(dev.device_id);
        }
    }

    inline std::int64_t device_count()
    {
        auto count = hetero::device::get_device_count();
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
        for (std::int64_t ii = 0; ii < gpu_count; ii++) {
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
            return std::hash<std::int64_t>{}(loc.device_id);
        }
    };
}   // namespace std
#endif   // DEVICE_HPP
