// =============================================================================
// device_guard.hpp
//
// raii device context management for multi-gpu execution.
// ensures device context is restored after operations complete.
//
// usage:
//   device_guard_t<cuda_space> guard(1);  // switches to device 1
//   // ... work on device 1
//   // guard destructor restores previous device
// =============================================================================
#pragma once

#include "execution_space.hpp"

#include <cstdint>

namespace simbi::xpu::exec {

    // =============================================================================
    // device context guard
    // =============================================================================

    template <execution_space_c ExecutionSpace>
    class device_guard_t
    {
      private:
        std::int64_t prev_device_;
        bool         active_ = false;

      public:
        explicit device_guard_t(std::int64_t device_id)
        {
            if constexpr (requires {
                              ExecutionSpace::get_device();
                              ExecutionSpace::set_device(device_id);
                          }) {
                prev_device_ = ExecutionSpace::get_device();
                if (prev_device_ != device_id) {
                    ExecutionSpace::set_device(device_id);
                    active_ = true;
                }
            }
        }

        ~device_guard_t() noexcept
        {
            if (active_) {
                if constexpr (requires { ExecutionSpace::set_device(prev_device_); }) {
                    try {
                        ExecutionSpace::set_device(prev_device_);
                    }
                    catch (...) {
                        // device context restoration failed - can't propagate from destructor
                        // this is a fatal error but terminating is worse than continuing
                    }
                }
            }
        }

        device_guard_t(const device_guard_t&)            = delete;
        device_guard_t& operator=(const device_guard_t&) = delete;
        device_guard_t(device_guard_t&&)                 = delete;
        device_guard_t& operator=(device_guard_t&&)      = delete;

        std::int64_t previous_device() const noexcept
        {
            return prev_device_;
        }

        bool is_active() const noexcept
        {
            return active_;
        }
    };

} // namespace simbi::xpu::exec
