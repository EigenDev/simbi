#include "build_options.hpp"
#include "core/types/monad/result.hpp"
#include "device_callable.hpp"
#include <string>
#include <unordered_map>

namespace simbi::jit {
    template <size_type D>
    class FunctionRegistry
    {
      private:
        std::unordered_map<std::string, DeviceCallable<D>> functions_;

      public:
        // immutable updates (functional approach bc I'm learning :D)
        FunctionRegistry
        with_function(std::string name, DeviceCallable<D> function) const
        {
            auto copy                        = *this;
            copy.functions_[std::move(name)] = std::move(function);
            return copy;
        }

        // lookup with optional result
        DEV Result<DeviceCallable<D>>
        get_function(const std::string& name) const
        {
            auto it = functions_.find(name);
            if (it == functions_.end()) {
                return Result<DeviceCallable<D>>::error(
                    "Function '" + name + "' not found"
                );
            }
            return Result<DeviceCallable<D>>::ok(it->second);
        }

        // Safe lookup - returns a no-op function if not found
        DEV DeviceCallable<D>
        get_function_or_noop(const std::string& name) const
        {
            auto it = functions_.find(name);
            if (it == functions_.end()) {
                return DeviceCallable<D>(
                    name,
                    nullptr
                );   // Returns non-callable function
            }
            return it->second;
        }
    };
}   // namespace simbi::jit
