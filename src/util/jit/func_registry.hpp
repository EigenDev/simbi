#include "build_options.hpp"
#include "core/types/monad/result.hpp"
#include "device_callable.hpp"
#include <string>
#include <unordered_map>
#include <variant>

namespace simbi::jit {
    class FunctionRegistry
    {
      private:
        using FunctionEntry = std::variant<
            DeviceCallable<void(real, real, real*)>,              // 1D source
            DeviceCallable<void(real, real, real, real*)>,        // 2D source
            DeviceCallable<void(real, real, real, real, real*)>   // 3D source
            >;

        std::unordered_map<std::string, FunctionEntry> functions_;

      public:
        // Add a function to the registry
        template <typename Signature>
        FunctionRegistry with_function(
            const std::string& name,
            const DeviceCallable<Signature>& func
        ) const
        {
            FunctionRegistry new_registry = *this;
            new_registry.functions_.insert_or_assign(name, func);
            return new_registry;
        }

        // Get a function from the registry
        template <typename Signature>
        Result<DeviceCallable<Signature>>
        get_function(const std::string& name) const
        {
            auto it = functions_.find(name);
            if (it == functions_.end()) {
                return Result<DeviceCallable<Signature>>::error(
                    "Function not found: " + name
                );
            }

            try {
                return Result<DeviceCallable<Signature>>::ok(
                    std::get<DeviceCallable<Signature>>(it->second)
                );
            }
            catch (const std::bad_variant_access&) {
                return Result<DeviceCallable<Signature>>::error(
                    "Function has incorrect signature: " + name
                );
            }
        }
    };
}   // namespace simbi::jit
