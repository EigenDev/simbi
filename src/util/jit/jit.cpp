#include "jit.hpp"
#include "build_options.hpp"
#include "core/types/monad/result.hpp"

namespace simbi::jit {
    // compile source to ir / ptx
    Result<std::string> compile_to_ir(const SourceCode& source)
    {
        if constexpr (global::on_gpu) {
            // implementation for GPU compilation
        }
        // placeholder for actual implementation
        return Result<std::string>::error("Not implemented");
    }

    // load ir / ptx into a module
    Result<devModule_t> load_module(const std::string& ir)
    {
        if constexpr (global::on_gpu) {
            // implementation for loading module on GPU
        }
        // placeholder for actual implementation
        return Result<devModule_t>::error("Not implemented");
    }

    // get a function from the module by name
    Result<devFunction_t>
    get_function(devModule_t module, const std::string& name)
    {
        if constexpr (global::on_gpu) {
            // implementation for getting function on GPU
        }
        // placeholder for actual implementation
        return Result<devFunction_t>::error("Not implemented");
    }

    // get the address of a device function
    Result<devFunction_t> get_device_function_address(
        devModule_t module,
        const std::string& func_name
    )
    {
        if constexpr (global::on_gpu) {
            // implementation for getting device function address on GPU
        }
        // placeholder for actual implementation
        return Result<devFunction_t>::error("Not implemented");
    }
}   // namespace simbi::jit
