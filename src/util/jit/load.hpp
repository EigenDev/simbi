#ifndef LOAD_HPP
#define LOAD_HPP

#include "core/types/monad/result.hpp"
#include "util/jit/device_callable.hpp"
#include "util/jit/jit.hpp"
#include "util/jit/source_code.hpp"

namespace simbi::jit {
    template <size_type Dims>
    Result<DeviceCallable<Dims>>
    compile_and_load(const SourceCode& source, const std::string& func_name)
    {
        return jit::compile_to_ir(source)
            .and_then([](const std::string& ir) { return load_module(ir); })
            .and_then([&func_name](devModule_t module) {
                return get_device_function_address(module, func_name);
            })
            .map([&func_name](void* func_addr) {
                return DeviceCallable<Dims>(func_name, func_addr);
            });
    }
}   // namespace simbi::jit

#endif
