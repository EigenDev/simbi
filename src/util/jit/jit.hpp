#ifndef JIT_HPP
#define JIT_HPP
#include "build_options.hpp"
#include "core/types/monad/result.hpp"   // for Result
#include "util/jit/source_code.hpp"      // for SourceCode

namespace simbi::jit {
    Result<std::string> compile_to_ir(const SourceCode& source);
    Result<devModule_t> load_module(const std::string& ir);
    Result<devFunction_t>
    get_function(devModule_t module, const std::string& name);
    Result<devFunction_t> get_device_function_address(
        devModule_t module,
        const std::string& func_name
    );
}   // namespace simbi::jit

#endif
