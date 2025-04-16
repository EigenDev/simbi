#include "jit.hpp"
#include "build_options.hpp"
#include "core/types/monad/result.hpp"
#include "util/tools/device_api.hpp"
#include <vector>

namespace simbi::jit {
    // compile source to ir / ptx
    Result<std::string> compile_to_ir(const SourceCode& source)
    {
        if constexpr (global::on_gpu) {
            // compile the source code into ptx code
            devProgram_t program;

            // create the program
            gpu::api::createProgram(
                &program,
                source.code.c_str(),
                source.name.c_str(),
                0,
                nullptr,
                nullptr
            );

            // compile the program
            if (gpu::api::program(program, 0, nullptr)) {
                // get the log
                size_t log_size;
                gpu::api::getProgramLogSize(program, &log_size);
                std::vector<char> log(log_size);
                gpu::api::getProgramLog(program, log.data());

                // throw an exception
                throw std::runtime_error(
                    "Failed to compile the program: " + std::string(log.data())
                );
            }

            // retrieve the intermediate representation
            size_t ir_size;
            gpu::api::getProgramIRSize(program, &ir_size);

            std::vector<char> ir(ir_size);
            gpu::api::getProgramIR(program, ir.data());
            return Result<std::string>::ok(std::string(ir.begin(), ir.end()));
        }
        return Result<std::string>::error("Not implemented on cpu");
    }

    // load ir / ptx into a module
    Result<devModule_t> load_module(const std::string& ir)
    {
        if constexpr (global::on_gpu) {
            // implementation for loading module on GPU
            try {
                devModule_t module;
                gpu::api::moduleLoadData(&module, ir.c_str());
                return Result<devModule_t>::ok(module);
            }
            catch (const std::exception& e) {
                return Result<devModule_t>::error(
                    "Failed to load module: " + std::string(e.what())
                );
            }
        }
        return Result<devModule_t>::error("Not implemented on cpu");
    }

    // get a function from the module by name
    Result<devFunction_t>
    get_function(devModule_t module, const std::string& name)
    {
        if constexpr (global::on_gpu) {
            // implementation for getting function on GPU
            devFunction_t function;
            gpu::api::getFunction(&function, module, name.c_str());
            if (function == nullptr) {
                return Result<devFunction_t>::error(
                    "Failed to get function: " + name
                );
            }
            return Result<devFunction_t>::ok(function);
        }
        return Result<devFunction_t>::error("Not implemented on cpu");
    }

    // get the address of a device function
    Result<devFunction_t> get_device_function_address(
        devModule_t module,
        const std::string& func_name
    )
    {
        if constexpr (!global::on_gpu) {
            return Result<void*>::error(
                "Cannot get device function address in CPU mode"
            );
        }

        // Declare a CUDA function handle
        devFunction_t function;

        // Get the function from the module
        gpu::api::getFunction(
            &function,
            static_cast<devModule_t>(module),
            func_name.c_str()
        );

        // Cast the device function and we're good
        return Result<devFunction_t>::ok(
            reinterpret_cast<devFunction_t>(function)
        );
    }
}   // namespace simbi::jit
