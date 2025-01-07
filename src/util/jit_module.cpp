#include "jit_module.hpp"
#include "device_api.hpp"
#include <stdexcept>
#include <vector>

using namespace simbi;
using namespace simbi::detail;

JITModule::JITModule() : module(nullptr) {}

JITModule::~JITModule() { gpu::api::gpuModuleUnload(module); }

std::string
JITModule::compile(const std::string& source, const std::string& program_name)
{
    // compile the source code into ptx code
    devProgram_t program;

    // create the program
    gpu::api::gpuCreateProgram(
        &program,
        source.c_str(),
        program_name.c_str(),
        0,
        nullptr,
        nullptr
    );

    // compile the program
    if (gpu::api::gpuProgram(program, 0, nullptr)) {
        // get the log
        size_t log_size;
        gpu::api::gpuGetProgramLogSize(program, &log_size);
        std::vector<char> log(log_size);
        gpu::api::gpuGetProgramLog(program, log.data());

        // throw an exception
        throw std::runtime_error(
            "Failed to compile the program: " + std::string(log.data())
        );
    }

    // retrieve the intermediate representation
    size_t ir_size;
    gpu::api::gpuGetProgramIRSize(program, &ir_size);

    std::vector<char> ir(ir_size);
    gpu::api::gpuGetProgramIR(program, ir.data());

    // return the ptx or llvm ir code
    return std::string(ir.begin(), ir.end());
}