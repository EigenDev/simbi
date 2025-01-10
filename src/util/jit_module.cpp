#include "jit_module.hpp"
#include "device_api.hpp"
#include <stdexcept>
#include <vector>

using namespace simbi;
using namespace simbi::detail;

JITModule::JITModule() : module(nullptr) {}

JITModule::~JITModule() { gpu::api::moduleUnload(module); }

std::string
JITModule::compile(const std::string& source, const std::string& program_name)
{
    // compile the source code into ptx code
    devProgram_t program;

    // create the program
    gpu::api::createProgram(
        &program,
        source.c_str(),
        program_name.c_str(),
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

    // return the ptx or llvm ir code
    return std::string(ir.begin(), ir.end());
}