#include "util/parallel/jit_module.hpp"
#include "util/tools/device_api.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace simbi;
using namespace simbi::detail;

JITModule::JITModule() : module(nullptr) { ensure_context_initialized(); }

JITModule::~JITModule()
{
    if (module) {
        gpu::api::moduleUnload(module);
        module = nullptr;
    }
}

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

bool JITModule::load_module_and_get_function(
    const std::string& ptx,
    const std::string& functionName,
    devFunction_t* function
)
{
    if (!ensure_context_initialized()) {
        return false;
    }

    try {
        // Clean up old module if present
        if (module) {
            gpu::api::moduleUnload(module);
            module = nullptr;
        }

        // Load the new module
        gpu::api::moduleLoadData(&module, ptx.c_str());

        // Get the function
        gpu::api::getFunction(function, module, functionName.c_str());

        // Store the function in our map
        functionMap[functionName] = ptx;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading module or function: " << e.what()
                  << std::endl;
        return false;
    }
}

bool JITModule::ensure_context_initialized()
{
    static bool initialized = false;
    if (!initialized) {
        simbi_device_t device;
        simbi_context_t context;

        auto status = simbi_init(0);
        if (status != simbi_success) {
            return false;
        }

        status = gpu::api::device_get(&device, 0);
        if (status != simbi_success) {
            return false;
        }

        status = gpu::api::context_create(&context, 0, device);
        if (status != simbi_success) {
            return false;
        }

        initialized = true;
    }
    return initialized;
}
