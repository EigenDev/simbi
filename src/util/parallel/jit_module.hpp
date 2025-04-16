/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            jit_module.hpp
 *  * @brief           a header file for the JIT compiler that turns the source
 * code into ptx (nvidia) or LLVM IR (amd) code
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef JIT_MODULE_HPP
#define JIT_MODULE_HPP

#include "build_options.hpp"
#include <string>
#include <unordered_map>

namespace simbi {
    namespace detail {

        class JITModule
        {
          private:
            devModule_t module;
            std::unordered_map<std::string, std::string> functionMap;

          public:
            JITModule();
            ~JITModule();
            std::string
            compile(const std::string& source, const std::string& prog_name);
            bool load_module_and_get_function(
                const std::string& ptx,
                const std::string& function_name,
                devFunction_t* function
            );
            bool ensure_context_initialized();
        };

    }   // namespace detail

}   // namespace simbi
#endif
