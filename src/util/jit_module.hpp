/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       jit_compile.hpp
 * @brief      a header file for the JIT compiler that turns the source code
 * into ptx (nvidia) or LLVM IR (amd) code
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Jan-06-2025     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
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
        };

    }   // namespace detail

}   // namespace simbi
#endif