#ifndef SOURCE_CODE_HPP
#define SOURCE_CODE_HPP

#include <string>
namespace simbi::jit {
    struct SourceCode {
        std::string code;
        std::string name;

        // ctor that effectively makes this bad boy immutable
        SourceCode(std::string code, std::string name)
            : code(std::move(code)), name(std::move(name))
        {
        }
    };
}   // namespace simbi::jit

#endif
