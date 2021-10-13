#ifndef PRINTB_HPP
#define PRINTB_HPP

#include "build_options.hpp"

#include <sstream>
#include <string>
#include <iostream>
#include <stdexcept>

namespace simbi
{
    namespace util
    {
        template<typename T>
        void print(T text)
        {
            std::cout << text << std::endl;
        };

        template<typename T, typename... Args>
        void print(T text, Args... args)
        {
            std::cout << text << " ";
            print(args...);
        };

        template<typename T>
        void print_benchmark(T dt, T t, T fold, T iteration);


        // lambdas not yet
        template <typename T> std::string to_string (T val) {
                std::stringstream ss;
                ss << val;
                return ss.str();
        }
        
        template <typename ...ARGS>
        void write (std::string const & fmt, ARGS... args) {
                const std::string argss[] = {to_string (args)...}; // <- indeed
                enum {argss_len = sizeof (argss) / sizeof(argss[0])};
        
                // no range based for loops yet ("for (auto it : fmt)")
                for (auto it: fmt) {
                        if (it == '{') {
                                auto const left = ++it;
                                if (&it != &fmt.back()) {
                                        // closing brace: fine
                                        if (it == '}')
                                                break;
                                        // check if numeric. if not, throw.
                                        switch (it) {
                                        default:
                                                throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "only numeric digits allowed between "
                                                "braces"
                                                );
                                        case '0':case '1':case '2':case '3':case '4':
                                        case '5':case '6':case '7':case '8':case '9':;
                                        };
                                }
                                if (it != '}') {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "missing closing brace"
                                        );
                                }
                                auto const right = it;
        
                                if (left == right) {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "no index given inside braces"
                                        );
                                }
        
                                std::stringstream ss;
                                ss << std::string(left,right);
                                size_t index;
                                ss >> index;
                                if (index >= argss_len) {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "index too big"
                                        );
                                }
                                std::cout << argss[index];
                        } else {
                                std::cout << it;
                        }
                }
        }
        
        void write (std::string const & str) {
                std::cout << str;
        }
        
        template <typename ...ARGS> void writeln (std::string const & fmt, ARGS... args) {
                write (fmt, args...);
                std::cout << '\n';
        }
        
        void writeln (std::string const & str) {
                std::cout << str << '\n';
    } // namespace util
    
} // namespace simbi

 

}


#endif