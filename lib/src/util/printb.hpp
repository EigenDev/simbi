#ifndef PRINT_HPP
#define PRINT_HPP

#include <sstream>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>

namespace simbi
{
    namespace util
    {   
        template <typename ...ARGS>
        void write(std::string const & fmt, ARGS... args);

        template <typename ...ARGS> 
        void writeln(std::string const & fmt, ARGS... args);

        template <typename ...ARGS> 
        void writefl(std::string const & fmt, ARGS... args);

        inline void write (std::string const & str) {
                std::cout << std::fixed << std::setprecision(3) << std::scientific << str;
        }
        inline void writeln(std::string const & str) {
                std::cout << std::fixed << std::setprecision(3) << std::scientific << str << '\n';
        }
        inline void writefl(std::string const & str) {
                std::cout << std::fixed << std::setprecision(3) << std::scientific <<  str << std::flush;
        }
    } // namespace util
    
} // namespace simbi


#include "printb.tpp"
#endif