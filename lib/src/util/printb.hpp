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
        // lambdas not yet
        template <typename T> std::string to_string (T val);
        
        template <typename ...ARGS>
        void write (std::string const & fmt, ARGS... args);

        template <typename ...ARGS> 
        void writeln (std::string const & fmt, ARGS... args);
        
        void write (std::string const & str);
        
        void writeln (std::string const & str);
    } // namespace util
    
} // namespace simbi


#include "printb.tpp"
#endif