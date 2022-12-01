#ifndef PRINTB_HPP
#define PRINTB_HPP

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <iomanip>

namespace simbi
{
    namespace util
    {   

        inline bool is_number(const std::string& s)
        {
            long double ld;
            return((std::istringstream(s) >> ld >> std::ws).eof());
        }
        template <typename ...ARGS>
        void write(std::string const & fmt, ARGS... args);

        template <typename ...ARGS> 
        void writeln(std::string const & fmt, ARGS... args);

        template <typename ...ARGS> 
        void writefl(std::string const & fmt, ARGS... args);
    } // namespace util
    
} // namespace simbi


#include "printb.tpp"
#endif