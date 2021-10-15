#include "printb.hpp"

namespace simbi
{
    namespace util
    {
        void write (std::string const & str) {
                std::cout << str;
        }
        
        void writeln (std::string const & str) {
                std::cout << str << '\n';
        }
    } // namespace util
    
} // namespace simbi
