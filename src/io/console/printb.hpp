/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            printb.hpp
 *  * @brief           printb - a simple, thread-safe, colorized print utility
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
#ifndef PRINTB_HPP
#define PRINTB_HPP

#include "core/types/utility/enums.hpp"   // for Color
#include <iomanip>                        // for scientific, precision
#include <iostream>                       // for operator <<
#include <map>                            // for allocator, map
#include <mutex>
#include <sstream>   // for operator>>, ws, basic_istream, basic_istringstream
#include <string>    // for string

namespace simbi {
    namespace util {

        inline bool is_number(const std::string& s)
        {
            long double ld;
            return ((std::istringstream(s) >> ld >> std::ws).eof());
        }

        template <Color C, typename... ARGS>
        void write(std::string const& fmt, ARGS... args);

        template <Color C = Color::DEFAULT, typename... ARGS>
        void writeln(std::string const& fmt, ARGS... args);

        template <Color C = Color::DEFAULT, typename... ARGS>
        void writefl(std::string const& fmt, ARGS... args);

        template <typename... Args>
        void sync_print(Args... args);
    }   // namespace util

}   // namespace simbi

#include "printb.ipp"
#endif