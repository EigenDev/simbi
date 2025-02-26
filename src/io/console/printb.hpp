/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       printb.hpp
 * @brief      implementation of custom rust-like print functions w/ formatting
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
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