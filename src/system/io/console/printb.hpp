/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            printb.hpp
 *  * @brief           printb - a simple, thread-safe, colorized prstd::int64_t
 * utility
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

#include "core/utility/enums.hpp"   // for Color
#include "core/utility/helpers.hpp"
#include <ctype.h>
#include <iomanip>
#include <iostream>   // for operator <<
#include <mutex>
#include <sstream>   // for operator>>, ws, basic_istream, basic_istringstream
#include <stdexcept>
#include <string>   // for string
namespace simbi::util {
    template <Color C, typename... ARGS>
    void write(std::string const& fmt, ARGS... args)
    {
        const std::string argss[] = {[](const auto& x) {
            std::stringstream ss;
            ss << x;
            return ss.str();
        }(args)...};

        auto argss_len        = sizeof(argss) / sizeof(argss[0]);
        std::string width_str = "";
        std::int64_t width    = 1;
        std::int64_t index    = 0;
        bool open_brace       = false;
        bool inserted         = false;
        unsigned cidx         = 0;
        for (const auto& ch : fmt) {
            if (ch == '{') {
                open_brace = true;
                inserted   = false;
            }
            else if (ch == '}') {
                open_brace = false;
                continue;
            }

            if (open_brace) {
                if (!inserted) {
                    auto const right = *(&ch + 1);
                    if (right == ':') {
                        bool scientific        = false;
                        std::int64_t precision = 0;
                        for (auto ii = cidx; ii < fmt.size(); ii++) {
                            const auto fmt_char = fmt[ii];
                            if (fmt_char == '>') {
                                const auto left_num  = fmt[ii + 1];
                                const auto right_num = fmt[ii + 2];
                                if (isdigit(left_num)) {
                                    width_str.push_back(left_num);
                                }

                                if (isdigit(right_num)) {
                                    width_str.push_back(right_num);
                                }

                                if (width_str.size() > 0) {
                                    width = std::stoi(width_str);
                                }
                            }
                            else if (fmt_char == '.') {
                                const auto left_num       = fmt[ii + 1];
                                const auto right_num      = fmt[ii + 2];
                                std::string precision_str = "";
                                if (isdigit(left_num)) {
                                    precision_str.push_back(left_num);
                                }

                                if (isdigit(right_num)) {
                                    precision_str.push_back(right_num);
                                }

                                if (precision_str.size() > 0) {
                                    precision = std::stoi(precision_str);
                                }
                                scientific =
                                    fmt[ii + 2] == 'e' || fmt[ii + 3] == 'e';
                            }
                            else if (fmt_char == '}') {
                                break;
                            }
                        }
                        const bool numeric = is_number(argss[index]);
                        if (scientific) {
                            if (numeric) {
                                std::cout << std::fixed << std::scientific
                                          << std::setprecision(precision)
                                          << std::setw(width)
                                          << std::stod(argss[index]);
                            }
                            else {
                                std::cout << argss[index];
                            }
                        }
                        else {
                            if (numeric) {
                                std::cout << std::fixed
                                          << std::setprecision(precision)
                                          << std::setw(width)
                                          << std::stod(argss[index]);
                            }
                            else {
                                std::cout << argss[index];
                            }
                        }

                        index++;
                        index %= argss_len;
                    }
                    else if (right == '}') {
                        std::cout << argss[index];
                        index++;
                        index %= argss_len;
                    }
                    else {
                        if (right != '}' && right != '>') {
                            throw std::invalid_argument(
                                "syntax error in format string, "
                                "missing closing brace"
                            );
                        }
                        else {
                            throw std::invalid_argument(
                                "syntax error in format string, "
                                "missing format signifier (:)"
                            );
                        }
                    }
                    inserted = true;
                }
            }
            else {
                width_str = "";
                std::cout << helpers::get_color_code(C) << ch
                          << helpers::get_color_code(Color::RESET);
            }
            cidx++;
        }
    }

    template <Color C, typename... ARGS>
    void writeln(std::string const& fmt, ARGS... args)
    {
        std::cout << "\n";
        write<C>(fmt, args...);
        std::cout << '\n';
    }

    template <Color C, typename... ARGS>
    void writefl(std::string const& fmt, ARGS... args)
    {
        write<C>(fmt, args...);
        std::cout << std::flush;
    }

    template <typename... Args>
    void sync_print(Args... args)
    {
        static std::mutex print_mutex;
        std::lock_guard<std::mutex> guard(print_mutex);
        std::ostringstream oss;
        (oss << ... << args) << '\n';
        std::cout << oss.str();
    }

}   // namespace simbi::util

#endif
