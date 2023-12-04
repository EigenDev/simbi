/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 * @file       printb.hpp
 * @brief      implementation of custom rust-like print functions w/ formatting
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */
#ifndef PRINTB_HPP
#define PRINTB_HPP

#include <iomanip>    // for scientific, precision
#include <iostream>   // for operator <<
#include <map>        // for allocator, map
#include <sstream>    // for operator>>, ws, basic_istream, basic_istringstream
#include <string>     // for string

enum class Color {
    DEFAULT,
    BLACK,
    BLUE,
    LIGHT_GREY,
    DARK_GREY,
    LIGHT_RED,
    LIGHT_GREEN,
    LIGHT_YELLOW,
    LIGHT_BLUE,
    LIGHT_MAGENTA,
    LIGHT_CYAN,
    WHITE,
    RED,
    GREEN,
    YELLOW,
    CYAN,
    MAGENTA,
    BOLD,
    RESET,
};

const std::string bold("\x1B[1m");
const std::string red("\x1B[0;31m");
const std::string green("\x1B[1;32m");
const std::string yellow("\x1B[1;33m");
const std::string cyan("\x1B[0;36m");
const std::string magenta("\x1B[0;35m");
const std::string def("\x1B[0;39m");
const std::string light_grey("\x1B[0;37m");
const std::string dark_grey("\x1B[0;90m");
const std::string light_red("\x1B[0;91m");
const std::string light_green("\x1B[0;92m");
const std::string light_yellow("\x1B[0;93m");
const std::string light_blue("\x1B[0;94m");
const std::string light_magenta("\x1B[0;95m");
const std::string light_cyan("\x1B[0;96m");
const std::string white("\x1B[0;97m");
const std::string blue("\x1B[0;34m");
const std::string reset("\x1B[0m");

const std::map<Color, std::string> color_map = {
  {Color::RED, red},
  {Color::DEFAULT, def},
  {Color::LIGHT_BLUE, light_blue},
  {Color::LIGHT_CYAN, light_cyan},
  {Color::LIGHT_GREEN, light_green},
  {Color::LIGHT_GREY, light_grey},
  {Color::LIGHT_MAGENTA, light_magenta},
  {Color::LIGHT_RED, light_red},
  {Color::LIGHT_YELLOW, light_yellow},
  {Color::WHITE, white},
  {Color::DARK_GREY, dark_grey},
  {Color::GREEN, green},
  {Color::YELLOW, yellow},
  {Color::CYAN, cyan},
  {Color::MAGENTA, magenta},
  {Color::BLUE, blue},
  {Color::RESET, reset},
  {Color::BOLD, bold}
};

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
    }   // namespace util

}   // namespace simbi

#include "printb.tpp"
#endif