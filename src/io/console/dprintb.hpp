// =============================================================================
// drintb.hpp
//
// minimal colorized debug print utility for c++20
//
// provides formatted output with ansi color codes:
//   - write()   : print without newline
//   - writeln() : print with newline
//   - writefl() : print and flush
//
// usage:
//   write<color_t::RED>("error: {}", msg);
//   writeln("iteration {}, error = {:.3e}", i, err);
//   writefl<color_t::GREEN>("progress: {:>10.2f}%", pct);
//
// supported format specs:
//   {}          - default formatting
//   {:>10}      - right align, width 10
//   {:<10}      - left align, width 10
//   {:.3f}      - 3 decimal places, fixed
//   {:.2e}      - 2 decimal places, scientific
//   {:>10.3e}   - combined: right align, width 10, 3 decimals, scientific
// =============================================================================

#ifndef PRINTB_HPP
#define PRINTB_HPP

#include "utility/enums.hpp"
#include "utility/helpers.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace simbi::io {

    namespace detail {

        struct fmt_spec_t
        {
            char align      = '\0';
            int  width      = -1;
            int  precision  = -1;
            char type       = '\0';
            bool scientific = false;
        };

        inline fmt_spec_t parse_spec(std::string_view spec)
        {
            fmt_spec_t  result;
            std::size_t pos = 0;

            if (pos < spec.size() && (spec[pos] == '<' || spec[pos] == '>' || spec[pos] == '^')) {
                result.align = spec[pos++];
            }

            if (pos < spec.size() && std::isdigit(spec[pos])) {
                auto start = spec.data() + pos;
                while (pos < spec.size() && std::isdigit(spec[pos])) {
                    ++pos;
                }
                std::from_chars(start, spec.data() + pos, result.width);
            }

            if (pos < spec.size() && spec[pos] == '.') {
                ++pos;
                if (pos < spec.size() && std::isdigit(spec[pos])) {
                    auto start = spec.data() + pos;
                    while (pos < spec.size() && std::isdigit(spec[pos])) {
                        ++pos;
                    }
                    std::from_chars(start, spec.data() + pos, result.precision);
                }
            }

            if (pos < spec.size() && (spec[pos] == 'f' || spec[pos] == 'e' || spec[pos] == 'g')) {
                result.type       = spec[pos];
                result.scientific = (result.type == 'e');
            }

            return result;
        }

        template <typename T>
        std::string apply_spec(T&& value, const fmt_spec_t& spec)
        {
            std::ostringstream oss;

            if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
                if (spec.scientific) {
                    oss << std::scientific;
                }
                else if (spec.type == 'f') {
                    oss << std::fixed;
                }

                if (spec.precision >= 0) {
                    oss << std::setprecision(spec.precision);
                }
            }

            if (spec.width > 0) {
                if (spec.align == '<') {
                    oss << std::left;
                }
                else if (spec.align == '>') {
                    oss << std::right;
                }
                oss << std::setw(spec.width);
            }

            oss << std::forward<T>(value);
            return oss.str();
        }

        template <typename Tuple, std::size_t... Is>
        std::string format_impl(std::string_view fmt, const Tuple& args, std::index_sequence<Is...>)
        {
            std::string result;
            result.reserve(fmt.size() + sizeof...(Is) * 10);

            std::size_t arg_idx = 0;
            std::size_t pos     = 0;

            while (pos < fmt.size()) {
                if (fmt[pos] == '{' && pos + 1 < fmt.size()) {
                    if (fmt[pos + 1] == '}') {
                        // simple {}
                        if (arg_idx < sizeof...(Is)) {
                            ((arg_idx == Is
                                  ? (result += apply_spec(std::get<Is>(args), fmt_spec_t{}), 0)
                                  : 0),
                             ...);
                            ++arg_idx;
                        }
                        pos += 2;
                    }
                    else if (fmt[pos + 1] == ':') {
                        // {:spec}
                        std::size_t spec_start = pos + 2;
                        std::size_t spec_end   = fmt.find('}', spec_start);

                        if (spec_end != std::string_view::npos && arg_idx < sizeof...(Is)) {
                            auto spec_str = fmt.substr(spec_start, spec_end - spec_start);
                            auto spec     = parse_spec(spec_str);

                            ((arg_idx == Is ? (result += apply_spec(std::get<Is>(args), spec), 0)
                                            : 0),
                             ...);

                            ++arg_idx;
                            pos = spec_end + 1;
                        }
                        else {
                            result += fmt[pos++];
                        }
                    }
                    else {
                        result += fmt[pos++];
                    }
                }
                else {
                    result += fmt[pos++];
                }
            }

            return result;
        }

        template <typename... Args>
        std::string format(std::string_view fmt, Args&&... args)
        {
            if constexpr (sizeof...(args) == 0) {
                return std::string(fmt);
            }
            else {
                auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
                return format_impl(fmt, args_tuple, std::index_sequence_for<Args...>{});
            }
        }

    } // namespace detail

    template <color_t C = color_t::DEFAULT, typename... Args>
    void write(std::string_view fmt, Args&&... args)
    {
        std::cout << helpers::get_color_code(C) << detail::format(fmt, std::forward<Args>(args)...)
                  << helpers::get_color_code(color_t::RESET);
    }

    template <color_t C = color_t::DEFAULT, typename... Args>
    void writeln(std::string_view fmt, Args&&... args)
    {
        write<C>(fmt, std::forward<Args>(args)...);
        std::cout << '\n';
    }

    template <color_t C = color_t::DEFAULT, typename... Args>
    void writefl(std::string_view fmt, Args&&... args)
    {
        write<C>(fmt, std::forward<Args>(args)...);
        std::cout << std::flush;
    }

} // namespace simbi::io

#endif
