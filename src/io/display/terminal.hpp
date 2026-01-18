// =============================================================================
// terminal.hpp
//
// terminal capability detection and ansi color palette.
// provides platform-independent terminal size queries and a gorgeous
// 256-color scheme based on catppuccin mocha + elegant gold accents.
//
// usage:
//   std::int64_t w = terminal_t::width();
//   std::cout << color::header() << "Title" << color::reset();
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace simbi::display {

    // ansi escape sequences
    namespace ansi {
        constexpr const char* CLEAR_SCREEN = "\033[H\033[J";
        constexpr const char* HIDE_CURSOR  = "\033[?25l";
        constexpr const char* SHOW_CURSOR  = "\033[?25h";
        constexpr const char* RESET        = "\033[0m";
        constexpr const char* BOLD         = "\033[1m";
    } // namespace ansi

        // elegant + modern 256-color palette
        // sophisticated gold headers with soft blue-gray borders
        namespace color
    {
        // structural elements
        inline std::string header()
        {
            return "\033[1;38;5;220m";
        } // bold gold/amber
        inline std::string border()
        {
            return "\033[38;5;67m";
        } // soft blue-gray
        inline std::string data()
        {
            return "\033[38;5;252m";
        } // light gray
        inline std::string title()
        {
            return "\033[1;38;5;183m";
        } // bold lavender

        // message types
        inline std::string info()
        {
            return "\033[38;5;117m";
        } // sky blue
        inline std::string success()
        {
            return "\033[38;5;158m";
        } // mint green
        inline std::string warning()
        {
            return "\033[38;5;215m";
        } // warm amber
        inline std::string error()
        {
            return "\033[38;5;210m";
        } // soft red

        // progress bar
        inline std::string progress_filled()
        {
            return "\033[38;5;183m";
        } // lavender
        inline std::string progress_mid()
        {
            return "\033[38;5;147m";
        } // light lavender
        inline std::string progress_empty()
        {
            return "\033[38;5;240m";
        } // dark gray

        // utility
        inline std::string reset()
        {
            return "\033[0m";
        }
        inline std::string bold()
        {
            return "\033[1m";
        }
    } // namespace color

    // terminal capability detection
    struct terminal_t
    {
        // get current terminal dimensions
        static std::int64_t width();
        static std::int64_t height();

        // feature detection
        static bool supports_unicode();
        static bool supports_256_colors();

        // adaptive padding based on terminal width
        // wide (>120): 4 spaces, normal (80-120): 3 spaces, narrow (<80): 2 spaces
        static std::int64_t adaptive_padding();
    };

} // namespace simbi::display


