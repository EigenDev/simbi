#include "terminal.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace simbi::display {

    std::int64_t terminal_t::width()
    {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            return static_cast<std::int64_t>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
        }
#else
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
            return static_cast<std::int64_t>(ws.ws_col);
        }
#endif
        return 80; // safe fallback
    }

    std::int64_t terminal_t::height()
    {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            return static_cast<std::int64_t>(csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
        }
#else
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
            return static_cast<std::int64_t>(ws.ws_row);
        }
#endif
        return 24; // safe fallback
    }

    bool terminal_t::supports_unicode()
    {
        // check for utf-8 in locale environment variables
        const char* lang     = std::getenv("LANG");
        const char* lc_all   = std::getenv("LC_ALL");
        const char* lc_ctype = std::getenv("LC_CTYPE");

        return (
            (lang && std::string(lang).find("UTF-8") != std::string::npos) ||
            (lc_all && std::string(lc_all).find("UTF-8") != std::string::npos) ||
            (lc_ctype && std::string(lc_ctype).find("UTF-8") != std::string::npos)
        );
    }

    bool terminal_t::supports_256_colors()
    {
        const char* term = std::getenv("TERM");
        if (!term) {
            return false;
        }

        std::string term_str(term);
        return (
            term_str.find("256") != std::string::npos ||
            term_str.find("xterm") != std::string::npos ||
            term_str.find("screen") != std::string::npos ||
            term_str.find("tmux") != std::string::npos
        );
    }

    std::int64_t terminal_t::adaptive_padding()
    {
        std::int64_t w = width();

        // wide terminal: spacious padding
        if (w > 120) {
            return 4;
        }
        // normal terminal: balanced padding
        else if (w >= 80) {
            return 3;
        }
        // narrow terminal: compact padding
        else {
            return 2;
        }
    }

} // namespace simbi::display
