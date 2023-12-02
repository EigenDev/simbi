#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <iostream>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif // Windows/Linux/Apple

namespace simbi
{
    namespace helpers
    {
        inline int get_term_width() {
            #if defined(_WIN32)
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
            const int columns  = csbi.srWindow.Right - csbi.srWindow.Left + 1;
            return columns
            #elif defined(__linux__) || defined(__APPLE__)
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            return w.ws_col;
            #endif
        }

        inline void progress_bar(double percentage) {
            static int barWidth = get_term_width() / 5;
            std::cout << "[";
            int pos = barWidth * percentage;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(percentage * 100.0) << " %\r";
            std::cout.flush();
        }
    } // namespace helpers
} // namespace simbi


#endif