/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       progress.hpp
 * @brief      implements a custom, portable progress bar
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
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <iostream>
#include <termios.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif   // Windows/Linux/Apple

namespace simbi {
    namespace helpers {
        inline int get_term_width()
        {
#if defined(_WIN32)
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
            const int columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
            return columns;
#elif defined(__linux__) || defined(__APPLE__)
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            return w.ws_col;
#endif
        }

        inline void progress_bar(double percentage)
        {
            static int termWidth = get_term_width();
            int cursorX          = 0;

#if defined(_WIN32)
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
            cursorX = csbi.dwCursorPosition.X;
#elif defined(__linux__) || defined(__APPLE__)
            struct termios term, oldTerm;
            tcgetattr(STDIN_FILENO, &term);
            oldTerm = term;
            term.c_lflag &= ~ICANON;
            term.c_lflag &= ~ECHO;
            tcsetattr(STDIN_FILENO, TCSANOW, &term);

            std::cout << "\x1B[6n";
            int cursorY;
            scanf("\x1B[%d;%dR", &cursorY, &cursorX);

            tcsetattr(STDIN_FILENO, TCSANOW, &oldTerm);
#endif

            // Hide the cursor
            std::cout << "\033[?25l";

            int availableWidth = termWidth - cursorX;
            std::cout << "\033[K";   // Clear from cursor to end of line
            std::cout << "[";
            int barWidth = availableWidth -
                           10;   // Adjust for percentage display and brackets
            int pos = barWidth * percentage;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) {
                    std::cout << "=";
                }
                else if (i == pos) {
                    std::cout << ">";
                }
                else {
                    std::cout << " ";
                }
            }
            std::cout << "] " << int(percentage * 100.0) << " %\r";
            std::cout.flush();
        }
    }   // namespace helpers
}   // namespace simbi

#endif