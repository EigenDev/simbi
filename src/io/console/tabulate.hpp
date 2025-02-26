/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            tabulate.hpp
 *  * @brief           tabulate - a simple, thread-safe, colorized table utility
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
#ifndef TABULATE_HPP
#define TABULATE_HPP

#include "core/types/utility/enums.hpp"   // for Color
#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace simbi {
    namespace helpers {
        // Forward declarations
        std::string getColorCode(Color color);
    };   // namespace helpers
}   // namespace simbi

using namespace simbi::helpers;

namespace simbi {
    enum class Alignment {
        Left,
        Center,
        Right
    };

    enum class BorderStyle {
        Simple,   // +---+---+
        Double,   // ╔═╗║╚═╝
        Fancy,    // ╔═══╗║╚═══╝
        Round,    // ╭─╮│╰─╯
        Thick,    // ┏━┓┃┗━┛
        Elegant   // ⎡⎺⎤⎢⎣⎯⎦
    };

    enum class LogLevel {
        Info,
        Error
    };

    enum class ProgressBarStyle {
        Bar,
        Block,
        Spinner,
        Arrow,
        Dots,
        Mixed,
        Gradient
    };

    struct Intializers {
        BorderStyle style              = BorderStyle::Double;
        int pad                        = 2;
        bool showProgress              = global::progress_bar_enabled;
        ProgressBarStyle progressStyle = ProgressBarStyle::Block;
        Color progressColor            = Color::LIGHT_YELLOW;
        Color textColor                = Color::WHITE;
        Color separatorColor           = Color::BOLD;
        Color infoColor                = Color::WHITE;
        Color errorColor               = Color::RED;
        Color titleColor               = Color::BOLD;
        Color messageBoardColor        = Color::LIGHT_CYAN;
    };

    struct CellFormat {
        Color fg_color = Color::DEFAULT;
        Color bg_color = Color::DEFAULT;
        bool bold      = false;
        bool italic    = false;
        bool underline = false;
    };

    enum class TableTheme {
        Default,
        Classic,
        Modern,
        Minimal,
        Cyberpunk
    };

    struct BorderCharacters {
        std::string top_left;
        std::string top_right;
        std::string bottom_left;
        std::string bottom_right;
        std::string horizontal;
        std::string vertical;
        std::string t_down;    // ╤
        std::string t_up;      // ╧
        std::string t_left;    // ╣
        std::string t_right;   // ╠
        std::string cross;     // ╬
    };

    class PrettyTable
    {
      private:
        bool has_header{false};
        Color header_color{Color::WHITE};
        std::vector<Alignment> column_alignments;
        TableTheme current_theme{TableTheme::Classic};

        std::vector<std::vector<std::string>> table;
        std::vector<int> columnWidths;
        BorderStyle borderStyle;
        int padding;
        std::string title;
        std::string messageBoardTitle;
        std::vector<std::pair<LogLevel, std::string>> messages;
        std::mutex mtx;
        int progress;
        bool showProgressBar;
        ProgressBarStyle progressBarStyle;
        Color progressBarColor;
        Color textColor;
        Color separatorColor;
        Color infoColor;
        Color errorColor;
        Color messageBoardColor;
        Color titleColor;
        std::vector<std::vector<CellFormat>> cell_formats;
        BorderCharacters current_border;

        // Add this method to set border characters based on style
        void updateBorderCharacters()
        {
            switch (borderStyle) {
                case BorderStyle::Simple:
                    current_border = {
                      "+",
                      "+",
                      "+",
                      "+",   // corners
                      "-",
                      "|",   // lines
                      "+",
                      "+",
                      "+",
                      "+",   // T-joints
                      "+"    // cross
                    };
                    break;
                case BorderStyle::Double:
                    current_border = {
                      "╔",
                      "╗",
                      "╚",
                      "╝",   // corners
                      "═",
                      "║",   // lines
                      "╤",
                      "╧",
                      "╣",
                      "╠",   // T-joints
                      "╬"    // cross
                    };
                    break;
                case BorderStyle::Round:
                    current_border = {
                      "╭",
                      "╮",
                      "╰",
                      "╯",   // corners
                      "─",
                      "│",   // lines
                      "┬",
                      "┴",
                      "┤",
                      "├",   // T-joints
                      "┼"    // cross
                    };
                    break;
                case BorderStyle::Thick:
                    current_border = {
                      "┏",
                      "┓",
                      "┗",
                      "┛",   // corners
                      "━",
                      "┃",   // lines
                      "┳",
                      "┻",
                      "┫",
                      "┣",   // T-joints
                      "╋"    // cross
                    };
                    break;
                case BorderStyle::Elegant:
                    current_border = {
                      "⎡",
                      "⎤",
                      "⎣",
                      "⎦",   // corners
                      "⎯",
                      "⎢",   // lines
                      "⎯",
                      "⎯",
                      "⎥",
                      "⎢",   // T-joints
                      "⎯"    // cross
                    };
                    break;
                case BorderStyle::Fancy:
                    current_border = {
                      "╔",
                      "╗",
                      "╚",
                      "╝",   // corners
                      "═",
                      "║",   // lines
                      "╦",
                      "╩",
                      "╣",
                      "╠",   // T-joints
                      "╬"    // cross
                    };
                    break;
                default:
                    current_border = {
                      "┌",
                      "┐",
                      "└",
                      "┘",   // corners
                      "─",
                      "│",   // lines
                      "┬",
                      "┴",
                      "┤",
                      "├",   // T-joints
                      "┼"    // cross
                    };
            }
        }

        void saveCursorPosition()
        {
            std::cout << "\033[s";   // Save cursor position
        }

        void restoreCursorPosition() const
        {
            std::cout << "\033[u";   // Restore cursor position
        }

        void moveCursorUp(int lines) const
        {
            if (lines > 0) {
                std::cout << "\033[" << lines << "A";
            }
        }

        void updateColumnWidths(const std::vector<std::string>& row)
        {
            if (columnWidths.size() < row.size()) {
                columnWidths.resize(row.size(), 0);
            }
            for (size_t i = 0; i < row.size(); ++i) {
                columnWidths[i] =
                    std::max(columnWidths[i], static_cast<int>(row[i].size()));
            }
        }

        void printSeparator(
            bool isTop        = false,
            bool isBottom     = false,
            bool include_t_up = true,
            bool at_middle    = false
        ) const
        {
            std::cout << getColorCode(separatorColor);

            // Choose the correct corner to start with
            if (isTop && !at_middle) {
                std::cout << current_border.top_left;
            }
            else if (isBottom && !at_middle) {
                std::cout << current_border.bottom_left;
            }
            else {
                std::cout << current_border.t_right;   // middle left T-joint
            }

            // Print horizontal lines and T-joints/crosses
            for (size_t i = 0; i < columnWidths.size(); ++i) {
                for (int j = 0; j < columnWidths[i] + 2 * padding + 2; ++j) {
                    std::cout << current_border.horizontal;
                }

                if (i < columnWidths.size() - 1) {
                    // Print internal joints
                    if (isTop) {
                        std::cout << current_border.t_down;
                    }
                    else if (isBottom) {
                        if (include_t_up) {
                            std::cout << current_border.t_up;
                        }
                        else {
                            std::cout << current_border.horizontal;
                        }
                    }
                    else {
                        std::cout << current_border.cross;
                    }
                }
            }

            // Choose the correct corner to end with
            if (isTop && !at_middle) {
                std::cout << current_border.top_right;
            }
            else if (isBottom && !at_middle) {
                std::cout << current_border.bottom_right;
            }
            else {
                std::cout << current_border.t_left;   // middle right T-joint
            }

            std::cout << std::endl;
            std::cout << getColorCode(Color::DEFAULT);
        }

        void printMessageBoard() const
        {
            int totalWidth =
                std::accumulate(columnWidths.begin(), columnWidths.end(), 0) +
                columnWidths.size() * (2 * padding + 3) + 1;

            // Top border
            std::cout << getColorCode(separatorColor)
                      << current_border.top_left;
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << current_border.horizontal;
            }
            std::cout << current_border.top_right << std::endl;

            // Title row
            std::cout
                << current_border.vertical << " "
                << getColorCode(messageBoardColor)
                << std::setw((totalWidth - 4 + messageBoardTitle.size()) / 2)
                << std::right << messageBoardTitle
                << std::setw((totalWidth + 5 - messageBoardTitle.size()) / 2)
                << getColorCode(Color::DEFAULT) << getColorCode(separatorColor)
                << " " << current_border.vertical << std::endl;

            // Separator after title
            std::cout << current_border.t_right;
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << current_border.horizontal;
            }
            std::cout << current_border.t_left << std::endl;
            std::cout << getColorCode(Color::DEFAULT);

            // Helper method for printing message lines
            auto printMessage = [&](const std::string& message, Color color) {
                std::istringstream stream(message);
                std::string line;
                while (std::getline(stream, line)) {
                    std::cout << getColorCode(separatorColor)
                              << current_border.vertical << " "
                              << getColorCode(color)
                              << std::setw(totalWidth - 4) << std::left << line
                              << getColorCode(Color::DEFAULT)
                              << getColorCode(separatorColor) << " "
                              << current_border.vertical
                              << getColorCode(Color::DEFAULT) << std::endl;
                }
            };

            // Print info messages
            for (const auto& message : messages) {
                if (message.first == LogLevel::Info) {
                    printMessage(message.second, infoColor);
                }
            }

            // Handle error messages
            bool hasErrors = false;
            for (const auto& message : messages) {
                if (message.first == LogLevel::Error) {
                    hasErrors = true;
                    break;
                }
            }

            if (hasErrors) {
                // Separator before errors
                std::cout << getColorCode(separatorColor)
                          << current_border.t_right;
                for (int i = 0; i < totalWidth - 2; ++i) {
                    std::cout << current_border.horizontal;
                }
                std::cout << current_border.t_left << std::endl;
                std::cout << getColorCode(Color::DEFAULT);

                bool firstError = true;
                for (const auto& message : messages) {
                    if (message.first == LogLevel::Error) {
                        if (!firstError) {
                            // Separator between errors
                            std::cout << getColorCode(separatorColor)
                                      << current_border.t_right;
                            for (int i = 0; i < totalWidth - 2; ++i) {
                                std::cout << current_border.horizontal;
                            }
                            std::cout << current_border.t_left << std::endl;
                            std::cout << getColorCode(Color::DEFAULT);
                        }
                        firstError = false;
                        printMessage(message.second, errorColor);
                    }
                }
            }

            // Bottom border
            std::cout << getColorCode(separatorColor)
                      << current_border.bottom_left;
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << current_border.horizontal;
            }
            std::cout << current_border.bottom_right << std::endl;
            std::cout << getColorCode(Color::DEFAULT);
        }

        void printProgressBar() const
        {
            if (!showProgressBar) {
                return;
            }
            int totalWidth =
                std::accumulate(columnWidths.begin(), columnWidths.end(), 0) +
                columnWidths.size() * (2 * padding + 3) + 1;
            int barWidth = totalWidth - 2;
            int pos      = barWidth * progress / 100;

            // Print separator before progress bar
            std::cout << getColorCode(separatorColor);
            std::cout << current_border.t_right;   // Left T-joint
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << current_border.horizontal;
            }
            std::cout << current_border.t_left << std::endl;   // Right T-joint

            // Print progress bar
            std::cout << current_border.vertical;
            std::cout << getColorCode(progressBarColor);
            switch (progressBarStyle) {
                case ProgressBarStyle::Bar:
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
                    break;
                case ProgressBarStyle::Block:
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << "█";
                        }
                        else if (i == pos) {
                            std::cout << ">";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    break;
                case ProgressBarStyle::Spinner:
                    static const char spinnerChars[] = {'|', '/', '-', '\\'};
                    std::cout << spinnerChars[progress % 4];
                    break;
                case ProgressBarStyle::Arrow:
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << ">";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    break;
                case ProgressBarStyle::Dots:
                    std::cout << "[";
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << ".";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    break;
                case ProgressBarStyle::Mixed:
                    std::cout << "[";
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << "■";
                        }
                        else if (i == pos) {
                            std::cout << ">";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    break;
                case ProgressBarStyle::Gradient:
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            // Gradient from red to green
                            float ratio = static_cast<float>(i) / pos;
                            std::cout << "\033[38;2;"
                                      << static_cast<int>(255 * (1 - ratio))
                                      << ";" << static_cast<int>(255 * ratio)
                                      << ";0m█";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    break;
            }
            std::cout << getColorCode(Color::DEFAULT)
                      << getColorCode(separatorColor) << current_border.vertical
                      << getColorCode(Color::DEFAULT) << progress << " %"
                      << std::endl;

            std::cout << getColorCode(Color::DEFAULT);   // Reset color
        }

      public:
        PrettyTable(Intializers init = {})
            : borderStyle(init.style),
              padding(init.pad),
              progress(0),
              showProgressBar(init.showProgress),
              progressBarStyle(init.progressStyle),
              progressBarColor(init.progressColor),
              textColor(init.textColor),
              separatorColor(init.separatorColor),
              infoColor(init.infoColor),
              errorColor(init.errorColor),
              messageBoardColor(init.messageBoardColor),
              titleColor(init.titleColor)
        {
            updateBorderCharacters();
        }

        void setTitle(const std::string& tableTitle) { title = tableTitle; }

        void setMessageBoardTitle(const std::string& boardTitle)
        {
            messageBoardTitle = boardTitle;
        }

        void addRow(const std::vector<std::string>& row)
        {
            std::lock_guard<std::mutex> lock(mtx);
            table.push_back(row);
            updateColumnWidths(row);
        }

        void updateRow(int index, const std::vector<std::string>& row)
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (static_cast<size_t>(index) < table.size()) {
                table[index] = row;
                updateColumnWidths(row);
            }
        }

        void postInfo(const std::string& message)
        {
            addMessage(LogLevel::Info, message);
        }

        void postError(const std::string& message)
        {
            addMessage(LogLevel::Error, message);
        }

        void addMessage(LogLevel level, const std::string& message)
        {
            std::lock_guard<std::mutex> lock(mtx);
            messages.emplace_back(level, message);
            if (messages.size() > 10) {
                messages.erase(messages.begin());
            }
        }

        void setProgress(int value)
        {
            std::lock_guard<std::mutex> lock(mtx);
            progress = value;
        }

        void setProgressBarColor(Color color) { progressBarColor = color; }

        void setTextColor(Color color) { textColor = color; }

        void setSeparatorColor(Color color) { separatorColor = color; }

        void setInfoColor(Color color) { infoColor = color; }

        void setErrorColor(Color color) { errorColor = color; }

        void setTheme(TableTheme theme)
        {
            current_theme = theme;
            switch (theme) {
                case TableTheme::Cyberpunk:
                    borderStyle      = BorderStyle::Thick;
                    progressBarColor = Color::CYAN;
                    textColor        = Color::LIGHT_YELLOW;
                    separatorColor   = Color::BOLD;
                    header_color     = Color::LIGHT_YELLOW;
                    progressBarStyle = ProgressBarStyle::Gradient;
                    break;
                case TableTheme::Modern:
                    borderStyle      = BorderStyle::Round;
                    progressBarColor = Color::BLUE;
                    textColor        = Color::WHITE;
                    separatorColor   = Color::LIGHT_GREY;
                    break;
                case TableTheme::Minimal:
                    borderStyle      = BorderStyle::Simple;
                    progressBarColor = Color::GREEN;
                    textColor        = Color::WHITE;
                    separatorColor   = Color::WHITE;
                    break;
                case TableTheme::Classic:
                    borderStyle      = BorderStyle::Double;
                    progressBarColor = Color::WHITE;
                    textColor        = Color::WHITE;
                    separatorColor   = Color::WHITE;
                    break;
                default: break;
            }
            updateBorderCharacters();
        }

        void setHeader(const std::vector<std::string>& header)
        {
            has_header = true;
            if (!table.empty()) {
                table.insert(table.begin(), header);
            }
            else {
                table.push_back(header);
            }
            updateColumnWidths(header);
        }

        void setColumnAlignment(size_t col, Alignment align)
        {
            if (col >= column_alignments.size()) {
                column_alignments.resize(col + 1, Alignment::Left);
            }
            column_alignments[col] = align;
        }

        void print()
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "\033[H\033[J";   // move cursor to home position
                                           // and clear screen
            if (!title.empty()) {
                int totalWidth = std::accumulate(
                                     columnWidths.begin(),
                                     columnWidths.end(),
                                     0
                                 ) +
                                 columnWidths.size() * (2 * padding + 3) + 1;
                std::cout << std::string((totalWidth - title.size()) / 2, ' ')
                          << getColorCode(titleColor) << title
                          << getColorCode(Color::DEFAULT) << std::endl;
            }
            printSeparator(true);   // top border

            for (size_t row_idx = 0; row_idx < table.size(); ++row_idx) {
                const auto& row = table[row_idx];
                // left broder
                std::cout << current_border.vertical;
                for (size_t i = 0; i < row.size(); ++i) {
                    // apply padding and cell content
                    std::cout << std::string(padding, ' ')
                              << getColorCode(
                                     row_idx == 0 && has_header ? header_color
                                                                : textColor
                                 )
                              << std::setw(columnWidths[i] + padding)
                              << std::left << row[i]
                              << std::string(padding, ' ')
                              << getColorCode(Color::DEFAULT);

                    // add column separator if not last column
                    if (i < row.size() - 1) {
                        std::cout << current_border.vertical;
                    }
                }

                // right border
                std::cout << current_border.vertical << std::endl;

                // print separator unless it's the last row
                if (row_idx < table.size() - 1) {
                    printSeparator(false, false);   // Middle separator
                }
                else {
                    if (showProgressBar) {
                        printSeparator(
                            false,
                            true,
                            true,
                            true
                        );   // Middle separator w/ t up enabled
                    }
                }
            }

            if (showProgressBar) {
                printProgressBar();
                printSeparator(false, true, false);
            }
            else {
                // print final bottom border
                printSeparator(false, true);
            }

            if (messages.size() > 0) {
                printMessageBoard();
            }
        }
    };
}   // namespace simbi
#endif