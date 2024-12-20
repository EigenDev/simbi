#ifndef TABULATE_HPP
#define TABULATE_HPP

#include "common/enums.hpp"   // for Color
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
    enum class BorderStyle {
        Simple,
        Double,
        Dashed
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
        Mixed
    };

    struct Intializers {
        BorderStyle style              = BorderStyle::Double;
        int pad                        = 2;
        bool showProgress              = true;
        ProgressBarStyle progressStyle = ProgressBarStyle::Bar;
        Color progressColor            = Color::WHITE;
        Color textColor                = Color::WHITE;
        Color separatorColor           = Color::WHITE;
        Color infoColor                = Color::WHITE;
        Color errorColor               = Color::RED;
        Color titleColor               = Color::WHITE;
        Color messageBoardColor        = Color::WHITE;
    };

    class PrettyTable
    {
      private:
        std::vector<std::vector<std::string>> table;
        std::vector<int> columnWidths;
        BorderStyle borderStyle;
        int padding;
        std::string title;
        std::string messageBoardTitle;
        std::vector<std::pair<LogLevel, std::string>> messages;
        std::mutex mtx;
        std::function<void()> customMessageHandler;
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

        void printSeparator() const
        {
            std::string corner, horizontal;
            switch (borderStyle) {
                case BorderStyle::Simple:
                    corner     = "+";
                    horizontal = "-";
                    break;
                case BorderStyle::Double:
                    corner     = "╬";
                    horizontal = "=";
                    break;
                case BorderStyle::Dashed:
                    corner     = "+";
                    horizontal = "-";
                    break;
            }
            std::cout << getColorCode(separatorColor);
            for (const auto& width : columnWidths) {
                std::cout << corner;
                std::cout
                    << std::string(width + 2 * padding + 2, horizontal[0]);
            }
            std::cout << corner << std::endl;
            std::cout << getColorCode(Color::DEFAULT);   // Reset color
        }

        void printMessageBoard() const
        {
            int totalWidth =
                std::accumulate(columnWidths.begin(), columnWidths.end(), 0) +
                columnWidths.size() * (2 * padding + 3) + 1;
            std::cout << getColorCode(separatorColor) << "╔";
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << "═";
            }
            std::cout << "╗" << std::endl;
            std::cout
                << "║ " << getColorCode(messageBoardColor)
                << std::setw((totalWidth - 4 + messageBoardTitle.size()) / 2)
                << std::right << messageBoardTitle
                << std::setw((totalWidth + 5 - messageBoardTitle.size()) / 2)
                << getColorCode(Color::DEFAULT) << getColorCode(separatorColor)
                << " ║" << std::endl;
            std::cout << "╠";
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << "═";
            }
            std::cout << "╣" << std::endl;
            std::cout << getColorCode(Color::DEFAULT);   // Reset color

            bool hasErrors = false;
            for (const auto& message : messages) {
                if (message.first == LogLevel::Error) {
                    hasErrors = true;
                    break;
                }
            }

            auto printMessage = [&](const std::string& message, Color color) {
                std::istringstream stream(message);
                std::string line;
                while (std::getline(stream, line)) {
                    std::cout << getColorCode(separatorColor) << "║ "
                              << getColorCode(color)
                              << std::setw(totalWidth - 4) << std::left << line
                              << getColorCode(Color::DEFAULT)
                              << getColorCode(separatorColor) << " ║"
                              << getColorCode(Color::DEFAULT) << std::endl;
                }
            };

            for (const auto& message : messages) {
                if (message.first == LogLevel::Info) {
                    printMessage(message.second, infoColor);
                }
            }

            if (hasErrors) {
                std::cout << getColorCode(separatorColor) << "╠";
                for (int i = 0; i < totalWidth - 2; ++i) {
                    std::cout << "═";
                }
                std::cout << "╣" << std::endl;
                std::cout << getColorCode(Color::DEFAULT);   // Reset color

                bool firstError = true;
                for (const auto& message : messages) {
                    if (message.first == LogLevel::Error) {
                        if (!firstError) {
                            std::cout << getColorCode(separatorColor) << "╠";
                            for (int i = 0; i < totalWidth - 2; ++i) {
                                std::cout << "═";
                            }
                            std::cout << "╣" << std::endl;
                            // Reset color
                            std::cout << getColorCode(Color::DEFAULT);
                        }
                        firstError = false;
                        printMessage(message.second, errorColor);
                    }
                }
            }

            std::cout << getColorCode(separatorColor) << "╚";
            for (int i = 0; i < totalWidth - 2; ++i) {
                std::cout << "═";
            }
            std::cout << "╝" << std::endl;
            std::cout << getColorCode(Color::DEFAULT);   // Reset color
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
            std::cout << getColorCode(progressBarColor);
            switch (progressBarStyle) {
                case ProgressBarStyle::Bar:
                    std::cout << "[";
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
                    std::cout << "] " << progress << " %\n";
                    break;
                case ProgressBarStyle::Block:
                    std::cout << "[";
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
                    std::cout << "] " << progress << " %\n";
                    break;
                case ProgressBarStyle::Spinner:
                    static const char spinnerChars[] = {'|', '/', '-', '\\'};
                    std::cout << spinnerChars[progress % 4] << " " << progress
                              << " %\n";
                    break;
                case ProgressBarStyle::Arrow:
                    std::cout << "[";
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << ">";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    std::cout << "] " << progress << " %\n";
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
                    std::cout << "] " << progress << " %\n";
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
                    std::cout << "] " << progress << " %\n";
                    break;
            }
            std::cout << getColorCode(Color::DEFAULT);   // Reset color
        }

      public:
        PrettyTable(Intializers init)
            : borderStyle(init.style),
              padding(init.pad),
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
            if (index < table.size()) {
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
        }

        void setCustomMessageHandler(std::function<void()> handler)
        {
            customMessageHandler = handler;
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

        void print()
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "\033[H\033[J";   // Move cursor to home position
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
            printSeparator();
            for (const auto& row : table) {
                for (size_t i = 0; i < row.size(); ++i) {
                    std::cout << getColorCode(textColor) << "| "
                              << std::setw(columnWidths[i] + padding)
                              << std::left << row[i]
                              << std::string(padding, ' ') << " ";
                }
                std::cout << "|" << std::endl;
                printSeparator();
            }
            printProgressBar();
            printMessageBoard();
        }

        void startMessageBoard()
        {
            std::thread([this]() {
                while (true) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    if (customMessageHandler) {
                        customMessageHandler();
                    }
                    print();
                }
            }).detach();
        }
    };
}   // namespace simbi
#endif