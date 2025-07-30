#include "table.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <sys/ttycom.h>
#include <unistd.h>
#endif

namespace simbi {
    namespace io {

        // initialize static members
        bool TerminalCapabilities::unicode_tested    = false;
        bool TerminalCapabilities::unicode_supported = false;

        // enhanced color functions with extended palette
        std::string get_color_code(Color color)
        {
            switch (color) {
                case Color::Default: return "\033[39m";
                case Color::Black: return "\033[30m";
                case Color::Red: return "\033[31m";
                case Color::Green: return "\033[32m";
                case Color::Yellow: return "\033[33m";
                case Color::Blue: return "\033[34m";
                case Color::Magenta: return "\033[35m";
                case Color::Cyan: return "\033[36m";
                case Color::White: return "\033[37m";
                case Color::BrightBlack: return "\033[90m";
                case Color::BrightRed: return "\033[91m";
                case Color::BrightGreen: return "\033[92m";
                case Color::BrightYellow: return "\033[93m";
                case Color::BrightBlue: return "\033[94m";
                case Color::BrightMagenta: return "\033[95m";
                case Color::BrightCyan: return "\033[96m";
                case Color::BrightWhite: return "\033[97m";

                // extended colors using 256-color palette
                case Color::DarkGray: return "\033[38;5;240m";
                case Color::LightGray: return "\033[38;5;250m";
                case Color::Orange: return "\033[38;5;208m";
                case Color::Purple: return "\033[38;5;129m";
                case Color::Pink: return "\033[38;5;205m";
                case Color::Lime: return "\033[38;5;154m";
                case Color::Teal: return "\033[38;5;80m";
                case Color::Navy: return "\033[38;5;17m";

                default: return "\033[0m";
            }
        }

        std::string get_bg_color_code(Color color)
        {
            switch (color) {
                case Color::Default: return "\033[49m";
                case Color::Black: return "\033[40m";
                case Color::Red: return "\033[41m";
                case Color::Green: return "\033[42m";
                case Color::Yellow: return "\033[43m";
                case Color::Blue: return "\033[44m";
                case Color::Magenta: return "\033[45m";
                case Color::Cyan: return "\033[46m";
                case Color::White: return "\033[47m";
                case Color::BrightBlack: return "\033[100m";
                case Color::BrightRed: return "\033[101m";
                case Color::BrightGreen: return "\033[102m";
                case Color::BrightYellow: return "\033[103m";
                case Color::BrightBlue: return "\033[104m";
                case Color::BrightMagenta: return "\033[105m";
                case Color::BrightCyan: return "\033[106m";
                case Color::BrightWhite: return "\033[107m";
                default: return "\033[49m";
            }
        }

        std::string reset_color() { return "\033[0m"; }
        std::string bold() { return "\033[1m"; }
        std::string italic() { return "\033[3m"; }
        std::string underline() { return "\033[4m"; }

        // terminal capabilities detection
        bool TerminalCapabilities::supports_unicode()
        {
            if (!unicode_tested) {
                // simple test: try to detect utf-8 support
                const char* lang     = std::getenv("LANG");
                const char* lc_all   = std::getenv("LC_ALL");
                const char* lc_ctype = std::getenv("LC_CTYPE");

                unicode_supported =
                    ((lang &&
                      std::string(lang).find("UTF-8") != std::string::npos) ||
                     (lc_all &&
                      std::string(lc_all).find("UTF-8") != std::string::npos) ||
                     (lc_ctype && std::string(lc_ctype).find("UTF-8") !=
                                      std::string::npos));

                unicode_tested = true;
            }
            return unicode_supported;
        }

        bool TerminalCapabilities::supports_256_colors()
        {
            const char* term = std::getenv("TERM");
            if (!term) {
                return false;
            }

            std::string term_str(term);
            return (
                term_str.find("256") != std::string::npos ||
                term_str.find("xterm") != std::string::npos ||
                term_str.find("screen") != std::string::npos
            );
        }

        bool TerminalCapabilities::supports_truecolor()
        {
            const char* colorterm = std::getenv("COLORTERM");
            if (!colorterm) {
                return false;
            }

            std::string colorterm_str(colorterm);
            return (colorterm_str == "truecolor" || colorterm_str == "24bit");
        }

        std::int64_t TerminalCapabilities::get_terminal_width()
        {
#ifdef _WIN32
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            if (GetConsoleScreenBufferInfo(
                    GetStdHandle(STD_OUTPUT_HANDLE),
                    &csbi
                )) {
                return csbi.srWindow.Right - csbi.srWindow.Left + 1;
            }
#else
            struct winsize ws;
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
                return ws.ws_col;
            }
#endif
            return 80;   // fallback
        }

        // sophisticated border character sets
        BorderChars get_border_chars(BorderStyle style)
        {
            bool use_unicode = TerminalCapabilities::supports_unicode();

            switch (style) {
                case BorderStyle::None:
                    return {
                      "",
                      "",
                      "",
                      "",
                      " ",
                      " ",
                      "",
                      "",
                      "",
                      "",
                      "",
                      "",
                      "",
                      "",
                      "",
                      ""
                    };

                case BorderStyle::Simple:
                    return {
                      "+",
                      "+",
                      "+",
                      "+",   // corners
                      "-",
                      "|",   // lines
                      "+",
                      "+",
                      "+",
                      "+",
                      "+",   // t-joints and cross
                      "+",
                      "+",
                      "+",   // header variants
                      "=",
                      "|"   // thick variants
                    };

                case BorderStyle::Elegant:
                    if (use_unicode) {
                        return {
                          "┌",
                          "┐",
                          "└",
                          "┘",   // corners
                          "─",
                          "│",   // lines
                          "┬",
                          "┴",
                          "├",
                          "┤",
                          "┼",   // t-joints and cross
                          "├",
                          "┤",
                          "┼",   // header variants
                          "━",
                          "┃"   // thick variants
                        };
                    }
                    else {
                        return {
                          "+",
                          "+",
                          "+",
                          "+",   // ascii fallback
                          "-",
                          "|",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "=",
                          "|"
                        };
                    }

                case BorderStyle::Modern:
                    if (use_unicode) {
                        return {
                          "╭",
                          "╮",
                          "╰",
                          "╯",   // rounded corners
                          "─",
                          "│",   // lines
                          "┬",
                          "┴",
                          "├",
                          "┤",
                          "┼",   // t-joints
                          "┝",
                          "┥",
                          "┿",   // header variants (slightly different)
                          "━",
                          "┃"   // thick variants
                        };
                    }
                    else {
                        return {
                          "+",
                          "+",
                          "+",
                          "+",
                          "-",
                          "|",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "=",
                          "|"
                        };
                    }

                case BorderStyle::Cyberpunk:
                    if (use_unicode) {
                        return {
                          "╔",
                          "╗",
                          "╚",
                          "╝",   // double line corners
                          "═",
                          "║",   // double lines
                          "╦",
                          "╩",
                          "╠",
                          "╣",
                          "╬",   // double t-joints
                          "╟",
                          "╢",
                          "╫",   // mixed variants for headers
                          "━",
                          "┃"   // thick single lines for accent
                        };
                    }
                    else {
                        return {
                          "#",
                          "#",
                          "#",
                          "#",   // ascii cyberpunk style
                          "=",
                          "#",
                          "#",
                          "#",
                          "#",
                          "#",
                          "#",
                          "#",
                          "#",
                          "#",
                          "=",
                          "#"
                        };
                    }

                case BorderStyle::Classic:
                    if (use_unicode) {
                        return {
                          "╔",
                          "╗",
                          "╚",
                          "╝",   // classic double lines
                          "═",
                          "║",
                          "╦",
                          "╩",
                          "╠",
                          "╣",
                          "╬",
                          "╠",
                          "╣",
                          "╬",
                          "═",
                          "║"
                        };
                    }
                    else {
                        return {
                          "+",
                          "+",
                          "+",
                          "+",
                          "=",
                          "|",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "+",
                          "=",
                          "|"
                        };
                    }

                case BorderStyle::Minimal:
                    return {
                      " ",
                      " ",
                      " ",
                      " ",   // no corners
                      "─",
                      " ",   // minimal lines
                      " ",
                      " ",
                      " ",
                      " ",
                      " ",   // no intersections
                      " ",
                      " ",
                      " ",
                      "─",
                      " "
                    };

                default: return get_border_chars(BorderStyle::Simple);
            }
        }

        // theme configurations for gorgeous styling
        ThemeConfig Table::get_theme_config(TableTheme theme) const
        {
            switch (theme) {
                case TableTheme::Cyberpunk:
                    return {
                      .header_color          = Color::BrightWhite,
                      .text_color            = Color::BrightCyan,
                      .border_color          = Color::BrightGreen,
                      .title_color           = Color::BrightYellow,
                      .accent_color          = Color::BrightMagenta,
                      .info_color            = Color::BrightCyan,
                      .success_color         = Color::BrightGreen,
                      .warning_color         = Color::BrightYellow,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::BrightBlack,
                      .progress_color        = Color::BrightGreen,
                      .progress_bg_color     = Color::DarkGray,
                      .border_style          = BorderStyle::Cyberpunk,
                      .use_bold_header       = true,
                      .use_italic_title      = false,
                      .use_gradient_progress = true,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };

                case TableTheme::Elegant:
                    return {
                      .header_color          = Color::Yellow,
                      .text_color            = Color::White,
                      .border_color          = Color::Yellow,
                      .title_color           = Color::BrightWhite,
                      .accent_color          = Color::Yellow,
                      .info_color            = Color::BrightWhite,
                      .success_color         = Color::BrightGreen,
                      .warning_color         = Color::Orange,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::LightGray,
                      .progress_color        = Color::Yellow,
                      .progress_bg_color     = Color::DarkGray,
                      .border_style          = BorderStyle::Elegant,
                      .use_bold_header       = true,
                      .use_italic_title      = true,
                      .use_gradient_progress = false,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };

                case TableTheme::Matrix:
                    return {
                      .header_color          = Color::BrightGreen,
                      .text_color            = Color::Green,
                      .border_color          = Color::BrightGreen,
                      .title_color           = Color::BrightGreen,
                      .accent_color          = Color::Lime,
                      .info_color            = Color::BrightGreen,
                      .success_color         = Color::Lime,
                      .warning_color         = Color::BrightYellow,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::DarkGray,
                      .progress_color        = Color::BrightGreen,
                      .progress_bg_color     = Color::Black,
                      .border_style          = BorderStyle::Modern,
                      .use_bold_header       = true,
                      .use_italic_title      = false,
                      .use_gradient_progress = true,
                      .use_background_colors = true,
                      .padding               = 4,
                      .title_spacing         = 0,
                      .section_spacing       = 0
                    };

                case TableTheme::Ocean:
                    return {
                      .header_color          = Color::BrightCyan,
                      .text_color            = Color::Cyan,
                      .border_color          = Color::Blue,
                      .title_color           = Color::BrightBlue,
                      .accent_color          = Color::Teal,
                      .info_color            = Color::BrightBlue,
                      .success_color         = Color::BrightCyan,
                      .warning_color         = Color::BrightYellow,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::LightGray,
                      .progress_color        = Color::BrightCyan,
                      .progress_bg_color     = Color::Navy,
                      .border_style          = BorderStyle::Modern,
                      .use_bold_header       = false,
                      .use_italic_title      = true,
                      .use_gradient_progress = true,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };

                case TableTheme::Sunset:
                    return {
                      .header_color          = Color::Orange,
                      .text_color            = Color::BrightYellow,
                      .border_color          = Color::Red,
                      .title_color           = Color::BrightRed,
                      .accent_color          = Color::Pink,
                      .info_color            = Color::BrightYellow,
                      .success_color         = Color::BrightGreen,
                      .warning_color         = Color::Orange,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::LightGray,
                      .progress_color        = Color::Orange,
                      .progress_bg_color     = Color::Red,
                      .border_style          = BorderStyle::Elegant,
                      .use_bold_header       = true,
                      .use_italic_title      = false,
                      .use_gradient_progress = true,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };

                case TableTheme::Monochrome:
                    return {
                      .header_color          = Color::BrightWhite,
                      .text_color            = Color::White,
                      .border_color          = Color::LightGray,
                      .title_color           = Color::BrightWhite,
                      .accent_color          = Color::LightGray,
                      .info_color            = Color::LightGray,
                      .success_color         = Color::White,
                      .warning_color         = Color::LightGray,
                      .error_color           = Color::BrightWhite,
                      .debug_color           = Color::DarkGray,
                      .progress_color        = Color::White,
                      .progress_bg_color     = Color::DarkGray,
                      .border_style          = BorderStyle::Minimal,
                      .use_bold_header       = true,
                      .use_italic_title      = true,
                      .use_gradient_progress = false,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };

                case TableTheme::Modern:
                default:
                    return {
                      .header_color          = Color::BrightCyan,
                      .text_color            = Color::White,
                      .border_color          = Color::BrightBlue,
                      .title_color           = Color::BrightBlue,
                      .accent_color          = Color::BrightCyan,
                      .info_color            = Color::BrightBlue,
                      .success_color         = Color::BrightGreen,
                      .warning_color         = Color::BrightYellow,
                      .error_color           = Color::BrightRed,
                      .debug_color           = Color::LightGray,
                      .progress_color        = Color::BrightBlue,
                      .progress_bg_color     = Color::DarkGray,
                      .border_style          = BorderStyle::Modern,
                      .use_bold_header       = false,
                      .use_italic_title      = false,
                      .use_gradient_progress = false,
                      .use_background_colors = false,
                      .padding               = 4,
                      .title_spacing         = 1,
                      .section_spacing       = 1
                    };
            }
        }

        // constructors
        Table::Table() { set_theme(TableTheme::Modern); }

        Table::Table(TableTheme theme) { set_theme(theme); }

        Table::Table(BorderStyle style, DisplayMode mode)
        {
            display_mode = mode;
            set_theme(TableTheme::Modern);
            theme_config.border_style = style;
            update_border_characters();
        }

        Table::~Table()
        {
            // show cursor if we're in dynamic mode
            if (display_mode == DisplayMode::Dynamic) {
                std::cout << "\033[?25h" << std::flush;
            }
        }

        Table::Table(Table&& other) noexcept
        {
            table_data        = std::move(other.table_data);
            column_alignments = std::move(other.column_alignments);
            column_widths     = std::move(other.column_widths);
            min_column_widths = std::move(other.min_column_widths);
            max_column_widths = std::move(other.max_column_widths);
            has_header        = other.has_header;
            current_theme     = other.current_theme;
            theme_config      = other.theme_config;
            border_chars      = other.border_chars;
            display_mode      = other.display_mode;
            title             = std::move(other.title);
            subtitle          = std::move(other.subtitle);
            footer            = std::move(other.footer);
            // copy other members...
        }

        Table& Table::operator=(Table&& other) noexcept
        {
            if (this != &other) {
                table_data        = std::move(other.table_data);
                column_alignments = std::move(other.column_alignments);
                column_widths     = std::move(other.column_widths);
                // copy other members...
            }
            return *this;
        }

        void Table::print_horizontal_border(
            std::ostream& os,
            const std::string& char_to_use,
            std::int64_t length
        ) const
        {
            // handle unicode characters properly by using loops instead of
            // string multiplication
            for (std::int64_t i = 0; i < length; ++i) {
                os << char_to_use;
            }
        }

        void Table::print_border_line(
            std::ostream& os,
            bool is_top,
            bool is_bottom,
            bool is_header_separator
        ) const
        {
            if (theme_config.border_style == BorderStyle::None) {
                return;
            }

            os << get_color_code(theme_config.border_color);

            // left corner
            if (is_top) {
                os << border_chars.top_left;
            }
            else if (is_bottom) {
                os << border_chars.bottom_left;
            }
            else if (is_header_separator) {
                os << border_chars.header_left;
            }
            else {
                os << border_chars.t_left;
            }

            // horizontal lines and intersections
            for (size_t i = 0; i < column_widths.size(); ++i) {
                std::int64_t line_length =
                    column_widths[i] + (2 * theme_config.padding);

                if (is_header_separator &&
                    !border_chars.thick_horizontal.empty()) {
                    print_horizontal_border(
                        os,
                        border_chars.thick_horizontal,
                        line_length
                    );
                }
                else {
                    print_horizontal_border(
                        os,
                        border_chars.horizontal,
                        line_length
                    );
                }

                if (i < column_widths.size() - 1) {
                    if (is_top) {
                        os << border_chars.t_down;
                    }
                    else if (is_bottom) {
                        os << border_chars.t_up;
                    }
                    else if (is_header_separator) {
                        os << border_chars.header_cross;
                    }
                    else {
                        os << border_chars.cross;
                    }
                }
            }

            // right corner
            if (is_top) {
                os << border_chars.top_right;
            }
            else if (is_bottom) {
                os << border_chars.bottom_right;
            }
            else if (is_header_separator) {
                os << border_chars.header_right;
            }
            else {
                os << border_chars.t_right;
            }

            os << reset_color() << "\n";
        }

        void Table::print_row(
            std::ostream& os,
            const std::vector<std::string>& row,
            bool is_header,
            size_t row_index
        ) const
        {
            // left border
            os << get_color_code(theme_config.border_color);
            if (!border_chars.thick_vertical.empty() && is_header) {
                os << border_chars.thick_vertical;
            }
            else {
                os << border_chars.vertical;
            }
            os << reset_color();

            for (size_t i = 0; i < column_widths.size(); ++i) {
                // select color and styling
                Color cell_color = is_header ? theme_config.header_color
                                             : theme_config.text_color;
                bool use_bold    = is_header && theme_config.use_bold_header;

                // apply styling
                std::string cell_prefix;
                std::string cell_suffix = reset_color();

                if (use_bold) {
                    cell_prefix += bold();
                }
                cell_prefix += get_color_code(cell_color);

                os << cell_prefix;

                // left padding
                for (std::int64_t p = 0; p < theme_config.padding; ++p) {
                    os << " ";
                }

                // cell content with alignment
                std::string cell_content = (i < row.size()) ? row[i] : "";
                Alignment align          = (i < column_alignments.size())
                                               ? column_alignments[i]
                                               : Alignment::Left;

                // apply zebra striping if enabled
                if (zebra_striping && !is_header) {
                    cell_content =
                        apply_zebra_styling(cell_content, row_index, true);
                }

                os << align_text(cell_content, column_widths[i], align);

                // right padding
                for (std::int64_t p = 0; p < theme_config.padding; ++p) {
                    os << " ";
                }

                os << cell_suffix;

                // vertical border
                os << get_color_code(theme_config.border_color);
                if (!border_chars.thick_vertical.empty() && is_header) {
                    os << border_chars.thick_vertical;
                }
                else {
                    os << border_chars.vertical;
                }
                os << reset_color();
            }

            os << "\n";
        }

        void Table::print_title_section(std::ostream& os) const
        {
            if (title.empty() && subtitle.empty()) {
                return;
            }

            size_t total_width = calculate_total_width();

            // prstd::int64_t title
            if (!title.empty()) {
                std::string title_text = title;

                if (theme_config.use_italic_title) {
                    title_text = italic() + title_text + reset_color();
                }

                title_text = get_color_code(theme_config.title_color) +
                             title_text + reset_color();

                // center the title
                if (title.length() < total_width) {
                    size_t left_padding = (total_width - title.length()) / 2;
                    for (size_t i = 0; i < left_padding; ++i) {
                        os << " ";
                    }
                }

                os << title_text << "\n";
            }

            // prstd::int64_t subtitle
            if (!subtitle.empty()) {
                std::string subtitle_text =
                    get_color_code(theme_config.accent_color) + subtitle +
                    reset_color();

                // center the subtitle
                if (subtitle.length() < total_width) {
                    size_t left_padding = (total_width - subtitle.length()) / 2;
                    for (size_t i = 0; i < left_padding; ++i) {
                        os << " ";
                    }
                }

                os << subtitle_text << "\n";
            }

            // add spacing based on theme
            for (std::int64_t i = 0; i < theme_config.title_spacing; ++i) {
                os << "\n";
            }
        }

        void Table::print_footer_section(std::ostream& os) const
        {
            if (footer.empty()) {
                return;
            }

            size_t total_width = calculate_total_width();

            // add spacing
            for (std::int64_t i = 0; i < theme_config.section_spacing; ++i) {
                os << "\n";
            }

            std::string footer_text =
                get_color_code(theme_config.accent_color) + footer +
                reset_color();

            // center the footer
            if (footer.length() < total_width) {
                size_t left_padding = (total_width - footer.length()) / 2;
                for (size_t i = 0; i < left_padding; ++i) {
                    os << " ";
                }
            }

            os << footer_text << "\n";
        }

        std::string Table::align_text(
            const std::string& text,
            size_t width,
            Alignment align
        ) const
        {
            if (text.length() >= width) {
                if (wrap_text) {
                    return text.substr(0, width);
                }
                else {
                    return text.substr(0, width);
                }
            }

            size_t padding = width - text.length();

            switch (align) {
                case Alignment::Right: return std::string(padding, ' ') + text;
                case Alignment::Center: {
                    size_t left_pad  = padding / 2;
                    size_t right_pad = padding - left_pad;
                    return std::string(left_pad, ' ') + text +
                           std::string(right_pad, ' ');
                }
                case Alignment::Left:
                default: return text + std::string(padding, ' ');
            }
        }

        std::string
        Table::wrap_text_to_width(const std::string& text, size_t width) const
        {
            if (!wrap_text || text.length() <= width) {
                return text;
            }

            // simple word wrapping
            std::string result;
            size_t pos = 0;

            while (pos < text.length()) {
                if (pos + width >= text.length()) {
                    result += text.substr(pos);
                    break;
                }

                // find the last space within the width
                size_t break_pos = pos + width;
                while (break_pos > pos && text[break_pos] != ' ') {
                    break_pos--;
                }

                if (break_pos == pos) {
                    // no space found, break at width
                    break_pos = pos + width;
                }

                result += text.substr(pos, break_pos - pos);
                if (break_pos < text.length()) {
                    result += "\n";
                }

                pos = break_pos;
                if (pos < text.length() && text[pos] == ' ') {
                    pos++;   // skip the space
                }
            }

            return result;
        }

        std::string Table::apply_zebra_styling(
            const std::string& text,
            size_t row_index,
            bool is_data_row
        ) const
        {
            if (!zebra_striping || !is_data_row) {
                return text;
            }

            // apply alternating background colors for zebra striping
            if (row_index % 2 == 0) {
                return text;   // normal background
            }
            else {
                // subtle background color for alternating rows
                return get_bg_color_code(Color::DarkGray) + text +
                       reset_color();
            }
        }

        std::string Table::create_gradient_text(
            const std::string& text,
            Color start_color,
            Color /* end_color */
        ) const
        {
            // simple gradient effect - can be enhanced with truecolor support
            if (!TerminalCapabilities::supports_256_colors()) {
                return get_color_code(start_color) + text + reset_color();
            }

            // for now, just use the start color
            return get_color_code(start_color) + text + reset_color();
        }

        std::int64_t TerminalCapabilities::get_terminal_height()
        {
#ifdef _WIN32
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            if (GetConsoleScreenBufferInfo(
                    GetStdHandle(STD_OUTPUT_HANDLE),
                    &csbi
                )) {
                return csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
            }
#else
            struct winsize ws;
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
                return ws.ws_row;
            }
#endif
            return 24;   // fallback
        }

        // theme and styling methods
        void Table::set_theme(TableTheme theme)
        {
            current_theme = theme;
            theme_config  = get_theme_config(theme);
            update_border_characters();
        }

        void Table::apply_theme(TableTheme theme) { set_theme(theme); }

        void Table::update_border_characters()
        {
            border_chars = get_border_chars(theme_config.border_style);
        }

        void Table::set_border_style(BorderStyle style)
        {
            theme_config.border_style = style;
            update_border_characters();
        }

        void Table::set_display_mode(DisplayMode mode) { display_mode = mode; }

        void Table::customize_theme(const ThemeConfig& config)
        {
            theme_config = config;
            update_border_characters();
        }

        // table metadata methods
        void Table::set_title(const std::string& table_title)
        {
            title = table_title;
        }

        void Table::set_subtitle(const std::string& table_subtitle)
        {
            subtitle = table_subtitle;
        }

        void Table::set_footer(const std::string& table_footer)
        {
            footer = table_footer;
        }

        // data manipulation methods
        void Table::set_header(const std::vector<std::string>& header_row)
        {
            if (table_data.empty() || !has_header) {
                table_data.insert(table_data.begin(), header_row);
                has_header = true;
            }
            else {
                table_data[0] = header_row;
            }

            // resize column alignments if needed
            if (column_alignments.size() < header_row.size()) {
                column_alignments.resize(header_row.size(), Alignment::Left);
            }

            calculate_column_widths();
        }

        void Table::add_row(const std::vector<std::string>& row)
        {
            table_data.push_back(row);

            // resize column alignments if needed
            if (column_alignments.size() < row.size()) {
                column_alignments.resize(row.size(), Alignment::Left);
            }

            calculate_column_widths();
        }

        void Table::update_row(
            size_t row_index,
            const std::vector<std::string>& new_data
        )
        {
            if (row_index >= table_data.size()) {
                return;
            }

            table_data[row_index] = new_data;

            // resize column alignments if needed
            if (column_alignments.size() < new_data.size()) {
                column_alignments.resize(new_data.size(), Alignment::Left);
            }

            calculate_column_widths();
        }

        void Table::update_cell(
            size_t row_index,
            size_t col_index,
            const std::string& new_data
        )
        {
            if (row_index >= table_data.size() ||
                col_index >= table_data[row_index].size()) {
                return;
            }

            table_data[row_index][col_index] = new_data;
            calculate_column_widths();
        }

        void
        Table::insert_row(size_t position, const std::vector<std::string>& row)
        {
            if (position > table_data.size()) {
                position = table_data.size();
            }

            table_data.insert(table_data.begin() + position, row);
            calculate_column_widths();
        }

        void Table::remove_row(size_t row_index)
        {
            if (row_index >= table_data.size()) {
                return;
            }

            // if removing header, update has_header flag
            if (row_index == 0 && has_header) {
                has_header = false;
            }

            table_data.erase(table_data.begin() + row_index);
            calculate_column_widths();
        }

        void Table::clear_rows()
        {
            if (has_header && !table_data.empty()) {
                // keep header but clear data rows
                std::vector<std::string> header = table_data[0];
                table_data.clear();
                table_data.push_back(header);
            }
            else {
                // clear everything
                table_data.clear();
                has_header = false;
            }

            calculate_column_widths();
        }

        void Table::sort_by_column(size_t col_index, bool ascending)
        {
            if (table_data.empty() || col_index >= column_count()) {
                return;
            }

            // determine start index (skip header if present)
            size_t start_idx = has_header ? 1 : 0;

            if (start_idx >= table_data.size()) {
                return;
            }

            // sort data rows
            std::sort(
                table_data.begin() + start_idx,
                table_data.end(),
                [col_index, ascending](
                    const std::vector<std::string>& a,
                    const std::vector<std::string>& b
                ) {
                    if (col_index >= a.size() || col_index >= b.size()) {
                        return false;
                    }

                    if (ascending) {
                        return a[col_index] < b[col_index];
                    }
                    else {
                        return a[col_index] > b[col_index];
                    }
                }
            );
        }

        // column configuration methods
        void Table::set_column_alignment(size_t col_index, Alignment alignment)
        {
            if (column_alignments.size() <= col_index) {
                column_alignments.resize(col_index + 1, Alignment::Left);
            }

            column_alignments[col_index] = alignment;
        }

        void Table::set_column_width(size_t col_index, std::int64_t width)
        {
            if (column_widths.size() <= col_index) {
                column_widths.resize(col_index + 1, 0);
            }

            column_widths[col_index] = width;
        }

        void
        Table::set_min_column_width(size_t col_index, std::int64_t min_width)
        {
            if (min_column_widths.size() <= col_index) {
                min_column_widths.resize(col_index + 1, 0);
            }

            min_column_widths[col_index] = min_width;
            calculate_column_widths();
        }

        void
        Table::set_max_column_width(size_t col_index, std::int64_t max_width)
        {
            if (max_column_widths.size() <= col_index) {
                max_column_widths.resize(col_index + 1, 0);
            }

            max_column_widths[col_index] = max_width;
            calculate_column_widths();
        }

        void Table::auto_resize_columns(bool enable)
        {
            auto_resize_columns_ = enable;
            if (enable) {
                calculate_column_widths();
            }
        }

        // advanced table features
        void Table::enable_zebra_striping(bool enable)
        {
            zebra_striping = enable;
        }

        void Table::enable_text_wrapping(bool enable) { wrap_text = enable; }

        void Table::set_max_table_width(std::int64_t width)
        {
            max_table_width = width;
            if (width > 0) {
                calculate_column_widths();
            }
        }

        void Table::center_table(bool enable) { center_table_ = enable; }

        // message board functionality
        void Table::enable_message_board(bool enable)
        {
            show_message_board = enable;
        }

        void Table::set_message_board_title(const std::string& mb_title)
        {
            message_board_title = mb_title;
        }

        void Table::set_max_messages(size_t max)
        {
            max_messages = max;

            // trim messages if needed
            while (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::enable_timestamps(bool enable) { show_timestamps = enable; }

        void Table::enable_compact_messages(bool enable)
        {
            compact_messages = enable;
        }

        void Table::post_info(const std::string& message)
        {
            messages.emplace_back(MessageType::Info, message);

            // enforce max messages
            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::post_success(const std::string& message)
        {
            messages.emplace_back(MessageType::Success, message);

            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::post_warning(const std::string& message)
        {
            messages.emplace_back(MessageType::Warning, message);

            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::post_error(const std::string& message)
        {
            messages.emplace_back(MessageType::Error, message);

            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::post_debug(const std::string& message)
        {
            messages.emplace_back(MessageType::Debug, message);

            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::post_custom(const std::string& message, Color color)
        {
            messages.emplace_back(MessageType::Info, message, color);

            if (messages.size() > max_messages) {
                messages.erase(messages.begin());
            }
        }

        void Table::clear_messages() { messages.clear(); }

        // progress bar functionality
        void Table::enable_progress(bool enable) { show_progress = enable; }

        void Table::set_progress(std::int64_t percent)
        {
            progress_percent = std::clamp<std::int64_t>(percent, 0, 100);
        }

        void Table::set_progress_style(ProgressStyle style)
        {
            progress_style = style;
        }

        void Table::set_progress_description(const std::string& desc)
        {
            progress_description = desc;
        }

        void Table::set_progress_units(const std::string& units)
        {
            progress_units = units;
        }

        void Table::set_progress_speed(double speed) { progress_speed = speed; }

        void
        Table::set_estimated_time_remaining(const std::chrono::seconds& time)
        {
            estimated_time_remaining = time;
        }

        std::string Table::format_timestamp(
            const std::chrono::system_clock::time_point& tp
        ) const
        {
            auto time  = std::chrono::system_clock::to_time_t(tp);
            std::tm tm = *std::localtime(&time);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%H:%M:%S");
            return oss.str();
        }

        void Table::calculate_column_widths()
        {
            if (table_data.empty()) {
                column_widths.clear();
                return;
            }

            // reset widths
            column_widths.resize(table_data[0].size(), 0);

            // calculate max width for each column
            for (const auto& row : table_data) {
                for (size_t i = 0; i < row.size(); ++i) {
                    if (i >= column_widths.size()) {
                        column_widths.resize(i + 1, 0);
                    }
                    column_widths[i] =
                        std::max<size_t>(column_widths[i], row[i].length());
                }
            }

            // apply min/max constraints
            for (size_t i = 0; i < column_widths.size(); ++i) {
                if (i < min_column_widths.size()) {
                    column_widths[i] =
                        std::max(column_widths[i], min_column_widths[i]);
                }
                if (i < max_column_widths.size()) {
                    column_widths[i] =
                        std::min(column_widths[i], max_column_widths[i]);
                }
            }

            // apply auto-resize if enabled
            if (auto_resize_columns_) {
                std::int64_t total_width = calculate_total_width();
                if (max_table_width > 0 && total_width > max_table_width) {
                    // scale down columns proportionally
                    double scale_factor =
                        static_cast<double>(max_table_width) / total_width;
                    for (auto& width : column_widths) {
                        width = static_cast<std::int64_t>(width * scale_factor);
                    }
                }
            }
        }

        std::string Table::format_timestamp_duration(
            const std::chrono::seconds& duration
        ) const
        {
            auto hours =
                std::chrono::duration_cast<std::chrono::hours>(duration);
            auto minutes =
                std::chrono::duration_cast<std::chrono::minutes>(duration) %
                std::chrono::hours(1);
            auto secs = duration % std::chrono::minutes(1);

            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(2) << hours.count() << ":"
                << std::setfill('0') << std::setw(2) << minutes.count() << ":"
                << std::setfill('0') << std::setw(2) << secs.count();
            return oss.str();
        }

        // main rendering method
        void Table::print()
        {
            std::ostringstream ss;

            // calc left padding if table is centered
            size_t left_padding = 0;
            // if (center_table_) {
            //     size_t terminal_width =
            //         TerminalCapabilities::get_terminal_width();
            //     size_t table_width = calculate_total_width();
            //     if (terminal_width > table_width) {
            //         left_padding = (terminal_width - table_width) / 2;
            //     }
            // }

            // add padding string once
            std::string padding =
                center_table_ ? std::string(left_padding, ' ') : "";

            // for dynamic mode, clear screen and move to home
            if (display_mode == DisplayMode::Dynamic) {
                ss << "\033[H\033[J";   // home and clear screen
                ss << "\033[?25l";      // hide cursor
            }

            // prstd::int64_t title
            if (!title.empty()) {
                size_t total_width   = calculate_total_width();
                size_t title_padding = (total_width - title.length()) / 2;
                ss << std::string(title_padding, ' ');
                ss << get_color_code(theme_config.title_color) << title
                   << reset_color() << "\n";
            }

            // prstd::int64_t table if not empty
            if (!table_data.empty()) {
                // top border
                print_separator(ss, true, false);

                // table rows
                for (size_t row_idx = 0; row_idx < table_data.size();
                     ++row_idx) {
                    const auto& row = table_data[row_idx];
                    bool is_header  = (row_idx == 0 && has_header);

                    // prstd::int64_t row
                    print_table_row(ss, row, is_header);

                    // prstd::int64_t separator unless it's the last row
                    if (row_idx < table_data.size() - 1) {
                        print_separator(ss, false, false);   // middle separator
                    }
                    else {
                        if (show_progress) {
                            print_separator(
                                ss,
                                false,
                                true,
                                true,
                                true
                            );   // special separator for progress
                        }
                    }
                }

                // progress bar integration (seamless extension)
                if (show_progress) {
                    print_progress_row(ss);
                    print_separator(
                        ss,
                        false,
                        true,
                        false
                    );   // bottom border after progress
                }
                else {
                    print_separator(ss, false, true);   // normal bottom border
                }
            }

            // message board as separate bordered section
            if (show_message_board && !messages.empty()) {
                print_message_board_section(ss);
            }

            // output everything at once
            std::cout << ss.str() << std::flush;
        }

        void Table::print_separator(
            std::ostream& os,
            bool is_top,
            bool is_bottom,
            bool include_t_up,
            bool at_middle
        ) const
        {
            if (theme_config.border_style == BorderStyle::None) {
                return;
            }

            os << get_color_code(theme_config.border_color);

            // choose the correct corner to start with
            if (is_top && !at_middle) {
                os << border_chars.top_left;
            }
            else if (is_bottom && !at_middle) {
                os << border_chars.bottom_left;
            }
            else {
                os << border_chars.t_left;   // middle left T-joint
            }

            // prstd::int64_t horizontal lines and T-joints/crosses
            for (size_t i = 0; i < column_widths.size(); ++i) {
                for (std::int64_t j = 0;
                     j < column_widths[i] + 2 * theme_config.padding;
                     ++j) {
                    os << border_chars.horizontal;
                }

                if (i < column_widths.size() - 1) {
                    // prstd::int64_t internal joints
                    if (is_top) {
                        os << border_chars.t_down;
                    }
                    else if (is_bottom) {
                        if (include_t_up) {
                            os << border_chars.t_up;
                        }
                        else {
                            os << border_chars.horizontal;
                        }
                    }
                    else {
                        os << border_chars.cross;
                    }
                }
            }

            // choose the correct corner to end with
            if (is_top && !at_middle) {
                os << border_chars.top_right;
            }
            else if (is_bottom && !at_middle) {
                os << border_chars.bottom_right;
            }
            else {
                os << border_chars.t_right;   // middle right T-joint
            }

            os << "\n";
            os << reset_color();
        }

        void Table::print_table_row(
            std::ostream& os,
            const std::vector<std::string>& row,
            bool is_header
        ) const
        {
            // left border
            os << get_color_code(theme_config.border_color)
               << border_chars.vertical << reset_color();

            for (size_t i = 0; i < column_widths.size(); ++i) {
                // padding and content
                os << std::string(theme_config.padding, ' ');

                // select color
                Color cell_color = is_header ? theme_config.header_color
                                             : theme_config.text_color;
                os << get_color_code(cell_color);

                // cell content with alignment
                std::string cell_content = (i < row.size()) ? row[i] : "";
                Alignment align          = (i < column_alignments.size())
                                               ? column_alignments[i]
                                               : Alignment::Left;
                os << align_text(cell_content, column_widths[i], align);

                os << reset_color();
                os << std::string(theme_config.padding, ' ');

                // column separator if not last column
                if (i < column_widths.size() - 1) {
                    os << get_color_code(theme_config.border_color)
                       << border_chars.vertical << reset_color();
                }
            }

            // right border
            os << get_color_code(theme_config.border_color)
               << border_chars.vertical << reset_color();
            os << "\n";
        }

        void Table::print_progress_row(std::ostream& os) const
        {
            if (!show_progress) {
                return;
            }

            size_t total_width     = calculate_total_width();
            std::int64_t bar_width = static_cast<std::int64_t>(total_width) -
                                     2;   // account for left/right borders
            std::int64_t pos = bar_width * progress_percent / 100;

            // prstd::int64_t T-jostd::int64_t separator before progress bar
            os << get_color_code(theme_config.border_color);
            os << border_chars.t_left;   // left T-joint
            for (std::int64_t i = 0; i < bar_width; ++i) {
                os << border_chars.horizontal;
            }
            os << border_chars.t_right << "\n";   // right T-joint

            // prstd::int64_t progress bar row
            os << border_chars.vertical;
            os << get_color_code(theme_config.progress_color);

            // render progress based on style
            switch (progress_style) {
                case ProgressStyle::Bar:
                    for (std::int64_t ii = 0; ii < bar_width; ++ii) {
                        if (ii < pos) {
                            os << "=";
                        }
                        else if (ii == pos) {
                            os << ">";
                        }
                        else {
                            os << " ";
                        }
                    }
                    break;
                case ProgressStyle::Blocks:
                    for (std::int64_t ii = 0; ii < bar_width; ++ii) {
                        if (ii < pos) {
                            os << "█";
                        }
                        else if (ii == pos) {
                            os << ">";
                        }
                        else {
                            os << " ";
                        }
                    }
                    break;
                case ProgressStyle::Dots:
                    for (std::int64_t ii = 0; ii < bar_width; ++ii) {
                        if (ii < pos) {
                            os << "●";
                        }
                        else if (ii == pos) {
                            os << "○";
                        }
                        else {
                            os << " ";
                        }
                    }
                    break;
                case ProgressStyle::Percentage:
                    os << std::setw(bar_width) << std::setfill(' ')
                       << progress_percent << "%";
                    break;
                case ProgressStyle::Arrow:
                    for (std::int64_t ii = 0; ii < bar_width; ++ii) {
                        if (ii < pos) {
                            os << "→";
                        }
                        else if (ii == pos) {
                            os << ">";
                        }
                        else {
                            os << " ";
                        }
                    }
                    break;
                case ProgressStyle::Gradient: {
                    for (std::int64_t ii = 0; ii < bar_width; ++ii) {
                        if (ii < pos) {
                            // create a gradient effect
                            float ratio = static_cast<float>(ii) / bar_width;
                            std::int64_t red =
                                static_cast<std::int64_t>(255 * (1 - ratio));
                            std::int64_t green =
                                static_cast<std::int64_t>(255 * ratio);
                            os << "\033[38;2;" << red << ";" << green << ";0m█";
                        }
                        else {
                            os << " ";
                        }
                    }
                    os << reset_color();
                    break;
                }
                default:
                    for (std::int64_t i = 0; i < bar_width; ++i) {
                        if (i < pos) {
                            os << "=";
                        }
                        else if (i == pos) {
                            os << ">";
                        }
                        else {
                            os << " ";
                        }
                    }
                    break;
            }

            os << reset_color() << get_color_code(theme_config.border_color)
               << border_chars.vertical;
            os << reset_color() << " " << progress_percent << " %" << "\n";
            os << reset_color();
        }

        void Table::print_message_board_section(std::ostream& os) const
        {
            if (!show_message_board || messages.empty()) {
                return;
            }

            size_t total_width = calculate_total_width();

            // top border
            os << get_color_code(theme_config.border_color);
            os << border_chars.top_left;
            for (size_t i = 0; i < total_width - 2; ++i) {
                os << border_chars.horizontal;
            }
            os << border_chars.top_right << "\n";

            // title row
            os << border_chars.vertical << " ";
            os << get_color_code(theme_config.title_color);
            size_t title_len = message_board_title.length();
            size_t left_pad  = (total_width - 4 - title_len) / 2;
            size_t right_pad = total_width - 4 - title_len - left_pad;
            os << std::string(left_pad, ' ') << message_board_title
               << std::string(right_pad, ' ');
            os << reset_color() << " "
               << get_color_code(theme_config.border_color)
               << border_chars.vertical << "\n";

            // separator after title
            os << border_chars.t_left;
            for (size_t i = 0; i < total_width - 2; ++i) {
                os << border_chars.horizontal;
            }
            os << border_chars.t_right << "\n";
            os << reset_color();

            // messages
            for (const auto& msg : messages) {
                Color msg_color;
                switch (msg.type) {
                    case MessageType::Info:
                        msg_color = theme_config.info_color;
                        break;
                    case MessageType::Success:
                        msg_color = theme_config.success_color;
                        break;
                    case MessageType::Warning:
                        msg_color = theme_config.warning_color;
                        break;
                    case MessageType::Error:
                        msg_color = theme_config.error_color;
                        break;
                    default: msg_color = theme_config.debug_color; break;
                }

                // split message by newlines and prstd::int64_t each line
                // separately
                std::istringstream stream(msg.text);
                std::string line;
                while (std::getline(stream, line)) {
                    os << get_color_code(theme_config.border_color)
                       << border_chars.vertical << " ";
                    os << get_color_code(msg_color);

                    if (line.length() > total_width - 4) {
                        line = line.substr(0, total_width - 7) + "...";
                    }
                    else {
                        line +=
                            std::string(total_width - 4 - line.length(), ' ');
                    }

                    os << line;
                    os << reset_color() << " "
                       << get_color_code(theme_config.border_color)
                       << border_chars.vertical << "\n";
                }
                os << reset_color();

                // add a separator line between messages (but not after the last
                // message)
                if (msg.type == MessageType::Error) {
                    if (&msg != &messages.back()) {
                        os << get_color_code(theme_config.border_color);
                        os << border_chars.vertical;
                        os << std::string(total_width - 2, ' ');
                        os << border_chars.vertical << "\n";
                        os << reset_color();
                    }
                }
            }

            // bottom border
            os << get_color_code(theme_config.border_color);
            os << border_chars.bottom_left;
            for (size_t i = 0; i < total_width - 2; ++i) {
                os << border_chars.horizontal;
            }
            os << border_chars.bottom_right << "\n";
            os << reset_color();
        }

        void Table::refresh()
        {
            // for dynamic tables, just call prstd::int64_t which handles screen
            // clearing
            print();
        }

        std::string Table::to_string() const
        {
            std::ostringstream ss;

            // create a temporary copy to avoid modifying state
            auto temp_display_mode                 = display_mode;
            const_cast<Table*>(this)->display_mode = DisplayMode::Static;

            // render to string stream
            const_cast<Table*>(this)->print();

            // restore original display mode
            const_cast<Table*>(this)->display_mode = temp_display_mode;

            return ss.str();
        }

        void Table::print_to_file(const std::string& filename) const
        {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return;
            }

            // save the table content without colors to file
            auto temp_display_mode                 = display_mode;
            const_cast<Table*>(this)->display_mode = DisplayMode::Static;

            // temporarily disable colors for file output
            // (this could be enhanced to strip ANSI codes)

            std::streambuf* orig = std::cout.rdbuf();
            std::cout.rdbuf(file.rdbuf());

            const_cast<Table*>(this)->print();

            std::cout.rdbuf(orig);
            const_cast<Table*>(this)->display_mode = temp_display_mode;

            file.close();
        }

        // utility methods
        size_t Table::calculate_total_width() const
        {
            if (column_widths.empty()) {
                return 0;
            }

            size_t total = 0;
            for (const auto& width : column_widths) {
                total += width + (2 * theme_config.padding);
            }

            // add borders
            total += column_widths.size() + 1;

            return total;
        }

        void Table::set_minimum_width(size_t width)
        {
            size_t current_width = calculate_total_width();
            if (width <= current_width) {
                return;
            }

            // distribute extra width among columns
            if (!column_widths.empty()) {
                size_t extra_width      = width - current_width;
                size_t extra_per_column = extra_width / column_widths.size();
                size_t remainder        = extra_width % column_widths.size();

                for (size_t i = 0; i < column_widths.size(); ++i) {
                    column_widths[i] += extra_per_column;
                    if (i < remainder) {
                        column_widths[i] += 1;
                    }
                }
            }
        }

        size_t Table::row_count() const { return table_data.size(); }

        size_t Table::column_count() const
        {
            if (table_data.empty()) {
                return 0;
            }

            size_t max_columns = 0;
            for (const auto& row : table_data) {
                max_columns = std::max(max_columns, row.size());
            }

            return max_columns;
        }

        bool Table::is_empty() const { return table_data.empty(); }

        void Table::reserve_rows(size_t count) { table_data.reserve(count); }

        // factory methods for beautiful themed tables
        Table TableFactory::create_cyberpunk_table(
            const std::string& title,
            DisplayMode display_mode
        )
        {
            Table table(TableTheme::Cyberpunk);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(true);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_elegant_table(
            const std::string& title,
            DisplayMode display_mode,
            ProgressBar progress_bar
        )
        {
            Table table(TableTheme::Elegant);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(progress_bar != ProgressBar::Disabled);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_matrix_table(
            const std::string& title,
            DisplayMode display_mode,
            ProgressBar progress_bar
        )
        {
            Table table(TableTheme::Matrix);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(progress_bar != ProgressBar::Disabled);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_ocean_table(
            const std::string& title,
            DisplayMode display_mode,
            ProgressBar progress_bar
        )
        {
            Table table(TableTheme::Ocean);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(progress_bar != ProgressBar::Disabled);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_sunset_table(
            const std::string& title,
            DisplayMode display_mode,
            ProgressBar progress_bar
        )
        {
            Table table(TableTheme::Sunset);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(progress_bar != ProgressBar::Disabled);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_minimal_table(
            const std::string& title,
            DisplayMode display_mode,
            ProgressBar progress_bar
        )
        {
            Table table(TableTheme::Monochrome);
            if (!title.empty()) {
                table.set_title(title);
            }
            table.set_display_mode(display_mode);
            table.enable_message_board(true);
            table.enable_progress(progress_bar != ProgressBar::Disabled);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Bar);
            return table;
        }

        // specialized tables for common use cases
        Table TableFactory::create_benchmark_table()
        {
            auto table = create_cyberpunk_table("Simulation Benchmarks");
            table.enable_progress(true);
            table.enable_message_board(true);
            table.set_message_board_title("Simulation Messages");
            table.set_progress_style(ProgressStyle::Gradient);
            return table;
        }

        Table TableFactory::create_system_info_table()
        {
            auto table = create_elegant_table();
            table.set_column_alignment(0, Alignment::Left);
            table.set_column_alignment(1, Alignment::Left);
            return table;
        }

        Table TableFactory::create_log_table()
        {
            auto table = create_matrix_table("System Log");
            table.enable_message_board(true);
            table.enable_compact_messages(true);
            return table;
        }

        Table TableFactory::create_status_table()
        {
            auto table = create_ocean_table("Status Monitor");
            table.enable_progress(true);
            table.set_progress_style(ProgressStyle::Bar);
            table.enable_zebra_striping(true);
            return table;
        }

        Table TableFactory::create_data_table()
        {
            auto table = create_minimal_table();
            table.enable_zebra_striping(true);
            table.auto_resize_columns(true);
            return table;
        }

    }   // namespace io
}   // namespace simbi
