#ifndef TABLE_HPP
#define TABLE_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace simbi {
    namespace io {

        // enhanced enums for beautiful styling
        enum class BorderStyle {
            None,
            Simple,      // ascii safe: + - |
            Elegant,     // ascii safe but prettier
            Modern,      // unicode with ascii fallback
            Cyberpunk,   // futuristic styling
            Classic,     // traditional double lines
            Minimal      // clean and minimal
        };

        enum class TableTheme {
            Classic,     // traditional blue/white
            Cyberpunk,   // neon green/cyan/magenta
            Modern,      // clean grays and blues
            Elegant,     // sophisticated gold/white
            Matrix,      // green matrix style
            Sunset,      // warm orange/red tones
            Ocean,       // cool blue/cyan tones
            Monochrome   // black and white
        };

        enum class Alignment {
            Left,
            Center,
            Right
        };

        enum class DisplayMode {
            Static,   // prstd::int64_t once at current cursor position
            Dynamic   // clear screen and update in place
        };

        enum class ProgressBar {
            Enabled,
            Disabled,
        };

        enum class Color {
            Default,
            Black,
            Red,
            Green,
            Yellow,
            Blue,
            Magenta,
            Cyan,
            White,
            BrightBlack,
            BrightRed,
            BrightGreen,
            BrightYellow,
            BrightBlue,
            BrightMagenta,
            BrightCyan,
            BrightWhite,
            // extended colors for theming
            DarkGray,
            LightGray,
            Orange,
            Purple,
            Pink,
            Lime,
            Teal,
            Navy
        };

        enum class MessageType {
            Info,
            Success,
            Warning,
            Error,
            Debug
        };

        enum class ProgressStyle {
            Bar,
            Spinner,
            Percentage,
            Blocks,
            Dots,
            Arrow,
            Gradient
        };

        // enhanced color helper functions
        std::string get_color_code(Color color);
        std::string get_bg_color_code(Color color);
        std::string reset_color();
        std::string bold();
        std::string italic();
        std::string underline();

        // sophisticated border character sets
        struct BorderChars {
            std::string top_left;
            std::string top_right;
            std::string bottom_left;
            std::string bottom_right;
            std::string horizontal;
            std::string vertical;
            std::string t_down;
            std::string t_up;
            std::string t_left;
            std::string t_right;
            std::string cross;

            // enhanced characters for better aesthetics
            std::string header_left;
            std::string header_right;
            std::string header_cross;
            std::string thick_horizontal;
            std::string thick_vertical;
        };

        // theme configuration structure
        struct ThemeConfig {
            // colors
            Color header_color;
            Color text_color;
            Color border_color;
            Color title_color;
            Color accent_color;

            // message colors
            Color info_color;
            Color success_color;
            Color warning_color;
            Color error_color;
            Color debug_color;

            // progress colors
            Color progress_color;
            Color progress_bg_color;

            // styling
            BorderStyle border_style;
            bool use_bold_header;
            bool use_italic_title;
            bool use_gradient_progress;
            bool use_background_colors;

            // spacing
            std::int64_t padding;
            std::int64_t title_spacing;
            std::int64_t section_spacing;
        };

        // message structure for the message board
        struct Message {
            MessageType type;
            std::string text;
            std::chrono::system_clock::time_point timestamp;
            Color custom_color    = Color::Default;
            bool use_custom_color = false;

            Message(MessageType t, const std::string& txt)
                : type(t),
                  text(txt),
                  timestamp(std::chrono::system_clock::now())
            {
            }

            Message(MessageType t, const std::string& txt, Color color)
                : type(t),
                  text(txt),
                  timestamp(std::chrono::system_clock::now()),
                  custom_color(color),
                  use_custom_color(true)
            {
            }
        };

        // utility class for terminal capabilities detection
        class TerminalCapabilities
        {
          private:
            static bool unicode_tested;
            static bool unicode_supported;

          public:
            static bool supports_unicode();
            static bool supports_256_colors();
            static bool supports_truecolor();
            static std::int64_t get_terminal_width();
            static std::int64_t get_terminal_height();
        };

        // main enhanced table class
        class Table
        {
          private:
            // table data
            std::vector<std::vector<std::string>> table_data;
            std::vector<Alignment> column_alignments;
            std::vector<std::int64_t> column_widths;
            std::vector<std::int64_t> min_column_widths;
            std::vector<std::int64_t> max_column_widths;
            bool has_header = false;

            // current theme and styling
            TableTheme current_theme = TableTheme::Modern;
            ThemeConfig theme_config;
            BorderChars border_chars;
            DisplayMode display_mode = DisplayMode::Static;

            // table metadata
            std::string title;
            std::string subtitle;
            std::string footer;

            // message board
            std::vector<Message> messages;
            std::string message_board_title = "Messages";
            size_t max_messages             = 10;
            bool show_message_board         = false;
            bool show_timestamps            = true;
            bool compact_messages           = false;

            // progress tracking
            std::int64_t progress_percent = 0;
            bool show_progress            = false;
            ProgressStyle progress_style  = ProgressStyle::Bar;
            std::string progress_description;
            std::string progress_units;
            double progress_speed = 0.0;
            std::chrono::seconds estimated_time_remaining{0};

            // advanced features
            bool auto_resize_columns_    = true;
            bool wrap_text               = false;
            bool zebra_striping          = false;
            std::int64_t max_table_width = 0;   // 0 = no limit
            bool center_table_           = false;

            // helper methods
            void apply_theme(TableTheme theme);
            void update_border_characters();
            void calculate_column_widths();
            void print_horizontal_border(
                std::ostream& os,
                const std::string& char_to_use,
                std::int64_t length
            ) const;
            void print_border_line(
                std::ostream& os,
                bool is_top,
                bool is_bottom,
                bool is_header_separator = false
            ) const;
            void print_row(
                std::ostream& os,
                const std::vector<std::string>& row,
                bool is_header,
                size_t row_index = 0
            ) const;
            void print_title_section(std::ostream& os) const;
            void print_footer_section(std::ostream& os) const;
            // unified rendering methods
            void print_separator(
                std::ostream& os,
                bool is_top       = false,
                bool is_bottom    = false,
                bool include_t_up = true,
                bool at_middle    = false
            ) const;
            void print_table_row(
                std::ostream& os,
                const std::vector<std::string>& row,
                bool is_header
            ) const;
            void print_progress_row(std::ostream& os) const;
            void print_message_board_section(std::ostream& os) const;

            // utility methods
            std::string format_timestamp(
                const std::chrono::system_clock::time_point& tp
            ) const;
            std::string align_text(
                const std::string& text,
                size_t width,
                Alignment align
            ) const;
            std::string format_timestamp_duration(
                const std::chrono::seconds& duration
            ) const;
            std::string
            wrap_text_to_width(const std::string& text, size_t width) const;
            std::string apply_zebra_styling(
                const std::string& text,
                size_t row_index,
                bool is_data_row
            ) const;
            std::string create_gradient_text(
                const std::string& text,
                Color start_color,
                Color end_color
            ) const;

            // theme definitions
            ThemeConfig get_theme_config(TableTheme theme) const;

          public:
            Table();
            explicit Table(TableTheme theme);
            Table(BorderStyle style, DisplayMode mode = DisplayMode::Static);

            ~Table();

            // disable copy constructor and assignment to avoid issues
            Table(const Table&)            = delete;
            Table& operator=(const Table&) = delete;

            // move constructor and assignment
            Table(Table&&) noexcept;
            Table& operator=(Table&&) noexcept;

            // theme and styling configuration
            void set_theme(TableTheme theme);
            void set_border_style(BorderStyle style);
            void set_display_mode(DisplayMode mode);
            void customize_theme(const ThemeConfig& config);

            // table metadata
            void set_title(const std::string& table_title);
            void set_subtitle(const std::string& table_subtitle);
            void set_footer(const std::string& table_footer);

            // data manipulation
            void set_header(const std::vector<std::string>& header_row);
            void add_row(const std::vector<std::string>& row);
            void update_row(
                size_t row_index,
                const std::vector<std::string>& new_data
            );
            void update_cell(
                size_t row_index,
                size_t col_index,
                const std::string& new_data
            );
            void
            insert_row(size_t position, const std::vector<std::string>& row);
            void remove_row(size_t row_index);
            void clear_rows();
            void sort_by_column(size_t col_index, bool ascending = true);

            // column configuration
            void set_column_alignment(size_t col_index, Alignment alignment);
            void set_column_width(size_t col_index, std::int64_t width);
            void set_min_column_width(size_t col_index, std::int64_t min_width);
            void set_max_column_width(size_t col_index, std::int64_t max_width);
            void auto_resize_columns(bool enable = true);

            // advanced table features
            void enable_zebra_striping(bool enable = true);
            void enable_text_wrapping(bool enable = true);
            void set_max_table_width(std::int64_t width);
            void center_table(bool enable = true);

            // message board functionality
            void enable_message_board(bool enable = true);
            void set_message_board_title(const std::string& mb_title);
            void set_max_messages(size_t max);
            void enable_timestamps(bool enable = true);
            void enable_compact_messages(bool enable = true);
            void post_info(const std::string& message);
            void post_success(const std::string& message);
            void post_warning(const std::string& message);
            void post_error(const std::string& message);
            void post_debug(const std::string& message);
            void post_custom(const std::string& message, Color color);
            void clear_messages();

            // progress bar functionality
            void enable_progress(bool enable = true);
            void set_progress(std::int64_t percent);
            void set_progress_style(ProgressStyle style);
            void set_progress_description(const std::string& desc);
            void set_progress_units(const std::string& units);
            void set_progress_speed(double speed);
            void set_estimated_time_remaining(const std::chrono::seconds& time);

            // rendering
            void print();
            void refresh();
            std::string to_string() const;
            void print_to_file(const std::string& filename) const;

            // utility
            size_t calculate_total_width() const;
            void set_minimum_width(size_t width);
            size_t row_count() const;
            size_t column_count() const;
            bool is_empty() const;
            void reserve_rows(size_t count);
        };

        // factory class for creating beautifully themed tables
        class TableFactory
        {
          public:
            // create themed tables with one function call
            static Table create_cyberpunk_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static
            );
            static Table create_elegant_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static,
                ProgressBar progress_bar = ProgressBar::Disabled
            );
            static Table create_matrix_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static,
                ProgressBar progress_bar = ProgressBar::Disabled
            );
            static Table create_ocean_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static,
                ProgressBar progress_bar = ProgressBar::Disabled
            );
            static Table create_sunset_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static,
                ProgressBar progress_bar = ProgressBar::Disabled
            );
            static Table create_minimal_table(
                const std::string& title = "",
                DisplayMode display_mode = DisplayMode::Static,
                ProgressBar progress_bar = ProgressBar::Disabled
            );

            // create specialized tables for common use cases
            static Table create_benchmark_table();
            static Table create_system_info_table();
            static Table create_log_table();
            static Table create_status_table();
            static Table create_data_table();
        };

    }   // namespace io
}   // namespace simbi

#endif   // MODERN_TABLE_HPP
