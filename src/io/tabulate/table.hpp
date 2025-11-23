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
        enum class border_style_t {
            None,
            Simple,      // ascii safe: + - |
            Elegant,     // ascii safe but prettier
            Modern,      // unicode with ascii fallback
            Cyberpunk,   // futuristic styling
            Classic,     // traditional double lines
            Minimal      // clean and minimal
        };

        enum class table_theme_t {
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

        enum class display_mode_t {
            Static,   // prstd::int64_t once at current cursor position
            Dynamic   // clear screen and update in place
        };

        enum class progress_bar_t {
            Enabled,
            Disabled,
        };

        enum class color_t {
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
        std::string get_color_code(color_t color);
        std::string get_bg_color_code(color_t color);
        std::string reset_color();
        std::string bold();
        std::string italic();
        std::string underline();

        // sophisticated border character sets
        struct border_chars_t {
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
        struct theme_config_t {
            // colors
            color_t header_color;
            color_t text_color;
            color_t border_color;
            color_t title_color;
            color_t accent_color;

            // message colors
            color_t info_color;
            color_t success_color;
            color_t warning_color;
            color_t error_color;
            color_t debug_color;

            // progress colors
            color_t progress_color;
            color_t progress_bg_color;

            // styling
            border_style_t border_style;
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
            color_t custom_color  = color_t::Default;
            bool use_custom_color = false;

            Message(MessageType t, const std::string& txt)
                : type(t),
                  text(txt),
                  timestamp(std::chrono::system_clock::now())
            {
            }

            Message(MessageType t, const std::string& txt, color_t color)
                : type(t),
                  text(txt),
                  timestamp(std::chrono::system_clock::now()),
                  custom_color(color),
                  use_custom_color(true)
            {
            }
        };

        // utility class for terminal capabilities detection
        class terminal_capabilities_t
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
        class table_t
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
            table_theme_t current_theme = table_theme_t::Modern;
            theme_config_t theme_config;
            border_chars_t border_chars;
            display_mode_t display_mode = display_mode_t::Static;

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
            void apply_theme(table_theme_t theme);
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
                color_t start_color,
                color_t end_color
            ) const;

            // theme definitions
            theme_config_t get_theme_config(table_theme_t theme) const;

          public:
            table_t();
            explicit table_t(table_theme_t theme);
            table_t(
                border_style_t style,
                display_mode_t mode = display_mode_t::Static
            );

            ~table_t();

            // disable copy constructor and assignment to avoid issues
            table_t(const table_t&)            = delete;
            table_t& operator=(const table_t&) = delete;

            // move constructor and assignment
            table_t(table_t&&) noexcept;
            table_t& operator=(table_t&&) noexcept;

            // theme and styling configuration
            void set_theme(table_theme_t theme);
            void set_border_style(border_style_t style);
            void set_display_mode(display_mode_t mode);
            void customize_theme(const theme_config_t& config);

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
            void post_custom(const std::string& message, color_t color);
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
        class table_factory_t
        {
          public:
            // create themed tables with one function call
            static table_t create_cyberpunk_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static
            );
            static table_t create_elegant_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static,
                progress_bar_t progress_bar = progress_bar_t::Disabled
            );
            static table_t create_matrix_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static,
                progress_bar_t progress_bar = progress_bar_t::Disabled
            );
            static table_t create_ocean_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static,
                progress_bar_t progress_bar = progress_bar_t::Disabled
            );
            static table_t create_sunset_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static,
                progress_bar_t progress_bar = progress_bar_t::Disabled
            );
            static table_t create_minimal_table(
                const std::string& title    = "",
                display_mode_t display_mode = display_mode_t::Static,
                progress_bar_t progress_bar = progress_bar_t::Disabled
            );

            // create specialized tables for common use cases
            static table_t create_benchmark_table();
            static table_t create_system_info_table();
            static table_t create_log_table();
            static table_t create_status_table();
            static table_t create_data_table();
        };

    }   // namespace io
}   // namespace simbi

#endif   // MODERN_TABLE_HPP
