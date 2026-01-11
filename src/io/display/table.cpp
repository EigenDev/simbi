#include "table.hpp"

#include "renderer.hpp"
#include "terminal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace simbi::display {

    struct message_t
    {
        std::chrono::system_clock::time_point timestamp;
        message_type_t                        type;
        std::string                           text;
    };

    struct table_t::impl_t
    {
        std::string title;
        bool        dynamic_mode;

        std::vector<std::string> headers;
        std::vector<std::string> data_row;
        std::int64_t             progress_percent;

        std::deque<message_t> messages;

        renderer_t renderer;

        impl_t(const std::string& t, bool dynamic)
            : title(t), dynamic_mode(dynamic), progress_percent(0)
        {
        }

        // calculate how many messages we can display based on terminal height
        std::int64_t max_messages() const
        {
            std::int64_t height = terminal_t::height();

            // reserve space for:
            // - table (title + header + separator + data + progress + borders): 8 lines
            // - message board (title + borders): 3 lines
            // - bottom margin: 2 lines
            std::int64_t reserved               = 13;
            std::int64_t available_for_messages = height - reserved;

            // clamp to reasonable range
            if (available_for_messages < 3) {
                return 3; // minimum: always show at least 3 messages
            }
            if (available_for_messages > 20) {
                return 20; // maximum: don't let message board dominate
            }

            return available_for_messages;
        }

        void render()
        {
            std::int64_t term_width = terminal_t::width();

            // recalculate layout based on current terminal size
            renderer.calculate_layout(headers, data_row, term_width);

            std::ostringstream buf;

            // clear screen and hide cursor if dynamic mode
            if (dynamic_mode) {
                buf << ansi::CLEAR_SCREEN << ansi::HIDE_CURSOR;
            }

            // render main table
            render_simulation_table(buf);

            // render message board if we have messages
            if (!messages.empty()) {
                buf << "\n";
                render_message_board(buf);
            }

            // flush to stdout
            std::cout << buf.str() << std::flush;
        }

        void render_simulation_table(std::ostream& os)
        {
            // title with decorative border
            renderer.render_title(os, title, renderer.total_width());

            // header row
            renderer.render_row(os, headers, true);

            // separator
            renderer.render_separator(os);

            // data row
            renderer.render_row(os, data_row, false);

            // progress bar
            renderer.render_progress_bar(os, progress_percent, renderer.total_width());

            // bottom border
            renderer.render_border_bottom(os);
        }

        void render_message_board(std::ostream& os)
        {
            std::int64_t max_msgs    = max_messages();
            std::int64_t width       = renderer.total_width();
            std::int64_t inner_width = width - 4; // account for borders and padding

            // only show most recent messages
            std::int64_t start_idx = 0;
            if (static_cast<std::int64_t>(messages.size()) > max_msgs) {
                start_idx = static_cast<std::int64_t>(messages.size()) - max_msgs;
            }

            // title
            renderer.render_title(os, "Messages", width);

            // render messages
            for (std::int64_t ii = start_idx; ii < static_cast<std::int64_t>(messages.size());
                 ++ii) {
                const auto& msg = messages[ii];

                std::string timestamp = format_timestamp(msg.timestamp);
                std::string type_str  = message_type_string(msg.type);
                std::string msg_color = message_color(msg.type);

                // format: "HH:MM:SS [TYPE   ] message text"
                std::ostringstream line;
                line << timestamp << " [" << std::left << std::setw(7) << type_str << "] "
                     << msg.text;

                std::string full_line = line.str();

                // truncate if too long
                if (static_cast<std::int64_t>(full_line.length()) > inner_width) {
                    full_line = full_line.substr(0, inner_width - 3) + "...";
                }

                // pad to full width
                std::int64_t pad_len = inner_width - static_cast<std::int64_t>(full_line.length());
                if (pad_len > 0) {
                    full_line += std::string(pad_len, ' ');
                }

                // render line with colors
                os << color::border() << "│" << color::reset() << " ";
                os << msg_color << full_line << color::reset();
                os << " " << color::border() << "│" << color::reset() << "\n";
            }

            // bottom border
            renderer.render_border_bottom(os);
        }

        std::string format_timestamp(const std::chrono::system_clock::time_point& tp) const
        {
            auto               time_t_val = std::chrono::system_clock::to_time_t(tp);
            std::tm            tm_val     = *std::localtime(&time_t_val);
            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(2) << tm_val.tm_hour << ":" << std::setw(2)
                << tm_val.tm_min << ":" << std::setw(2) << tm_val.tm_sec;
            return oss.str();
        }

        std::string message_type_string(message_type_t type) const
        {
            switch (type) {
                case message_type_t::INFO:
                    return "INFO";
                case message_type_t::SUCCESS:
                    return "SUCCESS";
                case message_type_t::WARNING:
                    return "WARNING";
                case message_type_t::ERROR:
                    return "ERROR";
                default:
                    return "UNKNOWN";
            }
        }

        std::string message_color(message_type_t type) const
        {
            switch (type) {
                case message_type_t::INFO:
                    return color::info();
                case message_type_t::SUCCESS:
                    return color::success();
                case message_type_t::WARNING:
                    return color::warning();
                case message_type_t::ERROR:
                    return color::error();
                default:
                    return color::data();
            }
        }
    };

    table_t::table_t(const std::string& title, bool dynamic_mode)
        : impl(std::make_unique<impl_t>(title, dynamic_mode))
    {
    }

    table_t::~table_t()
    {
        if (impl && impl->dynamic_mode) {
            std::cout << ansi::SHOW_CURSOR << std::flush;
        }
    }

    table_t::table_t(table_t&&) noexcept            = default;
    table_t& table_t::operator=(table_t&&) noexcept = default;

    void table_t::set_header(const std::vector<std::string>& headers)
    {
        impl->headers = headers;
    }

    void table_t::update_row(const std::vector<std::string>& data)
    {
        impl->data_row = data;
    }

    void table_t::set_progress(std::int64_t percent)
    {
        impl->progress_percent = std::clamp<std::int64_t>(percent, 0, 100);
    }

    void table_t::post_info(const std::string& msg)
    {
        impl->messages.push_back({std::chrono::system_clock::now(), message_type_t::INFO, msg});
    }

    void table_t::post_success(const std::string& msg)
    {
        impl->messages.push_back({std::chrono::system_clock::now(), message_type_t::SUCCESS, msg});
    }

    void table_t::post_warning(const std::string& msg)
    {
        impl->messages.push_back({std::chrono::system_clock::now(), message_type_t::WARNING, msg});
    }

    void table_t::post_error(const std::string& msg)
    {
        impl->messages.push_back({std::chrono::system_clock::now(), message_type_t::ERROR, msg});
    }

    void table_t::refresh()
    {
        impl->render();
    }

    void table_t::print()
    {
        impl->render();
    }

} // namespace simbi::display
