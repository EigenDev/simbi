#include "table.hpp"

#if CUDA_ENABLED
#include "xpu/vendors/cuda/device_queries.hpp"
#elif HIP_ENABLED
#include "xpu/vendors/hip/device_queries.hpp"
#endif

#include "io/console/statistics.hpp"
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
        bool        show_system_info;

        std::vector<std::vector<std::string>> static_info_rows;
        std::vector<std::string>              headers;
        std::vector<std::string>              data_row;
        std::int64_t                          progress_percent;

        std::deque<message_t> messages;

        renderer_t sys_info_renderer;
        renderer_t benchmark_renderer;

        impl_t(const std::string& t, bool dynamic, bool sys_info)
            : title(t), dynamic_mode(dynamic), show_system_info(sys_info), progress_percent(0)
        {
            if (show_system_info) {
                gather_system_info();
            }
        }

        void gather_system_info()
        {
#if CUDA_ENABLED
            using namespace xpu::vendors::cuda;
#elif HIP_ENABLED
            using namespace xpu::vendors::hip;
#endif
            using namespace statistics;

            cpu_info_t     cpu_info  = cpu_info_t::gather();
            os_info_t      os_info   = os_info_t::gather();
            memory_stats_t mem_stats = memory_stats_t::current();

            // build system info rows (3-column format: category, property, value)
            static_info_rows.push_back({"CPU", "Model", cpu_info.model_name});
            static_info_rows.push_back({"", "Cores", std::to_string(cpu_info.num_cores)});
            static_info_rows.push_back({"", "Threads", std::to_string(cpu_info.num_threads)});

            if (cpu_info.frequency_mhz > 0) {
                std::ostringstream freq_str;
                freq_str
                    << (cpu_info.frequency_mhz >= 1000
                            ? std::to_string(static_cast<int>(cpu_info.frequency_mhz / 1000)) +
                                  " GHz"
                            : std::to_string(static_cast<int>(cpu_info.frequency_mhz)) + " MHz");
                static_info_rows.push_back({"", "Frequency", freq_str.str()});
            }

            if (cpu_info.l3_cache_size > 0) {
                static_info_rows.push_back({"", "L3 Cache", format_bytes(cpu_info.l3_cache_size)});
            }

            // os info
            std::string os_version = os_info.name;
            if (!os_info.version.empty()) {
                os_version += " " + os_info.version;
            }
            static_info_rows.push_back({"System", "OS", os_version});

            // memory info
            std::ostringstream ram_usage;
            ram_usage << std::fixed << std::setprecision(1) << mem_stats.percent_used << "%";
            static_info_rows.push_back(
                {"Memory",
                 "System RAM",
                 format_bytes(mem_stats.total_physical) + " (" +
                     format_bytes(mem_stats.used_physical) + " used, " + ram_usage.str() + ")"}
            );

            static_info_rows.push_back({"", "Process", format_bytes(mem_stats.process_physical)});

            if (mem_stats.total_virtual > 0) {
                double swap_percent =
                    (static_cast<double>(mem_stats.used_virtual) / mem_stats.total_virtual) * 100.0;
                std::ostringstream swap_str;
                swap_str << std::fixed << std::setprecision(1) << swap_percent << "%";
                static_info_rows.push_back(
                    {"",
                     "Swap",
                     format_bytes(mem_stats.total_virtual) + " (" +
                         format_bytes(mem_stats.used_virtual) + " used, " + swap_str.str() + ")"}
                );
            }

#if GPU_ENABLED
            auto dev_count = get_device_count();
            if (dev_count > 0) {
                auto props = get_properties(0);
                static_info_rows.push_back({"GPU", "Device", props.name});
                static_info_rows.push_back(
                    {"",
                     "Compute",
                     std::to_string(props.compute_capability_major) + "." +
                         std::to_string(props.compute_capability_minor)}
                );
                static_info_rows.push_back({"", "Memory", format_bytes(props.total_memory)});

                int mem_clock_rate = 0;
                cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, 0);
                std::ostringstream bandwidth;
                bandwidth << std::fixed << std::setprecision(1)
                          << (2.0 * mem_clock_rate * (props.memory_bus_width_bits / 8) / 1.0e6)
                          << " GB/s";
                static_info_rows.push_back({"", "Bandwidth", bandwidth.str()});
            }
#endif
        }

        // calculate how many messages we can display based on terminal height
        std::int64_t max_messages() const
        {
            std::int64_t height = terminal_t::height();

            // reserve space for:
            // - main title box: 2 lines
            // - system info (if present): title + header + separator + rows + bottom: variable
            // - benchmark section: title + header + separator + data + progress + bottom: 7 lines
            // - message board: title + bottom border: 2 lines
            // - bottom margin: 2 lines

            std::int64_t reserved = 2 + 7 + 2 + 2; // 13 lines base

            if (show_system_info && !static_info_rows.empty()) {
                // system info: title + header + separator + rows + bottom
                reserved += 2 + 1 + static_info_rows.size() + 1 + 1; // +1 for spacing
            }

            std::int64_t available_for_messages = height - reserved;

            // clamp to reasonable range
            if (available_for_messages < 3) {
                return 3;
            }
            if (available_for_messages > 10) {
                return 10; // max 10 to prevent excessive scrolling
            }

            return available_for_messages;
        }

        void render()
        {
            std::int64_t term_width = terminal_t::width();

            // recalculate layouts independently for each section
            if (!static_info_rows.empty()) {
                sys_info_renderer.calculate_layout(
                    {"Category", "Property", "Value"},
                    static_info_rows[0],
                    term_width
                );
            }
            benchmark_renderer.calculate_layout(headers, data_row, term_width);

            std::ostringstream buf;

            // clear screen and hide cursor if dynamic mode
            if (dynamic_mode) {
                buf << ansi::CLEAR_SCREEN << ansi::HIDE_CURSOR;
            }

            // render unified table (system info + benchmark)
            render_unified_table(buf);

            // render message board if we have messages
            if (!messages.empty()) {
                buf << "\n";
                render_message_board(buf);
            }

            // flush to stdout
            std::cout << buf.str() << std::flush;
        }

        void render_unified_table(std::ostream& os)
        {
            // main title box
            benchmark_renderer.render_title(os, title, benchmark_renderer.total_width());
            benchmark_renderer.render_border_bottom(os);
            os << "\n";

            // system info section (if enabled)
            if (show_system_info && !static_info_rows.empty()) {
                sys_info_renderer
                    .render_title(os, "SYSTEM INFORMATION", sys_info_renderer.total_width());

                std::vector<std::string> sys_headers = {"Category", "Property", "Value"};
                sys_info_renderer.render_row(os, sys_headers, true);
                sys_info_renderer.render_separator(os);

                for (const auto& row : static_info_rows) {
                    sys_info_renderer.render_row(os, row, false);
                }

                sys_info_renderer.render_border_bottom(os);
                os << "\n";
            }

            // benchmark section
            benchmark_renderer.render_title(os, "BENCHMARKS", benchmark_renderer.total_width());
            benchmark_renderer.render_row(os, headers, true);
            benchmark_renderer.render_separator(os);
            benchmark_renderer.render_row(os, data_row, false);
            benchmark_renderer
                .render_progress_bar(os, progress_percent, benchmark_renderer.total_width());
            benchmark_renderer.render_border_bottom(os);
        }

        void render_message_board(std::ostream& os)
        {
            std::int64_t max_msgs    = max_messages();
            std::int64_t width       = benchmark_renderer.total_width();
            std::int64_t inner_width = width - 4; // account for borders and padding

            // only show most recent messages
            std::int64_t start_idx = 0;
            if (static_cast<std::int64_t>(messages.size()) > max_msgs) {
                start_idx = static_cast<std::int64_t>(messages.size()) - max_msgs;
            }

            // title
            benchmark_renderer.render_title(os, "Messages", width);

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
            benchmark_renderer.render_border_bottom(os);
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

    table_t::table_t(const std::string& title, bool dynamic_mode, bool show_system_info)
        : impl(std::make_unique<impl_t>(title, dynamic_mode, show_system_info))
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
