// =============================================================================
// table.hpp
//
// public api for terminal display system.
// provides a clean interface for progress tracking with adaptive message board.
//
// usage:
//   table_t table("Simulation", true);  // dynamic mode
//   table.set_header({"Iteration", "Time", "dt"});
//   table.update_row({"100", "1.2e-3", "5.4e-6"});
//   table.set_progress(45);
//   table.post_info("Checkpoint saved");
//   table.refresh();
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simbi::display {

    enum class message_type_t {
        INFO,
        SUCCESS,
        WARNING,
        ERROR
    };

    class table_t
    {
        struct impl_t;
        std::unique_ptr<impl_t> impl;

      public:
        table_t(const std::string& title, bool dynamic_mode, bool show_system_info = false);
        ~table_t();

        // non-copyable, movable
        table_t(const table_t&)            = delete;
        table_t& operator=(const table_t&) = delete;
        table_t(table_t&&) noexcept;
        table_t& operator=(table_t&&) noexcept;

        // table operations
        void set_header(const std::vector<std::string>& headers);
        void update_row(const std::vector<std::string>& data);

        // progress tracking
        void set_progress(std::int64_t percent);

        // message board
        void post_info(const std::string& msg);
        void post_success(const std::string& msg);
        void post_warning(const std::string& msg);
        void post_error(const std::string& msg);

        // rendering
        void refresh();
        void print(); // alias for refresh
    };

} // namespace simbi::display


