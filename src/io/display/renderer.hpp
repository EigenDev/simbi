// =============================================================================
// renderer.hpp
//
// rendering engine for terminal display system.
// handles layout calculation, box drawing, text alignment, and progress bars.
//
// key algorithm: proportional column width distribution
//   1. calculate content widths (max of header/data)
//   2. compute overhead (borders + padding)
//   3. scale all columns to fill available space
//   4. redistribute leftover after clamping to constraints
//
// usage:
//   renderer_t r;
//   r.calculate_layout(headers, data, terminal_width);
//   r.render_table(stream, headers, data);
// =============================================================================
#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace simbi::display {

    // text alignment
    enum class alignment_t {
        LEFT,
        RIGHT,
        CENTER
    };

    // box drawing characters
    struct box_chars_t
    {
        const char* top_left;
        const char* top_right;
        const char* bottom_left;
        const char* bottom_right;
        const char* horizontal;
        const char* vertical;
        const char* t_down;
        const char* t_up;
        const char* t_left;
        const char* t_right;
        const char* cross;

        static box_chars_t modern(); // unicode: ╭╮╰╯─│┬┴├┤┼
        static box_chars_t simple(); // ascii: +-+++++++
    };

    // column layout specification
    struct column_layout_t
    {
        std::vector<std::int64_t> widths;      // calculated widths
        std::vector<alignment_t>  alignments;  // per-column alignment
        std::int64_t              padding;     // adaptive padding (2-4 spaces)
        std::int64_t              total_width; // full table width including borders
    };

    // rendering engine
    class renderer_t
    {
        box_chars_t     box;
        column_layout_t layout;

      public:
        renderer_t();

        // calculate optimal column layout
        // this is the critical algorithm - must fill terminal width correctly
        void calculate_layout(
            const std::vector<std::string>& headers,
            const std::vector<std::string>& data,
            std::int64_t                    terminal_width
        );

        // rendering primitives
        void render_border_top(std::ostream& os) const;
        void render_border_bottom(std::ostream& os) const;
        void render_separator(std::ostream& os) const;

        void render_title(std::ostream& os, const std::string& title, std::int64_t width) const;

        void
        render_row(std::ostream& os, const std::vector<std::string>& cells, bool is_header) const;

        void render_progress_bar(std::ostream& os, std::int64_t percent, std::int64_t width) const;

        // text utilities
        std::string
        align_text(const std::string& text, std::int64_t width, alignment_t align) const;

        std::string truncate(const std::string& text, std::int64_t max_width) const;

        // accessors
        std::int64_t total_width() const
        {
            return layout.total_width;
        }
        std::int64_t padding() const
        {
            return layout.padding;
        }
    };

} // namespace simbi::display


