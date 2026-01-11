#include "renderer.hpp"

#include "terminal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace simbi::display {

    box_chars_t box_chars_t::modern()
    {
        return {
            .top_left     = "╭",
            .top_right    = "╮",
            .bottom_left  = "╰",
            .bottom_right = "╯",
            .horizontal   = "─",
            .vertical     = "│",
            .t_down       = "┬",
            .t_up         = "┴",
            .t_left       = "├",
            .t_right      = "┤",
            .cross        = "┼"
        };
    }

    box_chars_t box_chars_t::simple()
    {
        return {
            .top_left     = "+",
            .top_right    = "+",
            .bottom_left  = "+",
            .bottom_right = "+",
            .horizontal   = "-",
            .vertical     = "|",
            .t_down       = "+",
            .t_up         = "+",
            .t_left       = "+",
            .t_right      = "+",
            .cross        = "+"
        };
    }

    renderer_t::renderer_t()
    {
        box = terminal_t::supports_unicode() ? box_chars_t::modern() : box_chars_t::simple();
    }

    void renderer_t::calculate_layout(
        const std::vector<std::string>& headers,
        const std::vector<std::string>& data,
        std::int64_t                    terminal_width
    )
    {
        std::int64_t n_cols = static_cast<std::int64_t>(headers.size());

        // adaptive padding based on terminal width
        layout.padding = terminal_t::adaptive_padding();

        // initialize column widths and alignments
        layout.widths.resize(n_cols);
        layout.alignments.resize(n_cols);

        // ==================================================================
        // STEP 1: Calculate minimum required width for each column
        // This is the max of header length and data length
        // ==================================================================
        for (std::int64_t ii = 0; ii < n_cols; ++ii) {
            std::int64_t header_len = static_cast<std::int64_t>(headers[ii].length());
            std::int64_t data_len   = (ii < static_cast<std::int64_t>(data.size()))
                                          ? static_cast<std::int64_t>(data[ii].length())
                                          : 0;
            layout.widths[ii]       = std::max(header_len, data_len);

            // all columns right-aligned for numeric data
            layout.alignments[ii] = alignment_t::RIGHT;
        }

        // ==================================================================
        // STEP 2: Calculate overhead (borders + padding)
        // Borders: n_cols + 1 vertical bars (each is 1 character)
        // Padding: layout.padding * 2 * n_cols (both sides of each column)
        // ==================================================================
        std::int64_t borders_overhead = n_cols + 1;
        std::int64_t padding_overhead = layout.padding * 2 * n_cols;
        std::int64_t total_overhead   = borders_overhead + padding_overhead;

        // ==================================================================
        // STEP 3: Calculate available space for content
        // ==================================================================
        std::int64_t available = terminal_width - total_overhead;

        if (available <= 0) {
            // terminal too narrow, use minimum widths
            layout.total_width = terminal_width;
            return;
        }

        // ==================================================================
        // STEP 4: Scale ALL columns proportionally to fill available space
        // This is the key algorithm from the original table.cpp
        // ==================================================================
        std::int64_t current_content_width = 0;
        for (std::int64_t ii = 0; ii < n_cols; ++ii) {
            current_content_width += layout.widths[ii];
        }

        if (current_content_width == 0) {
            layout.total_width = terminal_width;
            return;
        }

        // scale factor to fill available space
        double scale_factor =
            static_cast<double>(available) / static_cast<double>(current_content_width);

        // apply scaling to all columns
        for (std::int64_t ii = 0; ii < n_cols; ++ii) {
            layout.widths[ii] = static_cast<std::int64_t>(layout.widths[ii] * scale_factor);

            // enforce minimum width of 6 characters
            if (layout.widths[ii] < 6) {
                layout.widths[ii] = 6;
            }

            // enforce maximum width to prevent one column dominating
            if (layout.widths[ii] > 40) {
                layout.widths[ii] = 40;
            }
        }

        // ==================================================================
        // STEP 5: Redistribute any leftover space
        // After clamping, we may have leftover space
        // ==================================================================
        std::int64_t allocated = 0;
        for (std::int64_t ii = 0; ii < n_cols; ++ii) {
            allocated += layout.widths[ii];
        }

        std::int64_t leftover = available - allocated;

        // distribute leftover evenly across columns that haven't hit max
        while (leftover > 0) {
            bool distributed = false;
            for (std::int64_t ii = 0; ii < n_cols && leftover > 0; ++ii) {
                if (layout.widths[ii] < 40) {
                    layout.widths[ii]++;
                    leftover--;
                    distributed = true;
                }
            }
            if (!distributed) {
                break; // all columns at max, stop
            }
        }

        layout.total_width = terminal_width;
    }

    void renderer_t::render_border_top(std::ostream& os) const
    {
        os << color::border();
        os << box.top_left;

        for (size_t ii = 0; ii < layout.widths.size(); ++ii) {
            // horizontal line for this column (width + padding on both sides)
            std::int64_t line_len = layout.widths[ii] + (2 * layout.padding);
            for (std::int64_t jj = 0; jj < line_len; ++jj) {
                os << box.horizontal;
            }

            // t-junction or top-right corner
            if (ii < layout.widths.size() - 1) {
                os << box.t_down;
            }
        }

        os << box.top_right;
        os << color::reset() << "\n";
    }

    void renderer_t::render_border_bottom(std::ostream& os) const
    {
        os << color::border();
        os << box.bottom_left;

        // clean horizontal line - no T-joints
        std::int64_t total_inner = 0;
        for (size_t ii = 0; ii < layout.widths.size(); ++ii) {
            total_inner += layout.widths[ii] + (2 * layout.padding);
        }
        total_inner += static_cast<std::int64_t>(layout.widths.size()) - 1; // add separator spaces

        for (std::int64_t ii = 0; ii < total_inner; ++ii) {
            os << box.horizontal;
        }

        os << box.bottom_right;
        os << color::reset() << "\n";
    }

    void renderer_t::render_separator(std::ostream& os) const
    {
        os << color::border();
        os << box.t_left;

        for (size_t ii = 0; ii < layout.widths.size(); ++ii) {
            std::int64_t line_len = layout.widths[ii] + (2 * layout.padding);
            for (std::int64_t jj = 0; jj < line_len; ++jj) {
                os << box.horizontal;
            }

            if (ii < layout.widths.size() - 1) {
                os << box.cross;
            }
        }

        os << box.t_right;
        os << color::reset() << "\n";
    }

    void
    renderer_t::render_title(std::ostream& os, const std::string& title, std::int64_t width) const
    {
        std::int64_t inner_width = width - 2; // account for borders

        // format: "─ Title ─────────"
        std::string  title_text = " " + title + " ";
        std::int64_t title_len  = static_cast<std::int64_t>(title_text.length());
        std::int64_t left_fill  = 1; // single dash before title
        std::int64_t right_fill = inner_width - title_len - left_fill;

        os << color::border() << box.top_left << box.horizontal;
        os << color::reset() << color::title() << title_text << color::reset();
        os << color::border();

        for (std::int64_t ii = 0; ii < right_fill; ++ii) {
            os << box.horizontal;
        }

        os << box.top_right << color::reset() << "\n";
    }

    void renderer_t::render_row(
        std::ostream&                   os,
        const std::vector<std::string>& cells,
        bool                            is_header
    ) const
    {
        // select color based on row type
        std::string cell_color = is_header ? color::header() : color::data();

        // left border
        os << color::border() << box.vertical << color::reset();

        for (size_t ii = 0; ii < layout.widths.size(); ++ii) {
            // left padding
            for (std::int64_t pp = 0; pp < layout.padding; ++pp) {
                os << " ";
            }

            // cell content
            std::string content = (ii < cells.size()) ? cells[ii] : "";
            os << cell_color << align_text(content, layout.widths[ii], layout.alignments[ii])
               << color::reset();

            // right padding
            for (std::int64_t pp = 0; pp < layout.padding; ++pp) {
                os << " ";
            }

            // vertical border
            os << color::border() << box.vertical << color::reset();
        }

        os << "\n";
    }

    void renderer_t::render_progress_bar(
        std::ostream& os,
        std::int64_t  percent,
        std::int64_t  width
    ) const
    {
        // separator before progress - clean horizontal only
        os << color::border() << box.t_left;
        for (std::int64_t ii = 0; ii < width - 2; ++ii) {
            os << box.horizontal;
        }
        os << box.t_right << color::reset() << "\n";

        // progress bar content
        // reserve space for: "│ " + bar + " XX% │"
        std::int64_t bar_width = width - 9; // 2 borders + 2 spaces + 5 for " XX% "
        std::int64_t filled    = (bar_width * percent) / 100;

        os << color::border() << box.vertical << color::reset() << " ";

        // filled portion with gradient effect
        os << color::progress_filled();
        for (std::int64_t ii = 0; ii < filled; ++ii) {
            if (ii > filled - 3 && filled < bar_width) {
                os << color::progress_mid() << "▒";
            }
            else {
                os << "▓";
            }
        }

        // empty portion
        os << color::progress_empty();
        for (std::int64_t ii = filled; ii < bar_width; ++ii) {
            os << "░";
        }

        // percentage text with proper spacing
        os << color::reset() << " ";
        if (percent < 10) {
            os << " ";
        }
        if (percent < 100) {
            os << " ";
        }
        os << color::data() << percent << "%" << color::reset() << " ";
        os << color::border() << box.vertical << color::reset() << "\n";
    }

    std::string
    renderer_t::align_text(const std::string& text, std::int64_t width, alignment_t align) const
    {
        std::int64_t text_len = static_cast<std::int64_t>(text.length());

        if (text_len >= width) {
            return text.substr(0, width);
        }

        std::int64_t padding = width - text_len;

        switch (align) {
            case alignment_t::RIGHT:
                return std::string(padding, ' ') + text;

            case alignment_t::CENTER: {
                std::int64_t left  = padding / 2;
                std::int64_t right = padding - left;
                return std::string(left, ' ') + text + std::string(right, ' ');
            }

            case alignment_t::LEFT:
            default:
                return text + std::string(padding, ' ');
        }
    }

    std::string renderer_t::truncate(const std::string& text, std::int64_t max_width) const
    {
        if (static_cast<std::int64_t>(text.length()) <= max_width) {
            return text;
        }

        if (max_width < 3) {
            return text.substr(0, max_width);
        }

        return text.substr(0, max_width - 3) + "...";
    }

} // namespace simbi::display
