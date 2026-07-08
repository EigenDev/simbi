// =============================================================================
// renderer.rs
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
//   let mut r = Renderer::new();
//   r.calculate_layout(&headers, &data, terminal::width());
//   r.render_border_top(&mut buf);
// =============================================================================

use crate::terminal::{self, color};

/// text alignment for cells.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Right,
    Center,
}

/// box-drawing character set.
pub struct BoxChars {
    pub top_left: &'static str,
    pub top_right: &'static str,
    pub bottom_left: &'static str,
    pub bottom_right: &'static str,
    pub horizontal: &'static str,
    pub vertical: &'static str,
    pub t_down: &'static str,
    pub t_up: &'static str,
    pub t_left: &'static str,
    pub t_right: &'static str,
    pub cross: &'static str,
}

impl BoxChars {
    pub fn unicode() -> Self {
        Self {
            top_left: "\u{256d}",     // ╭
            top_right: "\u{256e}",    // ╮
            bottom_left: "\u{2570}",  // ╰
            bottom_right: "\u{256f}", // ╯
            horizontal: "\u{2500}",   // ─
            vertical: "\u{2502}",     // │
            t_down: "\u{252c}",       // ┬
            t_up: "\u{2534}",         // ┴
            t_left: "\u{251c}",       // ├
            t_right: "\u{2524}",      // ┤
            cross: "\u{253c}",        // ┼
        }
    }

    pub fn ascii() -> Self {
        Self {
            top_left: "+",
            top_right: "+",
            bottom_left: "+",
            bottom_right: "+",
            horizontal: "-",
            vertical: "|",
            t_down: "+",
            t_up: "+",
            t_left: "+",
            t_right: "+",
            cross: "+",
        }
    }
}

struct Layout {
    widths: Vec<usize>,
    padding: usize,
    total_width: usize,
}

/// rendering engine. owns box chars and layout state.
pub struct Renderer {
    bx: BoxChars,
    layout: Layout,
}

impl Renderer {
    pub fn new() -> Self {
        let bx = if terminal::supports_unicode() {
            BoxChars::unicode()
        } else {
            BoxChars::ascii()
        };
        Self {
            bx,
            layout: Layout {
                widths: Vec::new(),
                padding: 3,
                total_width: 80,
            },
        }
    }

    pub fn total_width(&self) -> usize {
        self.layout.total_width
    }

    pub fn padding(&self) -> usize {
        self.layout.padding
    }

    /// proportional column width distribution algorithm.
    pub fn calculate_layout(&mut self, headers: &[&str], data: &[&str], term_width: usize) {
        let n_cols = headers.len();
        self.layout.padding = terminal::padding_for_width(term_width);
        self.layout.widths = vec![0; n_cols];

        // minimum widths = max(header_len, data_len). measure DISPLAY
        // columns (char count), not bytes — cells may carry multibyte glyphs
        // (em-dash, unicode paths) and a byte count would over-reserve width.
        for ii in 0..n_cols {
            let header_len = headers[ii].chars().count();
            let data_len = if ii < data.len() { data[ii].chars().count() } else { 0 };
            self.layout.widths[ii] = header_len.max(data_len);
        }

        // overhead = borders (n+1) + padding (2 * pad * n)
        let borders_overhead = n_cols + 1;
        let padding_overhead = self.layout.padding * 2 * n_cols;
        let total_overhead = borders_overhead + padding_overhead;

        // available space
        if term_width <= total_overhead {
            self.layout.total_width = term_width;
            return;
        }
        let available = term_width - total_overhead;

        // proportional scaling
        let current_content_width: usize = self.layout.widths.iter().sum();
        if current_content_width == 0 {
            self.layout.total_width = term_width;
            return;
        }

        let scale = available as f64 / current_content_width as f64;
        for ii in 0..n_cols {
            self.layout.widths[ii] = (self.layout.widths[ii] as f64 * scale) as usize;
            self.layout.widths[ii] = self.layout.widths[ii].clamp(6, 40);
        }

        // redistribute leftover
        let allocated: usize = self.layout.widths.iter().sum();
        let mut leftover = available.saturating_sub(allocated);

        while leftover > 0 {
            let mut distributed = false;
            for ii in 0..n_cols {
                if leftover == 0 {
                    break;
                }
                if self.layout.widths[ii] < 40 {
                    self.layout.widths[ii] += 1;
                    leftover -= 1;
                    distributed = true;
                }
            }
            if !distributed {
                break;
            }
        }

        self.layout.total_width = term_width;
    }

    /// top border: ╭─┬─╮
    pub fn render_border_top(&self, buf: &mut String) {
        buf.push_str(color::BORDER);
        buf.push_str(self.bx.top_left);

        for ii in 0..self.layout.widths.len() {
            let line_len = self.layout.widths[ii] + 2 * self.layout.padding;
            for _ in 0..line_len {
                buf.push_str(self.bx.horizontal);
            }
            if ii < self.layout.widths.len() - 1 {
                buf.push_str(self.bx.t_down);
            }
        }

        buf.push_str(self.bx.top_right);
        buf.push_str(color::RESET);
        buf.push('\n');
    }

    /// bottom border: ╰────╯ (clean, no T-joints)
    pub fn render_border_bottom(&self, buf: &mut String) {
        buf.push_str(color::BORDER);
        buf.push_str(self.bx.bottom_left);

        let mut total_inner: usize = self
            .layout
            .widths
            .iter()
            .map(|ww| ww + 2 * self.layout.padding)
            .sum();
        total_inner += self.layout.widths.len().saturating_sub(1);

        for _ in 0..total_inner {
            buf.push_str(self.bx.horizontal);
        }

        buf.push_str(self.bx.bottom_right);
        buf.push_str(color::RESET);
        buf.push('\n');
    }

    /// separator: ├─┼─┤
    pub fn render_separator(&self, buf: &mut String) {
        buf.push_str(color::BORDER);
        buf.push_str(self.bx.t_left);

        for ii in 0..self.layout.widths.len() {
            let line_len = self.layout.widths[ii] + 2 * self.layout.padding;
            for _ in 0..line_len {
                buf.push_str(self.bx.horizontal);
            }
            if ii < self.layout.widths.len() - 1 {
                buf.push_str(self.bx.cross);
            }
        }

        buf.push_str(self.bx.t_right);
        buf.push_str(color::RESET);
        buf.push('\n');
    }

    /// title row: ╭─ Title ────╮
    pub fn render_title(&self, buf: &mut String, title: &str, width: usize) {
        let inner_width = width.saturating_sub(2);
        let title_text = format!(" {} ", title);
        let title_len = title_text.chars().count();
        let left_fill = 1; // single dash before title
        let right_fill = inner_width.saturating_sub(title_len + left_fill);

        buf.push_str(color::BORDER);
        buf.push_str(self.bx.top_left);
        buf.push_str(self.bx.horizontal);
        buf.push_str(color::RESET);
        buf.push_str(color::TITLE);
        buf.push_str(&title_text);
        buf.push_str(color::RESET);
        buf.push_str(color::BORDER);

        for _ in 0..right_fill {
            buf.push_str(self.bx.horizontal);
        }

        buf.push_str(self.bx.top_right);
        buf.push_str(color::RESET);
        buf.push('\n');
    }

    /// data/header row with right-aligned cells.
    pub fn render_row(&self, buf: &mut String, cells: &[&str], is_header: bool) {
        let cell_color = if is_header {
            color::HEADER
        } else {
            color::DATA
        };

        buf.push_str(color::BORDER);
        buf.push_str(self.bx.vertical);
        buf.push_str(color::RESET);

        for ii in 0..self.layout.widths.len() {
            // left padding
            for _ in 0..self.layout.padding {
                buf.push(' ');
            }

            let content = if ii < cells.len() { cells[ii] } else { "" };
            buf.push_str(cell_color);
            buf.push_str(&align_text(
                content,
                self.layout.widths[ii],
                Alignment::Right,
            ));
            buf.push_str(color::RESET);

            // right padding
            for _ in 0..self.layout.padding {
                buf.push(' ');
            }

            buf.push_str(color::BORDER);
            buf.push_str(self.bx.vertical);
            buf.push_str(color::RESET);
        }

        buf.push('\n');
    }

    /// gradient progress bar: ▓▒░ with separator above.
    pub fn render_progress_bar(&self, buf: &mut String, percent: usize) {
        let width = self.layout.total_width;

        // separator before progress
        buf.push_str(color::BORDER);
        buf.push_str(self.bx.t_left);
        for _ in 0..width.saturating_sub(2) {
            buf.push_str(self.bx.horizontal);
        }
        buf.push_str(self.bx.t_right);
        buf.push_str(color::RESET);
        buf.push('\n');

        // bar content: "│ " + bar + " XX% │"
        let bar_width = width.saturating_sub(9); // 2 borders + 2 spaces + 5 for " XX% "
        let filled = (bar_width * percent) / 100;

        buf.push_str(color::BORDER);
        buf.push_str(self.bx.vertical);
        buf.push_str(color::RESET);
        buf.push(' ');

        // filled portion with gradient
        buf.push_str(color::PROGRESS_FILLED);
        for ii in 0..filled {
            if ii > filled.saturating_sub(3) && filled < bar_width {
                buf.push_str(color::PROGRESS_MID);
                buf.push('\u{2592}'); // ▒
            } else {
                buf.push('\u{2593}'); // ▓
            }
        }

        // empty portion
        buf.push_str(color::PROGRESS_EMPTY);
        for _ in filled..bar_width {
            buf.push('\u{2591}'); // ░
        }

        // percentage
        buf.push_str(color::RESET);
        buf.push(' ');
        if percent < 10 {
            buf.push(' ');
        }
        if percent < 100 {
            buf.push(' ');
        }
        buf.push_str(color::DATA);
        buf.push_str(&format!("{}%", percent));
        buf.push_str(color::RESET);
        buf.push(' ');
        buf.push_str(color::BORDER);
        buf.push_str(self.bx.vertical);
        buf.push_str(color::RESET);
        buf.push('\n');
    }
}

/// align text within a fixed display width (char-measured). truncates by
/// characters (never mid-codepoint) if the text exceeds the width.
pub fn align_text(text: &str, width: usize, align: Alignment) -> String {
    let text_len = text.chars().count();

    if text_len >= width {
        return text.chars().take(width).collect();
    }

    let pad = width - text_len;
    match align {
        Alignment::Right => {
            let mut result = String::with_capacity(width);
            for _ in 0..pad {
                result.push(' ');
            }
            result.push_str(text);
            result
        }
        Alignment::Center => {
            let left = pad / 2;
            let right = pad - left;
            let mut result = String::with_capacity(width);
            for _ in 0..left {
                result.push(' ');
            }
            result.push_str(text);
            for _ in 0..right {
                result.push(' ');
            }
            result
        }
        Alignment::Left => {
            let mut result = String::with_capacity(width);
            result.push_str(text);
            for _ in 0..pad {
                result.push(' ');
            }
            result
        }
    }
}

/// truncate text to a display width (char-measured), appending "..." if it
/// exceeds max_width. slices on character boundaries so multibyte glyphs never
/// panic the byte-index path.
pub fn truncate(text: &str, max_width: usize) -> String {
    if text.chars().count() <= max_width {
        return text.to_string();
    }
    if max_width < 3 {
        return text.chars().take(max_width).collect();
    }
    let kept: String = text.chars().take(max_width - 3).collect();
    format!("{kept}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- layout algorithm --

    #[test]
    fn layout_basic() {
        let mut r = Renderer::new();
        r.calculate_layout(
            &["Iteration", "Time", "dt"],
            &["100", "1.2e-3", "5.4e-6"],
            100,
        );
        assert_eq!(r.layout.widths.len(), 3);
        let total_content: usize = r.layout.widths.iter().sum();
        let overhead = 3 + 1 + 2 * r.layout.padding * 3;
        assert!(total_content + overhead <= 100);
    }

    #[test]
    fn layout_wide_terminal() {
        let mut r = Renderer::new();
        r.calculate_layout(&["A", "B"], &["x", "y"], 200);
        assert_eq!(r.layout.padding, 4);
        // columns should scale up but clamp at 40
        for ww in &r.layout.widths {
            assert!(*ww <= 40);
            assert!(*ww >= 6);
        }
    }

    #[test]
    fn layout_narrow_terminal() {
        let mut r = Renderer::new();
        r.calculate_layout(&["A", "B", "C"], &["x", "y", "z"], 40);
        assert_eq!(r.layout.padding, 2);
        for ww in &r.layout.widths {
            assert!(*ww >= 6);
        }
    }

    #[test]
    fn layout_min_clamp() {
        let mut r = Renderer::new();
        // tiny content, wide terminal — should scale up past min
        r.calculate_layout(&["A", "B"], &["x", "y"], 100);
        for ww in &r.layout.widths {
            assert!(*ww >= 6);
        }
    }

    #[test]
    fn layout_max_clamp() {
        let mut r = Renderer::new();
        // one huge column — should clamp at 40
        r.calculate_layout(
            &["A very long header name that is quite wide"],
            &["some data"],
            200,
        );
        assert!(r.layout.widths[0] <= 40);
    }

    #[test]
    fn layout_leftover_redistribution() {
        let mut r = Renderer::new();
        r.calculate_layout(&["A", "B", "C", "D"], &["1", "2", "3", "4"], 120);
        let total_content: usize = r.layout.widths.iter().sum();
        let overhead = 4 + 1 + 2 * r.layout.padding * 4;
        // leftover redistribution should fill available space
        assert!(total_content + overhead <= 120);
    }

    #[test]
    fn layout_single_column() {
        let mut r = Renderer::new();
        r.calculate_layout(&["Header"], &["data"], 80);
        assert_eq!(r.layout.widths.len(), 1);
        assert!(r.layout.widths[0] >= 6);
    }

    // -- align_text --

    #[test]
    fn align_text_left() {
        assert_eq!(align_text("hi", 6, Alignment::Left), "hi    ");
    }

    #[test]
    fn align_text_right() {
        assert_eq!(align_text("hi", 6, Alignment::Right), "    hi");
    }

    #[test]
    fn align_text_center() {
        assert_eq!(align_text("hi", 6, Alignment::Center), "  hi  ");
    }

    #[test]
    fn align_text_overflow() {
        assert_eq!(align_text("hello world", 5, Alignment::Left), "hello");
    }

    // -- truncate --

    #[test]
    fn truncate_short() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_exact() {
        assert_eq!(truncate("abcde", 5), "abcde");
    }

    #[test]
    fn truncate_long() {
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn truncate_tiny_max() {
        assert_eq!(truncate("hello", 2), "he");
    }

    // -- multibyte (display-width) regression: the em-dash in run messages is
    //    3 utf-8 bytes but 1 column; byte-measured padding/slicing drifts the
    //    box border and can panic on a mid-codepoint boundary. --

    #[test]
    fn align_text_pads_by_display_width_not_bytes() {
        // "a—b" is 1+3+1 = 5 bytes but 3 columns; right-align to 6 => 3 spaces.
        let out = align_text("a—b", 6, Alignment::Right);
        assert_eq!(out.chars().count(), 6);
        assert_eq!(out, "   a—b");
    }

    #[test]
    fn align_text_overflow_multibyte_no_panic() {
        // width cut falls inside the em-dash; must slice on a char boundary.
        let out = align_text("aa—bb", 3, Alignment::Right);
        assert_eq!(out.chars().count(), 3);
        assert_eq!(out, "aa—");
    }

    #[test]
    fn truncate_multibyte_is_char_bounded() {
        let out = truncate("done — final checkpoint", 8);
        assert_eq!(out.chars().count(), 8);
        assert!(out.ends_with("..."));
    }

    // -- border snapshot --

    #[test]
    fn border_top_snapshot() {
        let mut r = Renderer::new();
        r.bx = BoxChars::ascii(); // deterministic
        r.layout = Layout {
            widths: vec![6, 6],
            padding: 1,
            total_width: 20,
        };
        let mut buf = String::new();
        r.render_border_top(&mut buf);
        // strip ansi for content check
        let stripped = strip_ansi(&buf);
        assert!(stripped.starts_with('+'));
        assert!(stripped.trim_end().ends_with('+'));
    }

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut in_esc = false;
        for ch in s.chars() {
            if ch == '\x1b' {
                in_esc = true;
            } else if in_esc {
                if ch.is_ascii_alphabetic() {
                    in_esc = false;
                }
            } else {
                out.push(ch);
            }
        }
        out
    }
}
