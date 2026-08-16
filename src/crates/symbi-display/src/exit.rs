// =============================================================================
// exit.rs
//
// the run's final summary frame. after the live screen is left, one bounded,
// state-colored `Block` is rendered off-screen into a buffer, serialized to ansi,
// and printed onto the primary buffer so it persists in scrollback. rendering
// off-screen avoids any cursor-position query, which could block once
// ScreenGuard has restored the canonical termios.
//
// usage:
//   print!("{}", exit::render_exit_frame(ExitKind::Success, &summary, width));
// =============================================================================

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget, Wrap};

const C_DATA: Color = Color::Indexed(252); // light gray body text
const C_SUCCESS: Color = Color::Indexed(158); // mint green
const C_WARNING: Color = Color::Indexed(215); // warm amber
const C_ERROR: Color = Color::Indexed(210); // soft red

/// terminal state of a run, selecting the exit frame's heading and border color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitKind {
    /// reached t_final.
    Success,
    /// stopped by a caught signal; a restart checkpoint was written.
    Interrupt,
    /// halted on an unphysical state (cfl/wave-speed collapse).
    Crash,
}

/// border color for an exit state: green complete, amber interrupt, red crash.
fn exit_color(kind: ExitKind) -> Color {
    match kind {
        ExitKind::Success => C_SUCCESS,
        ExitKind::Interrupt => C_WARNING,
        ExitKind::Crash => C_ERROR,
    }
}

/// heading word for an exit state.
fn exit_heading(kind: ExitKind) -> &'static str {
    match kind {
        ExitKind::Success => "COMPLETE",
        ExitKind::Interrupt => "INTERRUPTED",
        ExitKind::Crash => "CRASHED",
    }
}

/// build the printable exit-frame: a bounded rounded `Block` with a state-colored
/// border and a wrapped one-line summary, sized to the wrapped content so it can
/// never overflow. pure: `(kind, summary, term_width) -> ansi String`, rendered
/// off-screen and serialized so it prints onto the primary buffer with no cursor
/// query (which would risk blocking once the live termios state is restored).
pub fn render_exit_frame(kind: ExitKind, summary: &str, term_width: u16) -> String {
    let color = exit_color(kind);
    let width = term_width.clamp(24, 76);
    let inner_w = width.saturating_sub(2);

    let body_style = Style::default().fg(C_DATA);
    let wrap = Wrap { trim: true };

    // measure the wrapped height with ratatui's own wrap (its line_count is gated
    // behind an unstable feature): render into a scratch buffer at the inner width
    // and count the non-blank rows.
    let measure = Paragraph::new(summary.to_string())
        .style(body_style)
        .wrap(wrap);
    let cap = (summary.chars().count() as u16).clamp(1, 256);
    let mut scratch = Buffer::empty(Rect::new(0, 0, inner_w, cap));
    measure.render(Rect::new(0, 0, inner_w, cap), &mut scratch);
    let body_h = last_content_row(&scratch).max(1);

    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .title(Span::styled(
            format!(" {} ", exit_heading(kind)),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    let para = Paragraph::new(summary.to_string())
        .style(body_style)
        .wrap(wrap);

    let height = body_h + 2; // top + bottom border
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    let inner = block.inner(area);
    block.render(area, &mut buf);
    para.render(inner, &mut buf);

    buffer_to_ansi(&buf)
}

/// one past the last row of a buffer that holds a non-blank cell (i.e. the number
/// of content rows ratatui's wrap produced).
fn last_content_row(buf: &Buffer) -> u16 {
    let area = *buf.area();
    let mut rows = 0;
    for yy in 0..area.height {
        let non_blank = (0..area.width).any(|xx| {
            buf.cell((xx, yy))
                .map(|c| !c.symbol().trim().is_empty())
                .unwrap_or(false)
        });
        if non_blank {
            rows = yy + 1;
        }
    }
    rows
}

/// serialize a buffer to ansi lines (reset + fg + optional bold per cell). handles
/// the indexed-palette colors and bold this crate emits; other colors fall back to
/// the terminal default. each row ends reset + newline so it prints standalone.
fn buffer_to_ansi(buf: &Buffer) -> String {
    let area = *buf.area();
    let mut out = String::with_capacity((area.width as usize + 12) * area.height as usize);
    for yy in area.y..area.y + area.height {
        for xx in area.x..area.x + area.width {
            let Some(cell) = buf.cell((xx, yy)) else {
                continue;
            };
            out.push_str("\x1b[0m");
            if cell.modifier.contains(Modifier::BOLD) {
                out.push_str("\x1b[1m");
            }
            if let Color::Indexed(n) = cell.fg {
                out.push_str("\x1b[38;5;");
                out.push_str(&n.to_string());
                out.push('m');
            }
            out.push_str(cell.symbol());
        }
        out.push_str("\x1b[0m\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the exit frame is bounded (clamped width), carries the heading + summary,
    /// and wraps a long summary onto extra rows.
    #[test]
    fn exit_frame_is_bounded_and_wraps() {
        let summary = "interrupted — 12000 steps, t = 4.3100 — restart checkpoint michel.interrupted.h5 written to disk for resume";
        let frame = render_exit_frame(ExitKind::Interrupt, summary, 200);
        let lines: Vec<&str> = frame.lines().collect();
        for line in &lines {
            assert!(strip_ansi(line).chars().count() <= 76);
        }
        let stripped: String = lines
            .iter()
            .map(|l| strip_ansi(l))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(stripped.contains("INTERRUPTED"));
        assert!(stripped.contains("michel.interrupted.h5"));
        assert!(lines.len() > 3);
    }

    /// each exit state paints its border in a distinct palette color.
    #[test]
    fn exit_frame_border_color_tracks_state() {
        let ok = render_exit_frame(ExitKind::Success, "complete", 60);
        let crash = render_exit_frame(ExitKind::Crash, "crashed", 60);
        assert!(ok.contains("\x1b[38;5;158m")); // mint green
        assert!(crash.contains("\x1b[38;5;210m")); // soft red
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
