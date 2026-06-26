// =============================================================================
// table.rs
//
// public api for terminal display system.
// provides progress tracking, adaptive message board, and optional log file.
//
// usage:
//   let mut table = Table::new("Simulation", true);
//   table.set_header(&["Iteration", "Time", "dt"]);
//   table.update_row(&["100", "1.2e-3", "5.4e-6"]);
//   table.set_progress(45);
//   table.post_info("Checkpoint saved");
//   table.refresh();
// =============================================================================

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::renderer::Renderer;
use crate::terminal::{self, ansi, color};

/// message severity level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageType {
    Info,
    Success,
    Warning,
    Error,
    Diagnostic,
}

struct Message {
    timestamp: String,
    kind: MessageType,
    text: String,
}

/// terminal display table with progress bar and message board.
pub struct Table {
    title: String,
    dynamic: bool,
    headers: Vec<String>,
    data: Vec<String>,
    progress: usize,
    messages: VecDeque<Message>,
    log_file: Option<BufWriter<File>>,
    renderer: Renderer,
    /// optional system-information rows (3-column: Category, Property, Value).
    /// rendered as a sub-table above the benchmark section. populated by
    /// `set_system_info` — the integration with `symbi-xpu::cuda::device_info`
    /// and CPU-side `/proc/{cpuinfo,meminfo}` happens example-side.
    system_info: Vec<[String; 3]>,
    /// optional problem-setup rows (3-column: Category, Property, Value).
    /// rendered between System Info and Benchmarks. holds the regime / coords
    /// / solver / reconstruction / timestepping / cfl identification a
    /// scientific user wants on screen.
    problem_setup: Vec<[String; 3]>,
}

impl Table {
    /// create a new table. if `dynamic` is true AND stdout is a terminal, each
    /// refresh clears the screen; when output is redirected to a file/pipe the
    /// table falls back to static output so no ansi escapes are embedded.
    pub fn new(title: &str, dynamic: bool) -> Self {
        Self {
            title: title.to_string(),
            dynamic: dynamic && terminal::is_tty(),
            headers: Vec::new(),
            data: Vec::new(),
            progress: 0,
            messages: VecDeque::new(),
            log_file: None,
            renderer: Renderer::new(),
            system_info: Vec::new(),
            problem_setup: Vec::new(),
        }
    }

    /// **B10** — set the static System Information sub-table (rendered above
    /// the benchmark section). each row is `[category, property, value]`.
    /// the renderer collapses repeated `category` entries to blanks (e.g.
    /// "CPU" + 4 properties = one "CPU" label + 3 blanks).
    pub fn set_system_info(&mut self, rows: &[[&str; 3]]) {
        self.system_info = rows.iter()
            .map(|r| [r[0].to_string(), r[1].to_string(), r[2].to_string()])
            .collect();
    }

    /// **B10** — set the static Problem Setup sub-table (rendered between
    /// System Info and Benchmarks). each row is `[category, property, value]`.
    pub fn set_problem_setup(&mut self, rows: &[[&str; 3]]) {
        self.problem_setup = rows.iter()
            .map(|r| [r[0].to_string(), r[1].to_string(), r[2].to_string()])
            .collect();
    }

    /// set column headers.
    pub fn set_header(&mut self, headers: &[&str]) {
        self.headers = headers.iter().map(|s| s.to_string()).collect();
    }

    /// update the data row.
    pub fn update_row(&mut self, row: &[&str]) {
        self.data = row.iter().map(|s| s.to_string()).collect();
    }

    /// set progress percentage (clamped to 0..=100).
    pub fn set_progress(&mut self, percent: usize) {
        self.progress = percent.min(100);
    }

    /// toggle the clearing redraw. flip to `false` to render a single STATIC
    /// frame (no screen clear) onto the primary buffer after an alternate-screen
    /// live session ends, so the run's final state persists in scrollback. only
    /// ever dynamic on a tty.
    pub fn set_dynamic(&mut self, on: bool) {
        self.dynamic = on && terminal::is_tty();
    }

    pub fn post_info(&mut self, msg: &str) {
        self.post_message(MessageType::Info, msg);
    }

    pub fn post_success(&mut self, msg: &str) {
        self.post_message(MessageType::Success, msg);
    }

    pub fn post_warning(&mut self, msg: &str) {
        self.post_message(MessageType::Warning, msg);
    }

    pub fn post_error(&mut self, msg: &str) {
        self.post_message(MessageType::Error, msg);
    }

    pub fn post_diagnostic(&mut self, msg: &str) {
        self.post_message(MessageType::Diagnostic, msg);
    }

    /// attach a log file. all posted messages are mirrored to disk.
    pub fn set_log_file(&mut self, path: &Path) -> io::Result<()> {
        let file = File::options().create(true).append(true).open(path)?;
        self.log_file = Some(BufWriter::new(file));
        Ok(())
    }

    /// render the full display to stdout.
    pub fn refresh(&mut self) {
        let term_width = terminal::width();
        let headers_ref: Vec<&str> = self.headers.iter().map(|s| s.as_str()).collect();
        let data_ref: Vec<&str> = self.data.iter().map(|s| s.as_str()).collect();
        self.renderer
            .calculate_layout(&headers_ref, &data_ref, term_width);

        let mut buf = String::with_capacity(4096);

        if self.dynamic {
            buf.push_str(ansi::CLEAR_SCREEN);
            buf.push_str(ansi::HIDE_CURSOR);
        }

        // main title box
        self.renderer
            .render_title(&mut buf, &self.title, self.renderer.total_width());
        self.renderer.render_border_bottom(&mut buf);
        buf.push('\n');

        // ---- System Info sub-table ------------------------------------
        if !self.system_info.is_empty() {
            self.render_info_section(&mut buf, "SYSTEM INFORMATION", &self.system_info);
            buf.push('\n');
        }
        // ---- Problem Setup sub-table ----------------------------------
        if !self.problem_setup.is_empty() {
            self.render_info_section(&mut buf, "PROBLEM SETUP", &self.problem_setup);
            buf.push('\n');
        }

        // benchmark section
        self.renderer.render_title(
            &mut buf,
            "BENCHMARKS",
            self.renderer.total_width(),
        );
        self.renderer.render_row(&mut buf, &headers_ref, true);
        self.renderer.render_separator(&mut buf);
        self.renderer.render_row(&mut buf, &data_ref, false);
        self.renderer.render_progress_bar(&mut buf, self.progress);
        self.renderer.render_border_bottom(&mut buf);

        // message board
        if !self.messages.is_empty() {
            buf.push('\n');
            self.render_message_board(&mut buf);
        }

        print!("{}", buf);
        // flush so the frame can never linger in the stdout buffer past an
        // alternate-screen leave (which would garble the primary buffer).
        let _ = io::stdout().flush();
    }

    /// **B10** — render a 3-col (Category | Property | Value) sub-table at
    /// the current renderer's total width. coalesces repeated `category`
    /// entries to blanks so the visual grouping is clear ("CPU" → 4 fields
    /// shows as one "CPU" label + 3 indented blanks).
    fn render_info_section(&self, buf: &mut String, title: &str, rows: &[[String; 3]]) {
        if rows.is_empty() { return; }
        let term_width = terminal::width();
        let mut sub = Renderer::new();
        let headers = ["Category", "Property", "Value"];
        // pick the worst-case (longest) per-column row for layout calc.
        let mut widest: [&str; 3] = headers;
        for r in rows {
            for i in 0..3 {
                // compare DISPLAY width (chars), not utf-8 byte length.
                if r[i].chars().count() > widest[i].chars().count() {
                    widest[i] = r[i].as_str();
                }
            }
        }
        sub.calculate_layout(&headers, &widest, term_width);
        sub.render_title(buf, title, sub.total_width());
        sub.render_row(buf, &headers, true);
        sub.render_separator(buf);
        // coalesce repeated category labels into blanks.
        let mut prev_cat: &str = "";
        for r in rows {
            let cat_render = if r[0] == prev_cat { "" } else { prev_cat = r[0].as_str(); &r[0] };
            sub.render_row(buf, &[cat_render, r[1].as_str(), r[2].as_str()], false);
        }
        sub.render_border_bottom(buf);
    }

    fn post_message(&mut self, kind: MessageType, msg: &str) {
        let ts = timestamp();
        self.log_to_file(kind, &ts, msg);
        self.messages.push_back(Message {
            timestamp: ts,
            kind,
            text: msg.to_string(),
        });

        // trim to capacity
        let cap = max_messages();
        while self.messages.len() > cap {
            self.messages.pop_front();
        }
    }

    fn log_to_file(&mut self, kind: MessageType, ts: &str, msg: &str) {
        if let Some(ref mut writer) = self.log_file {
            let type_str = message_type_string(kind);
            let _ = writeln!(writer, "{} [{:<7}] {}", ts, type_str, msg);
            let _ = writer.flush();
        }
    }

    fn render_message_board(&self, buf: &mut String) {
        let width = self.renderer.total_width();
        let inner_width = width.saturating_sub(4); // borders + padding

        let max_msgs = max_messages();
        let start_idx = if self.messages.len() > max_msgs {
            self.messages.len() - max_msgs
        } else {
            0
        };

        self.renderer.render_title(buf, "Messages", width);

        for ii in start_idx..self.messages.len() {
            let msg = &self.messages[ii];
            let msg_color = message_color(msg.kind);

            let mut line = format!(
                "{} [{:<7}] {}",
                msg.timestamp,
                message_type_string(msg.kind),
                msg.text,
            );

            // measure DISPLAY columns (char count), not bytes — messages carry
            // em-dashes / arrows whose utf-8 byte length exceeds their width,
            // which would otherwise under-pad and drift the right border left.
            let mut cols = line.chars().count();
            if cols > inner_width {
                line = if inner_width >= 3 {
                    let kept: String = line.chars().take(inner_width - 3).collect();
                    format!("{kept}...")
                } else {
                    line.chars().take(inner_width).collect()
                };
                cols = line.chars().count();
            }

            // pad to full width (in display columns)
            let pad_len = inner_width.saturating_sub(cols);

            buf.push_str(color::BORDER);
            buf.push_str("\u{2502}"); // │
            buf.push_str(color::RESET);
            buf.push(' ');
            buf.push_str(msg_color);
            buf.push_str(&line);
            for _ in 0..pad_len {
                buf.push(' ');
            }
            buf.push_str(color::RESET);
            buf.push(' ');
            buf.push_str(color::BORDER);
            buf.push_str("\u{2502}"); // │
            buf.push_str(color::RESET);
            buf.push('\n');
        }

        self.renderer.render_border_bottom(buf);
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if self.dynamic {
            print!("{}", ansi::SHOW_CURSOR);
        }
    }
}

fn message_type_string(kind: MessageType) -> &'static str {
    match kind {
        MessageType::Info => "INFO",
        MessageType::Success => "SUCCESS",
        MessageType::Warning => "WARNING",
        MessageType::Error => "ERROR",
        MessageType::Diagnostic => "DIAGNOSTIC",
    }
}

fn message_color(kind: MessageType) -> &'static str {
    match kind {
        MessageType::Info => color::INFO,
        MessageType::Success => color::SUCCESS,
        MessageType::Warning => color::WARNING,
        MessageType::Error => color::ERROR,
        MessageType::Diagnostic => color::DIAGNOSTIC,
    }
}

/// max displayable messages based on terminal height.
fn max_messages() -> usize {
    let hh = terminal::height();
    let reserved = 13; // title box + benchmark section + margins
    let available = hh.saturating_sub(reserved);
    available.clamp(3, 10)
}

/// format current local time as "HH:MM:SS".
fn timestamp() -> String {
    #[cfg(unix)]
    {
        #[repr(C)]
        struct Tm {
            tm_sec: i32,
            tm_min: i32,
            tm_hour: i32,
            _tm_mday: i32,
            _tm_mon: i32,
            _tm_year: i32,
            _tm_wday: i32,
            _tm_yday: i32,
            _tm_isdst: i32,
            _tm_gmtoff: isize,
            _tm_zone: *const u8,
        }

        unsafe extern "C" {
            unsafe fn time(t: *mut i64) -> i64;
            unsafe fn localtime_r(t: *const i64, result: *mut Tm) -> *mut Tm;
        }

        let mut now: i64 = 0;
        let mut tm = std::mem::MaybeUninit::<Tm>::zeroed();
        // SAFETY: standard posix time/localtime_r. writing to stack locals.
        unsafe {
            time(&mut now);
            localtime_r(&now, tm.as_mut_ptr());
            let tm = tm.assume_init();
            format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
        }
    }

    #[cfg(not(unix))]
    {
        "00:00:00".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let table = Table::new("Test", false);
        assert_eq!(table.title, "Test");
        assert!(!table.dynamic);
        assert_eq!(table.progress, 0);
        assert!(table.messages.is_empty());
    }

    #[test]
    fn header_and_row_update() {
        let mut table = Table::new("Test", false);
        table.set_header(&["A", "B", "C"]);
        table.update_row(&["1", "2", "3"]);
        assert_eq!(table.headers.len(), 3);
        assert_eq!(table.data.len(), 3);
        assert_eq!(table.data[1], "2");
    }

    #[test]
    fn progress_clamp() {
        let mut table = Table::new("Test", false);
        table.set_progress(150);
        assert_eq!(table.progress, 100);
        table.set_progress(0);
        assert_eq!(table.progress, 0);
        table.set_progress(50);
        assert_eq!(table.progress, 50);
    }

    #[test]
    fn message_ordering() {
        let mut table = Table::new("Test", false);
        table.post_info("first");
        table.post_warning("second");
        table.post_error("third");
        assert_eq!(table.messages.len(), 3);
        assert_eq!(table.messages[0].text, "first");
        assert_eq!(table.messages[1].kind, MessageType::Warning);
        assert_eq!(table.messages[2].text, "third");
    }

    #[test]
    fn message_capacity_trim() {
        let mut table = Table::new("Test", false);
        // post more than max_messages (at most 10)
        for ii in 0..20 {
            table.post_info(&format!("msg {}", ii));
        }
        assert!(table.messages.len() <= 10);
        // most recent should be last
        assert!(table.messages.back().unwrap().text.contains("19"));
    }

    #[test]
    fn message_type_colors() {
        assert_eq!(message_color(MessageType::Info), color::INFO);
        assert_eq!(message_color(MessageType::Success), color::SUCCESS);
        assert_eq!(message_color(MessageType::Warning), color::WARNING);
        assert_eq!(message_color(MessageType::Error), color::ERROR);
    }

    #[test]
    fn timestamp_format() {
        let ts = timestamp();
        assert_eq!(ts.len(), 8); // "HH:MM:SS"
        assert_eq!(&ts[2..3], ":");
        assert_eq!(&ts[5..6], ":");
    }
}
