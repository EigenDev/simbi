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
use std::io::{self, BufWriter, Stdout, Write};
use std::path::Path;
use std::time::Instant;

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::exit::{ExitKind, render_exit_frame};
use crate::input::Key;
use crate::live::{self, DiagnosticView};
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
    /// dim right-aligned title-bar subtitle (regime · zones); empty by default.
    subtitle: String,
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
    /// live ratatui terminal, present iff the run is dynamic on a tty. drives the
    /// in-alt-screen dashboard; `None` routes refresh to the static string path
    /// (headless, or the persisted final frame after `set_dynamic(false)`). it
    /// draws cells only — `ScreenGuard` owns alt-screen + termios state.
    terminal: Option<Terminal<CrosstermBackend<Stdout>>>,
    /// zone-cycle throughput history (oldest first), one instantaneous sample per
    /// benchmark-row update. a ring buffer capped at the chart width; built from a
    /// scalar already in hand, so it costs no device sync or field copy.
    throughput: VecDeque<f64>,
    /// live tabbed-dashboard state (design 40). the string/headless path ignores
    /// these; the ratatui path projects them into a `DiagnosticView`.
    start: Instant,
    regime: String,
    tab: usize,
    paused: bool,
    frame: u64,
    m_step: u64,
    m_t: f64,
    m_dt: f64,
    m_rate: f64, // zone-cyc/s (converted to Mzcups for display)
    dt_hist: VecDeque<f64>,
    cfl: f64,
    cfl_max: f64,
    blocks_per_level: Vec<u64>,
    /// t=0 baselines for relative conservation drift, set on the first sample.
    mass0: Option<f64>,
    energy0: Option<f64>,
    mass_drift: VecDeque<f64>,
    energy_drift: VecDeque<f64>,
    div_b: VecDeque<f64>,
    max_w: Option<f64>,
    field: Option<crate::live::FieldSlice>,
    /// smoothed (vmin, vmax) for the field colormap, so it doesn't flicker as the
    /// per-frame extrema jitter.
    field_range: Option<(f64, f64)>,
    /// number of selectable fields (for the `f`-key cycle).
    field_count: usize,
    /// compute-host + process resource sample (machine card); None until sampled.
    host: Option<crate::hostinfo::HostStats>,
}

/// throughput-history ring-buffer cap, sized to the chart's pixel width.
const THROUGHPUT_CAPACITY: usize = 240;

/// push onto a capped ring buffer, evicting the oldest when full.
fn push_ring(q: &mut VecDeque<f64>, v: f64) {
    if q.len() == THROUGHPUT_CAPACITY {
        q.pop_front();
    }
    q.push_back(v);
}

/// a history as `Some(vec)` when it holds data, else `None` (so the live view
/// omits the card until the solver has reduced it).
fn opt_hist(q: &VecDeque<f64>) -> Option<Vec<f64>> {
    if q.is_empty() {
        None
    } else {
        Some(q.iter().copied().collect())
    }
}

impl Table {
    /// create a new table. if `dynamic` is true AND stdout is a terminal, each
    /// refresh clears the screen; when output is redirected to a file/pipe the
    /// table falls back to static output so no ansi escapes are embedded.
    pub fn new(title: &str, dynamic: bool) -> Self {
        let dynamic = dynamic && terminal::is_tty();
        // the live ratatui terminal draws into the alternate screen ScreenGuard
        // enters later; constructing it here only wraps stdout (no escapes emitted)
        // and never enables crossterm's raw-mode/alt-screen helpers.
        let term = if dynamic {
            Terminal::new(CrosstermBackend::new(io::stdout())).ok()
        } else {
            None
        };
        Self {
            title: title.to_string(),
            subtitle: String::new(),
            dynamic,
            headers: Vec::new(),
            data: Vec::new(),
            progress: 0,
            messages: VecDeque::new(),
            log_file: None,
            renderer: Renderer::new(),
            system_info: Vec::new(),
            problem_setup: Vec::new(),
            terminal: term,
            throughput: VecDeque::with_capacity(THROUGHPUT_CAPACITY),
            start: Instant::now(),
            regime: String::new(),
            tab: 0,
            paused: false,
            frame: 0,
            m_step: 0,
            m_t: 0.0,
            m_dt: 0.0,
            m_rate: 0.0,
            dt_hist: VecDeque::with_capacity(THROUGHPUT_CAPACITY),
            cfl: 0.0,
            cfl_max: 1.0,
            blocks_per_level: Vec::new(),
            mass0: None,
            energy0: None,
            mass_drift: VecDeque::with_capacity(THROUGHPUT_CAPACITY),
            energy_drift: VecDeque::with_capacity(THROUGHPUT_CAPACITY),
            div_b: VecDeque::with_capacity(THROUGHPUT_CAPACITY),
            max_w: None,
            field: None,
            field_range: None,
            field_count: 1,
            host: None,
        }
    }

    /// how many fields the `f`-key can cycle through (density + any of pressure /
    /// Lorentz W / |B| the regime carries).
    pub fn set_field_count(&mut self, n: usize) {
        self.field_count = n.max(1);
    }

    /// set (or clear) the compute-host + process resource sample for the machine
    /// card. sampled by the solver each cadence so `mem_rss` tracks the footprint.
    pub fn set_host(&mut self, host: Option<crate::hostinfo::HostStats>) {
        self.host = host;
    }

    /// set (or clear) the live field-heatmap slice for the overview hero. cheap:
    /// the slice is already decimated to screen resolution by the caller. the
    /// colormap range is slow-followed (EMA) across frames so it doesn't flicker
    /// when the per-frame extrema jitter; values outside it clip to the endpoints.
    pub fn set_field(&mut self, field: Option<crate::live::FieldSlice>) {
        self.field = field.map(|mut f| {
            const A: f64 = 0.15;
            let (mn, mx) = match self.field_range {
                Some((mn, mx)) => (mn + A * (f.vmin - mn), mx + A * (f.vmax - mx)),
                None => (f.vmin, f.vmax),
            };
            self.field_range = Some((mn, mx));
            f.vmin = mn;
            f.vmax = mx;
            f
        });
    }

    /// record one conservation-reduction sample. mass/energy are tracked as drift
    /// relative to the first (t=0) sample; div B is the absolute peak monopole error.
    pub fn push_conservation(
        &mut self,
        mass: f64,
        energy: Option<f64>,
        div_b: Option<f64>,
        max_w: Option<f64>,
    ) {
        self.max_w = max_w;
        if self.mass0.is_none() {
            self.mass0 = Some(mass);
            self.energy0 = energy;
        }
        let drift = |q: f64, q0: f64| {
            if q0.abs() > 0.0 {
                (q - q0).abs() / q0.abs()
            } else {
                0.0
            }
        };
        if let Some(m0) = self.mass0 {
            push_ring(&mut self.mass_drift, drift(mass, m0));
        }
        if let (Some(e), Some(e0)) = (energy, self.energy0) {
            push_ring(&mut self.energy_drift, drift(e, e0));
        }
        if let Some(db) = div_b {
            push_ring(&mut self.div_b, db);
        }
    }

    /// append one zone-cycle throughput sample (zone-cyc/s) to the chart history,
    /// evicting the oldest once the buffer is full.
    pub fn push_throughput(&mut self, rate: f64) {
        if self.throughput.len() == THROUGHPUT_CAPACITY {
            self.throughput.pop_front();
        }
        self.throughput.push_back(rate);
    }

    /// the physics-regime badge shown in the live stat strip (e.g. "SRHD").
    pub fn set_regime(&mut self, regime: &str) {
        self.regime = regime.to_string();
    }

    /// paused state for the stat-strip indicator (amber "paused" vs green
    /// "integrating"); the driver parks the integrator, this only reflects it.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// the courant number and its nominal stability ceiling, for the CFL gauge.
    pub fn set_cfl(&mut self, cfl: f64, cfl_max: f64) {
        self.cfl = cfl;
        self.cfl_max = cfl_max;
    }

    /// interior cell (zone) count per amr level, for the blocks/level bars.
    pub fn set_blocks_per_level(&mut self, blocks: &[u64]) {
        self.blocks_per_level = blocks.to_vec();
    }

    /// record one diagnostic-cadence sample: the live scalars + a dt-history point.
    pub fn push_metrics(&mut self, step: u64, t: f64, dt: f64, rate_zcps: f64) {
        self.m_step = step;
        self.m_t = t;
        self.m_dt = dt;
        self.m_rate = rate_zcps;
        if self.dt_hist.len() == THROUGHPUT_CAPACITY {
            self.dt_hist.pop_front();
        }
        self.dt_hist.push_back(dt);
    }

    /// apply a navigation keypress; returns true when the frame should redraw.
    pub fn handle_key(&mut self, key: Key) -> bool {
        let n = live::tab_names(self.blocks_per_level.len() > 1).len();
        match key {
            Key::Tab | Key::Right => {
                self.tab = (self.tab + 1) % n;
                true
            }
            Key::BackTab | Key::Left => {
                self.tab = (self.tab + n - 1) % n;
                true
            }
            _ => false,
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

    /// set the dim right-aligned title-bar subtitle (e.g. "newtonian · 65536 zones").
    pub fn set_subtitle(&mut self, subtitle: &str) {
        self.subtitle = subtitle.to_string();
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
        // dropping the live terminal routes refresh to the static string path, so
        // the post-`leave()` final frame is rendered on the primary buffer.
        if !self.dynamic {
            self.terminal = None;
        }
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

    /// render the run's final summary frame. on a tty this is a bounded, state-
    /// colored box on the primary buffer that persists in scrollback; headless it
    /// is a single plain line. call after the live screen has been left — it does
    /// not touch alt-screen or termios state and issues no cursor query.
    pub fn exit_frame(&mut self, kind: ExitKind, summary: &str) {
        if terminal::is_tty() {
            let frame = render_exit_frame(kind, summary, terminal::width() as u16);
            print!("{frame}");
            let _ = io::stdout().flush();
        } else {
            println!("{summary}");
        }
    }

    /// attach a log file. all posted messages are mirrored to disk.
    pub fn set_log_file(&mut self, path: &Path) -> io::Result<()> {
        let file = File::options().create(true).append(true).open(path)?;
        self.log_file = Some(BufWriter::new(file));
        Ok(())
    }

    /// build an owned snapshot for the live tabbed render. owning the data lets it
    /// cross the channel to the render thread (tier 2a). throughput is converted
    /// from zone-cyc/s to Mzcups for display.
    pub fn diagnostic_view(&self) -> DiagnosticView {
        DiagnosticView {
            app_title: self.title.clone(),
            regime: self.regime.clone(),
            attached: String::new(),
            paused: self.paused,
            frame: self.frame,
            t: self.m_t,
            step: self.m_step,
            dt: self.m_dt,
            wall_secs: self.start.elapsed().as_secs_f64(),
            throughput_mzcups: self.m_rate / 1e6,
            tab: self.tab,
            throughput_hist: self.throughput.iter().map(|v| v / 1e6).collect(),
            dt_hist: self.dt_hist.iter().copied().collect(),
            mass_drift: opt_hist(&self.mass_drift),
            energy_drift: opt_hist(&self.energy_drift),
            div_b: opt_hist(&self.div_b),
            max_w: self.max_w,
            cfl: self.cfl,
            cfl_max: self.cfl_max,
            blocks_per_level: self.blocks_per_level.clone(),
            log: self
                .messages
                .iter()
                .map(|m| (m.timestamp.clone(), m.text.clone()))
                .collect(),
            config: self
                .problem_setup
                .iter()
                .map(|r| (r[1].clone(), r[2].clone()))
                .collect(),
            field: self.field.clone(),
            field_count: self.field_count,
            host: self.host.clone(),
        }
    }

    /// render the full display to stdout. the live (tty) frame is drawn with
    /// ratatui; the static/headless frame uses the string renderer below.
    pub fn refresh(&mut self) {
        // live tty path: draw the ratatui dashboard. snapshot first (immutable
        // borrow) so the draw closure owns its data and never aliases `self`.
        if self.terminal.is_some() {
            self.frame = self.frame.wrapping_add(1);
            let view = self.diagnostic_view();
            let term = self.terminal.as_mut().unwrap();
            let _ = term.draw(|frame| live::render(frame, &view));
            return;
        }

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
    fn throughput_ring_buffer_caps_and_evicts_oldest() {
        let mut table = Table::new("Test", false);
        for ii in 0..(THROUGHPUT_CAPACITY + 50) {
            table.push_throughput(ii as f64);
        }
        assert_eq!(table.throughput.len(), THROUGHPUT_CAPACITY);
        // the first 50 samples were evicted; the front is now sample 50.
        assert_eq!(*table.throughput.front().unwrap(), 50.0);
        assert_eq!(
            *table.throughput.back().unwrap(),
            (THROUGHPUT_CAPACITY + 49) as f64
        );
    }

    #[test]
    fn conservation_drift_is_relative_to_baseline() {
        let mut table = Table::new("Test", false);
        // first sample seeds the baseline; drift is zero, no energy row yet.
        table.push_conservation(100.0, None, Some(1e-16), Some(4.5));
        assert_eq!(*table.mass_drift.back().unwrap(), 0.0);
        assert!(table.energy_drift.is_empty());
        assert_eq!(*table.div_b.back().unwrap(), 1e-16);
        assert_eq!(table.max_w, Some(4.5));
        // a 1% mass change reads as 0.01 drift relative to the 100.0 baseline.
        table.push_conservation(101.0, None, Some(2e-16), Some(4.8));
        assert!((*table.mass_drift.back().unwrap() - 0.01).abs() < 1e-12);
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
