// =============================================================================
// dashboard_skeleton.rs
//
// throwaway driver for the design-40 tabbed dashboard. it owns a dummy xorshift
// integrator that random-walks a snapshot, projects it into a `live::DiagnosticView`,
// and renders through the SHARED production render (`symbi_display::live`) — so
// look-and-feel iterates here (compile ~0.4s, no maturin) and production inherits
// exactly the same output.
//
// it drives the terminal via the REAL production seam: SignalGuard owns the
// graceful-interrupt flag (Ctrl-C -> SIGINT, never a keystroke), ScreenGuard owns
// the alt screen + termios (ICANON/ECHO off, ISIG on), and poll_key reads the
// remaining keys over that mode. no crossterm raw mode is enabled anywhere.
//
// run:  cargo run -p symbi-display --example dashboard_skeleton
// keys: tab / shift-tab / arrows switch · space pause · s step · q/esc quit
// =============================================================================

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use symbi_display::live::{self, DiagnosticView, fmt_wall, tab_names};
use symbi_display::{Key, ScreenGuard, SignalGuard, poll_key_timeout};

const LEVELS: usize = 3;

/// tiny xorshift64 prng — deterministic dummy data, no `rand` dependency.
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// centered in [-0.5, 0.5).
    fn c(&mut self) -> f64 {
        self.f() - 0.5
    }
}

fn push_cap(q: &mut VecDeque<f64>, v: f64, cap: usize) {
    if q.len() == cap {
        q.pop_front();
    }
    q.push_back(v);
}

/// the dummy run state — a stand-in for the per-cadence DiagnosticSnapshot plus
/// ui state (active tab, pause).
struct App {
    tab: usize,
    paused: bool,
    frame: u64,
    rng: Rng,
    start: Instant,
    step: u64,
    cp_index: u64,
    t: f64,
    dt: f64,
    throughput: VecDeque<f64>,
    dt_hist: VecDeque<f64>,
    mass_drift: VecDeque<f64>,
    energy_drift: VecDeque<f64>,
    div_b: VecDeque<f64>,
    max_w: f64,
    cfl: f64,
    cfl_max: f64,
    blocks: [u64; LEVELS],
    log: VecDeque<(String, String)>,
}

impl App {
    fn new() -> Self {
        let mut app = App {
            tab: 0,
            paused: false,
            frame: 0,
            rng: Rng(0x9e3779b97f4a7c15),
            start: Instant::now(),
            step: 24_000,
            cp_index: 240,
            t: 1.8423,
            dt: 3.1e-4,
            throughput: VecDeque::new(),
            dt_hist: VecDeque::new(),
            mass_drift: VecDeque::new(),
            energy_drift: VecDeque::new(),
            div_b: VecDeque::new(),
            max_w: 4.82,
            cfl: 0.40,
            cfl_max: 0.80,
            blocks: [64, 210, 96],
            log: VecDeque::new(),
        };
        for _ in 0..120 {
            app.tick();
        }
        app.push_log("skeleton started — dummy integrator".to_string());
        app
    }

    /// advance one diagnostic cadence — the design's ~10-step reduction interval.
    fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.step += 10;
        self.dt = (self.dt * (1.0 + 0.05 * self.rng.c())).clamp(1e-5, 1e-2);
        self.t += self.dt * 10.0;

        let tp = (self.throughput.back().copied().unwrap_or(148.0) * (1.0 + 0.08 * self.rng.c()))
            .clamp(20.0, 260.0);
        push_cap(&mut self.throughput, tp, 240);
        push_cap(&mut self.dt_hist, self.dt, 120);

        let md = (self.mass_drift.back().copied().unwrap_or(2.4e-13) + self.rng.c() * 1e-14).abs();
        push_cap(&mut self.mass_drift, md, 120);
        let ed = (self.energy_drift.back().copied().unwrap_or(8.1e-9) + self.rng.c() * 4e-10).abs();
        push_cap(&mut self.energy_drift, ed, 120);
        let db = (self.div_b.back().copied().unwrap_or(4.6e-16) + self.rng.c() * 2e-17).abs();
        push_cap(&mut self.div_b, db, 120);

        self.max_w = (self.max_w + self.rng.c() * 0.05).clamp(1.0, 12.0);
        self.cfl = (self.cfl + self.rng.c() * 0.03).clamp(0.05, self.cfl_max);

        if self.rng.f() < 0.03 {
            let lvl = (self.rng.f() * LEVELS as f64) as usize % LEVELS;
            let delta = (self.rng.f() * 20.0) as i64 - 8;
            self.blocks[lvl] = (self.blocks[lvl] as i64 + delta).max(1) as u64;
            self.push_log(format!("regrid → level {lvl} : {delta:+} blocks"));
        }
        if self.step % 2000 < 10 {
            self.cp_index += 1;
            self.push_log(format!(
                "checkpoint written : 256x256.chkpt.{:03}.h5 (318 MB)",
                self.cp_index
            ));
        }
    }

    fn push_log(&mut self, text: String) {
        let ts = fmt_wall(self.start.elapsed().as_secs_f64());
        self.log.push_back((ts, text));
        while self.log.len() > 200 {
            self.log.pop_front();
        }
    }

    fn next_tab(&mut self) {
        let n = tab_names(true).len();
        self.tab = (self.tab + 1) % n;
    }
    fn prev_tab(&mut self) {
        let n = tab_names(true).len();
        self.tab = (self.tab + n - 1) % n;
    }

    /// project the dummy state into the render snapshot.
    fn view(&self) -> DiagnosticView {
        DiagnosticView {
            app_title: "hydroflux — kelvin_helmholtz.toml".into(),
            regime: "SRHD".into(),
            attached: "attached · rank 0 / 1".into(),
            paused: self.paused,
            frame: self.frame,
            t: self.t,
            step: self.step,
            dt: self.dt,
            wall_secs: self.start.elapsed().as_secs_f64(),
            throughput_mzcups: self.throughput.back().copied().unwrap_or(0.0),
            tab: self.tab,
            throughput_hist: self.throughput.iter().copied().collect(),
            dt_hist: self.dt_hist.iter().copied().collect(),
            mass_drift: Some(self.mass_drift.iter().copied().collect()),
            energy_drift: Some(self.energy_drift.iter().copied().collect()),
            div_b: Some(self.div_b.iter().copied().collect()),
            max_w: Some(self.max_w),
            cfl: self.cfl,
            cfl_max: self.cfl_max,
            blocks_per_level: self.blocks.to_vec(),
            log: self.log.iter().cloned().collect(),
            config: vec![
                ("regime".into(), "srhd".into()),
                ("eos".into(), "ideal gas (gamma = 1.6667)".into()),
                ("coords".into(), "cartesian · 2D".into()),
                ("resolution".into(), "256 x 256  (65536 zones)".into()),
                ("solver".into(), "hllc · plm · minmod-MC (theta = 1.50)".into()),
                ("timestepping".into(), "rk2 · cfl 0.10".into()),
                ("t_final".into(), "20.0000".into()),
                ("output".into(), "data/kh_config/".into()),
            ],
        }
    }
}

fn main() -> io::Result<()> {
    if !symbi_display::terminal::is_tty() {
        eprintln!("dashboard_skeleton needs an interactive terminal.");
        return Ok(());
    }
    // the production seam under test: SignalGuard owns the graceful-interrupt flag
    // (Ctrl-C -> SIGINT, never a keystroke), ScreenGuard owns the alt screen +
    // termios (ICANON/ECHO off, ISIG on), and poll_key reads the remaining keys
    // over that mode. no crossterm raw mode is enabled anywhere.
    let sig = SignalGuard::install();
    let mut screen = ScreenGuard::enter();
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let result = run(&mut terminal, &sig);

    screen.leave();
    let interrupted = result?;
    if interrupted {
        println!("interrupted — a real run would write a restart checkpoint here.");
    } else {
        println!("quit.");
    }
    Ok(())
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    sig: &SignalGuard,
) -> io::Result<bool> {
    let mut app = App::new();
    let mut last = Instant::now();
    let tick = Duration::from_millis(60); // dummy diagnostic cadence

    loop {
        if sig.stop_requested() {
            return Ok(true);
        }
        terminal.draw(|frame| live::render(frame, &app.view()))?;
        if sig.stop_requested() {
            return Ok(true);
        }
        // block up to 33ms for a key (frame pacing + prompt response); a signal
        // interrupts the poll and is caught by the stop_requested check above.
        match poll_key_timeout(33) {
            Some(Key::Char('q')) | Some(Key::Esc) => return Ok(false),
            Some(Key::Char(' ')) => app.paused = !app.paused,
            Some(Key::Char('s')) => app.tick(),
            Some(Key::Tab) | Some(Key::Right) => app.next_tab(),
            Some(Key::BackTab) | Some(Key::Left) => app.prev_tab(),
            _ => {}
        }

        if last.elapsed() >= tick {
            if app.paused {
                app.frame = app.frame.wrapping_add(1);
            } else {
                app.tick();
            }
            last = Instant::now();
        }
    }
}
