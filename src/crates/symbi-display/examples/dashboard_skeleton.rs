// =============================================================================
// dashboard_skeleton.rs
//
// throwaway driver for the tabbed dashboard. it owns a dummy xorshift
// integrator that random-walks a snapshot, projects it into a `live::DiagnosticView`,
// and renders through the SHARED production render (`symbi_display::live`) — so
// look-and-feel iterates here (compile ~0.4s, no maturin) and production inherits
// exactly the same output.
//
// it drives the terminal via the REAL production path: SignalGuard owns the
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

use symbi_display::live::{Colormap, DiagnosticView, FieldSlice, fmt_wall};
use symbi_display::{LiveDashboard, ScreenGuard, SignalGuard, signal_guard};

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

/// the dummy run state — a stand-in for the per-cadence DiagnosticSnapshot. tab +
/// pause are now render-thread state (owned by the LiveDashboard), not here.
struct App {
    frame: u64,
    field_kind: usize,
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
            frame: 0,
            field_kind: 0,
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

    /// advance one diagnostic cadence — the ~10-step reduction interval.
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

    /// a synthetic animated field, varied by `field_kind` to exercise the `f`-key
    /// cycle: 0 = density (2D heatmap), 1 = pressure (2D heatmap), 2 = a 1D line
    /// profile. the label is just the field name — the render thread appends the
    /// active colormap.
    fn field_slice(&self) -> FieldSlice {
        let t = self.frame as f64 * 0.05;
        let (label, width, height, data) = match self.field_kind {
            1 => {
                let (w, h) = (192usize, 108usize);
                let mut d = Vec::with_capacity(w * h);
                for j in 0..h {
                    for i in 0..w {
                        let x = i as f64 / w as f64 * 8.0;
                        let y = j as f64 / h as f64 * 6.0 - 3.0;
                        let v = 1.0 + 0.6 * (x * 1.5 - t).sin() * (y * 1.2 + t * 0.7).cos();
                        d.push(v as f32);
                    }
                }
                ("pressure", w, h, d)
            }
            2 => {
                // a 1D line-out (height = 1): a moving shock-like profile.
                let n = 220usize;
                let mut d = Vec::with_capacity(n);
                for i in 0..n {
                    let x = i as f64 / n as f64 * 10.0;
                    let v = 1.0 + 2.2 * (-(x - 3.0 - t * 0.25).powi(2)).exp()
                        + 0.25 * (x * 1.6 - t).sin();
                    d.push(v as f32);
                }
                ("density (line-out)", n, 1, d)
            }
            _ => {
                let (w, h) = (192usize, 108usize);
                let mut d = Vec::with_capacity(w * h);
                for j in 0..h {
                    for i in 0..w {
                        let x = i as f64 / w as f64 * 8.0;
                        let y = j as f64 / h as f64 * 6.0 - 3.0;
                        let shear = (y * 2.5).tanh();
                        let billow = 0.5 * (x * 2.0 + t).sin() * (-y * y).exp();
                        let ripple = 0.25 * ((x + y) * 3.0 - t * 1.3).sin();
                        d.push((shear + billow + ripple) as f32);
                    }
                }
                ("density", w, h, d)
            }
        };
        let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
        FieldSlice {
            label: label.into(),
            width,
            height,
            data,
            vmin: mn as f64,
            vmax: mx as f64,
            cmap: Colormap::Inferno,
            preserve_aspect: false,
            log_scale: false,
        }
    }

    /// project the dummy state into the render snapshot.
    fn view(&self) -> DiagnosticView {
        DiagnosticView {
            app_title: "hydroflux — kelvin_helmholtz.toml".into(),
            regime: "RHD".into(),
            attached: "attached · rank 0 / 1".into(),
            // tab / paused / frame are overridden by the render thread's ui state.
            paused: false,
            frame: self.frame,
            t: self.t,
            step: self.step,
            dt: self.dt,
            wall_secs: self.start.elapsed().as_secs_f64(),
            throughput_mzcups: self.throughput.back().copied().unwrap_or(0.0),
            progress: 63,
            tab: 0,
            config_scroll: 0,
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
                ("Physics".into(), "regime".into(), "rhd".into()),
                ("Physics".into(), "eos".into(), "ideal gas (gamma = 1.6667)".into()),
                ("Geometry".into(), "coords".into(), "cartesian · 2D".into()),
                ("Geometry".into(), "resolution".into(), "256 x 256  (65536 zones)".into()),
                ("Numerics".into(), "solver".into(), "hllc · plm · minmod-MC (theta = 1.50)".into()),
                ("Numerics".into(), "timestepping".into(), "rk2 · cfl 0.10".into()),
                ("Run".into(), "t_final".into(), "20.0000".into()),
                ("Run".into(), "output".into(), "data/kh_config/".into()),
            ],
            field: Some(self.field_slice()),
            field_count: 3, // density / pressure / line-out (dummy)
            host: Some(symbi_display::hostinfo::HostStats::sample()),
        }
    }
}

fn main() -> io::Result<()> {
    if !symbi_display::terminal::is_tty() {
        eprintln!("dashboard_skeleton needs an interactive terminal.");
        return Ok(());
    }
    // tier-2a architecture under test: SignalGuard owns the graceful-interrupt flag,
    // ScreenGuard owns the alt screen + termios, and a dedicated RENDER THREAD
    // (LiveDashboard) owns the terminal + input, drawing at ~30 fps. this main thread
    // is the (dummy) SOLVER: it steps, publishes a snapshot, and reads the render
    // thread's control flags — it never touches the terminal.
    let _sig = SignalGuard::install();
    let mut screen = ScreenGuard::enter();
    let mut dash = match LiveDashboard::spawn() {
        Some(d) => d,
        None => {
            screen.leave();
            return Ok(());
        }
    };

    let mut app = App::new();
    // a deliberately SLOW dummy step (200ms) so the responsiveness win is obvious:
    // tab / pause stay instant (30 fps render thread) while the "solver" crawls.
    let step_time = Duration::from_millis(200);

    let interrupted = loop {
        if signal_guard::stop_requested() {
            break true;
        }
        let c = dash.controls();
        if c.quit() {
            break false;
        }

        // pause: park (the render thread keeps drawing + accepting input).
        while c.paused() && !c.quit() && !signal_guard::stop_requested() {
            if c.take_step() {
                break; // single step while paused
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        if c.quit() {
            break false;
        }
        if signal_guard::stop_requested() {
            break true;
        }

        app.tick();
        app.field_kind = c.field_kind(); // the `f`-key selection (dummy: 0/1/2)
        if c.take_checkpoint() {
            app.push_log("manual checkpoint (dummy)".to_string());
        }
        dash.publish(app.view());

        std::thread::sleep(step_time); // simulate a heavy step
    };

    dash.shutdown();
    screen.leave();
    if interrupted {
        println!("interrupted — a real run would write a restart checkpoint here.");
    } else {
        println!("quit.");
    }
    Ok(())
}
