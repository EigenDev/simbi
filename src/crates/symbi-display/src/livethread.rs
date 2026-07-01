// =============================================================================
// livethread.rs
//
// render-thread decouple (design 40, tier 2a). the solver thread PUBLISHES a
// DiagnosticView snapshot at its cadence; a dedicated render thread owns the
// ratatui terminal, draws at ~30 fps, and handles keyboard input — so tab / pause
// respond instantly regardless of step rate, the ui never freezes on a heavy step,
// and the solver never blocks on a slow terminal.
//
// ScreenGuard (alt screen + termios) and SignalGuard (Ctrl-C) stay owned by the
// MAIN thread. the render thread only draws cells + reads keys over the mode they
// established, and exits the moment a signal is caught (checking
// `signal_guard::stop_requested`) so it stops touching a terminal the async-signal-
// safe handler has already restored. the main thread joins the render thread before
// leaving the alt screen, so terminal ownership is never shared concurrently.
//
// usage (solver thread):
//   let mut dash = LiveDashboard::spawn().unwrap();   // AFTER ScreenGuard::enter()
//   loop {
//       let c = dash.controls();
//       if c.quit() || signal_guard::stop_requested() { break; }
//       // ... step, then at the cadence:
//       dash.publish(view);
//   }
//   dash.shutdown();                                  // BEFORE ScreenGuard::leave()
// =============================================================================

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::input::{Key, poll_key_timeout};
use crate::live::{self, Colormap, DiagnosticView, FieldSlice};
use crate::{signal_guard, terminal};

/// the field-heatmap colormaps the `c`-key cycles through.
const COLORMAPS: [Colormap; 3] = [Colormap::Inferno, Colormap::Viridis, Colormap::Magma];

/// a published render frame. `fields` is the full bundle of selectable fields
/// (read-only attach): when non-empty the render thread picks `fields[field_kind]`
/// itself, so the `f`-key switches with no producer round-trip. the in-process
/// solver path leaves it empty and sets `view.field` directly.
struct Frame {
    view: DiagnosticView,
    fields: Vec<FieldSlice>,
}

/// solver-affecting control flags, set by the render thread's key handler and read
/// by the solver loop. tab selection is NOT here — it is render-only state the
/// render thread owns, so switching panels never round-trips through the solver.
#[derive(Default)]
pub struct Controls {
    paused: AtomicBool,
    quit: AtomicBool,
    step_once: AtomicBool,
    force_cp: AtomicBool,
    /// index of the field the `f`-key has selected; the solver decimates it and
    /// reports how many fields exist via `DiagnosticView::field_count`.
    field_kind: AtomicUsize,
}

impl Controls {
    /// the currently-selected field index (`f`-key cycle).
    pub fn field_kind(&self) -> usize {
        self.field_kind.load(Ordering::SeqCst)
    }
    /// paused: the solver should park (take no step) while keeping publishing.
    pub fn paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }
    /// a graceful quit was requested (`q`) — treat like a caught signal.
    pub fn quit(&self) -> bool {
        self.quit.load(Ordering::SeqCst)
    }
    /// consume a pending single-step request (`s` while paused).
    pub fn take_step(&self) -> bool {
        self.step_once.swap(false, Ordering::SeqCst)
    }
    /// consume a pending force-checkpoint request (`w`).
    pub fn take_checkpoint(&self) -> bool {
        self.force_cp.swap(false, Ordering::SeqCst)
    }
}

/// handle to the render thread + its shared control flags.
pub struct LiveDashboard {
    tx: SyncSender<Frame>,
    controls: Arc<Controls>,
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LiveDashboard {
    /// spawn the render thread. `ScreenGuard` must already have entered the alt
    /// screen. returns `None` off a tty (the caller renders synchronously/headless).
    pub fn spawn() -> Option<LiveDashboard> {
        if !terminal::is_tty() {
            return None;
        }
        let (tx, rx) = mpsc::sync_channel::<Frame>(1);
        let controls = Arc::new(Controls::default());
        let running = Arc::new(AtomicBool::new(true));
        let handle = thread::spawn({
            let controls = Arc::clone(&controls);
            let running = Arc::clone(&running);
            move || render_loop(rx, controls, running)
        });
        Some(LiveDashboard {
            tx,
            controls,
            running,
            handle: Some(handle),
        })
    }

    /// publish the latest snapshot (non-blocking, latest-wins — silently dropped if
    /// the render thread has not yet consumed the previous one). the in-process
    /// path: `view.field` is already the selected field, no bundle.
    pub fn publish(&self, view: DiagnosticView) {
        let _ = self.tx.try_send(Frame {
            view,
            fields: Vec::new(),
        });
    }

    /// publish a snapshot with the full field bundle (read-only attach): the render
    /// thread selects `fields[field_kind]` locally, so the `f`-key switches fields
    /// with no producer round-trip.
    pub fn publish_bundle(&self, view: DiagnosticView, fields: Vec<FieldSlice>) {
        let _ = self.tx.try_send(Frame { view, fields });
    }

    pub fn controls(&self) -> &Controls {
        &self.controls
    }

    /// stop the render thread and join it, so the main thread regains sole terminal
    /// ownership (to leave the alt screen + print the exit frame).
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for LiveDashboard {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// the render thread body: draw the latest published snapshot at ~30 fps and route
/// keys. owns the ratatui terminal for its lifetime; no other thread touches it.
fn render_loop(rx: Receiver<Frame>, controls: Arc<Controls>, running: Arc<AtomicBool>) {
    let mut terminal = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut latest: Option<Frame> = None;
    let mut tab = 0usize;
    let mut frame = 0u64;
    let mut cmap_idx = 0usize; // `c`-key colormap, applied render-side (no solver)

    // exit on shutdown OR a caught signal (so we never draw into a terminal the
    // async-signal-safe handler has already restored).
    while running.load(Ordering::SeqCst) && !signal_guard::stop_requested() {
        while let Ok(v) = rx.try_recv() {
            latest = Some(v);
        }

        // block up to ~30ms for a key: frame pacing (~33 fps) + instant key wake.
        if let Some(key) = poll_key_timeout(30) {
            let n = latest
                .as_ref()
                .map(|f| live::tab_names(f.view.blocks_per_level.len() > 1).len())
                .unwrap_or(1)
                .max(1);
            match key {
                Key::Tab | Key::Right => tab = (tab + 1) % n,
                Key::BackTab | Key::Left => tab = (tab + n - 1) % n,
                Key::Char(' ') => {
                    let p = controls.paused.load(Ordering::SeqCst);
                    controls.paused.store(!p, Ordering::SeqCst);
                }
                Key::Char('q') | Key::Esc => controls.quit.store(true, Ordering::SeqCst),
                Key::Char('s') => controls.step_once.store(true, Ordering::SeqCst),
                Key::Char('w') => controls.force_cp.store(true, Ordering::SeqCst),
                // c: cycle colormap (render-side, no solver round-trip).
                Key::Char('c') => cmap_idx = (cmap_idx + 1) % COLORMAPS.len(),
                // f: cycle the displayed field. bounded by the bundle length in
                // attach mode (client-side switch), else the solver's field_count.
                Key::Char('f') => {
                    let fc = latest
                        .as_ref()
                        .map(|f| {
                            if f.fields.is_empty() {
                                f.view.field_count
                            } else {
                                f.fields.len()
                            }
                        })
                        .unwrap_or(1)
                        .max(1);
                    let k = (controls.field_kind.load(Ordering::SeqCst) + 1) % fc;
                    controls.field_kind.store(k, Ordering::SeqCst);
                }
                _ => {}
            }
        }

        if let Some(Frame { view: v, fields }) = latest.as_mut() {
            // read-only attach: the producer sends every field; select the f-key's
            // choice here so the switch is instant (no round-trip). in-process runs
            // send an empty bundle and v.field is already the selected field.
            if !fields.is_empty() {
                let idx = controls.field_kind.load(Ordering::SeqCst) % fields.len();
                v.field = Some(fields[idx].clone());
                v.field_count = fields.len();
            }
            // inject the render-thread-owned ui state (tab / spinner / paused badge /
            // colormap).
            let n = live::tab_names(v.blocks_per_level.len() > 1).len();
            v.tab = tab.min(n.saturating_sub(1));
            v.frame = frame;
            v.paused = controls.paused.load(Ordering::SeqCst);
            if let Some(field) = v.field.as_mut() {
                field.cmap = COLORMAPS[cmap_idx];
            }
            let _ = terminal.draw(|f| live::render(f, v));
        }
        frame = frame.wrapping_add(1);
    }
}
