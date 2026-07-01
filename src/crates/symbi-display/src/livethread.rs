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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::input::{Key, poll_key_timeout};
use crate::live::{self, DiagnosticView};
use crate::{signal_guard, terminal};

/// solver-affecting control flags, set by the render thread's key handler and read
/// by the solver loop. tab selection is NOT here — it is render-only state the
/// render thread owns, so switching panels never round-trips through the solver.
#[derive(Default)]
pub struct Controls {
    paused: AtomicBool,
    quit: AtomicBool,
    step_once: AtomicBool,
    force_cp: AtomicBool,
}

impl Controls {
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
    tx: SyncSender<DiagnosticView>,
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
        let (tx, rx) = mpsc::sync_channel::<DiagnosticView>(1);
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
    /// the render thread has not yet consumed the previous one).
    pub fn publish(&self, view: DiagnosticView) {
        let _ = self.tx.try_send(view);
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
fn render_loop(rx: Receiver<DiagnosticView>, controls: Arc<Controls>, running: Arc<AtomicBool>) {
    let mut terminal = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut latest: Option<DiagnosticView> = None;
    let mut tab = 0usize;
    let mut frame = 0u64;

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
                .map(|v| live::tab_names(v.blocks_per_level.len() > 1).len())
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
                _ => {}
            }
        }

        if let Some(v) = latest.as_mut() {
            // inject the render-thread-owned ui state (tab / spinner / paused badge).
            let n = live::tab_names(v.blocks_per_level.len() > 1).len();
            v.tab = tab.min(n.saturating_sub(1));
            v.frame = frame;
            v.paused = controls.paused.load(Ordering::SeqCst);
            let _ = terminal.draw(|f| live::render(f, v));
        }
        frame = frame.wrapping_add(1);
    }
}
