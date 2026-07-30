// =============================================================================
// livethread.rs
//
// render-thread decouple. the solver thread PUBLISHES a
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
    /// the 3D slice orientation the `o`-key has selected: 0 = z mid-plane (x, y),
    /// 1 = y mid-plane (x, z), 2 = x mid-plane (y, z). ignored on 1D/2D runs.
    slice_orient: AtomicUsize,
    /// the heatmap zoom exponent (`+`/`-` keys): the slice samples a centered
    /// 1/2^k-extent window of the display plane, decimated to the same screen
    /// resolution — each step doubles the magnification about the domain center.
    zoom_level: AtomicUsize,
}

impl Controls {
    /// the selected field index (`f`-key cycle).
    pub fn field_kind(&self) -> usize {
        self.field_kind.load(Ordering::SeqCst)
    }
    /// the selected 3D slice orientation (`o`-key cycle): 0 = z, 1 = y, 2 = x mid-plane.
    pub fn slice_orient(&self) -> usize {
        self.slice_orient.load(Ordering::SeqCst)
    }
    /// the heatmap zoom exponent (`+`/`-`): magnification is 2^k about the center.
    pub fn zoom_level(&self) -> usize {
        self.zoom_level.load(Ordering::SeqCst)
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

/// which side owns the pause state the badge reports.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PauseSource {
    /// in-process: the space key toggles `Controls` and the integrator parks on that flag,
    /// so the live value is authoritative and the badge reads it at draw time. a parked
    /// producer publishes NOTHING, which is exactly why its published copy cannot be used.
    LocalControls,
    /// read-only attach: the solver is a different process, reachable only through the
    /// snapshot it writes. its pause state travels in the published view and no local key
    /// can change it, so the badge reports what the solver reported. a batch solver runs
    /// off-tty with no `Controls` at all and therefore reports never being paused.
    PublishedView,
}

/// the ui state the render thread owns between frames, applied to each received view.
struct RenderOwned {
    tab: usize,
    scroll: u16,
    frame: u64,
    cmap_idx: usize,
    log_scale: bool,
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
        Self::spawn_with(PauseSource::LocalControls)
    }

    /// spawn for read-only `simbi attach`: the pause badge comes from the published view
    /// and the pause / step / checkpoint keys are inert, since they cannot reach the
    /// solver process.
    pub fn spawn_read_only() -> Option<LiveDashboard> {
        Self::spawn_with(PauseSource::PublishedView)
    }

    fn spawn_with(pause_source: PauseSource) -> Option<LiveDashboard> {
        if !terminal::is_tty() {
            return None;
        }
        let (tx, rx) = mpsc::sync_channel::<Frame>(1);
        let controls = Arc::new(Controls::default());
        let running = Arc::new(AtomicBool::new(true));
        let handle = thread::spawn({
            let controls = Arc::clone(&controls);
            let running = Arc::clone(&running);
            move || render_loop(rx, controls, running, pause_source)
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

/// write the ui state the RENDER THREAD owns into a received view, just before drawing:
/// tab, scroll offset, spinner frame, the paused badge, and the colormap.
///
/// the paused badge belongs here rather than in the published view because a paused run
/// PUBLISHES NOTHING — the producer parks its integrator and stops sending frames, so a
/// producer-side copy of the flag would read "integrating" for exactly as long as the run
/// is paused. reading `Controls` at draw time makes the badge track the state the space
/// key actually toggles.
fn apply_render_owned_state(
    v: &mut live::DiagnosticView,
    controls: &Controls,
    owned: RenderOwned,
    pause_source: PauseSource,
) {
    let n = live::tab_names(v.blocks_per_level.len() > 1).len();
    v.tab = owned.tab.min(n.saturating_sub(1));
    v.config_scroll = owned.scroll;
    v.frame = owned.frame;
    if pause_source == PauseSource::LocalControls {
        v.paused = controls.paused();
    }
    if let Some(field) = v.field.as_mut() {
        field.cmap = COLORMAPS[owned.cmap_idx];
        field.log_scale = owned.log_scale;
    }
}

/// the render thread body: draw the latest published snapshot at ~30 fps and route
/// keys. owns the ratatui terminal for its lifetime; no other thread touches it.
fn render_loop(
    rx: Receiver<Frame>,
    controls: Arc<Controls>,
    running: Arc<AtomicBool>,
    pause_source: PauseSource,
) {
    let mut terminal = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut latest: Option<Frame> = None;
    let mut tab = 0usize;
    let mut frame = 0u64;
    let mut cmap_idx = 0usize; // `c`-key colormap, applied render-side (no solver)
    let mut log_scale = false; // `l`-key log10 colormap normalization, render-side
    let mut scroll = 0u16; // up/down scroll offset for a tall panel (the config listing); reset on tab

    // exit on shutdown OR a caught signal (so drawing never targets a terminal the
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
                Key::Tab | Key::Right => {
                    tab = (tab + 1) % n;
                    scroll = 0;
                }
                Key::BackTab | Key::Left => {
                    tab = (tab + n - 1) % n;
                    scroll = 0;
                }
                // scroll the active panel's listing (the config tab overflows). the renderer clamps
                // to the exact overflow; cap the state loosely at the row count to avoid runaway.
                Key::Up => scroll = scroll.saturating_sub(1),
                Key::Down => {
                    let cap = latest
                        .as_ref()
                        .map(|f| f.view.config.len() as u16)
                        .unwrap_or(0);
                    scroll = (scroll + 1).min(cap);
                }
                // pause / step / checkpoint act on the LOCAL integrator. under read-only
                // attach there is none, so they stay inert rather than moving a flag no
                // solver reads -- a local toggle would otherwise paint a paused badge over
                // a run that is still integrating.
                Key::Char(' ') if pause_source == PauseSource::LocalControls => {
                    let p = controls.paused.load(Ordering::SeqCst);
                    controls.paused.store(!p, Ordering::SeqCst);
                }
                Key::Char('q') | Key::Esc => controls.quit.store(true, Ordering::SeqCst),
                Key::Char('s') if pause_source == PauseSource::LocalControls => {
                    controls.step_once.store(true, Ordering::SeqCst)
                }
                Key::Char('w') if pause_source == PauseSource::LocalControls => {
                    controls.force_cp.store(true, Ordering::SeqCst)
                }
                // c: cycle colormap (render-side, no solver round-trip).
                Key::Char('c') => cmap_idx = (cmap_idx + 1) % COLORMAPS.len(),
                // l: toggle log10 colormap normalization (render-side) — fields
                // spanning decades (density, pressure) are unreadable linearly.
                Key::Char('l') => log_scale = !log_scale,
                // f: cycle the displayed field. bounded by the bundle length in
                // attach mode (client-side switch), else the solver's field_count.
                // o: cycle the 3D slice orientation (z -> y -> x mid-plane); the
                // solver re-decimates on its next publish. no-op on 1D/2D runs.
                Key::Char('o') => {
                    let k = (controls.slice_orient.load(Ordering::SeqCst) + 1) % 3;
                    controls.slice_orient.store(k, Ordering::SeqCst);
                }
                // +/-: zoom the heatmap about the domain center (2^k magnification,
                // clamped to 16x); the solver re-decimates on its next publish.
                Key::Char('+') | Key::Char('=') => {
                    let k = (controls.zoom_level.load(Ordering::SeqCst) + 1).min(4);
                    controls.zoom_level.store(k, Ordering::SeqCst);
                }
                Key::Char('-') => {
                    let k = controls.zoom_level.load(Ordering::SeqCst).saturating_sub(1);
                    controls.zoom_level.store(k, Ordering::SeqCst);
                }
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
            let owned = RenderOwned {
                tab,
                scroll,
                frame,
                cmap_idx,
                log_scale,
            };
            apply_render_owned_state(v, &controls, owned, pause_source);
            let _ = terminal.draw(|f| live::render(f, v));
        }
        frame = frame.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned() -> RenderOwned {
        RenderOwned {
            tab: 0,
            scroll: 0,
            frame: 0,
            cmap_idx: 0,
            log_scale: false,
        }
    }

    #[test]
    fn an_in_process_badge_follows_the_live_controls_not_the_published_view() {
        // the badge is fed at DRAW time from `Controls`, which is the state the space key
        // toggles and the integrator parks on. a published view carries no pause state, so
        // a renderer trusting the view would read "integrating" throughout a pause -- and
        // the absence of any producer calling a setter would look like a badge never wired.
        let mut view = crate::table::Table::new("badge probe", false).diagnostic_view();
        assert!(
            !view.paused,
            "an in-process view must not claim to carry pause state"
        );

        let controls = Controls::default();
        apply_render_owned_state(&mut view, &controls, owned(), PauseSource::LocalControls);
        assert!(!view.paused, "an unpaused run must render as integrating");

        controls.paused.store(true, Ordering::SeqCst);
        apply_render_owned_state(&mut view, &controls, owned(), PauseSource::LocalControls);
        assert!(
            view.paused,
            "a paused run must render the paused badge even though the view said otherwise"
        );
    }

    #[test]
    fn an_attached_badge_reports_the_solver_state_and_ignores_local_keys() {
        // read-only attach: the solver is another process and the snapshot is the only
        // channel. a local key cannot pause it, so a locally-set flag must NOT reach the
        // badge -- otherwise pressing space paints "paused" over a run that is still
        // integrating, which is the whole failure this separation exists to prevent.
        let mut view = crate::table::Table::new("attach probe", false).diagnostic_view();
        let controls = Controls::default();
        controls.paused.store(true, Ordering::SeqCst);
        apply_render_owned_state(&mut view, &controls, owned(), PauseSource::PublishedView);
        assert!(
            !view.paused,
            "a local pause must not reach the badge of a remote run"
        );

        // and the solver's own reported state is preserved rather than overwritten.
        view.paused = true;
        let idle = Controls::default();
        apply_render_owned_state(&mut view, &idle, owned(), PauseSource::PublishedView);
        assert!(
            view.paused,
            "the solver reported paused; the badge must keep it"
        );
    }
}
