// =============================================================================
// signal_guard.rs
//
// run-scoped graceful-interrupt + terminal recovery. mirrors the c++
// `helpers::catch_signals()` / `InterruptException` flow: a caught signal does
// NOT kill the process outright — it sets a stop flag the evolution loop polls,
// so the driver can write a restart checkpoint and unwind cleanly before
// exiting. a SECOND signal force-kills (the escape hatch when a checkpoint
// write hangs), because each handler re-arms the default disposition.
//
// unlike `term_guard` (a global, last-resort cursor restore that re-raises
// SIG_DFL), `SignalGuard` is RAII and run-scoped:
//   - `install()` saves the previous dispositions and traps the signals.
//   - `stop_requested()` is polled by the loop.
//   - `Drop` restores the previous dispositions (e.g. python's handlers) and
//     shows the cursor, so a caught signal NEVER leaves the terminal broken
//     and never permanently steals python's Ctrl-C.
//
// the trapped set covers interactive quits (INT/QUIT), terminations
// (TERM/HUP), and cluster pre-emption warnings (USR1/USR2 — slurm `--signal`,
// pbs, lsf), so a scheduler eviction saves state instead of losing the run.
//
// usage:
//   let guard = symbi_display::SignalGuard::install();   // before the loop
//   loop {
//       if guard.stop_requested() { write_restart_checkpoint(); break; }
//       step();
//   }
//   // Drop restores python's handlers + shows the cursor.
// =============================================================================

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

const SHOW_CURSOR: &[u8] = b"\x1b[?25h";

// alternate-screen control (btop / vim / htop style). entering:
//   ?1049h  switch to the scratch buffer
//   ?7l     disable auto-wrap (a glyph at the right margin overwrites, never
//           wraps+scrolls — otherwise a too-wide row smears the redraw)
//   ?1002h  enable mouse button/drag tracking and ?1006h SGR encoding. this is
//           what actually PINS the view: the terminal forwards wheel events to
//           the app as input instead of scrolling its buffer, so the user cannot
//           scroll the live TUI (the alternate screen alone does NOT stop the
//           wheel on macOS Terminal.app / iTerm2).
//   ?25l    hide the cursor; then clear + home.
// leaving reverses every mode and restores the primary buffer + cursor. the
// `_STR` forms drive the normal buffered path; the `_BYTES` form is the
// async-signal-safe restore written from the handler.
const ENTER_ALT_STR: &str = "\x1b[?1049h\x1b[?7l\x1b[?1002h\x1b[?1006h\x1b[?25l\x1b[2J\x1b[H";
const LEAVE_ALT_STR: &str = "\x1b[?1002l\x1b[?1006l\x1b[?7h\x1b[?25h\x1b[?1049l";
const LEAVE_ALT_BYTES: &[u8] = b"\x1b[?1002l\x1b[?1006l\x1b[?7h\x1b[?25h\x1b[?1049l";

// whether the alternate screen is currently active, so the signal handler knows
// to leave it (not merely show the cursor) on an interrupt.
static IN_ALT: AtomicBool = AtomicBool::new(false);

/// the trapped signals: interactive (INT/QUIT), termination (TERM/HUP), and the
/// cluster pre-emption warnings (USR1/USR2).
const TRAPPED: [i32; 6] = [
    libc::SIGINT,
    libc::SIGTERM,
    libc::SIGHUP,
    libc::SIGQUIT,
    libc::SIGUSR1,
    libc::SIGUSR2,
];

// process-global because a posix signal handler is a `extern "C" fn` that
// closes over nothing. a single run is active at a time (the GIL-released
// `run_simulation`), so one flag suffices.
static STOP: AtomicBool = AtomicBool::new(false);
static GOT: AtomicI32 = AtomicI32::new(0);

/// SAFETY: installed only by `install()`. the body does ONLY async-signal-safe
/// work — a `libc::write` of a fixed const buffer to fd 2, atomic integer
/// stores, and `libc::signal` to re-arm the default disposition. it never
/// allocates, never touches the rust runtime, and never re-raises (the loop
/// observes `STOP` and exits gracefully; a second signal hits SIG_DFL).
unsafe extern "C" fn handler(signum: i32) {
    unsafe {
        // restore the terminal: leave the alternate screen if we entered it
        // (the sequence also shows the cursor), else just show the cursor. this
        // runs even for the SECOND signal's SIG_DFL kill window — the buffer is
        // already restored, so a hard kill never strands the shell in alt mode.
        if IN_ALT.swap(false, Ordering::SeqCst) {
            libc::write(2, LEAVE_ALT_BYTES.as_ptr() as *const _, LEAVE_ALT_BYTES.len());
            // drop any queued mouse/scroll reports so they don't surface at the
            // shell prompt after we exit (tcflush is async-signal-safe).
            libc::tcflush(0, libc::TCIFLUSH);
        } else {
            libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
        }
        libc::signal(signum, libc::SIG_DFL);
    }
    GOT.store(signum, Ordering::SeqCst);
    STOP.store(true, Ordering::SeqCst);
}

/// run-scoped trap for the evolution loop. restores prior dispositions + cursor
/// on `Drop`.
pub struct SignalGuard {
    saved: [libc::sighandler_t; 6],
}

impl SignalGuard {
    /// trap the signal set, clearing any stale stop state. returns a guard whose
    /// `Drop` restores the previous dispositions.
    pub fn install() -> Self {
        STOP.store(false, Ordering::SeqCst);
        GOT.store(0, Ordering::SeqCst);
        let mut saved = [0 as libc::sighandler_t; 6];
        for (ii, &sig) in TRAPPED.iter().enumerate() {
            // SAFETY: `handler` is `extern "C"`, captures nothing, and is
            // async-signal-safe; `libc::signal` returns the prior disposition.
            saved[ii] = unsafe { libc::signal(sig, handler as *const () as libc::sighandler_t) };
        }
        SignalGuard { saved }
    }

    /// whether a signal has asked the run to stop (polled by the loop).
    pub fn stop_requested(&self) -> bool {
        STOP.load(Ordering::SeqCst)
    }

    /// the name of the most recently caught signal (for the log/restart message).
    pub fn signal_name(&self) -> &'static str {
        match GOT.load(Ordering::SeqCst) {
            libc::SIGINT => "SIGINT",
            libc::SIGTERM => "SIGTERM",
            libc::SIGHUP => "SIGHUP",
            libc::SIGQUIT => "SIGQUIT",
            libc::SIGUSR1 => "SIGUSR1",
            libc::SIGUSR2 => "SIGUSR2",
            _ => "signal",
        }
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        // restore the previous dispositions (e.g. python's Ctrl-C handler) and
        // make the cursor visible regardless of how the run ended.
        for (ii, &sig) in TRAPPED.iter().enumerate() {
            // SAFETY: restoring a disposition captured from `libc::signal`.
            unsafe {
                libc::signal(sig, self.saved[ii]);
            }
        }
        // SAFETY: async-signal-safe write of a const buffer to stderr.
        unsafe {
            libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
        }
    }
}

/// alternate-screen session for the live dashboard. on a tty, `enter()` switches
/// to the scratch buffer (clean full-screen TUI, nothing pollutes scrollback);
/// `leave()` / `Drop` restores the primary buffer so the TUI "goes away" like
/// btop. NON-tty (piped) output is left untouched. pair with a final static
/// render after `leave()` to persist the run's result on the primary screen.
pub struct ScreenGuard {
    active: bool,
}

impl ScreenGuard {
    /// enter the alternate screen (tty only); no-op when output is redirected.
    pub fn enter() -> Self {
        let active = crate::terminal::is_tty();
        if active {
            IN_ALT.store(true, Ordering::SeqCst);
            print!("{ENTER_ALT_STR}");
            let _ = std::io::stdout().flush();
        }
        ScreenGuard { active }
    }

    /// leave the alternate screen, restoring the primary buffer + cursor + mouse
    /// modes. idempotent — safe to call before `Drop` (e.g. to print a final
    /// frame on the primary screen). if a signal already restored the buffer the
    /// escape write is skipped, but stdin is still drained.
    pub fn leave(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if IN_ALT.swap(false, Ordering::SeqCst) {
            print!("{LEAVE_ALT_STR}");
            let _ = std::io::stdout().flush();
        }
        // drain queued mouse/scroll reports so they don't leak to the shell.
        // SAFETY: tcflush on the stdin fd; ENOTTY on a pipe is harmless.
        unsafe {
            libc::tcflush(0, libc::TCIFLUSH);
        }
    }

    /// whether the alternate screen is (still) active.
    pub fn is_active(&self) -> bool {
        IN_ALT.load(Ordering::SeqCst)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        self.leave();
    }
}
