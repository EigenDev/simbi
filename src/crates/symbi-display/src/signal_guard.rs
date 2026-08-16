// =============================================================================
// signal_guard.rs
//
// run-scoped graceful-interrupt + terminal recovery. a caught signal does
// not kill the process outright — it sets a stop flag the evolution loop polls,
// so the driver can write a restart checkpoint and unwind cleanly before
// exiting. a second signal force-kills (the escape hatch when a checkpoint
// write hangs), because each handler re-arms the default disposition.
//
// unlike `term_guard` (a global, last-resort cursor restore that re-raises
// SIG_DFL), `SignalGuard` is raii and run-scoped:
//   - `install()` saves the previous dispositions and traps the signals.
//   - `stop_requested()` is polled by the loop.
//   - `Drop` restores the previous dispositions (e.g., python's handlers) and
//     shows the cursor, so a caught signal never leaves the terminal broken
//     and never permanently steals python's Ctrl-C.
//
// the trapped set covers interactive quits (INT/quit), terminations
// (term/hup), and cluster pre-emption warnings (USR1/USR2 — slurm `--signal`,
// pbs, lsf), so a scheduler eviction saves state, preserving the run.
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
//   ?25l    hide the cursor; then clear + home.
// mouse tracking (?1002h + sgr ?1006h) pins the wheel to the app (btop-style):
// without it, macOS Terminal maps wheel scrolls to viewport scrollback and the
// live dashboard "scrolls away" until the next repaint. the resulting mouse
// reports arrive as input bytes; the key parser discards any unrecognized
// escape sequence (and the input read buffer holds a whole sgr report), so
// clicks/scrolls never alias onto a key binding. echo is disabled in termios,
// so nothing is ever printed. both leave paths (normal + the async-signal-safe
// handler restore) disable tracking, so the iTerm2 "mouse reporting left on"
// nag can fire only on an unhandleable sigkill.
// leaving reverses the modes and restores the primary buffer + cursor. the `_STR`
// forms drive the normal buffered path; the `_BYTES` form is the
// async-signal-safe restore written from the handler.
const ENTER_ALT_STR: &str = "\x1b[?1049h\x1b[?7l\x1b[?25l\x1b[?1002h\x1b[?1006h\x1b[2J\x1b[H";
const LEAVE_ALT_STR: &str = "\x1b[?1006l\x1b[?1002l\x1b[?7h\x1b[?25h\x1b[?1049l";
const LEAVE_ALT_BYTES: &[u8] = b"\x1b[?1006l\x1b[?1002l\x1b[?7h\x1b[?25h\x1b[?1049l";

// whether the alternate screen is active, so the signal handler knows
// to leave the alternate screen on an interrupt.
static IN_ALT: AtomicBool = AtomicBool::new(false);

// terminal line-discipline control. the alternate screen + mouse tracking pin
// the view, but with the tty's default echo on, every keystroke and every mouse
// /scroll report (which `?1002h` forwards as input) is echoed onto the
// live dashboard as ascii garbage. disabling echo + icanon on stdin suppresses
// that echo for both keys and mouse reports; isig is left set so Ctrl-C still
// raises sigint for the graceful-interrupt path. the original discipline is
// saved and restored on leave and from the signal handler, so neither a clean
// exit nor a hard-kill ever strands the shell with echo off.
static RAW_ACTIVE: AtomicBool = AtomicBool::new(false);
static mut SAVED_TERMIOS: std::mem::MaybeUninit<libc::termios> = std::mem::MaybeUninit::uninit();

/// disable echo + canonical input on stdin (fd 0), preserving the prior
/// discipline for restore. no-op when stdin is not a tty.
unsafe fn disable_input_echo() {
    unsafe {
        let saved = std::ptr::addr_of_mut!(SAVED_TERMIOS) as *mut libc::termios;
        if libc::tcgetattr(0, saved) != 0 {
            return; // not a terminal
        }
        RAW_ACTIVE.store(true, Ordering::SeqCst);
        // apply to a copy so the saved original is untouched for restore.
        let mut raw = std::ptr::read(saved);
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        libc::tcsetattr(0, libc::TCSANOW, &raw);
    }
}

/// restore the saved line discipline. idempotent (gated on `RAW_ACTIVE`) and
/// async-signal-safe (a `tcsetattr` from a static buffer), so the signal handler
/// can call it on a hard-kill.
unsafe fn restore_input_echo() {
    unsafe {
        if RAW_ACTIVE.swap(false, Ordering::SeqCst) {
            let saved = std::ptr::addr_of!(SAVED_TERMIOS) as *const libc::termios;
            libc::tcsetattr(0, libc::TCSANOW, saved);
        }
    }
}

/// the trapped signals: interactive (INT/quit), termination (term/hup), and the
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
// closes over nothing. a single run is active at a time (the gil-released
// `run_simulation`), so one flag suffices.
static STOP: AtomicBool = AtomicBool::new(false);
static GOT: AtomicI32 = AtomicI32::new(0);

/// whether a stop has been requested (a caught signal), readable without holding
/// the `SignalGuard` — the render thread polls this to stop drawing the moment the
/// async-signal-safe handler has restored the terminal.
pub fn stop_requested() -> bool {
    STOP.load(Ordering::SeqCst)
}

/// safety: installed only by `install()`. the body does only async-signal-safe
/// work — a `libc::write` of a fixed const buffer to fd 2, atomic integer
/// stores, and `libc::signal` to re-arm the default disposition. it never
/// allocates, never touches the rust runtime, and never re-raises (the loop
/// observes `STOP` and exits gracefully; a second signal hits SIG_DFL).
unsafe extern "C" fn handler(signum: i32) {
    unsafe {
        // restore the terminal: leave the alternate screen if it was entered
        // (the sequence also shows the cursor), else just show the cursor. this
        // runs even for the second signal's SIG_DFL kill window — the buffer is
        // already restored, so a hard kill never strands the shell in alt mode.
        if IN_ALT.swap(false, Ordering::SeqCst) {
            libc::write(
                2,
                LEAVE_ALT_BYTES.as_ptr() as *const _,
                LEAVE_ALT_BYTES.len(),
            );
            // drop any queued mouse/scroll reports so they don't surface at the
            // shell prompt after exit (tcflush is async-signal-safe).
            libc::tcflush(0, libc::TCIFLUSH);
        } else {
            libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
        }
        // re-enable echo on every exit path, so a hard-kill never strands the
        // shell with a silent (no-echo) prompt.
        restore_input_echo();
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
            // safety: `handler` is `extern "C"`, captures nothing, and is
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
        // restore the previous dispositions (e.g., python's Ctrl-C handler) and
        // make the cursor visible regardless of how the run ended.
        for (ii, &sig) in TRAPPED.iter().enumerate() {
            // safety: restoring a disposition captured from `libc::signal`.
            unsafe {
                libc::signal(sig, self.saved[ii]);
            }
        }
        // safety: async-signal-safe write of a const buffer to stderr.
        unsafe {
            libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
        }
    }
}

/// alternate-screen session for the live dashboard. on a tty, `enter()` switches
/// to the scratch buffer (clean full-screen tui, nothing pollutes scrollback);
/// `leave()` / `Drop` restores the primary buffer so the tui "goes away" like
/// btop. non-tty (piped) output is left untouched. pair with a final static
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
            // suppress echo of keystrokes (and any wheel/arrow bytes the terminal
            // forwards under alternate-scroll mode), so input never prints onto the
            // live dashboard. safety: single run at a time (gil released); no-op off a tty.
            unsafe { disable_input_echo() };
        }
        ScreenGuard { active }
    }

    /// leave the alternate screen, restoring the primary buffer + cursor + mouse
    /// modes. idempotent — safe to call before `Drop` (e.g., to print a final
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
        // restore echo, then drain queued mouse/scroll reports so they don't
        // leak to the shell. safety: stdin fd; enotty on a pipe is harmless.
        unsafe {
            restore_input_echo();
            libc::tcflush(0, libc::TCIFLUSH);
        }
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        self.leave();
    }
}
