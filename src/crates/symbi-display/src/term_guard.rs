// =============================================================================
// term_guard.rs
//
// terminal-state recovery on abnormal exit. the live `Table` widget hides the
// cursor on startup and restores it via `Drop`, which is sufficient for normal
// exits and unwinding panics. but Drop does not run on:
//   - sigint  (Ctrl-C)        — kernel kills the process before unwind
//   - sigterm (kill / ide stop) — same
//   - sighup  (terminal closed) — same
//   - panic = abort           — unwinding is skipped entirely
//   - abort() via sigsegv/sigbus — same
//
// `install()` registers a panic hook + signal handlers that each write the
// SHOW_CURSOR escape sequence to stderr (fd 2) via an async-signal-safe path.
// signal handlers chain to SIG_DFL after running so the second Ctrl-C kills
// the process the same way it always would. idempotent: calling `install()`
// more than once is a no-op after the first.
//
// usage:
//   symbi_display::term_guard::install();      // once at program startup
//   let mut t = symbi_display::Table::new(...); // hides the cursor
//   // ... cursor restored on Drop, panic, Ctrl-C, etc.
// =============================================================================

use std::sync::OnceLock;

const SHOW_CURSOR: &[u8] = b"\x1b[?25h";

static INSTALLED: OnceLock<()> = OnceLock::new();

/// install the panic-hook + signal-handlers that restore the terminal cursor
/// on any abnormal exit. idempotent — safe to call multiple times.
pub fn install() {
    INSTALLED.get_or_init(|| {
        install_panic_hook();
        // safety: each handler only writes a fixed const buffer to fd 2 via
        // `libc::write` (which is async-signal-safe per posix) and then
        // restores the default disposition for the signal.
        unsafe {
            install_signal(libc::SIGINT);
            install_signal(libc::SIGTERM);
            install_signal(libc::SIGHUP);
        }
    });
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // restore cursor before the previous hook prints, so the panic
        // banner doesn't render onto an invisible-cursor line.
        let _ = write_show_cursor();
        prev(info);
    }));
}

/// async-signal-safe write of SHOW_CURSOR to stderr. uses libc::write
/// directly (never println!, never any allocator) so it's safe from a
/// signal handler. ignores partial writes / eintr — best-effort recovery.
fn write_show_cursor() -> std::io::Result<()> {
    // safety: SHOW_CURSOR is a static byte slice with stable ptr+len; fd 2
    // is the process's stderr; libc::write with these args has no ub
    // potential beyond a returned error code.
    unsafe {
        libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
    }
    Ok(())
}

/// safety: caller must invoke from `install()` only — the handler closes
/// over no captured state and re-arms SIG_DFL before re-raising, so a
/// second signal terminates the process via the default action.
unsafe extern "C" fn handler(signum: i32) {
    // safety: write to fd 2 is async-signal-safe.
    unsafe {
        libc::write(2, SHOW_CURSOR.as_ptr() as *const _, SHOW_CURSOR.len());
        libc::signal(signum, libc::SIG_DFL);
        libc::raise(signum);
    }
}

/// safety: must run before any thread spawn that could be killed mid-write.
/// `libc::signal` is portable across posix platforms (and works on Linux).
unsafe fn install_signal(signum: i32) {
    // safety: handler has C ABI, no captured state, only does
    // async-signal-safe work.
    unsafe {
        libc::signal(signum, handler as *const () as libc::sighandler_t);
    }
}
