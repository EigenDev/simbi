// =============================================================================
// terminal.rs
//
// terminal capability detection and ansi color palette.
// provides platform-independent terminal size queries and a
// 256-color scheme based on catppuccin mocha + elegant gold accents.
//
// usage:
//   let w = terminal::width();
//   print!("{}{}{}", color::header, "Title", color::reset);
// =============================================================================

// ansi escape sequences
pub mod ansi {
    pub const CLEAR_SCREEN: &str = "\x1b[H\x1b[J";
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    pub const SHOW_CURSOR: &str = "\x1b[?25h";
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
}

// 256-color palette
pub mod color {
    // structural elements
    pub const HEADER: &str = "\x1b[1;38;5;220m"; // bold gold/amber
    pub const BORDER: &str = "\x1b[38;5;67m"; // soft blue-gray
    pub const DATA: &str = "\x1b[38;5;252m"; // light gray
    pub const TITLE: &str = "\x1b[1;38;5;183m"; // bold lavender

    // message types
    pub const INFO: &str = "\x1b[38;5;117m"; // sky blue
    pub const SUCCESS: &str = "\x1b[38;5;158m"; // mint green
    pub const WARNING: &str = "\x1b[38;5;215m"; // warm amber
    pub const ERROR: &str = "\x1b[38;5;210m"; // soft red
    pub const DIAGNOSTIC: &str = "\x1b[38;5;141m"; // medium purple

    // progress bar
    pub const PROGRESS_FILLED: &str = "\x1b[38;5;183m"; // lavender
    pub const PROGRESS_MID: &str = "\x1b[38;5;147m"; // light lavender
    pub const PROGRESS_EMPTY: &str = "\x1b[38;5;240m"; // dark gray

    pub const RESET: &str = "\x1b[0m";
}

/// query terminal width (columns). fallback: 80.
pub fn width() -> usize {
    terminal_size().0
}

/// query terminal height (rows). fallback: 24.
pub fn height() -> usize {
    terminal_size().1
}

/// check if the locale environment suggests utf-8 support.
pub fn supports_unicode() -> bool {
    for var in &["LANG", "LC_ALL", "LC_CTYPE"] {
        if let Ok(val) = std::env::var(var) {
            if val.contains("UTF-8") || val.contains("utf-8") {
                return true;
            }
        }
    }
    false
}

/// check if stdout is connected to a terminal. when output is piped to a file or
/// pipe this is false, so the dynamic clear-screen / cursor-hide escapes must be
/// suppressed (otherwise they get embedded in the redirected stream).
pub fn is_tty() -> bool {
    #[cfg(unix)]
    {
        // safety: isatty on the stdout fd; returns 1 for a terminal, 0 otherwise.
        unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// check if term suggests 256-color support.
pub fn supports_256_color() -> bool {
    if let Ok(term) = std::env::var("TERM") {
        return term.contains("256")
            || term.contains("xterm")
            || term.contains("screen")
            || term.contains("tmux");
    }
    false
}

/// adaptive padding based on terminal width.
/// wide (>120): 4, normal (>=80): 3, narrow (<80): 2.
pub fn padding_for_width(ww: usize) -> usize {
    if ww > 120 {
        4
    } else if ww >= 80 {
        3
    } else {
        2
    }
}

fn terminal_size() -> (usize, usize) {
    #[cfg(unix)]
    {
        // ioctl tiocgwinsz
        #[cfg(target_os = "macos")]
        const TIOCGWINSZ: libc::c_ulong = 0x40087468;
        #[cfg(target_os = "linux")]
        const TIOCGWINSZ: libc::c_ulong = 0x5413;

        #[repr(C)]
        struct Winsize {
            ws_row: libc::c_ushort,
            ws_col: libc::c_ushort,
            ws_xpixel: libc::c_ushort,
            ws_ypixel: libc::c_ushort,
        }

        let mut ws = Winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };

        // safety: standard posix ioctl on stdout fd. Winsize is repr(C)
        // and zeroed, so the kernel writes valid data or returns -1.
        let ret = unsafe { libc::ioctl(libc::STDOUT_FILENO, TIOCGWINSZ, &mut ws) };
        if ret == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            return (ws.ws_col as usize, ws.ws_row as usize);
        }
    }

    (80, 24) // safe fallback
}

// private: raw libc bindings (zero deps — no libc crate)
#[cfg(unix)]
mod libc {
    #[allow(non_camel_case_types)]
    pub type c_int = i32;
    #[allow(non_camel_case_types)]
    pub type c_ulong = u64;
    #[allow(non_camel_case_types)]
    pub type c_ushort = u16;

    pub const STDOUT_FILENO: c_int = 1;

    unsafe extern "C" {
        pub unsafe fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        pub unsafe fn isatty(fd: c_int) -> c_int;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_positive() {
        assert!(width() > 0);
    }

    #[test]
    fn height_is_positive() {
        assert!(height() > 0);
    }

    #[test]
    fn padding_wide() {
        assert_eq!(padding_for_width(200), 4);
        assert_eq!(padding_for_width(121), 4);
    }

    #[test]
    fn padding_normal() {
        assert_eq!(padding_for_width(120), 3);
        assert_eq!(padding_for_width(80), 3);
    }

    #[test]
    fn padding_narrow() {
        assert_eq!(padding_for_width(79), 2);
        assert_eq!(padding_for_width(40), 2);
    }

    #[test]
    fn supports_unicode_parses_env() {
        // just verify it doesn't panic — actual result depends on env
        let _ = supports_unicode();
    }

    #[test]
    fn supports_256_color_parses_env() {
        let _ = supports_256_color();
    }
}
