// =============================================================================
// input.rs
//
// non-blocking keyboard reader for the live dashboard. relies on the terminal
// being in non-canonical mode — `ScreenGuard` disables ICANON + ECHO but LEAVES
// ISIG set — so bytes arrive per keystroke while Ctrl-C still raises SIGINT for
// the graceful-interrupt path instead of surfacing here as a key. the reader
// never enables crossterm raw mode and never changes termios; it only reads. it
// is a no-op off a tty, and never blocks beyond an explicit poll timeout.
//
// usage:
//   while let Some(key) = poll_key() {
//       match key { Key::Tab => next_tab(), Key::Char('q') => quit(), _ => {} }
//   }
// =============================================================================

/// a decoded keypress — the minimal set the dashboard navigates with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Tab,
    BackTab,
    Left,
    Right,
    Up,
    Down,
    Enter,
    Esc,
}

/// read one pending keypress without blocking. returns `None` when no input is
/// available; no-op off a tty. the sim loop uses this so it never stalls a step.
pub fn poll_key() -> Option<Key> {
    poll_key_timeout(0)
}

/// like [`poll_key`], but waits up to `timeout_ms` for input (0 = non-blocking).
/// a ui-driven loop uses a small timeout for frame pacing + prompt key response;
/// a poll interrupted by a signal (Ctrl-C) returns `None`, so the caller's
/// stop-flag check runs on the next iteration.
pub fn poll_key_timeout(timeout_ms: i32) -> Option<Key> {
    if !crate::terminal::is_tty() {
        return None;
    }
    let mut pfd = libc::pollfd {
        fd: 0,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: poll on the stdin fd for `timeout_ms`; a single valid pollfd.
    let n = unsafe { libc::poll(&mut pfd, 1 as libc::nfds_t, timeout_ms) };
    if n <= 0 || (pfd.revents & libc::POLLIN) == 0 {
        return None;
    }
    let mut buf = [0u8; 8];
    // SAFETY: read the pending keystroke bytes into a stack buffer. ICANON is off
    // so this returns immediately with whatever the terminal has queued.
    let r = unsafe { libc::read(0, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
    if r <= 0 {
        return None;
    }
    parse_key(&buf[..r as usize])
}

/// decode the leading keypress from a slice of raw terminal input. handles plain
/// printable chars, tab/enter, a bare escape, and the CSI arrow / shift-tab
/// sequences; control bytes and unrecognized escape sequences yield `None` (so a
/// stray Ctrl-C byte, were it ever delivered, is never mistaken for a key).
fn parse_key(b: &[u8]) -> Option<Key> {
    match b {
        [] => None,
        [0x09, ..] => Some(Key::Tab),
        [0x0d, ..] | [0x0a, ..] => Some(Key::Enter),
        [0x1b, b'[', b'A', ..] => Some(Key::Up),
        [0x1b, b'[', b'B', ..] => Some(Key::Down),
        [0x1b, b'[', b'C', ..] => Some(Key::Right),
        [0x1b, b'[', b'D', ..] => Some(Key::Left),
        [0x1b, b'[', b'Z', ..] => Some(Key::BackTab),
        [0x1b] => Some(Key::Esc),
        [0x1b, ..] => None,
        [c, ..] if *c >= 0x20 && *c < 0x7f => Some(Key::Char(*c as char)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_chars() {
        assert_eq!(parse_key(b" "), Some(Key::Char(' ')));
        assert_eq!(parse_key(b"q"), Some(Key::Char('q')));
        assert_eq!(parse_key(b"s"), Some(Key::Char('s')));
    }

    #[test]
    fn tab_and_enter() {
        assert_eq!(parse_key(&[0x09]), Some(Key::Tab));
        assert_eq!(parse_key(&[0x0d]), Some(Key::Enter));
        assert_eq!(parse_key(&[0x0a]), Some(Key::Enter));
    }

    #[test]
    fn csi_arrows_and_backtab() {
        assert_eq!(parse_key(&[0x1b, b'[', b'A']), Some(Key::Up));
        assert_eq!(parse_key(&[0x1b, b'[', b'B']), Some(Key::Down));
        assert_eq!(parse_key(&[0x1b, b'[', b'C']), Some(Key::Right));
        assert_eq!(parse_key(&[0x1b, b'[', b'D']), Some(Key::Left));
        assert_eq!(parse_key(&[0x1b, b'[', b'Z']), Some(Key::BackTab));
    }

    #[test]
    fn bare_escape_vs_unknown_sequence() {
        assert_eq!(parse_key(&[0x1b]), Some(Key::Esc));
        assert_eq!(parse_key(&[0x1b, b'[', b'?']), None);
    }

    #[test]
    fn control_bytes_are_not_keys() {
        // Ctrl-C (0x03) never reaches us — ISIG turns it into SIGINT — but even if
        // it did it must not decode as a printable key.
        assert_eq!(parse_key(&[0x03]), None);
        assert_eq!(parse_key(&[]), None);
    }
}
