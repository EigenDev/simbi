// =============================================================================
// attach.rs
//
// read-only `simbi attach` client. a headless solver writes a Snapshot to
// `<rundir>/.simbi-live/snapshot.bin` each cadence (see snapshot.rs); this polls
// that file's mtime and, on change, republishes it to a LiveDashboard render
// thread — so a running batch/cluster job is monitored from a login node or
// laptop over a shared filesystem, no sockets or ports.
//
// the transport is strictly one-way: the client never writes back. tab / field /
// colormap are all render-thread-local (the field bundle travels in the
// snapshot), so panels and the f/c keys respond instantly; pause/step/checkpoint
// do not apply to a remote run and are absent.
//
// the same terminal seam as the in-process dashboard: SignalGuard owns Ctrl-C,
// ScreenGuard owns the alt screen + termios, the render thread owns the terminal.
//
// usage:
//   attach::run_attach(Path::new("data/kh_config/"), 250)?;
// =============================================================================

use std::io;
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime};

use crate::livethread::LiveDashboard;
use crate::signal_guard::{ScreenGuard, SignalGuard};
use crate::snapshot::{Snapshot, snapshot_path};
use crate::{signal_guard, terminal};

/// poll `<rundir>/.simbi-live/snapshot.bin` and render it live until the user
/// quits (`q`) or Ctrl-C. `poll_ms` is the filesystem poll interval; snapshots
/// arrive at the solver's cadence, so a few hundred ms is ample and keeps shared-
/// filesystem metadata traffic low.
pub fn run_attach(rundir: &Path, poll_ms: u64) -> io::Result<()> {
    if !terminal::is_tty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "attach needs an interactive terminal",
        ));
    }
    let path = snapshot_path(rundir);
    let poll = Duration::from_millis(poll_ms.max(1));

    // own Ctrl-C before the wait loop so it aborts cleanly while we watch for the
    // first snapshot (the run may not have reached its first cadence yet).
    let _sig = SignalGuard::install();
    eprintln!("attach: waiting for {} …", path.display());
    while !path.exists() {
        if signal_guard::stop_requested() {
            return Ok(());
        }
        thread::sleep(poll);
    }

    // the file exists: take over the screen and stream it.
    let mut screen = ScreenGuard::enter();
    let mut dash = match LiveDashboard::spawn() {
        Some(d) => d,
        None => {
            screen.leave();
            return Ok(());
        }
    };

    let mut last_mtime: Option<SystemTime> = None;
    loop {
        if signal_guard::stop_requested() || dash.controls().quit() {
            break;
        }
        // republish only when the snapshot's mtime changes; a torn read (writer
        // mid-rename) surfaces as an error and is retried on the next tick.
        if let Ok(mtime) = std::fs::metadata(&path).and_then(|m| m.modified()) {
            if last_mtime != Some(mtime) {
                if let Ok(snap) = Snapshot::read(&path) {
                    last_mtime = Some(mtime);
                    dash.publish_bundle(snap.view, snap.fields);
                }
            }
        }
        thread::sleep(poll);
    }

    dash.shutdown();
    screen.leave();
    Ok(())
}
