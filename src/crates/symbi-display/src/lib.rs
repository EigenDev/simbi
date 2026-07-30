// =============================================================================
// lib.rs
//
// terminal display crate for live simulation monitoring.
// provides box-drawn tables, 256-color palette, gradient progress bars,
// and a scrolling message board. the live (tty) frame is drawn with ratatui;
// the static/headless frame uses the bundled string renderer.
//
// usage:
//   use symbi_display::{Table, MessageType};
//   let mut table = Table::new("Simulation", true);
//   table.set_header(&["Iteration", "Time", "dt"]);
//   table.update_row(&["100", "1.2e-3", "5.4e-6"]);
//   table.set_progress(45);
//   table.post_info("Checkpoint saved");
//   table.refresh();
// =============================================================================

pub mod attach;
pub mod exit;
pub mod hostinfo;
pub mod input;
pub mod live;
pub mod livethread;
pub mod meta_table;
pub mod renderer;
pub mod signal_guard;
pub mod snapshot;
pub mod table;
pub mod term_guard;
pub mod terminal;

pub use attach::run_attach;
pub use exit::ExitKind;
pub use hostinfo::HostStats;
pub use input::{Key, poll_key, poll_key_timeout};
pub use live::{Colormap, DiagnosticView, FieldSlice};
pub use livethread::{Controls, LiveDashboard};
pub use meta_table::{render_metadata, render_tree_buf};
pub use renderer::{Alignment, Renderer, align_text, truncate};
pub use signal_guard::{ScreenGuard, SignalGuard};
pub use snapshot::{Snapshot, snapshot_path};
pub use table::{MessageType, Table};
