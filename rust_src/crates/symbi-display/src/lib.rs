// =============================================================================
// lib.rs
//
// terminal display crate for live simulation monitoring.
// provides box-drawn tables, 256-color palette, gradient progress bars,
// and a scrolling message board. zero external dependencies.
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

pub mod terminal;
pub mod renderer;
pub mod table;
pub mod meta_table;
pub mod term_guard;

pub use renderer::{Alignment, Renderer, align_text, truncate};
pub use table::{MessageType, Table};
pub use meta_table::{render_metadata, render_tree_buf};
