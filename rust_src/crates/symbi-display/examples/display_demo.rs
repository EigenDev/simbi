// =============================================================================
// display_demo.rs
//
// exercises the full symbi-display api: headers, row updates,
// progress bar, and all four message types.
//
// usage:
//   cargo run -p symbi-display --example display_demo
// =============================================================================

use symbi_display::Table;

fn main() {
    let mut table = Table::new("Simulation Dashboard", false);

    table.set_header(&["Iteration", "Time", "dt", "CFL", "Max |v|"]);
    table.update_row(&["5000", "1.234e-02", "2.47e-06", "0.40", "1.83e+03"]);
    table.set_progress(42);

    table.post_info("Checkpoint written: chkpt_5000.h5");
    table.post_success("AMR regrid complete: 12 blocks -> 18 blocks");
    table.post_warning("CFL dropped below 0.1 on level 3");
    table.post_error("Negative pressure detected at (128, 64, 32)");

    table.refresh();
}
