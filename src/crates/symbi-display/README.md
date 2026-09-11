# symbi-display

The terminal display for a running simulation: tables, colors, progress bars, and
scrolling messages. Interactive runs use ratatui. Headless runs and redirected
output use a plain text renderer to keep the logs readable.

## Dependencies

Reads the output schema through `symbi-io`. The physics crates don't depend on
this one.

## Start here

`table.rs` has the layout, and `live.rs` draws the interactive display.
