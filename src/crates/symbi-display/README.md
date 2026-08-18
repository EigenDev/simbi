# symbi-display

The terminal front end for a running simulation. Box-drawn tables, a color
palette, progress bars, and a scrolling message board. A live terminal is drawn
with ratatui, and a headless or redirected run falls back to a plain string
renderer so that log files stay readable.

## Where it sits

Above `symbi-io`, which it reads the schema through. Nothing in the physics
depends on it.

## Where to start reading

`table.rs` for the layout, `live.rs` for the interactive frame.
