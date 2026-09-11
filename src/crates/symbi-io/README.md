# symbi-io

Simulation output from a shared schema. A `Tree` describes the data, and the
HDF5 writer, JSON backend, and terminal table renderer all walk that tree.
HDF5 is used for checkpoints; JSON lets you inspect the schema.

Field names come from the regime specification in one place, so readers and
writers use the same on-disk names.

## Dependencies

Uses algebra, grid, and hydro. The display crate and afterglow adapter read
through it.

## Start here

`tree.rs` defines the schema, `field_layout.rs` handles field names, and
`hdf5.rs` handles checkpoint output.
