# symbi-io

Serialization driven by a schema. One `Tree` describes the output, and every
channel walks that same tree: the HDF5 backend that writes production checkpoints,
the JSON backend that exposes the schema for introspection, and the table renderer
that the terminal display reads through.

The reason for the arrangement is that a writer and a reader which each spell the
field names themselves will eventually disagree. Here the on-disk naming derives
from the regime specification, in one place, and both directions consult it.

## Where it sits

Above algebra, grid, and hydro. The display crate and the afterglow adapter read
through it.

## Where to start reading

`tree.rs` for the schema, `field_layout.rs` for the naming, `hdf5.rs` for the
production path.
