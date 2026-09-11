# symbi-grid

Field storage and access. A `Field` owns a `symbi-xpu` memory block and uses a
domain from `symbi-algebra`. Most reads and writes use views; coordinate-indexed
access is also available for host-side probes.

`Centering` records whether a field lives at cell centers, on faces, or on edges.
This is especially useful for keeping track of fields in constrained transport.

## Dependencies

Uses `symbi-algebra` and `symbi-xpu`. Generated substrate kernels work directly
on this storage.

## Start here

`field.rs` defines the fields, and `ghost.rs` describes the halo regions.

## Notes

Primitive storage includes the halo, but owned index bounds are relative to the
interior. Account for that offset when slicing. Missing it shifts the grid by the
halo width and can look like a physical asymmetry.
