# symbi-grid

Field storage. A `Field` owns memory through a `symbi-xpu` memory block and is
bound to a domain from `symbi-algebra`. Reads and writes go through views, or
through coordinate-indexed access for the occasional host-side probe.

`Centering` records whether a field lives at cell centers, on faces, or on edges,
which matters a great deal once constrained transport enters the picture.

## Where it sits

Above `symbi-algebra` and `symbi-xpu`. The generated substrate kernels operate on
this storage directly.

## Where to start reading

`field.rs`, then `ghost.rs` for the halo regions.

## Things worth knowing before you change it

Primitives are stored with their halo included, while the owned index range is
interior-relative. A reader that slices with the owned bounds without accounting
for the halo gets a lattice displaced by the halo width, which looks like a
physical asymmetry rather than an indexing error.
