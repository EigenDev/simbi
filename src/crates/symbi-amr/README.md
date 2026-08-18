# symbi-amr

Static mesh refinement. A hierarchy of levels, each with its own simulation state
and kernel set, plus the transfer operators that move data between them and the
registers that keep the coarse-fine interface conservative.

The transfer operators are where the care goes. Restriction has to conserve, and
prolongation has to avoid manufacturing structure that was never there. The
coarse-fine flux and EMF registers exist so that a face shared between two levels
carries one flux rather than two slightly different ones.

## Where it sits

Above `symbi-sim`, whose driver primitives it reuses, and above `symbi-substrate`,
whose kernel sets it dispatches per level. The `symbi` crate drives it. It is a
sibling of the single-grid driver rather than a layer above it.

## Where to start reading

`refinement.rs`, which holds `SmrHierarchy`.

## Things worth knowing before you change it

The coarse-fine ghost transfer is balance-aware. On a stratified background, moving
a raw state across the interface leaves an entropy signature at the seam, so what
crosses is the departure from the local hydrostatic isentrope instead. The
equivalent device kernels are baked and are bit-identical to the host path.

Seeding a fine level that spans a decomposition cut needs the conserved exchange
and the decomposed seeding path, and it has to happen before the level is primed.
