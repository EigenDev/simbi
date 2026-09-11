# symbi-ib

Bodies inside the fluid: black holes, planets, rigid bodies of arbitrary shape,
and bonded assemblies.

One part of the crate handles their motion: positions, gravity, collisions, and
bonds between fragments. The other handles their interaction with the fluid.
Shapes are described by signed distance functions, combined with
constructive-solid-geometry operations. Volume penalization couples the bodies
to the fluid based on how far a cell lies inside a body.

Accretion, drainage, and horizon excision are also handled here.

## Dependencies

Uses algebra, geometry, hydro, and the IR. The discretization traces penalization
into kernels, and the simulation crates store the bodies.

## Start here

`body.rs` defines a body, `sdf.rs` describes its geometry, and `penalize.rs`
handles the fluid coupling.

## Notes

Gravitational softening is part of the physical setup. Plummer and compact
softening give different flows near an accretor. If you're unsure which was used
in an old run, check the field implied by the flow rather than guessing from the
run date.
