# symbi-ib

Discrete objects living inside the fluid. Black holes, planets, rigid bodies of
arbitrary shape, and the bonded assemblies built from them. The crate splits
cleanly into two halves that are worth keeping separate in your head.

The first is kinematics. Where a body is, how it moves, how bodies attract each
other, how they collide, and how bonds hold fragments together.

The second is how a body meets the fluid. Geometry is exact, expressed as signed
distance functions that compose through the usual constructive-solid-geometry
operations, and the surface physics is volume penalization over that geometry. A
cell knows how deeply it lies inside a body, and the penalty is applied in
proportion.

Accretion, drainage, and horizon excision also live here, since each is a statement
about what a body does to the fluid that reaches it.

## Where it sits

Above algebra, geometry, hydro, and the IR. The discretization traces its
penalization into kernels, and the simulation crates carry the bodies.

## Where to start reading

`body.rs` for what a body is, `sdf.rs` for the geometry, and `penalize.rs` for the
coupling.

## Things worth knowing before you change it

The softening of a body's gravitational field is a configuration choice with a
long reach. A Plummer-softened point mass and a compact one produce visibly
different flows near the accretor, and the honest way to know which a given run
used is to measure the field from the flow itself rather than to infer it from when
the run was launched.
