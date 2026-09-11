# symbi-sim

Simulation data and routines shared by the drivers. `FieldStore` and `SimState`
hold the data in structure-of-arrays form. This crate also handles Runge-Kutta
stage bookkeeping, checkpoint reading and writing, domain decomposition across
devices, radial census diagnostics, and passive tracers.

The `KernelSet` and `RegimeSubstrate` traits describe how a simulation uses its
kernels. They're defined here and implemented in higher-level crates, so both the
substrate and time integrators can use the same interface without depending on
each other. The enums that classify a run also live here.

## Dependencies

Uses algebra, geometry, grid, hydro, IO, IR, and the execution abstraction. It
doesn't depend on a concrete kernel set or executor.

## Start here

`state.rs` has the data containers, `substrate_seam.rs` defines the traits, and
`driver.rs` has stepping routines shared by the single-grid and refined drivers.

## Notes

When adding a step to either driver, check whether the other needs it too. Shared
routines belong in `driver.rs`.

Checkpoint time and logarithmic output cadence have separate anchors. A restart
resumes the existing cadence instead of starting a new one.
