# symbi-sim

The hub. `FieldStore` and `SimState` hold the simulation's data in
structure-of-arrays form, and around them sit the pieces that every driver needs
regardless of how it steps in time: the stage bookkeeping for a Runge-Kutta
substep, checkpoint reading and writing, the domain decomposition for multiple
devices, the radial census that science runs accumulate, and the passive tracers.

It also carries the seam between a simulation and a substrate, meaning the
`KernelSet` and `RegimeSubstrate` traits together with the enums that classify a
run. Those traits are declared here and implemented above, which is what lets the
hub sit below both the substrate and the integrator so that they depend downward on
it rather than sideways on each other.

## Where it sits

Above the dependency floor of algebra, geometry, grid, hydro, IO, the IR, and the
execution abstraction. It names no concrete kernel set and no executor.

## Where to start reading

`state.rs` for the containers, `substrate_seam.rs` for the traits, and `driver.rs`
for the stepping primitives that both the single-grid and refined drivers share.

## Things worth knowing before you change it

The single-grid driver and the refined driver are siblings. Both consume the
primitives in `driver.rs`, and neither depends on the other. When you add a step to
one of them, the question to ask is whether the other needs it too, because the
shared primitive is usually the right home.

Checkpoint time and the logarithmic output cadence are anchored separately, so a
restart resumes the cadence rather than restarting it.
