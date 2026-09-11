# symbi-substrate

Kernel sets for each fluid regime. This connects simulation state to the kernels,
provides their buffers and parameters, and arranges the calls.

`SimSubstrate` is the main interface. The isothermal, adiabatic, relativistic,
and magnetized regimes have their own kernel sets. They share routines for CFL
reduction, ghost filling, and runtime sources.

## Dependencies

Uses `symbi-sim` for `FieldStore` and `symbi-exec` to launch kernels. The
top-level `symbi` crate drives it. Time integration and refinement are handled
by the crates above it.

## Start here

`regimes/regime_substrate.rs` maps regimes to kernel sets.
`regimes/substrate.rs` contains the concrete implementations.

## Notes

Pointwise sources run inside the Godunov kernel. Tests check that this gives
bit-exact results against running them separately. When adding a source, check
whether it belongs in that kernel or needs a separate pass.
