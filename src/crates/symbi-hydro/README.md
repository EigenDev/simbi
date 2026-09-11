# symbi-hydro

The fluid physics: equations of state, primitive and conservative states,
Riemann solvers, source terms, and checks for physically admissible states.
Regimes include Newtonian, isothermal, relativistic, and magnetized fluids.

The physics is generic over `S: Scalar`. With `S = f64`, you can evaluate it
numerically and call it directly from a test. With `S = Gv`, the same operations
record a computation graph that becomes a kernel. This lets the CPU, CUDA, and
HIP implementations share the equations.

## Dependencies

Uses algebra, geometry, IR, JIT, and the expression language. The discretization
and simulation crates build on it; it has no runtime dependency on the simulation
drivers.

Before editing the physics, read [WRITING_PHYSICS.md](WRITING_PHYSICS.md). It's a
short guide to writing code that works with each scalar type.

## Start here

`state.rs` and `regime.rs` define fluid states and regimes. `riemann/` has the
solvers, `eos.rs` has the closures, and `source_spec.rs` with `expr_bridge.rs`
connects user expressions to source terms.

`state_law.rs` is a small example of the generic style. It converts primitives
to conserved variables across regimes and backgrounds. The sponge source uses it
to share one configuration format across Newtonian, relativistic, and curved
spacetime calculations.

## Notes

An ordinary `if` can't branch on a traced value because that value isn't known
when the graph is recorded. Use `S::branch`, `S::select`, or a `GvMask`.

The kernels receive the equation of state as a value. If another part of the code
fixes it through a type parameter, the two choices can disagree and corrupt the
initial conditions. When changing EOS selection, check `substrate_param()` and
`gamma()` too.

Stored velocity is physical in Newtonian regimes. The Valencia formulation's
contravariant `v^i` is for general relativity only.
