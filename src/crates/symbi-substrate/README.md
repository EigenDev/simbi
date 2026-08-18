# symbi-substrate

The live kernel sets, one per regime, and the machinery that binds a simulation
state to them. It answers the question of which kernels this run needs, what
buffers and parameters they take, and in what order they fire.

`SimSubstrate` is the front door. Below it sit the per-regime kernel sets for the
isothermal, adiabatic, relativistic, and magnetized cases, along with the shared
support that none of them should duplicate, such as the CFL reduction, the ghost
filling driver, and the runtime source path.

## Where it sits

Above `symbi-sim`, whose `FieldStore` it implements kernel sets over, and above
`symbi-exec`, through which it dispatches. Below the top-level `symbi` crate, which
drives it. It names no time integrator and no refinement strategy, since both of
those depend downward on it.

## Where to start reading

`regimes/regime_substrate.rs` for the map from a regime to its kernel set, then any
one of the concrete substrates in `regimes/substrate.rs`.

## Things worth knowing before you change it

Pointwise sources ride inside the Godunov kernel rather than in a separate pass,
and that fusion is gated to be bit-exact against the unfused path. If you add a
source, decide consciously which side of that seam it belongs on.
