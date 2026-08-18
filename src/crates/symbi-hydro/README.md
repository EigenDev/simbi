# symbi-hydro

The physics, written once. Equations of state, the primitive and conservative
state types, the regimes (Newtonian, isothermal, relativistic, and their
magnetized counterparts), the Riemann solvers, the source terms, and the
constraints a state has to satisfy to be admissible.

Everything here is generic over a carrier `S: Scalar`. Instantiated at `f64` it is
ordinary numerical code that you can call from a test and reason about directly.
Instantiated at `Gv` the very same text records a computation graph instead, and
that graph becomes the GPU kernel. This is the arrangement that lets one written
statement of a Riemann solver serve the CPU, CUDA, and HIP paths without three
copies drifting apart.

Because of that, this crate is the one place to state a piece of physics.

## Where it sits

Above algebra, geometry, the IR, the JIT, and the expression language. Below the
discretization and everything that runs a simulation. It has no runtime dependency
on the orchestration crates.

Before changing anything here, read `WRITING_PHYSICS.md` in this directory. It is
the four things carrier-generic code costs you, and it is short.

## Where to start reading

`state.rs` and `regime.rs` for what a fluid state is, `riemann/` for the solvers,
`eos.rs` for the closures, and `source_spec.rs` together with `expr_bridge.rs` for
how a user's expression becomes a source term.

`state_law.rs` is a good short read if you want to see the carrier-generic style
doing real work. It converts primitives to conserved variables for any regime and
any supported background, and the sponge source uses it so that one configuration
wire serves a Newtonian gas, a relativistic one, and a curved spacetime.

## Things worth knowing before you change it

Control flow needs care in carrier-generic code. An ordinary `if` on a traced value
cannot work, because at trace time there is no value to branch on. Use `S::branch`,
`S::select`, or a `GvMask` instead.

The equation of state reaches the kernels as a value rather than as a type
parameter. Pinning it to a type in one place while the kernels select from a value
elsewhere is a mistake that has happened, and it corrupts initial conditions
quietly. When you change anything about which EOS is in play, check
`substrate_param()` and `gamma()` along with the obvious sites.

Stored velocity is physical for the Newtonian regimes. The contravariant `v^i` of
the Valencia formulation belongs to general relativity only.
