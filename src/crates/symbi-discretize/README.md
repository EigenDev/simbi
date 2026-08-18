# symbi-discretize

The bridge between physics and code generation. It runs the carrier-generic
physics of `symbi-hydro` at `S = Gv` and collects the resulting stencil graph,
which `symbi-ir` then lowers and a backend renders.

The main data flow is:

    symbi-hydro       what the physics is        (generic over S: Scalar)
         |
         |            evaluate it at S = Gv
         v
    symbi-ir          what the kernel computes   (a stencil graph)
         |
         |            lower and emit
         v
    CPU Rust, CUDA, HIP

Every production kernel comes through here. Conserved-to-primitive inversion, face
fluxes, wave speeds, the Godunov update, ghost filling, the constrained-transport
curl, refinement transfer, viscous terms, and the immersed-boundary penalization.

## Where it sits

Above hydro, geometry, the immersed bodies, and the IR. Below the ahead-of-time
kernel library that bakes what it produces.

## Where to start reading

`gv/flux.rs` for a representative builder, since a face flux exercises
reconstruction, the Riemann solve, and the geometry all at once. Then `coords.rs`
for how a chart and a spacing reach the trace, and `kernel_slug.rs` for how a
kernel's name is assembled from its configuration.

## Things worth knowing before you change it

A kernel's name encodes its configuration, and the dispatch side reconstructs that
name at runtime. When you add an axis of variation, the baking side and the
dispatch side have to learn about it together, or a run ends with a panic about an
unbaked kernel. There is a coverage gate in CI for exactly this.

Grid spacing is a runtime property rather than a baked one. A recurring source of
confusion is treating the bake-time `spacing` as though it decided the runtime map,
when the runtime `map_kind_d` is what actually governs.

For behavior-preserving refactors, compare the emitted kernels before and after in
addition to running the tests.
