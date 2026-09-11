# symbi-discretize

This turns the physics in `symbi-hydro` into stencil graphs. It evaluates the
generic physics code at `S = Gv`, recording the operations for `symbi-ir` to
lower and turn into code.

    symbi-hydro       physics, generic over S: Scalar
         |
         |            evaluate at S = Gv in symbi-discretize
         v
    symbi-ir          stencil graph
         |
         |            lower and generate code
         v
    CPU Rust, CUDA, HIP

All production kernels are traced here: conserved-to-primitive inversion, face
fluxes, wave speeds, Godunov updates, ghost filling, constrained-transport curls,
refinement transfer, viscous terms, and immersed-boundary penalization.

## Dependencies

Uses hydro, geometry, immersed bodies, and the IR. `symbi-aot` builds the kernel
library from these graphs.

## Start here

`gv/flux.rs` is a useful example because a face flux brings together
reconstruction, a Riemann solve, and geometry. `coords.rs` passes the chart and
spacing into the trace. `kernel_slug.rs` builds a kernel's name from its
configuration.

## Notes

Kernel names encode their configuration. If you add a configuration option that
changes the name, update both generation and runtime dispatch. Otherwise, a run
can ask for a kernel that wasn't built. CI checks kernel coverage.

Grid spacing is a runtime property. The runtime `map_kind_d` determines the map;
the build-time `spacing` doesn't select it.

For refactors that should preserve behavior, compare generated kernels before and
after as well as running the tests.
