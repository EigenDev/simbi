# symbi-aot

The kernel library generated ahead of time. During the build, `build.rs` traces
and lowers each kernel in the kernel matrix, then writes CPU Rust source and a
serialized, backend-neutral IR blob. The blob can be turned into CUDA or HIP code
at runtime without tracing the physics again.

The build needs the physics and discretization crates. At runtime, this crate
only needs `symbi-algebra` and `symbi-ir`; the generated CPU kernels work on
plain slices.

## Dependencies

At build time, this sits above `symbi-discretize`. At runtime, it sits just above
the IR.

## Start here

`build.rs`, especially the kernel matrix near the top. Add a kernel to the matrix
and the generated registry will include its function automatically.

## Notes

Generating the full library takes a few minutes. Use `cargo check -p symbi-hydro`
and targeted kernel runs for quicker feedback while working on the physics.
