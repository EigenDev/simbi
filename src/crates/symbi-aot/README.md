# symbi-aot

The kernel library, baked ahead of time. During its build, `build.rs` runs the
substrate, lowers every kernel in the matrix to IR, and writes out two things for
each one. The first is compilable Rust source for the CPU. The second is a
serialized backend-neutral blob that can be rendered to CUDA or HIP later, at
runtime, without the tracing machinery being present.

That second artifact explains the shape of this crate's dependencies, which look
strange at first glance. At build time it needs the physics and the discretization.
At runtime it needs only `symbi-algebra` and `symbi-ir`. The generated kernels are
self-contained Rust over plain slices, so nothing downstream carries the compiler
around with it.

## Where it sits

Its build sits above the discretization. Its runtime sits just above the IR.

## Where to start reading

`build.rs`, and specifically the kernel matrix near the top. Adding a kernel means
adding an entry there and nothing else, because the registry module that
`include!`s every generated function is itself generated.

## Things worth knowing before you change it

The full bake takes a few minutes. During development, use
`cargo check -p symbi-hydro` and targeted kernel runs for quicker feedback.
