# symbi-jit

This crate compiles a scalarized IR function (`LoweredFn`) to native CPU machine
code with Cranelift. The interpreter accepts the same representation.

It accepts a deliberately small subset, which is exactly what a user's source
expression compiles to. Anything outside that subset, such as a stencil, a
reduction, or a generic-dimension loop, is refused with `JitError::Unsupported` so
the caller falls back to the interpreter. Unsupported input is returned as an
error rather than compiled with different semantics.

## Where it sits

Above `symbi-ir` and `symbi-algebra`. `symbi-hydro` uses it to make user
expressions fast.

## Things worth knowing before you change it

The compiled code agrees with the interpreter bit for bit, and that property is
maintained on purpose. Arithmetic stays as plain IEEE operations, because Cranelift
does not contract a multiply and an add into an FMA on its own, so `a*b + c`
matches the interpreter's separate steps. Every transcendental call is routed
through a Rust shim wrapping the same `std` function the interpreter calls, which
avoids depending on a platform `libm` whose last bit might differ.
