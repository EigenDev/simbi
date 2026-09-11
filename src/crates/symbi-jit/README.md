# symbi-jit

This compiles a scalarized IR function (`LoweredFn`) to native CPU code with
Cranelift. The interpreter accepts the same representation.

It supports the subset of IR needed for user source expressions. Stencils,
reductions, and generic-dimension loops return `JitError::Unsupported`, letting
the caller fall back to the interpreter.

## Dependencies

Uses `symbi-ir` and `symbi-algebra`. `symbi-hydro` uses it to speed up user
expressions.

## Notes

The compiled code is kept bit-for-bit consistent with the interpreter. Arithmetic
uses plain IEEE operations. Cranelift doesn't automatically combine a multiply
and add into an FMA, so `a*b + c` follows the interpreter's separate steps.

Transcendental functions go through Rust shims calling the same `std` functions
as the interpreter. That avoids differences in the last bit from using a
platform `libm`.
