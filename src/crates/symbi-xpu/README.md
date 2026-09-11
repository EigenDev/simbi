# symbi-xpu

Memory, execution, and synchronization for CPU, CUDA, and HIP. This manages
memory lifetimes while callers choose the layout. The executor holds a stream
that orders the work, and this crate loads the compiled kernels.

## Dependencies

Uses `symbi-algebra`. Code generation happens elsewhere, but this crate calls
`nvrtc` and `hiprtc` when kernels need to be compiled at runtime.

## Start here

`runtime.rs` describes execution. Then look at `cuda.rs` or `hip.rs`, depending
on the hardware you're using.

## Notes

AMD managed memory needs `HSA_XNACK=1`. An MI250X run can be roughly 24 times
slower without it, so check `rocminfo` when investigating poor AMD performance.

Dispatch is resolved at compile time without `dyn`. Fallible operations return
`Result<T, XpuError>` instead of panicking.
