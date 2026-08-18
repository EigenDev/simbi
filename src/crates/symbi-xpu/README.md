# symbi-xpu

This crate manages memory, execution, and synchronization for CPU, CUDA, and HIP.
It owns memory
lifetime and leaves layout to its callers, it orders execution through a stream
that the executor holds, and it loads kernels that were compiled elsewhere.

## Where it sits

Just above `symbi-algebra`. It generates no code itself, though it does drive
`nvrtc` and `hiprtc` when a kernel needs compiling at runtime.

## Where to start reading

`runtime.rs` for the execution model, then whichever of `cuda.rs` or `hip.rs`
matches your machine.

## Things worth knowing before you change it

On AMD hardware, managed memory needs `HSA_XNACK=1`. Without it an MI250X run
can be roughly twenty-four times slower. Check `rocminfo` when diagnosing poor AMD
performance.

Dispatch is resolved at compile time, with no `dyn` anywhere, and every fallible
operation returns `Result<T, XpuError>` rather than panicking.
