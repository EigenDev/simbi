# symbi-exec

This crate handles kernel dispatch. Given a kernel name resolved through the ahead-of-time
registry, a set of field buffers over a domain, the packed integer and scalar
tails, and an execution policy, it launches the work. It also holds the CPU
parallelism policy, including the cache-blocking traversal.

It does not contain physics. No regime name and no
simulation state crosses this boundary, which is what allows it to depend only on
the lower crates and keeps the layering free of a cycle back into the orchestration.

## Where it sits

Above the ahead-of-time kernels, the grid, the IR, and the execution abstraction.
Below the substrate that calls it.

## Where to start reading

`policy.rs` for the CPU side, `engine.rs` for the neutral dispatch.
