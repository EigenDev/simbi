# symbi-exec

This launches kernels from the ahead-of-time registry. It takes a kernel name,
field buffers over a domain, packed integer and scalar arguments, and an
execution policy. The CPU parallelism policy, including cache-blocking traversal,
also lives here.

Dispatch doesn't need to know the fluid regime or hold simulation state, so this
crate can depend on the lower-level crates without a dependency cycle.

## Dependencies

Uses the ahead-of-time kernels, grid, IR, and execution abstraction.
`symbi-substrate` calls it to launch work.

## Start here

`policy.rs` handles CPU parallelism, and `engine.rs` handles dispatch.
