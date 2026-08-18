# symbi-ir

The computation graph, and the compiler that turns it into something a machine can
run. Physics enters as a traced graph, passes rewrite that graph, and backends
render it to CPU Rust, to CUDA or HIP source, or to a serialized blob that can be
rendered later.

This is the crate where you are reading a compiler rather than reading physics, and
it is worth knowing that when you open it. Someone adding a source term or a
Riemann solver should be able to work for a long time without coming in here.

## Where it sits

Above `symbi-algebra` and `symbi-abi`, and below everything that generates a
kernel. It shells no toolchain of its own. Driving `nvcc` and `hiprtc` belongs to
`symbi-xpu`.

## Where to start reading

`gv.rs` first, because `Gv` is the whole front end. It is a `Scalar` implementation
whose arithmetic operators record nodes instead of computing numbers, so running
ordinary generic physics code at `S = Gv` leaves a graph behind. Then `graph.rs`
for the data structure, `passes/scalarize.rs` for the lowering, and
`backends/` for the emitters.

## Things worth knowing before you change it

Two distinctions cause most of the confusion here, and neither is visible in the
types yet.

The first is trace time against lowering time. A kernel's parameter list is
collected while the trace runs, so it records everything the builder touched. Dead
code elimination happens later, during lowering. The parameter list is therefore a
superset of the parameters the lowered code actually reads, and code that assumes
the two agree will be right until the day it is not.

The second is that there are two lowering entry points with different behavior.
`scalarize` takes a single output and lowers every node in the graph.
`scalarize_kernel` takes several outputs and prunes what none of them reach. Both
are correct for their callers, and a graph handed to the wrong one produces a
parameter list of the wrong length rather than a wrong answer, which at least fails
loudly.

There is also the ambient trace. `with_trace` and `in_isolated_trace` mean a
builder can behave differently depending on whether something above it has already
opened a trace, and the fused source path does exactly that. This is the standing
cost of a tracing front end, and `in_isolated_trace` is the sanctioned way to
re-enter.
