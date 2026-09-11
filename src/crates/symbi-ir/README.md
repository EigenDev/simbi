# symbi-ir

Computation graphs and the compiler that turns them into kernels. Physics arrives
as a traced graph. Compiler passes rewrite it, and backends generate CPU Rust,
CUDA or HIP source, or a serialized blob for later code generation.

Start here if you're working on the compiler. For a new source term or Riemann
solver, `symbi-hydro` is usually the place to look.

## Dependencies

Uses `symbi-algebra` and `symbi-abi`. Kernel generation builds on this crate.
External compiler calls are handled by `symbi-xpu`.

## Start here

`gv.rs` defines `Gv`, a `Scalar` implementation whose arithmetic records graph
nodes. Running the generic physics with `S = Gv` builds the graph.

Then try `graph.rs` for the data structure, `passes/scalarize.rs` for lowering,
and `backends/` for code generation.

## Notes

A kernel's parameter list is collected during tracing and includes everything the
builder touched. Dead code elimination happens later, during lowering, so the
list can include parameters the generated code never reads. Don't assume those
sets are identical.

There are two lowering entry points. `scalarize` takes one output and lowers
every node. `scalarize_kernel` takes several outputs and prunes nodes unreachable
from them. Using the wrong entry point can produce a parameter list of the wrong
length.

Check whether a trace is already active when working with builders. `with_trace`
and `in_isolated_trace` can behave differently in that case, as in the fused
source path. Use `in_isolated_trace` when starting a separate trace inside an
existing one.
