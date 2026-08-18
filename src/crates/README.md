# The symbi workspace

The workspace contains twenty-two crates organized into a few layers. This page
summarizes their responsibilities and dependencies.

## Where to begin

Physics work, such as adding a source term, trying a different Riemann solver,
changing a boundary, or setting up a new problem, usually belongs in
`symbi-hydro`, `symbi-geometry`, `symbi-ib`, or the Python frontend. Most of this
work does not require opening the compiler crates.

Compiler and code-generation work lives in `symbi-ir`, `symbi-discretize`, and
`symbi-aot`. These crates turn the physics definitions into executable CPU and GPU
kernels. Most physics changes do not require changes in this layer.

## Background

A limited amount of graph theory is used: directed acyclic graphs, topological
order, and reachability from a set of outputs. Register allocation in `symbi-expr`
also relies on graph scheduling.
The quantity under pressure there is how many values are simultaneously live, and
that crate once refused any expression past about 252 nodes because a depth-first
schedule kept values alive long after their last use.

The main organizing idea is to write an expression once and evaluate it in more
than one algebra. The same Riemann solver code runs at
`S = f64` and produces numbers, at `S = Gv` and produces a computation graph, and
at `S = Dual` and carries derivatives alongside values. A graph appears because one
of those algebras has no behavior except to remember what it was asked to do, so
the graph is a consequence of the arrangement rather than the goal of it.

Automatic differentiation through dual numbers is exactly this pattern. So is
carrying units through a calculation, where the algebra tracks dimensions alongside
magnitudes and declines a sum that makes no sense. Formal perturbation theory in
powers of a small parameter has the same shape. Anyone who has the instinct from
one of those has the instinct that matters here.

The remaining concepts use standard compiler vocabulary. Lowering
means rewriting something into a simpler form on the way to machine code. Common
subexpression elimination is a question about when two expressions are the same
thing. Dead code elimination is reachability from the outputs. A graph and a
schedule are separate objects with separate costs, which is why the order in which
nodes are visited can matter as much as how many of them there are.

## Data flow

The physics is
written once over `S: Scalar`, the discretization evaluates it at `S = Gv` to
obtain a stencil graph, and the IR lowers that graph and renders it for whichever
backend is in play. The same Riemann solver definition serves the CPU, CUDA, and
HIP paths.

    symbi-hydro        the physics, generic over S: Scalar
         |
         |             evaluate at S = Gv
         v
    symbi-discretize   trace it into a stencil graph
         |
         v
    symbi-ir           lower the graph, rewrite it, render it
         |
         v
    symbi-aot          bake CPU Rust and a neutral IR blob at build time
         |
         v
    symbi-exec         launch it
         |
         v
    symbi-substrate    the live kernel sets, one per regime

The same IR carries user expressions from a configuration file, which is why it has
to exist at all. A source term written in Python cannot be compiled ahead of time,
so something has to lower it at runtime, and once that machinery is present it may
as well serve the baked kernels too.

## The crates, from the bottom up

**Foundations, depending on little or nothing**

| Crate | What it holds |
| --- | --- |
| `symbi-algebra` | Tensors, domains, memory layout. No dependencies at all. |
| `symbi-abi` | The names a trace and a dispatch must agree on. |
| `symbi-expr` | The user expression language and its register machine. |
| `symbi-xpu` | Where data lives and how work runs, on CPU, CUDA, and HIP. |
| `symbi-ir` | The computation graph, its passes, and its backends. |
| `symbi-jit` | Cranelift compilation of user expressions on the CPU. |
| `symbi-geometry` | Coordinate maps, metrics, finite-volume geometry. |
| `symbi-grid` | Field storage, views, and halos. |

**Physics**

| Crate | What it holds |
| --- | --- |
| `symbi-hydro` | Equations of state, regimes, Riemann solvers, sources. |
| `symbi-ib` | Immersed bodies, signed-distance geometry, penalization. |

**Code generation and dispatch**

| Crate | What it holds |
| --- | --- |
| `symbi-discretize` | The physics traced at `S = Gv` into stencil graphs. |
| `symbi-aot` | The kernel library, baked at build time. |
| `symbi-exec` | Neutral dispatch and the CPU parallelism policy. |
| `symbi-substrate` | The live per-regime kernel sets. |

**Running a simulation**

| Crate | What it holds |
| --- | --- |
| `symbi-sim` | The state containers, the stepping primitives, checkpoints. |
| `symbi-amr` | Static mesh refinement and conservative level transfer. |
| `symbi` | The builder and the evolution driver a user calls. |

**Output and post-processing**

| Crate | What it holds |
| --- | --- |
| `symbi-io` | Schema-driven HDF5 and JSON serialization. |
| `symbi-display` | The terminal view of a running simulation. |
| `symbi-afterglow` | Synchrotron light curves from relativistic blast waves. |
| `symbi-afterglow-io` | Reading checkpoints of any geometry into that module. |
| `symbi-py` | The Python extension module. |

## A few conventions that hold everywhere

Loop indices are doubled, so `ii`, `jj`, `kk`. This is partly to avoid collisions
and partly so that searching for a loop variable returns loops.

Comment prose is lowercase, and it states what the code does rather than what it
avoids doing. A comment should still make sense to somebody who has never seen the
conversation or the document that produced it, which rules out references to task
numbers and internal notes.

Warnings are denied across the workspace. A deliberate exception is written as an
`#[allow(...)]` at the specific site, with a reason.

## Working here

The fast inner loop is `cargo check -p <crate>`. A full bake takes a few minutes
and a Python install takes rather longer, so it pays to know which of the three a
given question actually needs.

When a refactor is meant to preserve behavior, diffing the emitted kernels before
and after settles the question more convincingly than a passing test suite does.

Tests are run in debug. The release profile is for measuring performance, and
performance claims should be based on measurements.
