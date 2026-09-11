# The symbi workspace

The Rust code is split into twenty-two crates. This page is a guide to what's
where and how the pieces fit together.

## Where to begin

If you're adding a source term, trying a Riemann solver, changing a boundary, or
setting up a problem, start with `symbi-hydro`, `symbi-geometry`, `symbi-ib`, or
the Python frontend. You usually won't need to touch the compiler code.

For compiler and code generation work, start with `symbi-ir`,
`symbi-discretize`, and `symbi-aot`. They turn the physics definitions into CPU
and GPU kernels.

## A bit of background

The physics code works with several kinds of scalar. Run a Riemann solver with
`S = f64` and you get numbers. Run it with `S = Gv` and the arithmetic records a
computation graph. With `S = Dual`, it carries derivatives alongside the values.
The equations stay the same; what changes is how the arithmetic is evaluated.

If you've used dual numbers for automatic differentiation, this should feel
familiar. Tracking units through a calculation or keeping terms in a perturbation
series follows a similar idea.

The graphs are directed and acyclic. We use topological ordering to schedule
operations and reachability from the outputs to find the parts we need. In
compiler terms, removing unused operations is *dead code elimination*. Finding
repeated expressions is *common subexpression elimination*, and *lowering* means
rewriting the graph into a simpler form on the way to executable code.

Graph size isn't the only thing that matters. In `symbi-expr`, register use
depends on how many intermediate values need to stay alive at once. A previous
depth-first schedule kept values around too long and hit a limit at about 252
nodes. Changing the order of evaluation can make a big difference without
changing the calculation.

## How the physics becomes a kernel

`symbi-discretize` evaluates the physics at `S = Gv` to record a stencil graph.
The IR lowers that graph and generates code for the chosen backend. This lets the
CPU, CUDA, and HIP paths share the same Riemann solver definition.

    symbi-hydro        physics, generic over S: Scalar
         |
         |             evaluate at S = Gv
         v
    symbi-discretize   record a stencil graph
         |
         v
    symbi-ir           lower, simplify, and generate code
         |
         v
    symbi-aot          generate CPU Rust and a neutral IR blob at build time
         |
         v
    symbi-exec         launch the kernel
         |
         v
    symbi-substrate    manage the kernel sets for each regime

User expressions from configuration files go through the same IR. Those
expressions aren't known when the library is built, so they need runtime
compilation. The kernels generated at build time use the same machinery.

## The crates

### Basic types, storage, and compilation

| Crate | What's in it |
| --- | --- |
| `symbi-algebra` | Tensors, domains, and memory layout. No dependencies. |
| `symbi-abi` | Shared names for kernel parameters and buffers. |
| `symbi-expr` | User expressions and their register machine. |
| `symbi-xpu` | Memory and execution on CPU, CUDA, and HIP. |
| `symbi-ir` | Computation graphs, compiler passes, and code generation. |
| `symbi-jit` | CPU compilation of user expressions with Cranelift. |
| `symbi-geometry` | Coordinate maps, metrics, and finite-volume geometry. |
| `symbi-grid` | Field storage, views, and halos. |

### Physics

| Crate | What's in it |
| --- | --- |
| `symbi-hydro` | Equations of state, regimes, Riemann solvers, and sources. |
| `symbi-ib` | Immersed bodies, signed-distance geometry, and penalization. |

### Building and running kernels

| Crate | What's in it |
| --- | --- |
| `symbi-discretize` | Stencil graphs traced from the physics at `S = Gv`. |
| `symbi-aot` | The kernel library generated at build time. |
| `symbi-exec` | Kernel dispatch and CPU parallelism. |
| `symbi-substrate` | Kernel sets for each regime. |

### Running a simulation

| Crate | What's in it |
| --- | --- |
| `symbi-sim` | Simulation state, shared stepping routines, and checkpoints. |
| `symbi-refinement` | Fixed mesh refinement and conservative level transfer. |
| `symbi` | The builder and evolution driver. |

### Output and post-processing

| Crate | What's in it |
| --- | --- |
| `symbi-io` | HDF5 and JSON serialization from a shared schema. |
| `symbi-display` | The terminal display for a running simulation. |
| `symbi-afterglow` | Synchrotron light curves from relativistic blast waves. |
| `symbi-afterglow-io` | Checkpoint reading and geometry conversion for afterglow calculations. |
| `symbi-py` | The Python extension module. |

## A few conventions

Loop indices use doubled letters: `ii`, `jj`, `kk`. This helps avoid name
collisions and makes them easier to search for.

Comments use lowercase prose and explain the code in terms of the physics or
algorithm. Leave out task numbers and references to internal discussions so the
comment makes sense on its own.

Warnings are denied across the workspace. If an exception is needed, use
`#[allow(...)]` at the relevant spot and explain why.

## Working here

`cargo check -p <crate>` is useful for quick feedback. Generating the full kernel
library takes a few minutes, and a Python install takes longer.

For refactors that should preserve behavior, compare the generated kernels before
and after as well as running the tests.

Run tests in debug mode. Use the release profile when measuring performance.
