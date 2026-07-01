// =============================================================================
// symbi-ir
//
// the substrate's target-independent computation-graph IR — graph, carrier, and
// the codegen substrate's mathematical contract, all in one crate.
//
// symbi-discretize lowers carrier-generic physics into this IR by tracing it at
// `S = Gv`; symbi-aot's build.rs emits kernels (CPU Rust + CUDA source + neutral
// IR blobs) from the resulting graph. nvcc orchestration lives in symbi-xpu, not
// here — this crate does not shell a toolchain.
//
// organization:
//   algebra              — codegen substrate's mathematical contract
//                          (Op signature + Mask + Scalar + SourceLoc + RenderPolicy)
//   gv                   — the Gv carrier (Scalar impl that traces into the IR)
//   {element, symbol,    — IR data structures the closed signature operates on
//    dim, variance, ty,    (rank-0/rank-N types, dimensions, einsum, hash-cons keys)
//    error, einsum,
//    morphism, graph}
//   passes/              — IR-to-IR transformations (scalarize, splice, cse)
//   backends/            — IR-to-target emitters (cpu, cuda, kernel, render, interp)
//   class                — ClassWitness determinism lattice (consumed by ty)
//   emit                 — high-level emit entry (Target enum, render_from_ir)
// =============================================================================

// codegen substrate constitution + carrier
pub mod algebra;
pub mod dual;
pub mod gv;

// IR data structures (the closed signature operates over these)
pub mod element;
pub mod symbol;
pub mod dim;
pub mod variance;
pub mod ty;
pub mod error;
pub mod einsum;
pub mod morphism;
pub mod graph;

// IR-to-IR passes
pub mod passes;
// IR-to-target backends
pub mod backends;

// determinism lattice + high-level emit entry
pub mod class;
pub mod emit;

// keystone: symbolic div(curl B) = 0 checker over the traced curl DAG.
pub mod proof;

// the typed trace <-> dispatch ABI vocabulary (field-buffer / scalar-param / mesh
// names) lives in the leaf `symbi-abi` crate so this IR carries the generic
// `FieldBind`/`ScalarBind` containers without spelling the closed domain vocabulary
// itself. re-exported below (`symbi_ir::FieldRef`, ...).

// typed kernel registry names for the amr-transfer / field-op family.
pub mod kernel_id;

// Layer 0 universal primitives — categorical structures over the substrate:
// variance (re-exported), Scope/Scoped (multi-rank discipline), LinearSpace
// (opt-in algebraic structure), Geometry (metric-bearing manifold).
pub mod primitives;

// ───── re-exports — the IR surface at the crate root.

pub use class::ClassWitness;
// the ABI vocabulary lives in `symbi-abi`; re-exported here so downstream callers
// use the `symbi_ir::FieldRef` / `symbi_ir::ScalarBind` paths.
pub use symbi_abi::{
    BodyScalar, FieldBind, FieldRef, MeshScalar, ScalarBind, ScalarRef, StateComp, StateSlot,
};
pub use kernel_id::{KernelId, ProlongTag};

// carrier-side surface
pub use gv::{
    Gv, GvKernel, GvTrace, TileSpec, begin_trace, end_trace, in_isolated_trace, with_trace,
};

// IR data surface
pub use element::ElementTy;
pub use symbol::Symbol;
pub use dim::{DimExpr, Shape, shapes_equal, broadcasts_to, broadcast_shape};
pub use variance::VarianceTag;
pub use ty::{TensorTy, DetClass, join_class};
pub use error::ShapeError;
pub use einsum::{Atom, AtomList, EinsumSpec, EinsumParseError, parse_einsum_spec, MAX_LABELS_PER_SPEC};
pub use graph::{Graph, Node, NodeId, Op, ConstValue, DimIndex, ElementWiseOp, TranscendentalOp, ReduceOp, FnDef, FnId};

// passes surface
pub use passes::scalarize::{
    LoweredFn, LoweredParam, ScalarExpr, ScalarStmt, BinaryKind, UnaryKind,
    scalarize, scalarize_kernel, KernelScalarized,
};
pub use passes::splice::{splice_graph, SpliceError};
pub use passes::cse::{cse_lowered_fn, cse_kernel};

// backends surface
pub use backends::cpu::emit_cpu;
pub use backends::cuda::emit_cuda;
pub use backends::kernel::{
    emit_kernel_from_lowering, prepared_from_ir, prepared_to_ir, render_field_reduction,
    render_from_ir, KernelEmitInputs, REDUCTION_BLOCK_SIZE,
};
pub use backends::kernel_cpu::{emit_kernel_cpu, emit_kernel_cpu_serial};
pub use backends::render::{
    emit_kernel_render, kernel_bindings_from_ir, kernel_scalar_params_typed_from_ir, prepare,
    render, KernelRenderer, Prepared,
};
pub use backends::interp::{Backend, Cpu, CpuField, CpuFieldMut};
