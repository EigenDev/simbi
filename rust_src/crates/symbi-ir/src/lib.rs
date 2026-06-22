// =============================================================================
// symbi-ir
//
// the substrate's target-independent computation-graph IR — graph, carrier, and
// the codegen substrate's mathematical contract, all in one crate (consolidated
// 2026-05-30: symbi-core was folded in here so the IR machine is one layer, not
// two; the `tensor/` submodule was flattened the same day — its name had become
// vestigial after the legacy scalar IR was deleted).
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

// typed kernel scalar-parameter names (the trace <-> dispatch ABI, minted once).
pub mod scalar_param;

// typed kernel field-buffer names (the trace <-> dispatch ABI, minted once).
pub mod field_ref;

// typed kernel scalar-parameter names (the closed half of the trace <-> dispatch
// ABI, minted once — the scalar analog of field_ref).
pub mod scalar_ref;

// typed kernel registry names for the amr-transfer / field-op family.
pub mod kernel_id;

// Layer 0 universal primitives — categorical structures over the substrate:
// variance (re-exported), Scope/Scoped (multi-rank discipline), LinearSpace
// (opt-in algebraic structure), Geometry (metric-bearing manifold).
pub mod primitives;

// ───── re-exports — the pre-flatten `tensor::*` surface, now at the crate root.
// callers that did `symbi_ir::Symbol` now do `symbi_ir::Symbol`; explicit
// submodule paths (`symbi_ir::graph::NodeId`) become `symbi_ir::graph::NodeId`.

pub use class::ClassWitness;
pub use scalar_param::MeshScalar;
pub use field_ref::{FieldBind, FieldRef, StateComp, StateSlot};
pub use scalar_ref::{BodyScalar, ScalarBind, ScalarRef};
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
