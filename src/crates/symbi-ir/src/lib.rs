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
//    dim, ty,              (rank-0/rank-N types, dimensions, hash-cons keys)
//    error, graph}
//   passes/              — IR-to-IR transformations (scalarize, splice, cse)
//   backends/            — IR-to-target emitters (cpu, cuda, kernel, render, interp)
//   emit                 — high-level emit entry (Target enum, render_from_ir)
// =============================================================================

// codegen substrate constitution + carrier
pub mod algebra;
pub mod dual;
pub mod gv;

// IR data structures (the closed signature operates over these)
pub mod dim;
pub mod element;
pub mod error;
pub mod graph;
pub mod symbol;
pub mod ty;

// IR-to-IR passes
pub mod passes;
// IR-to-target backends
pub mod backends;

// high-level emit entry
pub mod emit;

// keystone: symbolic div(curl B) = 0 checker over the traced curl DAG.
pub mod proof;

// the typed trace <-> dispatch ABI vocabulary (field-buffer / scalar-param / mesh
// names) lives in the leaf `symbi-abi` crate so this IR carries the generic
// `FieldBind`/`ScalarBind` containers without spelling the closed domain vocabulary
// itself. re-exported below (`symbi_ir::FieldRef`, ...).

// typed kernel registry names for the amr-transfer / field-op family.
pub mod kernel_id;
pub mod support;
pub mod support_infer;

// ───── re-exports — the IR surface at the crate root.

// the ABI vocabulary lives in `symbi-abi`; re-exported here so downstream callers
// use the `symbi_ir::FieldRef` / `symbi_ir::ScalarBind` paths.
pub use kernel_id::{KernelId, ProlongTag};
pub use symbi_abi::{
    BodyScalar, FieldBind, FieldRef, MeshScalar, ScalarBind, ScalarRef, StateComp, StateSlot,
};

// carrier-side surface
pub use gv::{
    Gv, GvKernel, GvMask, GvTrace, KernelWrite, KernelWrites, NumericalPolicy, RewriteClass,
    TileSpec, TraceCx, trace, trace_for_domain, trace_with,
};

// IR data surface
pub use dim::{DimExpr, Shape, broadcast_shape, broadcasts_to, shapes_equal};
pub use element::ElementTy;
pub use error::ShapeError;
pub use graph::{
    ConstValue, DimIndex, ElementWiseOp, FnDef, FnId, Graph, Node, NodeId, Op, ReduceOp,
};
pub use symbol::{InputKey, OutputKey, ScalarParam, Symbol};
pub use ty::TensorTy;

// passes surface
pub use passes::cse::{cse_kernel, cse_lowered_fn};
pub use passes::scalarize::{
    BinaryKind, KernelScalarized, LoweredFn, LoweredParam, ScalarExpr, ScalarStmt, UnaryKind,
    scalarize, scalarize_kernel,
};
pub use passes::splice::{SpliceError, splice_graph};
pub use passes::stencil_reach::{AxisReach, ReachReport, stencil_reach};
pub use support::{ParamExpr, Support};

// backends surface
pub use backends::cpu::emit_cpu;
pub use backends::cuda::emit_cuda;
pub use backends::interp::{Backend, Cpu, CpuField, CpuFieldMut};
pub use backends::kernel::{
    KernelEmitInputs, REDUCTION_BLOCK_SIZE, SEGMENT_EXCLUDED_OFFSET, SEGMENTED_LDS_BUDGET_BYTES,
    SEGMENTED_MAX_BLOCKS, emit_kernel_from_lowering, prepared_from_ir, prepared_to_ir,
    render_field_reduction, render_field_segmented_reduction, render_from_ir, segmented_privatizes,
};
pub use backends::kernel_cpu::{emit_kernel_cpu, emit_kernel_cpu_serial};
pub use backends::render::{
    KernelRenderer, Prepared, emit_kernel_render, kernel_bindings_from_ir,
    kernel_output_support_from_ir, kernel_scalar_params_typed_from_ir, prepare, render,
};
