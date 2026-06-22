// =============================================================================
// ir_blobs_render.rs
//
// the CPU-side regression for the serialized backend-neutral IR (docs/design/15
// §3): every kernel build.rs emits is stored as a `<KERNEL>_IR` blob — the
// serialized `Prepared`. this test deserializes the REAL generated blobs and
// renders them to CUDA source on the CPU (no GPU, no cuda feature, no nvcc).
//
// it guards two things the synthetic round-trip proof (symbi-ir) cannot:
//   - serde DESERIALIZATION of the deepest real graphs — the 100-iter RMHD c2p
//     bracketed iterate — does not trip serde_json's recursion limit.
//   - the emitted blobs are well-formed: each renders to a named __global__.
//
// the actual on-GPU render+launch+compare is the step-3c distrobox gate.
// =============================================================================

use symbi_aot::{
    ISO_C2P_1D_IR, RMHD_C2P_3D_IR, RMHD_FACE_FLUX_3D_0_IR, RMHD_GHOST_FILL_3D_IR, SRHD_C2P_1D_IR,
    SRHD_C2P_2D_IR, SRHD_FACE_FLUX_2D_0_IR, SRHD_GODUNOV_STAGE_2D_IR,
};
use symbi_ir::emit::{Precision, Target};
use symbi_ir::render_from_ir;

// render one blob to CUDA source and assert it is a well-formed named kernel.
fn renders(ir: &str, name: &str) {
    let desc = render_from_ir(ir, Target::Cuda, Precision::F64);
    assert_eq!(desc.kernel_name, name, "kernel name mismatch");
    assert!(desc.source.contains(name), "{name}: source missing the kernel name:\n{}", desc.source);
    assert!(desc.source.contains("__global__"), "{name}: not a CUDA __global__:\n{}", desc.source);
    assert!(!desc.field_bindings.is_empty(), "{name}: no buffer bindings");
}

#[test]
fn aot_ir_blobs_deserialize_and_render_to_cuda() {
    // a shallow pointwise kernel, an iterative one, and the three deepest RMHD
    // kernels (the 100-iter c2p, the quartic-wave-speed flux, the ghost fill).
    renders(ISO_C2P_1D_IR, "iso_c2p_1d");
    renders(SRHD_C2P_1D_IR, "srhd_c2p_1d");
    renders(RMHD_C2P_3D_IR, "rmhd_c2p_3d");
    renders(RMHD_FACE_FLUX_3D_0_IR, "rmhd_face_flux_3d_0");
    renders(RMHD_GHOST_FILL_3D_IR, "rmhd_ghost_fill_3d");
}

// the dimension-invariance proof: the SAME dim-generic builders, instantiated at 2D,
// emit well-formed kernels — c2p (iterative), the HLLE flux (2-component momentum),
// and the godunov update (divergence over 2 sweep axes). once build.rs derives the
// generated filename from kernel_name, going from 1D to 2D is just another instance.
#[test]
fn aot_dim_generic_srhd_2d_blobs_render() {
    renders(SRHD_C2P_2D_IR, "srhd_c2p_2d");
    renders(SRHD_FACE_FLUX_2D_0_IR, "srhd_face_flux_2d_0");
    renders(SRHD_GODUNOV_STAGE_2D_IR, "srhd_godunov_stage_2d");
}
