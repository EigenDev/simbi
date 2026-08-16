// =============================================================================
// emit_rmhd_cuda.rs
//
// emit the substrate RMHD kernels as CUDA source — the GPU portability gate.
// the three production builders cover every RMHD-specific feature:
//   - rmhd_c2p:        the KKC false-position (inline iterate loop) + recovery
//   - rmhd_hlle_flux:  the quartic wave speeds (sinh/asinh/cosh/acosh/cos/acos/pow
//                      + (double)INFINITY / nan("") sentinels) + RMHD U/F + HLLE
//   - rmhd_ct_curl_2d: the constrained-transport curl (integer-offset load_at)
// nvcc-compile the output to PTX (sm_75) to prove the GPU codegen is well-formed.
//
// usage: cargo run -p symbi-discretize --example emit_rmhd_cuda -- <out_dir>
// =============================================================================

use std::fs;

use symbi_discretize::GvKernel;
use symbi_discretize::Spacing;
use symbi_discretize::{
    rmhd_c2p_gv, rmhd_ct_curl_2d_dir_gv, rmhd_flux_gv, rmhd_resistive_emf_2d_gv,
    rmhd_resistive_emf_3d_dir_gv, rmhd_resistive_emf_cyl_rz_gv,
};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

type Writes = Vec<(String, symbi_ir::FieldBind, NodeId)>;

// emit a Gv-traced kernel (graph + ABI manifest already carried) -> CUDA source.
fn emit_gv(out_dir: &str, name: &str, ndim: u8, k: GvKernel, writes: Writes) {
    assert!(
        !k.graph.has_errors(),
        "{name} graph errors: {:?}",
        k.graph.errors()
    );
    // thread the kernel's declared smem tile intent so the emitted CUDA
    // exercises the smem prelude + redirected stencil reads through the PTX gate.
    let tile_spec = k.infer_tile_spec();
    let desc = emit_kernel_from_lowering(
        &k.graph,
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &k.field_inputs,
            scalar_params: &k.scalar_params,
            field_writes: &writes,
            coord_components: &k.coord_components,
            device_preamble: &[],
            tile_spec: tile_spec.as_ref(),
        },
    );
    let path = format!("{out_dir}/{name}.cu");
    fs::write(&path, &desc.source).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("emitted {path}");
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: emit_rmhd_cuda <out_dir>");

    // the KKC false-position c2p is the gv single-source physics (symbi-hydro's
    // `rmhd_recover` at S=Gv — the 6-state bracketed iterate -> multi-acc IterateInline).
    let (rmhd_k, rmhd_writes) = rmhd_c2p_gv(100);
    emit_gv(&out_dir, "rmhd_c2p", 1, rmhd_k, rmhd_writes);

    // theta-MC PLM (Gv stencil) + riemann::hlle at the Rmhd regime — the gv single source.
    let (rmhd_f, rmhd_fw) = rmhd_flux_gv(1, 0, 0);
    emit_gv(&out_dir, "rmhd_hlle_flux", 1, rmhd_f, rmhd_fw);

    // the 3D direction-0 flux — the hottest GPU kernel + the smem slab target.
    let (rmhd_f3, rmhd_f3w) = rmhd_flux_gv(3, 0, 0);
    emit_gv(&out_dir, "rmhd_face_flux_3d_0", 3, rmhd_f3, rmhd_f3w);

    // the constrained-transport curl — the gv staggered stencil (div(B)=0 preserved). the 2d curl
    // is split per in-plane direction (dir=0 -> B_x, dir=1 -> B_y, both from the corner E_z).
    // emit both.
    for dir in 0..2 {
        let (ct_k, ct_w) = rmhd_ct_curl_2d_dir_gv(dir);
        emit_gv(&out_dir, &format!("rmhd_ct_curl_2d_{dir}"), 2, ct_k, ct_w);
    }

    // the ohmic resistive edge EMF (eta * J_z added to Ez) — the same Gv trace the CPU bakes, so
    // the CUDA lowering is the GPU portability gate for generic resistive MHD.
    let (res_k, res_w) = rmhd_resistive_emf_2d_gv();
    emit_gv(&out_dir, "rmhd_resistive_emf_2d", 2, res_k, res_w);
    for dir in 0..3 {
        let (r3_k, r3_w) = rmhd_resistive_emf_3d_dir_gv(dir);
        emit_gv(
            &out_dir,
            &format!("rmhd_resistive_emf_3d_{dir}"),
            3,
            r3_k,
            r3_w,
        );
    }

    // the cylindrical r-z resistive EMF: the mimetic adjoint of the cyl induction curl, carrying the
    // face-position geom scalars through the CUDA lowering — the GPU gate for curvilinear resistive MHD.
    let (rcyl_k, rcyl_w) = rmhd_resistive_emf_cyl_rz_gv(&[Spacing::Uniform; 2]);
    emit_gv(&out_dir, "rmhd_resistive_emf_cyl_rz", 2, rcyl_k, rcyl_w);

    // the immersed-body localized resistive EMF: the masked current eta*chi(x)*J with the body-mask
    // SDF (tanh mollifier + body position scalars) traced in-kernel — the GPU gate for resistive sinks.
    let (rbody_k, rbody_w) =
        symbi_discretize::body_resistive_emf_2d_gv(symbi_discretize::coords::Coords::Cartesian);
    emit_gv(&out_dir, "body_resistive_emf_2d", 2, rbody_k, rbody_w);
}
