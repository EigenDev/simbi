// =============================================================================
// emit_ppm_hip.rs
//
// emit the ppm face-flux twins and the quartic coarse-fine prolong kernels as
// HIP source — the same IR graphs the CPU AOT path compiles, rendered for the
// device backend. these are the only kernel families whose first device
// execution is a ppm+fmr run, so a standalone hipcc compile (and eyeball) of
// the rendered source separates a render-level defect from a launch/runtime
// one. the ppm-order prolong sweeps are emitted alongside as the known-good
// baseline: they have run on the device for months under plm production, so
// any structural difference between the two renders is signal.
//
// usage: cargo run -p symbi-discretize --example emit_ppm_hip -- <out_dir>
//        then on the target machine: hipcc -c <out_dir>/*.hip.cpp
// =============================================================================

use std::fs;

use symbi_discretize::GvKernel;
use symbi_discretize::gv::{adiabatic_hllc_plus_flux_gv, rhd_c2p_gv};
use symbi_discretize::{
    EosArm, ProlongOrder, Recon, field_lerp_multi_gv, refine_prolong_multi_1t_gv,
    refine_prolong_sweep_multi_gv,
};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

type Writes = Vec<(String, symbi_ir::FieldBind, NodeId)>;

fn emit_gv(out_dir: &str, name: &str, ndim: u8, k: GvKernel, writes: Writes) {
    assert!(
        !k.graph.has_errors(),
        "{name} graph errors: {:?}",
        k.graph.errors()
    );
    let desc = emit_kernel_from_lowering(
        &k.graph,
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim,
            target: TargetConfig {
                target: Target::Hip,
                precision: Precision::F64,
            },
            field_inputs: &k.field_inputs,
            scalar_params: &k.scalar_params,
            field_writes: &writes,
            coord_components: &k.coord_components,
            device_preamble: &[],
            tile_spec: None,
        },
    );
    let path = format!("{out_dir}/{name}.hip.cpp");
    fs::write(&path, &desc.source).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("emitted {path}");
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/ppm_hip".to_string());
    fs::create_dir_all(&out).expect("create out dir");

    // the ppm face-flux twins, 3d, HLLC+ (the production sink solver): the -3..+2
    // parabola stencil, the two velocity-jump dissipation rescalings and the
    // compression-gated flatten, all in one graph — the widest-stencil flux the device runs.
    for dir in 0..3u8 {
        let (k, w) = adiabatic_hllc_plus_flux_gv::<3>(
            dir,
            Recon::Ppm,
            symbi_discretize::coords::Balance::Plain,
            symbi_discretize::coords::Coords::Cartesian,
            &[0, 1, 2],
        );
        emit_gv(
            &out,
            &format!("adiabatic_face_flux_hllc_plus_ppm_3d_{dir}"),
            3,
            k,
            w,
        );
    }

    // the quartic coarse-fine prolong family at the production component count
    // (rho + 3 velocities + pressure): the coarse-side time lerp, the fused
    // one-tile prolong, and the three axis-split sweeps.
    let nd = 3usize;
    let ncomp = 5usize;
    let (k, w) = field_lerp_multi_gv(nd, ncomp);
    emit_gv(&out, "field_lerp_multi_5c_3d", 3, k, w);
    let (k, w) = refine_prolong_multi_1t_gv(nd, 2, ProlongOrder::Quartic, ncomp);
    emit_gv(&out, "refine_prolong_quartic_3d_5c_1t", 3, k, w);
    for axis in 0..3usize {
        let (k, w) = refine_prolong_sweep_multi_gv(nd, 2, ProlongOrder::Quartic, axis, ncomp);
        emit_gv(
            &out,
            &format!("refine_prolong_quartic_3d_5c_sw{axis}"),
            3,
            k,
            w,
        );
        // the ppm-order sweep: the device-proven baseline for the same pass.
        let (k, w) = refine_prolong_sweep_multi_gv(nd, 2, ProlongOrder::Ppm, axis, ncomp);
        emit_gv(&out, &format!("refine_prolong_ppm_3d_5c_sw{axis}"), 3, k, w);
    }

    // the c2p that reported the NaN, for completeness of the standalone set.
    let (k, w) = rhd_c2p_gv::<3>(20, EosArm::IdealGamma);
    emit_gv(&out, "rhd_c2p_3d", 3, k, w);

    println!("done -> {out}");
}
