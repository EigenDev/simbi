// =============================================================================
// penalize_shaped_cuda_smoke.rs
//
// the device twin of the shaped-wall JIT gate: the runtime-built shaped
// penalization kernel must render to CUDA and survive NVRTC. its surface normal
// is the SDF gradient (Dual-derived CSG min/max branches), so this pins that the
// shaped kernel's op set lowers to a compilable __global__ — the device analogue
// of the cranelift-subset gate. min/max lower to the ternary form (matching the
// cranelift + carrier spelling), tanh/sqrt/sin/cos to libdevice; none reject.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi-substrate --features cuda --test penalize_shaped_cuda_smoke
// =============================================================================
#![cfg(feature = "cuda")]

use symbi_discretize::coords::Coords;
use symbi_discretize::{
    GvKernel, penalize_porous_gv_shaped, penalize_porous_gv_spinning,
    penalize_porous_iso_gv_shaped, penalize_porous_iso_gv_spinning,
};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::{KernelEmitInputs, KernelWriteEffect, emit_kernel_from_lowering, legacy_writes};
use symbi_xpu::nvrtc::compile_ptx;

// render the runtime GvKernel to CUDA at f64 (the shaped ABI is raw f64) and
// NVRTC-compile it, exactly as the device dispatch path will. the name mirrors
// the AOT penalize convention: coalesce_layout is false (penalize buffers do not
// share one layout), tile_spec None (the smem path is gated + unimplemented).
fn nvrtc_ok<W: KernelWriteEffect>(name: &str, ndim: u8, k: &GvKernel, writes: &[W]) {
    let writes = legacy_writes(writes);
    let inputs = KernelEmitInputs {
        kernel_name: name,
        ndim,
        target: TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        },
        coalesce_layout: false,
        field_inputs: &k.field_inputs,
        scalar_params: &k.scalar_params,
        field_writes: &writes,
        coord_components: &k.coord_components,
        device_preamble: &[],
        tile_spec: None,
    };
    let desc = emit_kernel_from_lowering(&k.graph, &inputs);
    let ptx = compile_ptx(&desc.source, &desc.kernel_name);
    assert!(
        ptx.is_ok(),
        "the shaped penalize kernel '{name}' failed NVRTC:\n{:?}\n--- source ---\n{}",
        ptx.err(),
        desc.source,
    );
}

#[test]
fn shaped_porous_penalize_nvrtc_compiles_3d() {
    // a box unioned with an offset sphere — a genuine CSG with min/max kinks and a
    // Dual-autodiff normal, in the body-local frame.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
        .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
    let (k, w) = penalize_porous_gv_shaped(Coords::Cartesian, 3, 3, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_3d", 3, &k, &w);
}

#[test]
fn shaped_porous_penalize_nvrtc_compiles_2d() {
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.4, 0.6, 1.0]);
    let (k, w) = penalize_porous_gv_shaped(Coords::Cartesian, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_2d", 2, &k, &w);
}

#[test]
fn shaped_porous_penalize_nvrtc_compiles_2p5d() {
    // dof = 3 on a 2d grid (2.5d): the out-of-plane momentum channel rides the
    // same shaped kernel and must lower to CUDA.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.4, 0.6, 1.0]);
    let (k, w) = penalize_porous_gv_shaped(Coords::Cartesian, 2, 3, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_2p5d", 2, &k, &w);
}

#[test]
fn shaped_penalize_nvrtc_compiles_curvilinear() {
    // the mask distance is physical: on a cylindrical grid the kernel maps the
    // coordinate centroid to Cartesian first (centroid_to_cartesian +
    // vector_from_cartesian). that path must lower to CUDA for the r-phi wall gate.
    let shape = SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], 0.35);
    let (k, w) = penalize_porous_gv_shaped(Coords::Cylindrical, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_cyl", 2, &k, &w);
    let (k, w) = penalize_porous_gv_shaped(Coords::Spherical, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_sph", 2, &k, &w);
}

#[test]
fn spinning_penalize_nvrtc_compiles() {
    // the spinning wall: the mask is rotated by R(angle) built from Gv cos/sin and
    // the surface velocity carries omega x r. cos/sin lower to libdevice.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.2, 0.3]);
    let (k, w) = penalize_porous_gv_spinning(Coords::Cartesian, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_spin", 2, &k, &w);
    let (k, w) = penalize_porous_iso_gv_spinning(Coords::Cartesian, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_spin_iso", 2, &k, &w);
}

#[test]
fn shaped_iso_porous_penalize_nvrtc_compiles() {
    // the energy-free shaped wall (iso obstacle flows): same CSG normal, no nrg
    // channel — the dropped energy buffer must not desync the binding order.
    let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
        .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
    let (k, w) = penalize_porous_iso_gv_shaped(Coords::Cartesian, 2, 2, &shape, false);
    nvrtc_ok("shaped_penalize_cuda_iso", 2, &k, &w);
}
