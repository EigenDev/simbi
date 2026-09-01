// =============================================================================
// emit_iso_cuda.rs
//
// emit the substrate iso/adiabatic kernels as CUDA source (the same IR graphs the
// CPU AOT path compiles, run through emit_kernel_from_lowering instead of
// emit_kernel_cpu) and write them as .cu files. proves the backend axis: one
// physics graph -> two backends. nvcc-compile the output to PTX (the GPU
// portability gate); the kernels chosen exercise the GPU-risky features —
// integer source-coord select (ghost_fill), sqrt + HLLE branches + integer
// stencil shifts (flux), reciprocal (c2p).
//
// usage: cargo run -p symbi-discretize --example emit_iso_cuda -- <out_dir>
// =============================================================================

use std::fs;

use symbi_discretize::GvKernel;
use symbi_discretize::{
    Coords, EosArm, GeoSource, Recon, Spacetime, Spacing, adiabatic_c2p_gv, adiabatic_flux_gv,
    godunov_mass_gv, godunov_stage_gv, iso_c2p_gv, iso_flux_gv, iso_ghost_fill_gv,
    iso_wave_speed_map_gv, rhd_c2p_gv, rhd_flux_gv, snapshot_gv,
};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::{KernelEmitInputs, KernelWrites, emit_kernel_from_lowering};

// emit a Gv-traced kernel (graph + ABI manifest already carried by the GvKernel) -> CUDA source.
fn emit_gv(out_dir: &str, name: &str, ndim: u8, k: GvKernel, writes: KernelWrites) {
    assert!(
        !k.graph().has_errors(),
        "{name} graph errors: {:?}",
        k.graph().errors()
    );
    let desc = emit_kernel_from_lowering(
        k.graph(),
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: k.field_inputs(),
            scalar_params: k.scalar_params(),
            field_writes: &writes,
            coord_components: k.coord_components(),
            device_preamble: &[],
            tile_spec: None,
        },
    );
    let path = format!("{out_dir}/{name}.cu");
    fs::write(&path, &desc.source).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("emitted {path}");
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/iso_cuda".to_string());
    fs::create_dir_all(&out).expect("create out dir");

    // ghost_fill: the gv lattice-map pullback — integer source-coord select branch.
    let (gk, gw) = iso_ghost_fill_gv(1, 1, &[0]);
    emit_gv(&out, "iso_ghost_fill_1d", 1, gk, gw);

    // flux: PLM reconstruction (Gv stencil) + symbi-hydro's riemann::hlle traced at S=Gv —
    // the gv single source. iso (IsoNewtonian, no energy) + adiabatic (newtonian, energy).
    let (iso_f, iso_fw) = iso_flux_gv::<1>(0);
    emit_gv(&out, "iso_face_flux_1d", 1, iso_f, iso_fw);
    let (adi_f, adi_fw) = adiabatic_flux_gv::<1>(0, Recon::Plm);
    emit_gv(&out, "adiabatic_face_flux_1d", 1, adi_f, adi_fw);

    // c2p: the EOS closures (reciprocal multiply; adiabatic adds the energy term). both
    // are the gv single-source physics (symbi-hydro at S=Gv).
    let (iso_k, iso_writes) = iso_c2p_gv::<1>();
    emit_gv(&out, "iso_c2p_1d", 1, iso_k, iso_writes);
    let (adi_k, adi_writes) = adiabatic_c2p_gv::<1>();
    emit_gv(&out, "adiabatic_c2p_1d", 1, adi_k, adi_writes);

    // cfl wave-speed map: per-cell sqrt(gamma*p/rho) + the max fold — the fully-gv timestep
    // kernel (Newtonian::wave_speeds_axis at S=Gv + the in-kernel cartesian-uniform width).
    let (iso_ws, iso_wsw) = iso_wave_speed_map_gv(Coords::Cartesian, &[Spacing::Uniform], &[0], 1);
    emit_gv(&out, "iso_wave_speed_map_1d", 1, iso_ws, iso_wsw);

    // rhd c2p: the first iterative kernel — a 20-step newton (Op::IterateInline, body
    // once) + sqrt (lorentz factor). the GPU-risky construct is the deep iterate; this is
    // the on-device proof it emits compilable CUDA. the gv single-source physics
    // (symbi-hydro's `rhd_recover` at S=Gv), like iso.
    let (rhd_k, rhd_writes) = rhd_c2p_gv::<1>(20, EosArm::IdealGamma);
    emit_gv(&out, "rhd_c2p_1d", 1, rhd_k, rhd_writes);

    // rhd flux: reconstruction + the canonical HLLE with relativistic physics (lorentz
    // factor + relativistic enthalpy/sound speed + the mignone-bodo wave speeds). gv single
    // source (riemann::hlle at the Rhd regime).
    let (rhd_f, rhd_fw) = rhd_flux_gv::<1>(0, EosArm::IdealGamma);
    emit_gv(&out, "rhd_face_flux_1d", 1, rhd_f, rhd_fw);

    // the conserved-update family — godunov step, RK2, snapshot — EOS-generic gv kernels
    // (has_energy=false is iso; true covers adiabatic and rhd alike, whose godunov is the identical
    // regime-agnostic kernel). cartesian-1D here: the stencil divergence + in-place update.
    let cart = (Coords::Cartesian, [Spacing::Uniform; 1], [0usize; 1]);
    let src = GeoSource::Hydro { inertial: true };
    let (mk, mw) = godunov_mass_gv(cart.0, &cart.1, &cart.2, 1);
    emit_gv(&out, "godunov_mass_1d", 1, mk, mw);
    // one godunov-stage kernel per regime (runtime (a0, ac) SSP coefficients serve euler/rk2/rk3).
    let (iek, iew) = godunov_stage_gv(
        cart.0,
        Spacetime::Minkowski,
        &cart.1,
        &cart.2,
        1,
        1,
        false,
        src,
    );
    emit_gv(&out, "iso_godunov_stage_1d", 1, iek, iew);
    let (aek, aew) = godunov_stage_gv(
        cart.0,
        Spacetime::Minkowski,
        &cart.1,
        &cart.2,
        1,
        1,
        true,
        src,
    );
    emit_gv(&out, "adiabatic_godunov_stage_1d", 1, aek, aew);
    let (isk, isw) = snapshot_gv(1, false);
    emit_gv(&out, "iso_snapshot_1d", 1, isk, isw);
    let (ask, asw) = snapshot_gv(1, true);
    emit_gv(&out, "adiabatic_snapshot_1d", 1, ask, asw);

    println!("done: {} kernels -> {out}", 13);
}
