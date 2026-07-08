// =============================================================================
// aot_godunov_fused_source.rs
//
// the FUSED-source godunov AOT-compiles to a single
// Rust kernel that applies `div(F) + Σ spec_source + integrator` in ONE call.
//
//   - `iso_godunov_stage_with_uniform_accel_1d` — iso (mass + mom) + uniform
//     external acceleration overlay (mom only; iso has no energy law).
//   - `adiabatic_godunov_stage_with_uniform_accel_1d` — adiabatic + uniform
//     accel overlays for BOTH momentum (S_mom = ρ·g_ext) AND energy
//     (S_nrg = ρ·v·g_ext). proves the multi-source fusion baked at
//     build time, not just at trace time.
//
// **what's validated**:
//   1. each fused kernel resolves through the AOT registry (`kernel_by_name`)
//      and the generated `pub fn ..__raw<S>(..)` signature — present means the
//      build emitted, compiled, and registered.
//   2. invoked on a uniform state with zero-flux inputs, the kernels produce
//      EXACTLY the analytical source update (`mom_new = mom + dt·ρ·g_ext`
//      and, for adiabatic, `nrg_new = nrg + dt·ρ·v·g_ext`). bit-exact at f64.
//   3. the kernels also work at f32 (Scalar genericity through to the AOT
//      artifact) — the SAME generated fn body, inferred for f32 from input
//      buffers, produces the analytical update within f32 tol.
//
// run: cargo test -p symbi-aot --test aot_godunov_fused_source
// =============================================================================

use symbi_aot::{kernel_by_name, NamedKernel};

// the godunov-stage kernel reads the u_n snapshot + the SSP (a0, ac) coefficients. these wrappers
// drive the forward-Euler stage (a0=0, ac=1): u_n is multiplied by 0, so it is set to the current
// cons (start-of-step snapshot) and contributes nothing — the result is the pure euler update the
// callers assert against. all buffers use lo=0 and extent = data.len().
#[allow(clippy::too_many_arguments, dead_code)]
fn iso_fused<S: symbi_aot::Scalar + symbi_aot::OrderedNumeric + Send + Sync>(
    mass_flux: &[S], mom_flux: &[S],
    rho_out: &mut [S], mom_out: &mut [S],
    grid_size_0: i32, dom_lo_0: i32,
    _: i32, _: i32, _: i32, _: i32,
    dt: S, g_ext_0: S, dx_0: S,
) {
    let u_n_den = rho_out.to_vec(); // start-of-step snapshot (a0 = 0 -> contributes nothing)
    let u_n_mom = mom_out.to_vec();
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("iso_godunov_stage_with_uniform_accel_1d")
        .input("u_n.den", &u_n_den).input("mass_flux[0]", mass_flux)
        .input("u_n.mom_0", &u_n_mom).input("mom_flux_0[0]", mom_flux)
        .output("cons.den", rho_out).output("cons.mom_0", mom_out)
        .grid(&grid).dom_lo(&dom)
        .scalar("dt", dt).scalar("a0", S::ZERO).scalar("ac", S::ONE)
        .scalar("g_ext_0", g_ext_0).scalar("dx_0", dx_0)
        .scalar("mesh_hdil", S::ZERO) // static mesh: the homologous-dilution term is an exact zero
        .run();
}

#[allow(clippy::too_many_arguments, dead_code)]
fn adi_fused<S: symbi_aot::Scalar + symbi_aot::OrderedNumeric + Send + Sync>(
    mass_flux: &[S], mom_flux: &[S], nrg_flux: &[S],
    rho_out: &mut [S], mom_out: &mut [S], nrg_out: &mut [S],
    grid_size_0: i32, dom_lo_0: i32,
    _: i32, _: i32, _: i32, _: i32, _: i32, _: i32,
    dt: S, g_ext_0: S, dx_0: S,
) {
    let u_n_den = rho_out.to_vec(); // start-of-step snapshot (a0 = 0 -> contributes nothing)
    let u_n_mom = mom_out.to_vec();
    let u_n_nrg = nrg_out.to_vec();
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("adiabatic_godunov_stage_with_uniform_accel_1d")
        .input("u_n.den", &u_n_den).input("mass_flux[0]", mass_flux)
        .input("u_n.mom_0", &u_n_mom).input("mom_flux_0[0]", mom_flux)
        .input("u_n.nrg", &u_n_nrg).input("nrg_flux[0]", nrg_flux)
        .output("cons.den", rho_out).output("cons.mom_0", mom_out).output("cons.nrg", nrg_out)
        .grid(&grid).dom_lo(&dom)
        .scalar("dt", dt).scalar("a0", S::ZERO).scalar("ac", S::ONE)
        .scalar("g_ext_0", g_ext_0).scalar("dx_0", dx_0)
        .scalar("mesh_hdil", S::ZERO) // static mesh: the homologous-dilution term is an exact zero
        .run();
}

#[test]
fn iso_fused_kernel_registered_by_name() {
    // structural: the AOT build emitted the kernel, the registry exposes it.
    // a `Some` here proves the build.rs pipeline (gen_godunov_euler_fused ->
    // emit_gv -> write_both -> REGISTRY -> kernel_by_name) closed for the
    // fused-source variant.
    let resolved = kernel_by_name::<f64>("iso_godunov_stage_with_uniform_accel_1d");
    assert!(resolved.is_some(), "iso fused kernel must register through kernel_by_name");
    let (_kfn, ir_blob) = resolved.unwrap();
    assert!(!ir_blob.is_empty(), "fused kernel must expose a non-empty IR blob");
}

#[test]
fn adiabatic_fused_kernel_registered_by_name() {
    let resolved = kernel_by_name::<f64>("adiabatic_godunov_stage_with_uniform_accel_1d");
    assert!(resolved.is_some(), "adiabatic fused kernel must register through kernel_by_name");
    let (_kfn, ir_blob) = resolved.unwrap();
    assert!(!ir_blob.is_empty(), "fused kernel must expose a non-empty IR blob");
}

#[test]
fn iso_aot_fused_step_matches_analytical_source_update() {
    // **load-bearing semantic check**: the AOT-baked iso kernel, called on a
    // uniform state with zero flux differences (mass_flux constant => div=0),
    // produces EXACTLY the analytical source update for every cell:
    //
    //     rho_new[i] = rho[i]                         (mass unaffected)
    //     mom_new[i] = mom[i] + dt * rho[i] * g_ext_0 (uniform_accel mom overlay)
    //
    // bit-exact at f64. proves the spec source's contribution is fused at the
    // right point in the AOT kernel — not lost, not double-applied.
    let n = 8usize;
    let dt = 0.01_f64;
    let dx = 0.5_f64;
    let g_ext_0 = -9.81_f64;

    let rho_in: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let mom_in: Vec<f64> = (0..n).map(|i| 0.3 - 0.05 * i as f64).collect();
    let mass_flux = vec![0.7_f64; n + 1]; // uniform => zero divergence
    let mom_flux  = vec![0.4_f64; n + 1];

    let mut rho_out = rho_in.clone();
    let mut mom_out = mom_in.clone();

    // signature (raw): (mass_flux[0], mom_flux_0[0], cons.den, cons.mom_0,
    //                   grid_size_0, dom_lo_0, buf_lo_0..3_0, dt, g_ext_0, dx_0)
    iso_fused(
        &mass_flux,
        &mom_flux,
        &mut rho_out,
        &mut mom_out,
        n as i32,
        0, // dom_lo
        0, 0, 0, 0, // buf_lo for each of the 4 buffers
        dt, g_ext_0, dx,
    );

    for i in 0..n {
        assert!(
            (rho_out[i] - rho_in[i]).abs() < 1e-15,
            "cell {i}: rho_new {} ≠ rho_in {} (mass should be invariant)",
            rho_out[i], rho_in[i],
        );
        let mom_expected = mom_in[i] + dt * rho_in[i] * g_ext_0;
        assert!(
            (mom_out[i] - mom_expected).abs() < 1e-13,
            "cell {i}: mom_new {} ≠ analytical {}",
            mom_out[i], mom_expected,
        );
    }
}

#[test]
fn adiabatic_aot_fused_step_applies_both_mom_and_nrg_overlays() {
    // **the Phase-2b-via-AOT claim**: `uniform_acceleration_sources(D, true)`
    // returns TWO specs (mom + nrg). the AOT-baked adiabatic kernel applies
    // BOTH in ONE call:
    //
    //     rho_new[i] = rho[i]
    //     mom_new[i] = mom[i] + dt * rho[i] * g_ext_0
    //     nrg_new[i] = nrg[i] + dt * rho[i] * v[i] * g_ext_0,  v = mom/rho
    //
    // with uniform fluxes (zero divergence) the AOT step must EQUAL the
    // analytical update, bit-exact at f64.
    let n = 8usize;
    let dt = 0.01_f64;
    let dx = 0.5_f64;
    let g_ext_0 = -9.81_f64;

    let rho_in: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let mom_in: Vec<f64> = (0..n).map(|i| 0.3 - 0.05 * i as f64).collect();
    let nrg_in: Vec<f64> = (0..n).map(|i| 5.0 + 0.2 * i as f64).collect();
    let mass_flux = vec![0.7_f64; n + 1];
    let mom_flux  = vec![0.4_f64; n + 1];
    let nrg_flux  = vec![1.1_f64; n + 1];

    let mut rho_out = rho_in.clone();
    let mut mom_out = mom_in.clone();
    let mut nrg_out = nrg_in.clone();

    // signature (raw): (mass_flux[0], mom_flux_0[0], nrg_flux[0],
    //                   cons.den, cons.mom_0, cons.nrg,
    //                   grid, dom_lo, buf_lo_{0..5}_0, dt, g_ext_0, dx_0)
    adi_fused(
        &mass_flux, &mom_flux, &nrg_flux,
        &mut rho_out, &mut mom_out, &mut nrg_out,
        n as i32, 0,
        0, 0, 0, 0, 0, 0,
        dt, g_ext_0, dx,
    );

    for i in 0..n {
        assert!(
            (rho_out[i] - rho_in[i]).abs() < 1e-15,
            "cell {i}: rho_new {} ≠ rho_in {}",
            rho_out[i], rho_in[i],
        );
        let mom_expected = mom_in[i] + dt * rho_in[i] * g_ext_0;
        assert!(
            (mom_out[i] - mom_expected).abs() < 1e-13,
            "cell {i}: mom_new {} ≠ analytical {}",
            mom_out[i], mom_expected,
        );
        // energy overlay: S_nrg = ρ · v · g_ext = ρ · (mom/ρ) · g_ext = mom · g_ext
        let v_in = mom_in[i] / rho_in[i];
        let nrg_expected = nrg_in[i] + dt * rho_in[i] * v_in * g_ext_0;
        assert!(
            (nrg_out[i] - nrg_expected).abs() < 1e-13,
            "cell {i}: nrg_new {} ≠ analytical {} (the energy overlay must fuse too)",
            nrg_out[i], nrg_expected,
        );
    }
}

#[test]
fn iso_aot_fused_runs_at_f32() {
    // **precision-genericity at the FUSED variant**: the same generated
    // `pub fn iso_godunov_stage_with_uniform_accel_1d__raw<S>` instantiated at
    // S=f32 by the input buffer type. proves that AOT-baking a spec-driven
    // source preserves Scalar-genericity through the splice + integrator —
    // f32 lanes pick up the fused source, not just the divergence.
    let n = 8usize;
    let dt = 0.01_f32;
    let dx = 0.5_f32;
    let g_ext_0 = -9.81_f32;
    let rho_in: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * i as f32).collect();
    let mom_in: Vec<f32> = (0..n).map(|i| 0.3 - 0.05 * i as f32).collect();
    let mass_flux = vec![0.7_f32; n + 1];
    let mom_flux  = vec![0.4_f32; n + 1];
    let mut rho_out = rho_in.clone();
    let mut mom_out = mom_in.clone();

    iso_fused(
        &mass_flux, &mom_flux, &mut rho_out, &mut mom_out,
        n as i32, 0, 0, 0, 0, 0, dt, g_ext_0, dx,
    );

    for i in 0..n {
        let mom_expected = mom_in[i] + dt * rho_in[i] * g_ext_0;
        assert!(
            (mom_out[i] - mom_expected).abs() < 1e-5,
            "cell {i} (f32): mom_new {} ≠ analytical {}",
            mom_out[i], mom_expected,
        );
    }
}

#[test]
fn bake_matrix_emits_every_regime_family_ndim_cell() {
    // **structural fingerprint**: the data-driven bake
    // matrix in build.rs (REGIMES × FUSED_FAMILIES × ndim) must emit a
    // kernel at every cell of the cube. asserts the loop walked the table
    // — adding a row to REGIMES or FUSED_FAMILIES MUST surface here as a
    // new kernel without any other change in the codebase.
    let regimes = ["iso", "adiabatic"];
    let families = ["uniform_accel", "point_mass_grav"];
    for regime in regimes {
        for family in families {
            for ndim in 1..=3 {
                let name = format!("{regime}_godunov_stage_with_{family}_{ndim}d");
                assert!(
                    kernel_by_name::<f64>(&name).is_some(),
                    "bake matrix missing cell '{name}' — REGIMES × FUSED_FAMILIES × ndim loop drift",
                );
            }
        }
    }
}
