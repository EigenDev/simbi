// =============================================================================
// fused_source_full_stack.rs
//
// **B6-iv (Phase 4c + 3b + 2c) end-to-end**: SimulationLaws declares a
// `FusedSourceFamily`, derives the substrate `FusedSourceBinding`, the
// kernel-set routes the production `evolve()` loop through the AOT-baked
// fused godunov. Phase 3b extended the AOT bake matrix to 1D/2D/3D
// cartesian × {uniform_accel, point_mass_grav}; Phase 2c bound the spec's
// `x_k` Params to in-kernel cell centroids, so position-dependent overlays
// (point-mass gravity) fuse the same way as uniform accel.
//
// **what this validates** as a single layered claim:
//
//   1. `SimulationLaws::new(&NEWTONIAN_SPEC).with_fused_family(uniform_accel)`
//      → `derive_fused_binding()` → `FusedSourceBinding::from_pair()` →
//      `AdiabaticSubstrateKernelSet::with_fused_source()` produces a kernel
//      set whose evolve loop accelerates an initially-stationary gas in 2D
//      (the Phase 3b 2D AOT kernel exists + the data-driven derivation
//      reaches it through the substrate).
//
//   2. `FusedSourceFamily::PointMassGravity` with a body at the origin
//      produces a kernel whose Param manifest includes `xm_k` and `gm`
//      (NOT `x_k` — those bind to the in-kernel centroid via Phase 2c), so
//      the position-dependent overlay actually fuses with the godunov.
//
//   3. backwards-compat — a `SimulationLaws` with no families derives None,
//      the kernel-set runs the unfused godunov, gas stays at rest.
//
// run: cargo test -p symbi --test fused_source_full_stack
// =============================================================================

use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_aot::kernel_by_name;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{FusedSourceFamily, SimulationLaws, NEWTONIAN_SPEC};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
const GAMMA: f64 = 1.4;

#[test]
fn simulation_laws_drives_2d_evolve_via_uniform_accel_family() {
    // **Phase 4c + Phase 3b layered claim**: declare `uniform_accel` at the
    // SimulationLaws layer, derive the binding, wire it through a 2D kernel
    // set, march evolve(). gas accelerates in +x, density stays stable, no
    // manual kernel-name spelling or scalar binding at the call site —
    // every part of the pipeline (laws → binding → AOT kernel) is data-
    // driven.
    let n = 16usize;
    let dx = 1.0 / n as f64;
    // uniform stationary gas everywhere.
    let mut sim = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n]).spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_x| Prim { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 })
        .build();

    // **Phase 4c**: declare the family, derive the binding from the data layer.
    let g_ext = vec![0.4_f64, 0.0]; // accelerate in +x only
    let laws = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_fused_family(
            FusedSourceFamily::UniformAcceleration { g_ext: g_ext.clone() },
            2,
        );
    let pair = laws.derive_fused_binding().expect("a fused family produces a binding");
    assert_eq!(pair.0, "uniform_accel", "the derived slug matches the AOT kernel name");
    let binding = FusedSourceBinding::from_pair(pair);

    // wire through the kernel set + evolve via the production loop.
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated)
        .with_fused_source(binding);
    let t_final = 0.05_f64;
    evolve(&mut sim, &sub, t_final).expect("2D evolve with fused source failed");

    // gas accelerated in +x to ≈ g·t; +y stayed at rest.
    let cells: Vec<[isize; 2]> = sim.geom.interior.iter().collect();
    let cnt = cells.len() as f64;
    let mean_vx: f64 = cells.iter()
        .map(|c| *sim.fields.prim.vel[0].view().at(*c))
        .sum::<f64>() / cnt;
    let mean_vy: f64 = cells.iter()
        .map(|c| *sim.fields.prim.vel[1].view().at(*c))
        .sum::<f64>() / cnt;
    let analytical_vx = g_ext[0] * sim.time;
    assert!(mean_vx > 0.0, "+x acceleration did not occur (mean vx = {mean_vx})");
    assert!(
        (mean_vx - analytical_vx).abs() < 0.5 * analytical_vx.abs(),
        "mean vx = {mean_vx} too far from analytical g·t = {analytical_vx}",
    );
    assert!(mean_vy.abs() < 1e-6, "+y picked up motion ({mean_vy}); gravity component leaked");
}

#[test]
fn point_mass_gravity_aot_kernel_binds_x_to_centroid_not_scalar() {
    // **Phase 2c structural claim**: the `point_mass_grav` 2D AOT kernel
    // declares `xm_k` + `gm` as scalar params (body parameters, runtime-
    // bound), but DOES NOT declare `x_k` (cell position) — those Params
    // were bound to the in-kernel centroid via Phase 2c, computed from
    // `x_lo + i*dx`. proves the position-dependent overlay actually fused
    // (the spec's `x_k` Params resolved INSIDE the trace, not as runtime
    // scalars).
    let (_kfn, ir_blob) = kernel_by_name::<f64>("adiabatic_godunov_stage_with_point_mass_grav_2d")
        .expect("Phase 3b should have AOT-baked adiabatic point_mass_grav 2D");
    assert!(!ir_blob.is_empty());

    // the kernel manifest lists scalars. xm_0, xm_1, gm MUST appear (body
    // parameters); x_0, x_1 MUST NOT appear (those bind to centroid IN
    // the kernel from x_lo + i*dx, not as runtime scalars).
    let scalars = symbi_ir::kernel_scalar_params_typed_from_ir(ir_blob);
    let scalar_names: Vec<String> = scalars.iter().map(|(b, _)| b.name()).collect();
    let scalar_names: Vec<&str> = scalar_names.iter().map(|s| s.as_str()).collect();
    assert!(scalar_names.contains(&"xm_0"), "xm_0 missing from {scalar_names:?}");
    assert!(scalar_names.contains(&"xm_1"), "xm_1 missing from {scalar_names:?}");
    assert!(scalar_names.contains(&"gm"), "gm missing from {scalar_names:?}");
    assert!(
        !scalar_names.contains(&"x_0"),
        "x_0 leaked into scalars — Phase 2c centroid binding broke; scalars = {scalar_names:?}",
    );
    assert!(
        !scalar_names.contains(&"x_1"),
        "x_1 leaked into scalars — Phase 2c centroid binding broke; scalars = {scalar_names:?}",
    );

    // and the geometry scalars ARE present (the centroid is computed from them).
    assert!(scalar_names.contains(&"x_lo_0"), "x_lo_0 missing (centroid needs the grid origin)");
    assert!(scalar_names.contains(&"dx_0"),   "dx_0 missing (centroid needs the grid step)");
}

#[test]
fn empty_simulation_laws_derives_no_binding() {
    // backwards-compat: no families declared => `derive_fused_binding` => None
    // => substrate routes through the unfused godunov (the prior default).
    let laws = SimulationLaws::new(&NEWTONIAN_SPEC);
    assert!(
        laws.derive_fused_binding().is_none(),
        "an empty SimulationLaws must derive None — backwards compat",
    );
}

#[test]
fn fused_source_family_round_trip_uniform_accel() {
    // **Phase 4c contract test**: the family's `into_binding_pair` produces
    // exactly the scalar pairs the AOT kernel manifest expects, for every
    // dimension we've baked.
    for ndim in 1usize..=3 {
        let g_ext: Vec<f64> = (0..ndim).map(|k| 0.5 * (k as f64 + 1.0)).collect();
        let family = FusedSourceFamily::UniformAcceleration { g_ext: g_ext.clone() };
        let (slug, pairs) = family.into_binding_pair();
        assert_eq!(slug, "uniform_accel");
        assert_eq!(pairs.len(), ndim);
        for (k, (name, value)) in pairs.iter().enumerate() {
            assert_eq!(name, &format!("g_ext_{k}"));
            assert_eq!(*value, g_ext[k]);
        }
        // and the corresponding AOT kernel exists.
        let kname = format!("adiabatic_godunov_stage_with_uniform_accel_{ndim}d");
        assert!(
            kernel_by_name::<f64>(&kname).is_some(),
            "Phase 3b AOT kernel {kname} missing — fan-out regressed",
        );
    }
}

#[test]
fn fused_source_family_round_trip_point_mass_grav() {
    // same contract for point-mass gravity (Phase 2c position-dependent).
    for ndim in 1usize..=3 {
        let xm: Vec<f64> = vec![0.5; ndim];
        let gm = 1.0_f64;
        let family = FusedSourceFamily::PointMassGravity { gm, xm: xm.clone(), eps: 0.0 };
        let (slug, pairs) = family.into_binding_pair();
        assert_eq!(slug, "point_mass_grav");
        // pairs: xm_0..xm_{D-1}, then gm, then eps
        assert_eq!(pairs.len(), ndim + 2);
        for (k, x) in xm.iter().enumerate() {
            assert_eq!(pairs[k].0, format!("xm_{k}"));
            assert_eq!(pairs[k].1, *x);
        }
        assert_eq!(pairs[ndim].0, "gm");
        assert_eq!(pairs[ndim].1, gm);
        assert_eq!(pairs[ndim + 1].0, "eps");
        let kname = format!("adiabatic_godunov_stage_with_point_mass_grav_{ndim}d");
        assert!(
            kernel_by_name::<f64>(&kname).is_some(),
            "Phase 2c AOT kernel {kname} missing",
        );
    }
}
