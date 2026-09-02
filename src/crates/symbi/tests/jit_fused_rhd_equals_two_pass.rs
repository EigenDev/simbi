// =============================================================================
// jit_fused_rhd_equals_two_pass.rs
//
// fused-source path in the relativistic-hydro regime: a flat (SRHD) run with a raw user
// source (the only kind the RHD bridge accepts) must produce a bit-for-bit identical trajectory whether
// the source is fused into the godunov stage (one Cranelift-JIT'd host kernel) or run as the two-pass
// (plain AOT godunov + `apply_runtime_source`). RHD carries no Newtonian immersed body, so this is the
// source-only fusion; GR backgrounds keep the two-pass (the fused builder traces the flat geo).
//
// the godunov combine is regime-agnostic on a cartesian flat grid (pure flux divergence — the
// relativistic physics lives in the flux + c2p), so the fused stage matches the
// `rhd` AOT godunov exactly and the raw source rides additively, bit-for-bit.
//
// run: cargo test -p symbi --test jit_fused_rhd_equals_two_pass
// =============================================================================

use symbi::prelude::*;
use symbi_algebra::Domain;
use symbi_grid::Field;
use symbi_source_compile::expr_bridge::build_user_source;
use symbi_hydro::RHD_SPEC;
use symbi_source_compile::SourceConfig;
use symbi_xpu::HostMemory;

fn assert_cons_bit_identical<const D: usize>(
    interior: &Domain<D>,
    a: &Field<f64, D, HostMemory>,
    b: &Field<f64, D, HostMemory>,
    label: &str,
) {
    for c in interior.iter() {
        let (va, vb) = (*a.view().at(c), *b.view().at(c));
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{label} differs at {c:?}: fused={va:?} two_pass={vb:?} (delta={:?})",
            va - vb,
        );
    }
}

#[test]
fn rhd_raw_source_fused_equals_two_pass_rk2() {
    // the two-pass source application is the default policy; opt the fused kernel
    // in before the policy OnceLock latches so this test exercises the fused path.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    type Sim = SimCpu<Rhd, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 4.0 / 3.0;
    let n = 24usize;
    let t_final = 0.02;

    // a raw radiative-cooling-style source S_nrg = -(C * rho * pre) — reads the per-cell state, the
    // FP-sensitive path (rho/pre binding must match between fused + two-pass). outputs=[5], target=nrg.
    let json = r#"{
        "kind": "raw", "dim": 2, "outputs": [5], "params": [0.25], "target": "nrg",
        "nodes": [ {"op":"PARAMETER","param_idx":0}, {"op":"VARIABLE_RHO"},
                   {"op":"VARIABLE_PRESSURE"}, {"op":"MULTIPLY","left":0,"right":1},
                   {"op":"MULTIPLY","left":3,"right":2}, {"op":"NEG","left":4} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse raw config");

    let build = || -> Sim {
        let sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n])
            .bounds([0.0, 0.0], [1.0, 1.0])
            .boundaries(BoundaryType::Outflow)
            .finish()
            .unwrap();
        // non-uniform density (nonzero divergence) + subluminal velocity (live relativistic flux).
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim {
                rho,
                vel: Tensor::new([0.1, -0.05]),
                pre: 1.0,
            }
        });
        sim
    };

    // two-pass: plain AOT rhd godunov + the per-cell apply_runtime_source pass.
    let mut sim_two = build();
    let sub_two = sim_two.substrate().with_runtime_source(
        build_user_source(&cfg, &RHD_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_two, &sub_two, t_final).expect("rhd two-pass evolve");

    // fused: one JIT'd godunov+source launch.
    let mut sim_fused = build();
    let sub_fused = sim_fused.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &RHD_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_fused, &sub_fused, t_final).expect("rhd fused evolve");

    assert_eq!(
        sub_fused.runtime_source.as_ref().unwrap().fused_cpu_state(),
        Some(true),
        "rhd fused godunov+source kernel did not compile — fell back to two-pass",
    );

    let interior = &sim_fused.geom.interior;
    assert_cons_bit_identical(
        interior,
        &sim_fused.fields.cons.den,
        &sim_two.fields.cons.den,
        "cons.den",
    );
    for k in 0..2 {
        assert_cons_bit_identical(
            interior,
            &sim_fused.fields.cons.mom[k],
            &sim_two.fields.cons.mom[k],
            "cons.mom",
        );
    }
    let (nf, nt) = (
        sim_fused.fields.cons.nrg_field().unwrap(),
        sim_two.fields.cons.nrg_field().unwrap(),
    );
    assert_cons_bit_identical(interior, nf, nt, "cons.nrg");
}
