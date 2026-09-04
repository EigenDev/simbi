// =============================================================================
// jit_fused_body_equals_two_pass.rs
//
// with the fused runtime-source path on by default, a Newtonian run carrying both a
// user source and an immersed body must produce a bit-for-bit identical trajectory whether the update
// is fused (one Cranelift-JIT'd godunov that folds the user source and the body wrap) or run as the
// two-pass (plain AOT godunov -> `apply_runtime_source` -> `body_source`, three separate cons sweeps).
//
// resolving the RHS once must generalize over every contribution: geometric source, active user
// source, and immersed body all ride the single update sweep and match the standalone chain to the
// bit. `jit_fused_equals_two_pass` covers the source-only case; this adds the body fold + the
// `body_source`-skip the fused path relies on.
//
// run: cargo test -p symbi --test jit_fused_body_equals_two_pass
// =============================================================================

use symbi::prelude::*;
use symbi_algebra::Domain;
use symbi_grid::Field;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_ib::{Body, BodyCollection};
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_user_source;
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

// a central accreting mass at the domain center: gravity (softened) + a Bondi-Hoyle sink, so both
// body operators — the additive gravity force and the multiplicative accretion drain — are live.
fn central_black_hole() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5, 0.5]),
        Tensor::zeros(),
        1.0,  // mass
        0.05, // radius
        0.15, // softening
        10.0, // sink_rate
        1e-3, // sink_delta
        0.2,  // accretion_radius
    ))
}

#[test]
fn adiabatic_source_and_body_fused_equals_two_pass_rk2() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 1.4;
    let n = 24usize;
    let t_final = 0.04;

    // external acceleration (force kind) -> mom + energy overlays; exercised alongside the body.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5, -0.3],
        "vocabulary":{"reads":[],"params":[0,1]},
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "PARAMETER", "param_idx": 1} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");

    let build = |with_body: bool| -> Sim {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n])
            .bounds([0.0, 0.0], [1.0, 1.0])
            .boundaries(BoundaryType::Outflow)
            .finish()
            .unwrap();
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim::adiabatic(Density(rho), Tensor::new([0.1, -0.05]), Pressure(1.0))
        });
        if with_body {
            sim.with_bodies(central_black_hole())
        } else {
            sim
        }
    };

    // two-pass: plain AOT godunov + the per-cell apply_runtime_source pass + the standalone body_source.
    let mut sim_two = build(true);
    let sub_two = sim_two.substrate().with_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_two, &sub_two, t_final).expect("two-pass evolve");

    // fused: one JIT'd godunov folding the user source and the immersed-body wrap; source_apply and
    // body_source both skip.
    let mut sim_fused = build(true);
    let sub_fused = sim_fused.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_fused, &sub_fused, t_final).expect("fused evolve");

    // guard: the fused kernel actually JIT-compiled (else this compares two-pass vs two-pass).
    assert_eq!(
        sub_fused.runtime_source.as_ref().unwrap().fused_cpu_state(),
        Some(true),
        "fused godunov+source+body kernel did not compile — fused path silently fell back to two-pass",
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

    // non-vacuity: the body must actually change the trajectory (else the fused body wrap could be a
    // no-op and the equivalence would prove nothing). compare against the same fused run without a body.
    let mut sim_nobody = build(false);
    let sub_nobody = sim_nobody.substrate().with_fused_runtime_source(
        build_user_source(&cfg, &NEWTONIAN_SPEC).unwrap(),
        cfg.params.clone(),
    );
    evolve(&mut sim_nobody, &sub_nobody, t_final).expect("no-body evolve");
    let body_changed = interior.iter().any(|c| {
        sim_fused.fields.cons.den.view().at(c).to_bits()
            != sim_nobody.fields.cons.den.view().at(c).to_bits()
    });
    assert!(
        body_changed,
        "the immersed body left the state unchanged — the oracle is vacuous"
    );
}

#[test]
fn adiabatic_body_only_fused_equals_two_pass_rk2() {
    // the two-pass is the default; this test pins the fused kernel as live, so
    // opt in before the policy OnceLock latches.
    unsafe { std::env::set_var("SYMBI_FUSE", "1") };
    // the body-without-a-user-source path: a pure gravity/accretion run. `with_source_fusion()` folds
    // the immersed body into godunov (one launch, no user source to carry it) and must match the
    // standalone `body_source` pass bit-for-bit. proves the fused path is not gated on a runtime source.
    type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
    const GAMMA: f64 = 1.4;
    let n = 24usize;
    let t_final = 0.04;

    let build = || -> Sim {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([n, n])
            .bounds([0.0, 0.0], [1.0, 1.0])
            .boundaries(BoundaryType::Outflow)
            .finish()
            .unwrap();
        sim.seed_cells(|p| {
            let (x, y) = (p[0], p[1]);
            let rho =
                1.0 + 0.2 * (std::f64::consts::TAU * x).sin() * (std::f64::consts::TAU * y).cos();
            Prim::adiabatic(Density(rho), Tensor::new([0.1, -0.05]), Pressure(1.0))
        });
        sim.with_bodies(central_black_hole())
    };

    // two-pass: plain godunov + the standalone body_source pass (no fusion flag).
    let mut sim_two = build();
    let sub_two = sim_two.substrate();
    evolve(&mut sim_two, &sub_two, t_final).expect("two-pass evolve");

    // fused: body folded into godunov via with_source_fusion (no user source).
    let mut sim_fused = build();
    let sub_fused = sim_fused.substrate().with_source_fusion();
    evolve(&mut sim_fused, &sub_fused, t_final).expect("body-only fused evolve");

    assert_eq!(
        sub_fused.body_only_fused_state(),
        Some(true),
        "body-only fused kernel did not compile — fell back to two-pass",
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
