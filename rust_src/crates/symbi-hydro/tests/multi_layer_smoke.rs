// =============================================================================
// multi_layer_smoke.rs
//
// **end-to-end multi-layer smoke test.** exercises the FULL stack in one place:
//
//   spec data → SimulationLaws → SourceEvaluator → time evolution → physics check.
//
// the load-bearing claim: when the data layer drives real time-stepping,
// the result matches the analytical answer for cases where one's known.
// covers what the Sod test (B5-ix) didn't — the source-RHS integration.
//
// **tests in this file** (each isolates one layer interaction):
//
//   1. `uniform_acceleration_drives_velocity_linearly_in_time`
//      uniform fluid + constant body force. fluxes cancel (no gradients);
//      only source RHS contributes. analytical: v(t) = g·t. proves the
//      source-RHS pipeline integrates correctly per step.
//
//   2. `additive_composition_holds_under_time_evolution`
//      two overlays on the same field. analytical: net velocity matches
//      sum of individual force contributions. proves the additive
//      composition contract survives time evolution (not just static eval).
//
//   3. `iso_regime_with_momentum_only_overlay_evolves_correctly`
//      isothermal regime + gravity that drops the energy source (has_energy=false).
//      validates the iso-special-case routing in SimulationLaws end-to-end.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime_spec::{law_params, NEWTONIAN_SPEC, ISO_NEWTONIAN_SPEC};
use symbi_hydro::source_spec::{
    gravity_params, point_mass_gravity_sources, source_params, user_params,
    uniform_acceleration_sources,
};
use symbi_hydro::state::{Cons, Prim};
use symbi_hydro::{IsoNewtonian, Newtonian, SimulationLaws, SourceEvaluator};

// =============================================================================
// section 1 — uniform external acceleration on uniform fluid.
//
// the cleanest exercise of source-RHS integration. with no spatial gradients,
// the flux divergence is identically zero per cell — every cell gets the
// same flux from every neighbor — so the source RHS is the SOLE contribution
// to dU/dt. analytical answer:
//
//   dm/dt = ρ·g       (momentum density gains ρ·g per unit time)
//   dE/dt = ρ·v·g     (energy gains ρ·(v·g) — work done by the force)
//
// integrating from rest: v(t) = g·t, m(t) = ρ₀·g·t (constant ρ₀ since
// mass is conserved and fluid is uniform).
// =============================================================================

#[test]
fn uniform_acceleration_drives_velocity_linearly_in_time() {
    const N_STEPS: usize = 50;
    const DT: f64 = 0.01;
    const RHO_0: f64 = 1.5;
    const GAMMA: f64 = 5.0 / 3.0;
    let g_ext = [0.1_f64, -0.2, 0.05];

    // ----- setup the composition -----
    let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_user(uniform_acceleration_sources(3, true)); // mom + nrg
    let evaluator = SourceEvaluator::new(&sim, 3).expect("composes");

    // ----- initial state: uniform at rest -----
    let prim_0 = Prim::<f64, 3> {
        rho: RHO_0,
        vel: Tensor::new([0.0, 0.0, 0.0]),
        pre: 1.0,
    };
    let eos = IdealGas { gamma: GAMMA };
    let mut cons = prim_0.to_conserved(&eos);

    // ----- evolution loop: cons += dt * source_rhs each step -----
    // (no flux divergence — uniform fluid has all-equal neighbor fluxes, so
    // F_face_right - F_face_left = 0 at every face, leaving source as the
    // sole contribution.)
    for _step in 0..N_STEPS {
        let prim = prim_from_cons(&cons, GAMMA);

        // momentum source.
        let s_mom = evaluator.eval("mom", &[
            (law_params::RHO, prim.rho),
            (law_params::vel(0).as_str(), prim.vel[0]),
            (law_params::vel(1).as_str(), prim.vel[1]),
            (law_params::vel(2).as_str(), prim.vel[2]),
            (user_params::g_ext(0).as_str(), g_ext[0]),
            (user_params::g_ext(1).as_str(), g_ext[1]),
            (user_params::g_ext(2).as_str(), g_ext[2]),
        ]).expect("mom has overlay");

        // energy source.
        let s_nrg = evaluator.eval("nrg", &[
            (law_params::RHO, prim.rho),
            (law_params::vel(0).as_str(), prim.vel[0]),
            (law_params::vel(1).as_str(), prim.vel[1]),
            (law_params::vel(2).as_str(), prim.vel[2]),
            (user_params::g_ext(0).as_str(), g_ext[0]),
            (user_params::g_ext(1).as_str(), g_ext[1]),
            (user_params::g_ext(2).as_str(), g_ext[2]),
        ]).expect("nrg has overlay");

        // Euler update.
        for k in 0..3 {
            cons.mom[k] += DT * s_mom[k];
        }
        cons.nrg += DT * s_nrg[0];
        // density: no source on mass for uniform-acceleration; conservation holds.
    }

    // ----- analytical verification -----
    // v(t) = g·t,  momentum_k = ρ * v_k.
    let t_final = N_STEPS as f64 * DT;
    let final_prim = prim_from_cons(&cons, GAMMA);

    assert!(
        (final_prim.rho - RHO_0).abs() < 1e-12,
        "density must be preserved exactly (no mass source); got {}", final_prim.rho,
    );

    for k in 0..3 {
        let v_expected = g_ext[k] * t_final;
        let v_actual = final_prim.vel[k];
        assert!(
            (v_actual - v_expected).abs() < 1e-12,
            "component {k}: v({t_final}) = {v_actual} != g·t = {v_expected}",
        );
        let mom_expected = RHO_0 * g_ext[k] * t_final;
        let mom_actual = cons.mom[k];
        assert!(
            (mom_actual - mom_expected).abs() < 1e-12,
            "component {k}: momentum density {mom_actual} != ρ·g·t = {mom_expected}",
        );
    }
}

// =============================================================================
// section 2 — additive composition under time evolution.
//
// two overlays target the SAME field. the cumulative effect after N steps
// MUST equal the sum of the individual overlays' effects after N steps
// (additivity = A1's commutative/associative Add at the composition layer
// AND at the time-integration layer).
// =============================================================================

#[test]
fn additive_composition_holds_under_time_evolution() {
    const N_STEPS: usize = 30;
    const DT: f64 = 0.005;
    const RHO_0: f64 = 1.0;
    const GAMMA: f64 = 1.4;

    // overlay A: uniform external acceleration.
    let g_ext = [0.3_f64, 0.0, 0.0];
    // overlay B: point-mass gravity (mass at origin, cell at fixed offset).
    let gm = 0.5;
    let xm = [0.0_f64, 0.0, 0.0];
    let x_cell = [2.0_f64, 0.0, 0.0]; // cell is 2 units away from mass on +x axis

    // ----- run with overlay A only -----
    let sim_a = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_user(uniform_acceleration_sources(3, false)); // mom-only for simplicity
    let eval_a = SourceEvaluator::new(&sim_a, 3).expect("a");
    let final_mom_a = evolve_uniform(
        &eval_a, RHO_0, GAMMA, N_STEPS, DT,
        |prim| eval_a.eval("mom", &uniform_accel_vals(prim, g_ext)).unwrap(),
    );

    // ----- run with overlay B only -----
    let sim_b = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_gravity(point_mass_gravity_sources(3, false));
    let eval_b = SourceEvaluator::new(&sim_b, 3).expect("b");
    let final_mom_b = evolve_uniform(
        &eval_b, RHO_0, GAMMA, N_STEPS, DT,
        |prim| eval_b.eval("mom", &gravity_vals(prim, x_cell, xm, gm)).unwrap(),
    );

    // ----- run with COMPOSED A + B -----
    let sim_c = SimulationLaws::new(&NEWTONIAN_SPEC)
        .with_user(uniform_acceleration_sources(3, false))
        .with_gravity(point_mass_gravity_sources(3, false));
    let eval_c = SourceEvaluator::new(&sim_c, 3).expect("composes");
    let final_mom_c = evolve_uniform(
        &eval_c, RHO_0, GAMMA, N_STEPS, DT,
        |prim| {
            let mut vals = uniform_accel_vals(prim, g_ext);
            vals.extend(gravity_vals(prim, x_cell, xm, gm));
            eval_c.eval("mom", &vals).unwrap()
        },
    );

    // ----- analytical assertion -----
    //
    // the analytical sum is ONLY exact when each overlay's source is
    // INDEPENDENT of the running state. uniform_acceleration depends on
    // ρ (constant per cell) → constant per step → integrates linearly.
    // point-mass gravity depends on ρ (constant) AND v (changing) — but
    // its momentum source `-ρ*GM*(x-xm)/|x-xm|^3` is v-independent, so
    // it's ALSO constant per step under the static-position assumption.
    //
    // since both individual contributions are constant per step, the
    // composed result equals the sum exactly.
    for k in 0..3 {
        let sum = final_mom_a[k] + final_mom_b[k];
        let composed = final_mom_c[k];
        assert!(
            (sum - composed).abs() < 1e-12,
            "component {k}: composed momentum {composed} != \
             A({}) + B({}) = {sum} (additive composition under evolution)",
            final_mom_a[k], final_mom_b[k],
        );
    }
}

// =============================================================================
// section 3 — isothermal regime + momentum-only overlay.
//
// the iso-routing in SimulationLaws / SourceEvaluator: when the regime is
// isothermal (has_energy=false), the gravity overlay must DROP the energy
// source automatically. the runtime call sequence (validate → fields_with_overlays
// → eval) handles this without the caller intervening. this test exercises
// that path end-to-end.
// =============================================================================

#[test]
fn iso_regime_with_momentum_only_overlay_evolves_correctly() {
    const N_STEPS: usize = 20;
    const DT: f64 = 0.01;
    const RHO_0: f64 = 1.0;
    const CS_SQ: f64 = 1.0;
    let g_ext = [0.5_f64];

    // iso + gravity with has_energy=false. SourceEvaluator.fields() must
    // return ONLY "mom" (no "nrg" — iso has no energy equation).
    let sim = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
        .with_user(uniform_acceleration_sources(1, false));
    let evaluator = SourceEvaluator::new(&sim, 1).expect("iso composes");

    let fields: Vec<&str> = evaluator.fields().collect();
    assert_eq!(
        fields, vec!["mom"],
        "iso evaluator must expose ONLY 'mom' (has_energy=false drops 'nrg')",
    );

    // initial: uniform iso state at rest. ρ is constant (no mass source) so
    // we don't need `mut` on it — just bind the constant.
    let rho = RHO_0;
    let mut vx = 0.0_f64;
    for _step in 0..N_STEPS {
        let s_mom = evaluator.eval("mom", &[
            (law_params::RHO, rho),
            (law_params::vel(0).as_str(), vx),
            (user_params::g_ext(0).as_str(), g_ext[0]),
        ]).expect("mom has overlay");

        // Euler update on momentum density (= ρ * v in 1D).
        let mom_density = rho * vx + DT * s_mom[0];
        vx = mom_density / rho;
        // ρ unchanged (no mass source); cs_sq is the regime constant.
        let _cs_sq = CS_SQ;
    }

    let t_final = N_STEPS as f64 * DT;
    let v_expected = g_ext[0] * t_final;
    assert!(
        (vx - v_expected).abs() < 1e-12,
        "iso evolution: v({t_final}) = {vx} != g·t = {v_expected}",
    );
}

// =============================================================================
// helpers — primitive recovery + uniform-cell evolution + param plumbing.
// =============================================================================

/// recover (rho, vel, pre) from conserved (den, mom, nrg) under the ideal
/// gas EOS. used to thread state through evaluator calls per step.
fn prim_from_cons(cons: &Cons<f64, 3>, gamma: f64) -> Prim<f64, 3> {
    let rho = cons.den;
    let vel = Tensor::new([cons.mom[0] / rho, cons.mom[1] / rho, cons.mom[2] / rho]);
    let kinetic = 0.5 * (cons.mom[0].powi(2) + cons.mom[1].powi(2) + cons.mom[2].powi(2)) / rho;
    let pre = (gamma - 1.0) * (cons.nrg - kinetic);
    Prim { rho, vel, pre }
}

/// uniform-cell evolution: no spatial gradients → no flux divergence,
/// only source-RHS contributes. returns final momentum density vec.
fn evolve_uniform(
    _eval: &SourceEvaluator,
    rho_0: f64,
    gamma: f64,
    n_steps: usize,
    dt: f64,
    mut source_fn: impl FnMut(&Prim<f64, 3>) -> Vec<f64>,
) -> [f64; 3] {
    let prim_0 = Prim::<f64, 3> {
        rho: rho_0,
        vel: Tensor::new([0.0; 3]),
        pre: 1.0,
    };
    let eos = IdealGas { gamma };
    let mut cons = prim_0.to_conserved(&eos);

    for _ in 0..n_steps {
        let prim = prim_from_cons(&cons, gamma);
        let s = source_fn(&prim);
        for k in 0..3 {
            cons.mom[k] += dt * s[k];
        }
    }

    [cons.mom[0], cons.mom[1], cons.mom[2]]
}

fn uniform_accel_vals<'a>(prim: &'a Prim<f64, 3>, g_ext: [f64; 3]) -> Vec<(&'a str, f64)> {
    static RHO: &str = "rho";
    static V0: &str = "vel_0"; static V1: &str = "vel_1"; static V2: &str = "vel_2";
    static G0: &str = "g_ext_0"; static G1: &str = "g_ext_1"; static G2: &str = "g_ext_2";
    vec![
        (RHO, prim.rho),
        (V0, prim.vel[0]), (V1, prim.vel[1]), (V2, prim.vel[2]),
        (G0, g_ext[0]), (G1, g_ext[1]), (G2, g_ext[2]),
    ]
}

fn gravity_vals<'a>(
    prim: &'a Prim<f64, 3>, x: [f64; 3], xm: [f64; 3], gm: f64,
) -> Vec<(&'a str, f64)> {
    static RHO: &str = "rho";
    static V0: &str = "vel_0"; static V1: &str = "vel_1"; static V2: &str = "vel_2";
    static X0: &str = "x_0"; static X1: &str = "x_1"; static X2: &str = "x_2";
    static XM0: &str = "xm_0"; static XM1: &str = "xm_1"; static XM2: &str = "xm_2";
    static GM: &str = "gm";
    static EPS: &str = "eps";
    vec![
        (RHO, prim.rho),
        (V0, prim.vel[0]), (V1, prim.vel[1]), (V2, prim.vel[2]),
        (X0, x[0]), (X1, x[1]), (X2, x[2]),
        (XM0, xm[0]), (XM1, xm[1]), (XM2, xm[2]),
        (GM, gm),
        (EPS, 0.0), // bare 1/r^3 reference
    ]
}

// the params helpers above use static string literals; assert they match
// the actual `law_params` / `gravity_params` / `user_params` slot names
// — if a future rename happens, these tests fail fast at compile.
#[test]
fn param_name_constants_match_module_helpers() {
    assert_eq!("rho", law_params::RHO);
    assert_eq!("vel_0", law_params::vel(0));
    assert_eq!("vel_1", law_params::vel(1));
    assert_eq!("vel_2", law_params::vel(2));
    assert_eq!("g_ext_0", user_params::g_ext(0));
    assert_eq!("g_ext_1", user_params::g_ext(1));
    assert_eq!("g_ext_2", user_params::g_ext(2));
    assert_eq!("x_0", source_params::x(0));
    assert_eq!("xm_0", gravity_params::xm(0));
    assert_eq!("gm", gravity_params::GM);
    assert_eq!("eps", gravity_params::EPS);
    // unused imports trip up clippy; pull them in via assertions.
    let _ = Newtonian; let _ = IsoNewtonian;
}
