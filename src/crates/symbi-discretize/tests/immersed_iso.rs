// =============================================================================
// immersed_iso.rs
//
// validates the isothermal immersed-body source/feedback builders against the
// same spec as the adiabatic kernels (immersed_source.rs). the body physics is
// EOS-independent — gravity + bondi-hoyle accretion are functions of (den, mom,
// cs) only — so the iso kernels must:
//   - produce den/mom updates bitwise-identical to the adiabatic kernels when
//     the sound speed matches (the shared `body_contribution` is the single
//     source of truth);
//   - match the analytic accretion spec with cs from prim.pre (= cs^2*rho),
//     the isothermal closure for the sound speed;
//   - write den and mom alone, the isothermal conserved set.
// =============================================================================

mod harness;
use harness::{KernelRun, Out};

use symbi_discretize::{
    Coords, body_feedback_gv, body_feedback_iso_gv, body_source_gv, body_source_iso_gv,
};

const NX: usize = 6;
const NY: usize = 5;
const X0: f64 = -0.5;
const Y0: f64 = -0.4;
const DX: f64 = 0.18;
const DY: f64 = 0.22;
const DT: f64 = 0.01;
const CS: f64 = 0.5; // isothermal sound speed (constant)

const M0: f64 = 1.2;
const SOFT0: f64 = 0.1;
const RACC0: f64 = 0.6;
const SINK0: f64 = 5.0;
const DELTA0: f64 = 0.3;

// iso scalar binding: dt + grid + per-body params. gamma is absent because the iso
// kernel reads cs from the prim.pre field.
fn iso_scalars(sink0: f64) -> Vec<(&'static str, f64)> {
    vec![
        ("dt", DT),
        ("x_lo_0", X0),
        ("dx_0", DX),
        ("map_kind_0", 0.0),
        ("map_param_0", 1.0),
        ("x_lo_1", Y0),
        ("dx_1", DY),
        ("map_kind_1", 0.0),
        ("map_param_1", 1.0),
        ("body_0_mass", M0),
        ("body_0_soft", SOFT0), ("body_0_softkind", 0.0),
        ("body_0_pos_0", 0.0),
        ("body_0_pos_1", 0.0),
        ("body_0_racc", RACC0),
        ("body_0_sink", sink0),
        ("body_0_delta", DELTA0),
        ("body_0_vel_0", 0.0),
        ("body_0_vel_1", 0.0),
        ("body_1_mass", 0.0),
        ("body_1_soft", 1.0), ("body_1_softkind", 0.0),
        ("body_1_pos_0", 5.0),
        ("body_1_pos_1", -3.0),
        ("body_1_racc", 1.0),
        ("body_1_sink", 0.0),
        ("body_1_delta", 1.0),
        ("body_1_vel_0", 0.0),
        ("body_1_vel_1", 0.0),
    ]
}

// the adiabatic binding adds gamma and supplies `nrg` as a field.
fn adiabatic_scalars(sink0: f64) -> Vec<(&'static str, f64)> {
    let mut s = iso_scalars(sink0);
    s.push(("gamma", 1.4));
    s
}

fn run_iso(sink0: f64, den: f64, m0: f64, m1: f64) -> Out {
    let pre = CS * CS * den; // prim.pre = cs^2 * rho
    KernelRun::new(body_source_iso_gv(2, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid([NX, NY])
        .fields(&[
            ("den", den),
            ("mom_0", m0),
            ("mom_1", m1),
            ("pre", pre),
            // the body is evaluated at the stage input; driven directly with no prior flux
            // update, the stage input is the bound state itself.
            ("us_den", den),
            ("us_mom_0", m0),
            ("us_mom_1", m1),
        ])
        .scalars(&iso_scalars(sink0))
        .run()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

#[test]
fn iso_gravity_matches_adiabatic_exactly() {
    // gravity-only (sink=0): den_dot = 0, so the softened gravity alone drives the den/mom
    // updates, and it is EOS-independent. choose an adiabatic energy whose
    // recovered cs equals the iso CS so the two kernels are on equal footing — then
    // their den/mom writes must be bitwise-identical.
    let (den, m0, m1) = (2.0, 0.6, -0.4);
    // adiabatic: cs = sqrt(gamma*(gamma-1)*e_int) ... easiest: pick nrg s.t. p matches.
    // for gravity-only the den/mom update is cs-independent, so any nrg works.
    let iso = run_iso(0.0, den, m0, m1);
    let adia = KernelRun::new(body_source_gv(2, Coords::Cartesian, 2, 2, &[0, 1], false))
        .grid([NX, NY])
        .fields(&[
            ("den", den),
            ("mom_0", m0),
            ("mom_1", m1),
            ("nrg", 5.0),
            ("us_den", den),
            ("us_mom_0", m0),
            ("us_mom_1", m1),
            ("us_nrg", 5.0),
        ])
        .scalars(&adiabatic_scalars(0.0))
        .run();
    for i in 0..NX {
        for j in 0..NY {
            let c = [i, j];
            for w in ["den_new", "mom_0_new", "mom_1_new"] {
                let (a, b) = (iso.get(c, w), adia.get(c, w));
                assert!(
                    (a - b).abs() < 1e-14,
                    "iso vs adiabatic {w} ({i},{j}): {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn iso_source_gravity_only_matches_analytic() {
    // sink=0: only gravity. den unchanged; mom += dt * den * g (g from the softened potential).
    let out = run_iso(0.0, 2.0, 0.0, 0.0);
    for i in 0..NX {
        for j in 0..NY {
            let x = X0 + (i as f64 + 0.5) * DX;
            let y = Y0 + (j as f64 + 0.5) * DY;
            let r_eff = (x * x + y * y + SOFT0 * SOFT0).sqrt();
            let inv_r3 = 1.0 / r_eff.powi(3);
            let c = [i, j];
            assert!(
                rel(out.get(c, "den_new"), 2.0) < 1e-12,
                "den unchanged ({i},{j})"
            );
            assert!(
                rel(out.get(c, "mom_0_new"), DT * 2.0 * (-M0 * x * inv_r3)) < 1e-12,
                "mom0 ({i},{j})"
            );
            assert!(
                rel(out.get(c, "mom_1_new"), DT * 2.0 * (-M0 * y * inv_r3)) < 1e-12,
                "mom1 ({i},{j})"
            );
        }
    }
}

#[test]
fn iso_source_accretion_matches_spec() {
    // gravity + the well-posed uniform-scaling DRAIN (docs/ideas/accretor.md), cs = CS from
    // prim.pre, no energy. per cell: gravity is an additive momentum source, then each conserved
    // component is scaled by f = exp(-drain_rate*dt) -- so den and mom shrink by the same factor
    // (the velocity is invariant, the design invariant).
    let (den, m0, m1) = (1.5, 0.4, -0.3);
    let out = run_iso(SINK0, den, m0, m1);

    let cs = CS;
    let min_w = DX.min(DY);
    let sound_rate = cs / min_w;
    for i in 0..NX {
        for j in 0..NY {
            let x = X0 + (i as f64 + 0.5) * DX;
            let y = Y0 + (j as f64 + 0.5) * DY;
            let r_dist2 = x * x + y * y;
            let r_eff3 = (r_dist2 + SOFT0 * SOFT0).powf(1.5);
            let r_mag = r_dist2.sqrt();
            let g = [-M0 * x / r_eff3, -M0 * y / r_eff3];
            // gravity additive, then the uniform drain factor.
            let mom_grav = [m0 + DT * den * g[0], m1 + DT * den * g[1]];
            let chi = 0.5 * (1.0 - ((r_mag - RACC0) / min_w).tanh());
            let drain_rate = chi * SINK0.min(sound_rate);
            let f = (-drain_rate * DT).exp();
            let c = [i, j];
            assert!(rel(out.get(c, "den_new"), den * f) < 1e-11, "den ({i},{j})");
            assert!(
                rel(out.get(c, "mom_0_new"), mom_grav[0] * f) < 1e-11,
                "mom0 ({i},{j})"
            );
            assert!(
                rel(out.get(c, "mom_1_new"), mom_grav[1] * f) < 1e-11,
                "mom1 ({i},{j})"
            );
            // the design invariant: velocity (mom/den) is invariant under the drain (gravity aside).
            // check the drained state's velocity equals the gravity-updated velocity.
            let v_drained = out.get(c, "mom_0_new") / out.get(c, "den_new");
            let v_grav = mom_grav[0] / den;
            assert!(
                (v_drained - v_grav).abs() < 1e-11,
                "velocity not drain-invariant ({i},{j})"
            );
        }
    }
    let any_removed = out.values("den_new").iter().any(|&d| d < den - 1e-9);
    assert!(any_removed, "the drain removed no mass");
}

#[test]
fn iso_feedback_matches_adiabatic_when_cs_matches() {
    // the feedback force/torque/mass are EOS-independent given the same cs. pick an
    // adiabatic state whose recovered cs == CS, then the iso + adiabatic feedback writes
    // must agree. recovered cs = sqrt(gamma*p/den), p=(gamma-1)(nrg-ke). with mom=0:
    // cs = sqrt(gamma*(gamma-1)*nrg/den). solve nrg for cs=CS, gamma=1.4.
    let (den, gamma) = (1.5, 1.4);
    let nrg = CS * CS * den / (gamma * (gamma - 1.0)); // mom=0 -> ke=0
    let iso = KernelRun::new(body_feedback_iso_gv(2, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid([NX, NY])
        .fields(&[
            ("den", den),
            ("mom_0", 0.0),
            ("mom_1", 0.0),
            ("pre", CS * CS * den),
        ])
        .scalars(&iso_scalars(SINK0))
        .run();
    let adia = KernelRun::new(body_feedback_gv(2, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid([NX, NY])
        .fields(&[("den", den), ("mom_0", 0.0), ("mom_1", 0.0), ("nrg", nrg)])
        .scalars(&adiabatic_scalars(SINK0))
        .run();
    for i in 0..NX {
        for j in 0..NY {
            let c = [i, j];
            for w in ["b0_f0", "b0_f1", "b0_t2", "b0_m"] {
                let (a, b) = (iso.get(c, w), adia.get(c, w));
                assert!(
                    (a - b).abs() < 1e-11,
                    "iso vs adiabatic feedback {w} ({i},{j}): {a} vs {b}"
                );
            }
        }
    }
}
