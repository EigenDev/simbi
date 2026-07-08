// =============================================================================
// immersed_source.rs
//
// validates the substrate body_source builder (docs/design/19) against an
// independent inline implementation of the same spec: an interpreter run on a known
// state with one ACTIVE body + one INACTIVE body (mass=0, sink=0) to prove the
// branch-free MAX_BODIES loop contributes exactly zero for inactive slots.
//
//   gravity:   g = -mass (x - x_b) / (|x-x_b|^2 + soft^2)^{3/2}
//   accretion: den_dot = den * min(sink, 1/t_nat, 1/dt) * exp(-(r/(0.5 racc))^2),
//              mass removed at the torque-controlled sink velocity v_star.
//   cons += dt * (S_grav + S_accretion)
// =============================================================================

mod harness;
use harness::{KernelRun, Out};

use symbi_discretize::{body_feedback_gv, body_source_gv, Coords};

const NX: usize = 6;
const NY: usize = 5;
const X0: f64 = -0.5;
const Y0: f64 = -0.4;
const DX: f64 = 0.18;
const DY: f64 = 0.22;
const DT: f64 = 0.01;
const GAMMA: f64 = 1.4;

// active body 0 at origin (gravity + accretion); body 1 inactive.
const M0: f64 = 1.2;
const SOFT0: f64 = 0.1;
const RACC0: f64 = 0.6;
const SINK0: f64 = 5.0;
const DELTA0: f64 = 0.3;

// the two-body scalar binding: active body 0 + inactive body 1 (mass 0 -> no gravity,
// sink 0 -> no accretion, safe soft/racc/delta). `sink0` lets the gravity-only case
// disable body 0's accretion.
fn body_scalars(sink0: f64) -> Vec<(&'static str, f64)> {
    vec![
        ("dt", DT), ("gamma", GAMMA),
        ("x_lo_0", X0), ("dx_0", DX), ("x_lo_1", Y0), ("dx_1", DY),
        ("body_0_mass", M0), ("body_0_soft", SOFT0), ("body_0_pos_0", 0.0), ("body_0_pos_1", 0.0),
        ("body_0_racc", RACC0), ("body_0_sink", sink0), ("body_0_delta", DELTA0),
        ("body_0_vel_0", 0.0), ("body_0_vel_1", 0.0),
        ("body_1_mass", 0.0), ("body_1_soft", 1.0), ("body_1_pos_0", 5.0), ("body_1_pos_1", -3.0),
        ("body_1_racc", 1.0), ("body_1_sink", 0.0), ("body_1_delta", 1.0),
        ("body_1_vel_0", 0.0), ("body_1_vel_1", 0.0),
    ]
}

// build + interpret the body_source kernel; the writes (den_new / mom_0_new / mom_1_new /
// nrg_new) are out-of-place so they read back the post-update cons.* per cell.
// `sink0` lets the gravity-only case disable accretion.
fn run(sink0: f64, den_in: f64, m0_in: f64, m1_in: f64, nrg_in: f64) -> Out {
    KernelRun::new(body_source_gv(2, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid([NX, NY])
        .fields(&[("den", den_in), ("mom_0", m0_in), ("mom_1", m1_in), ("nrg", nrg_in)])
        .scalars(&body_scalars(sink0))
        .run()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

#[test]
fn body_source_gravity_only_matches_analytic() {
    // sink=0 -> accretion off; only gravity acts. fluid at rest (mom=0) so nrg is unchanged.
    let out = run(0.0, 2.0, 0.0, 0.0, 5.0);
    for i in 0..NX {
        for j in 0..NY {
            let x = X0 + (i as f64 + 0.5) * DX;
            let y = Y0 + (j as f64 + 0.5) * DY;
            let r_eff = (x * x + y * y + SOFT0 * SOFT0).sqrt();
            let inv_r3 = 1.0 / r_eff.powi(3);
            let gx = -M0 * x * inv_r3;
            let gy = -M0 * y * inv_r3;
            let c = [i, j];
            assert!(rel(out.get(c, "den_new"), 2.0) < 1e-12, "den unchanged by gravity ({i},{j}): {}", out.get(c, "den_new"));
            assert!(rel(out.get(c, "mom_0_new"), DT * 2.0 * gx) < 1e-12, "mom0 ({i},{j})");
            assert!(rel(out.get(c, "mom_1_new"), DT * 2.0 * gy) < 1e-12, "mom1 ({i},{j})");
            assert!(rel(out.get(c, "nrg_new"), 5.0) < 1e-12, "nrg (mom=0 -> no work) ({i},{j}): {}", out.get(c, "nrg_new"));
        }
    }
}

#[test]
fn body_source_accretion_matches_spec() {
    // a moving, varying-density fluid so gravity + accretion both act non-trivially.
    let (den_in, m0_in, m1_in, nrg_in) = (1.5, 0.4, -0.3, 6.0);
    let out = run(SINK0, den_in, m0_in, m1_in, nrg_in);

    // independent inline implementation of the same spec (gravity + Bondi-Hoyle sink).
    let v = [m0_in / den_in, m1_in / den_in];
    let ke = 0.5 * (m0_in * v[0] + m1_in * v[1]);
    let p = (GAMMA - 1.0) * (nrg_in - ke);
    let cs = (GAMMA * p / den_in).sqrt();
    let e_int = (nrg_in - ke) / den_in;
    let min_w = DX.min(DY);

    for i in 0..NX {
        for j in 0..NY {
            let x = X0 + (i as f64 + 0.5) * DX;
            let y = Y0 + (j as f64 + 0.5) * DY;
            let rvec = [x, y];
            let r_dist2 = x * x + y * y;
            let r_eff3 = (r_dist2 + SOFT0 * SOFT0).powf(1.5);
            let r_mag = r_dist2.sqrt();

            // gravity rates
            let g = [-M0 * rvec[0] / r_eff3, -M0 * rvec[1] / r_eff3];
            let mut d_mom = [den_in * g[0], den_in * g[1]];
            let mut d_nrg = m0_in * g[0] + m1_in * g[1];
            let mut d_den = 0.0;

            // accretion
            let r_kernel = 0.5 * RACC0;
            let weight = (-(r_mag / r_kernel).powi(2)).exp();
            let sound_crossing = min_w / cs;
            let t_ff = (r_mag.powi(3) / (2.0 * M0 + 1e-30)).sqrt();
            let t_nat = sound_crossing.min(t_ff);
            let nat_rate = 1.0 / (t_nat + 1e-30);
            let sr_base = SINK0.min(nat_rate).min(1.0 / DT);
            let den_dot = den_in * sr_base * weight;
            let safe_r = (r_dist2 + 1e-24).sqrt();
            let rhat = [rvec[0] / safe_r, rvec[1] / safe_r];
            let vrel = [v[0], v[1]]; // body at rest
            let vrad_comp = vrel[0] * rhat[0] + vrel[1] * rhat[1];
            let mut vstar2 = 0.0;
            for ax in 0..2 {
                let vrad = vrad_comp * rhat[ax];
                let vang = vrel[ax] - vrad;
                let vstar = vrad + DELTA0 * vang; // + body_vel = 0
                d_mom[ax] -= vstar * den_dot;
                vstar2 += vstar * vstar;
            }
            d_den -= den_dot;
            d_nrg -= (0.5 * vstar2 + e_int) * den_dot;

            let want_den = den_in + DT * d_den;
            let want_m0 = m0_in + DT * d_mom[0];
            let want_m1 = m1_in + DT * d_mom[1];
            let want_nrg = nrg_in + DT * d_nrg;

            let c = [i, j];
            assert!(rel(out.get(c, "den_new"), want_den) < 1e-11, "den ({i},{j}): got {} want {want_den}", out.get(c, "den_new"));
            assert!(rel(out.get(c, "mom_0_new"), want_m0) < 1e-11, "mom0 ({i},{j}): got {} want {want_m0}", out.get(c, "mom_0_new"));
            assert!(rel(out.get(c, "mom_1_new"), want_m1) < 1e-11, "mom1 ({i},{j}): got {} want {want_m1}", out.get(c, "mom_1_new"));
            assert!(rel(out.get(c, "nrg_new"), want_nrg) < 1e-11, "nrg ({i},{j}): got {} want {want_nrg}", out.get(c, "nrg_new"));
        }
    }

    // sanity: accretion removes mass somewhere inside the kernel (den strictly drops).
    let any_removed = out.values("den_new").iter().any(|&d| d < den_in - 1e-9);
    assert!(any_removed, "accretion removed no mass anywhere");
}

#[test]
fn body_feedback_matches_spec() {
    // build + interpret body_feedback; the gv builder emits 6 writes per body in order
    // [force_0, force_1, torque_0, torque_1, torque_2, mass] (named b{b}_f{ax} / b{b}_t{t} /
    // b{b}_m). this 2D test reads b0_f0=force_0, b0_f1=force_1, b0_t2=torque_2, b0_m=mass
    // (the in-plane force + z-torque), and checks every inactive-body-1 write is zero.
    let (den_in, m0_in, m1_in, nrg_in) = (1.5, 0.4, -0.3, 6.0);
    let out = KernelRun::new(body_feedback_gv(2, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid([NX, NY])
        .fields(&[("den", den_in), ("mom_0", m0_in), ("mom_1", m1_in), ("nrg", nrg_in)])
        .scalars(&body_scalars(SINK0))
        .run();

    let dv = DX * DY;
    let v = [m0_in / den_in, m1_in / den_in];
    let ke = 0.5 * (m0_in * v[0] + m1_in * v[1]);
    let p = (GAMMA - 1.0) * (nrg_in - ke);
    let cs = (GAMMA * p / den_in).sqrt();
    let min_w = DX.min(DY);

    for i in 0..NX {
        for j in 0..NY {
            let x = X0 + (i as f64 + 0.5) * DX;
            let y = Y0 + (j as f64 + 0.5) * DY;
            let r_dist2 = x * x + y * y;
            let r_eff3 = (r_dist2 + SOFT0 * SOFT0).powf(1.5);
            let r_mag = r_dist2.sqrt();
            let g = [-M0 * x / r_eff3, -M0 * y / r_eff3];
            let weight = (-(r_mag / (0.5 * RACC0)).powi(2)).exp();
            let t_ff = (r_mag.powi(3) / (2.0 * M0 + 1e-30)).sqrt();
            let nat_rate = 1.0 / ((min_w / cs).min(t_ff) + 1e-30);
            let sr = SINK0.min(nat_rate).min(1.0 / DT);
            let den_dot = den_in * sr * weight;
            let safe_r = (r_dist2 + 1e-24).sqrt();
            let rhat = [x / safe_r, y / safe_r];
            let vrad_comp = v[0] * rhat[0] + v[1] * rhat[1];
            let vstar = [
                vrad_comp * rhat[0] + DELTA0 * (v[0] - vrad_comp * rhat[0]),
                vrad_comp * rhat[1] + DELTA0 * (v[1] - vrad_comp * rhat[1]),
            ];
            let want_fx = -(den_in * g[0] + vstar[0] * den_dot) * dv;
            let want_fy = -(den_in * g[1] + vstar[1] * den_dot) * dv;
            let fa = [-(vstar[0] * den_dot) * dv, -(vstar[1] * den_dot) * dv];
            let want_tz = x * fa[1] - y * fa[0];
            let want_m = den_dot * dv * DT;

            let c = [i, j];
            assert!(rel(out.get(c, "b0_f0"), want_fx) < 1e-11, "f0x ({i},{j}): {} vs {want_fx}", out.get(c, "b0_f0"));
            assert!(rel(out.get(c, "b0_f1"), want_fy) < 1e-11, "f0y ({i},{j}): {} vs {want_fy}", out.get(c, "b0_f1"));
            assert!(rel(out.get(c, "b0_t2"), want_tz) < 1e-11, "t0z ({i},{j}): {} vs {want_tz}", out.get(c, "b0_t2"));
            assert!(rel(out.get(c, "b0_m"), want_m) < 1e-11, "m0 ({i},{j}): {} vs {want_m}", out.get(c, "b0_m"));
            // inactive body 1 contributes nothing across all six of its writes.
            for w in ["b1_f0", "b1_f1", "b1_t0", "b1_t1", "b1_t2", "b1_m"] {
                assert!(out.get(c, w).abs() < 1e-14, "inactive body 1 {w} ({i},{j}) = {}", out.get(c, w));
            }
        }
    }
}
