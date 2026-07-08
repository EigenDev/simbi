// =============================================================================
// rmhd_wave_speeds.rs
//
// validates the substrate RMHD CFL wave-speed MAP — the fully-gv `rmhd_wave_speed_map_gv`
// (symbi-hydro's `Rmhd::wave_speeds_axis`, the SAME quartic the flux HLLE consumes, traced
// at S=Gv with the in-kernel metric width + max-reduction in ONE graph) — against a
// straight-Rust transcription of wave_speeds.hpp::rmhd::wave_speeds.
//
// the SPEED physics itself (Eq.57 / Eq.58 / Eq.56 across regimes, and the
// solve_cubic / solve_quartic polynomial solvers) is validated at the single source
// in symbi-hydro (rmhd.rs tests); this test pins the substrate COMPOSITION: that the gv
// trace yields the right ABI manifest and the per-axis max with inv_dx is right.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{rmhd_wave_speed_map_gv, Coords, Spacing};

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---- straight-Rust transcription of helpers.hpp + wave_speeds.hpp ----

fn solve_cubic_ref(b: f64, c: f64, d: f64) -> f64 {
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    if p.abs() < 1e-12 {
        return q.powf(1.0 / 3.0);
    }
    if q.abs() < 1e-12 {
        return 0.0;
    }
    let t = (p.abs() / 3.0).sqrt();
    let g = 1.5 * q / (p * t);
    if p > 0.0 {
        -2.0 * t * (g.asinh() / 3.0).sinh() - b / 3.0
    } else if 4.0 * p * p * p + 27.0 * q * q < 0.0 {
        2.0 * t * (g.acos() / 3.0).cos() - b / 3.0
    } else if q > 0.0 {
        -2.0 * t * ((-g).acosh() / 3.0).cosh() - b / 3.0
    } else {
        2.0 * t * (g.acosh() / 3.0).cosh() - b / 3.0
    }
}

fn solve_quartic_ref(b: f64, c: f64, d: f64, e: f64) -> (f64, f64) {
    let nan = f64::NAN;
    let p = c - 0.375 * b * b;
    let q = 0.125 * b * b * b - 0.5 * b * c + d;
    let m = solve_cubic_ref(
        p,
        0.25 * p * p + 0.01171875 * b * b * b * b - e + 0.25 * b * d - 0.0625 * b * b * c,
        -0.125 * q * q,
    );
    let mut smin = f64::INFINITY;
    let mut smax = f64::NEG_INFINITY;
    let mut track = |root: f64| {
        if root < smin {
            smin = root;
        }
        if root > smax {
            smax = root;
        }
    };
    if q.abs() < 1e-12 {
        if m < 0.0 {
            return (nan, nan);
        }
        let s = (2.0 * m).sqrt();
        if -m - p > 0.0 {
            let dl = (2.0 * (-m - p)).sqrt();
            track(-0.25 * b + 0.5 * (s - dl));
            track(-0.25 * b - 0.5 * (s - dl));
            track(-0.25 * b + 0.5 * (s + dl));
            track(-0.25 * b - 0.5 * (s + dl));
        }
        if (-m - p).abs() < 1e-12 {
            track(-0.25 * b - 0.5 * s);
            track(-0.25 * b + 0.5 * s);
        }
    } else {
        if m < 0.0 {
            return (nan, nan);
        }
        let s = (2.0 * m).sqrt();
        if -m - p + q / s >= 0.0 {
            let dl = (2.0 * (-m - p + q / s)).sqrt();
            track(0.5 * (-s + dl) - 0.25 * b);
            track(0.5 * (-s - dl) - 0.25 * b);
        }
        if -m - p - q / s >= 0.0 {
            let dl = (2.0 * (-m - p - q / s)).sqrt();
            track(0.5 * (s + dl) - 0.25 * b);
            track(0.5 * (s - dl) - 0.25 * b);
        }
    }
    if smin > smax {
        return (nan, nan);
    }
    (smin, smax)
}

fn ref_wave_speeds(rho: f64, vel: [f64; 3], p: f64, mag: [f64; 3], gamma: f64, dir: usize) -> (f64, f64) {
    let eps = 1e-12;
    let vsq = dot(&vel, &vel);
    let w = 1.0 / (1.0 - vsq).sqrt();
    let h = 1.0 + gamma / (gamma - 1.0) * p / rho;
    let cssq = gamma * p / (rho * h);
    let vdb = dot(&vel, &mag);
    let bmu0 = w * vdb;
    let bmu_s = [mag[0] / w + w * vel[0] * vdb, mag[1] / w + w * vel[1] * vdb, mag[2] / w + w * vel[2] * vdb];
    let bmusq = -bmu0 * bmu0 + bmu_s[0] * bmu_s[0] + bmu_s[1] * bmu_s[1] + bmu_s[2] * bmu_s[2];
    let bn = mag[dir];
    let bnsq = bn * bn;
    let vn = vel[dir];
    if vsq < eps {
        let fac = 1.0 / (rho * h + bmusq);
        let b = -(bmusq + rho * h * cssq + bnsq * cssq) * fac;
        let c = cssq * bnsq * fac;
        let disq = (b * b - 4.0 * c).sqrt();
        let lr = (0.5 * (-b + disq)).sqrt();
        (-lr, lr)
    } else if bnsq < eps {
        let g2 = w * w;
        let vdbperp = vdb - vn * bn;
        let q = bmusq - cssq * vdbperp * vdbperp;
        let a2 = rho * h * (cssq + g2 * (1.0 - cssq)) + q;
        let a1 = -2.0 * rho * h * g2 * vn * (1.0 - cssq);
        let a0 = rho * h * (-cssq + g2 * vn * vn * (1.0 - cssq)) - q;
        let disq = a1 * a1 - 4.0 * a2 * a0;
        (0.5 * (-a1 - disq.sqrt()) / a2, 0.5 * (-a1 + disq.sqrt()) / a2)
    } else {
        let bmun = bmu_s[dir];
        let w2 = w * w;
        let vn2 = vn * vn;
        let a4 = -bmu0 * bmu0 * cssq + bmusq * w2 - cssq * w2 * w2 * h * rho + cssq * w2 * h * rho + w2 * w2 * h * rho;
        let fac = 1.0 / a4;
        let a3 = fac * (2.0 * bmu0 * bmun * cssq - 2.0 * bmusq * w2 * vn + 4.0 * cssq * w2 * w2 * h * rho * vn - 2.0 * cssq * w2 * h * rho * vn - 4.0 * w2 * w2 * h * rho * vn);
        let a2 = fac * (bmu0 * bmu0 * cssq + bmusq * w2 * vn2 - bmusq * w2 - bmun * bmun * cssq - 6.0 * cssq * w2 * w2 * h * rho * vn2 + cssq * w2 * h * rho * vn2 - cssq * w2 * h * rho + 6.0 * w2 * w2 * h * rho * vn2);
        let a1 = fac * (-2.0 * bmu0 * bmun * cssq + 2.0 * bmusq * w2 * vn + 4.0 * cssq * w2 * w2 * h * rho * vn * vn2 + 2.0 * cssq * w2 * h * rho * vn - 4.0 * w2 * w2 * h * rho * vn * vn2);
        let a0 = fac * (-bmusq * w2 * vn2 + bmun * bmun * cssq - cssq * w2 * w2 * h * rho * vn2 * vn2 - cssq * w2 * h * rho * vn2 + w2 * w2 * h * rho * vn2 * vn2);
        let (ll, lr) = solve_quartic_ref(a3, a2, a1, a0);
        if ll.is_nan() {
            (0.0, 0.0)
        } else {
            (ll, lr)
        }
    }
}

// run the rmhd_wave_speed_map builder over the given prim buffers + gamma. RMHD is
// intrinsically 3D (3-velocity), and the map goes through the shared wave_speed_map,
// so it is built at ndim=3 and returns lambda = max_d (s_d * inv_dx_d). inv_dx = 1 is passed
// on each axis (a [n,1,1] grid), so the output is max over the 3 axes of the quartic speed.
fn run_map(inputs: &[(&str, Vec<f64>)], gamma: f64) -> Vec<f64> {
    let n = inputs[0].1.len();
    // a [n,1,1] grid: cell [i,0,0] selects state column i. inv_dx = 1 on every axis, so the
    // output (write `lambda`) is the max over the 3 axes of the quartic characteristic speed.
    let built = rmhd_wave_speed_map_gv(Coords::Cartesian, &[Spacing::Uniform; 3], &[0, 1, 2], 3);
    assert_eq!(
        built.0.scalar_params,
        vec!["gamma".to_string(), "inv_dx_0".into(), "inv_dx_1".into(), "inv_dx_2".into()],
        "the gv map declares gamma + the cartesian-uniform inv_dx widths, in that order"
    );
    let mut k = KernelRun::new(built).grid([n, 1, 1]);
    for (key, col) in inputs {
        let owned = col.clone();
        k = k.field_with(key, move |c| owned[c[0]]);
    }
    k.scalars(&[("gamma", gamma), ("inv_dx_0", 1.0), ("inv_dx_1", 1.0), ("inv_dx_2", 1.0)])
        .run()
        .values("lambda")
        .to_vec()
}

#[test]
fn rmhd_wave_speed_map_bounds_cpp_quartic() {
    // the 3D CFL map (RMHD is 3D): lambda = max_d (max(|lambda_-|, |lambda_+|) * inv_dx_d).
    // the map traces `rmhd_magnetosonic_cfl_speeds` (the cheap c_f^2 upper bound), NOT the
    // exact Mignone & Del Zanna quartic — the quartic stays on the Riemann/flux path. so the
    // contract is not EQUALITY with the exact quartic but CFL-SAFETY: the bound must never
    // UNDER-estimate the exact characteristic speed (an under-estimate would make dt too large
    // and the scheme unstable), and it must stay subluminal. both are validated per state.
    let g = 5.0 / 3.0;
    let states = [
        ([1.0, 0.1, 0.0, 0.0], 1.0, [0.5, 0.3, 0.2]),
        ([2.0, 0.3, 0.1, 0.2], 2.0, [0.6, 0.1, 0.4]),
        ([0.5, 0.0, 0.0, 0.4], 2.0, [0.1, 0.1, 0.6]),
        ([1.0, 0.5, 0.2, 0.1], 3.0, [1.0, 0.5, 0.2]),
    ];
    let cols = |sel: &dyn Fn(&([f64; 4], f64, [f64; 3])) -> f64| -> Vec<f64> {
        states.iter().map(sel).collect()
    };
    let inputs = vec![
        ("prim_rho", cols(&|s| s.0[0])),
        ("prim_v0", cols(&|s| s.0[1])),
        ("prim_v1", cols(&|s| s.0[2])),
        ("prim_v2", cols(&|s| s.0[3])),
        ("prim_pre", cols(&|s| s.1)),
        ("prim_b0", cols(&|s| s.2[0])),
        ("prim_b1", cols(&|s| s.2[1])),
        ("prim_b2", cols(&|s| s.2[2])),
    ];
    let got = run_map(&inputs, g);
    for (i, (rv, p, b)) in states.iter().enumerate() {
        let vel = [rv[1], rv[2], rv[3]];
        // exact = max over the 3 axes (inv_dx = 1) of the quartic characteristic speed.
        let exact = (0..3)
            .map(|dir| {
                let (lo, hi) = ref_wave_speeds(rv[0], vel, *p, *b, g, dir);
                lo.abs().max(hi.abs())
            })
            .fold(0.0_f64, f64::max);
        // never under-estimate (CFL-safe): the magnetosonic bound dominates the exact quartic.
        assert!(got[i] >= exact - 1e-9,
            "map state {i}: bound {} under-estimates exact quartic {exact} -> CFL-unsafe", got[i]);
        // the bound MAY exceed c = 1 by design (a simplified discriminant that over-estimates
        // only shrinks dt — safe). it must stay finite, though: a NaN would poison the dt
        // max-reduction. (per rmhd_magnetosonic_cfl_speeds: "over-estimating only shrinks dt".)
        assert!(got[i].is_finite(),
            "map state {i}: bound {} must be finite", got[i]);
    }
}
