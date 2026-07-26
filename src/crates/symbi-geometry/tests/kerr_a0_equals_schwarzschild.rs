// =============================================================================
// kerr_a0_equals_schwarzschild.rs
//
// at zero spin the kerr-schild kerr chart IS the schwarzschild kerr-schild
// chart: the oblate-spheroidal radius solve
//   r^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 + a^2 z^2)
// collapses to r = R, the null covector l^i = (r x + a y, r y - a x, z r)/(r^2 + a^2)
// collapses to x^i / r, and 2H = 2 M r^3 / (r^4 + a^2 z^2) collapses to 2M/r.
// so every metric quantity from the two implementations must agree to roundoff.
//
// this is a LAW, not a sampled comparison: the two are separate code paths that
// carry separate baked kernels (Spacetime::KerrKS vs Spacetime::SchwarzschildKS), so a
// divergence here is silent — both paths keep running and merely disagree about
// the same spacetime.
//
// run: cargo test -p symbi-geometry --test kerr_a0_equals_schwarzschild
// =============================================================================

use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCartesian, Metric, SchwarzschildKSCartesian};

const M: f64 = 1.0;

/// sample points spanning the exterior, the near-horizon region, and inside the
/// r < M/2 clamp both implementations apply.
fn points() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for &x in &[1.0_f64, 2.5, 6.0, -4.0] {
        for &y in &[0.0_f64, 0.7, -3.0] {
            for &z in &[0.0_f64, 1.3, -5.0] {
                v.push([x, y, z]);
            }
        }
    }
    // INSIDE the r < M/2 radius floor as well: both charts contract the rank-1 forms with the
    // ACTUAL |l|^2, which the floor drives below 1, so the two continue the clamped core
    // identically rather than diverging. sampling the floor is the point — the a = 0 chart once
    // assumed the unit-l forms there and disagreed with the spinning chart by ~19% in the lapse,
    // decaying outward about two decades per shell into the region the accretion gates read.
    for &s in &[0.45_f64, 0.3, 0.12, 0.02] {
        v.push([s, 0.0, 0.0]);
        v.push([0.0, -s, 0.0]);
        v.push([0.0, 0.0, s]);
        let t = s / 3.0_f64.sqrt();
        v.push([t, -t, t]);
    }
    v
}

fn close(a: f64, b: f64, what: &str, p: [f64; 3]) {
    let tol = 1e-12 * a.abs().max(b.abs()).max(1.0);
    assert!(
        (a - b).abs() <= tol,
        "{what} disagrees at {p:?}: kerr(a=0) {a:.17e} vs schwarzschild {b:.17e} (diff {:.3e})",
        (a - b).abs()
    );
}

#[test]
fn lapse_shift_and_spatial_metric_agree_at_zero_spin() {
    let kerr = KerrKSCartesian { mass: M, spin: 0.0 };
    let schw = SchwarzschildKSCartesian { mass: M };
    for p in points() {
        let x = Tensor::<f64, 3>::new(p);
        close(
            <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse(&schw, x),
            "lapse",
            p,
        );
        close(
            <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse_sq(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse_sq(&schw, x),
            "lapse_sq",
            p,
        );
        let (bk, bs) = (
            <KerrKSCartesian<f64> as Metric<f64, 3>>::shift(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::shift(&schw, x),
        );
        for ii in 0..3 {
            close(bk[ii], bs[ii], &format!("shift[{ii}]"), p);
        }
        let (gk, gs) = (
            <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&schw, x),
        );
        let (ik, is) = (
            <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric_inv(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric_inv(&schw, x),
        );
        for ii in 0..3 {
            for jj in 0..3 {
                close(gk[(ii, jj)], gs[(ii, jj)], &format!("gamma[{ii}{jj}]"), p);
                close(
                    ik[(ii, jj)],
                    is[(ii, jj)],
                    &format!("gamma_inv[{ii}{jj}]"),
                    p,
                );
            }
        }
        close(
            <KerrKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&kerr, x),
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&schw, x),
            "sqrt_det_gamma",
            p,
        );
    }
}
