// =============================================================================
// kerr_schild_unit_determinant.rs
//
// the cartesian kerr-schild charts have unit four-volume: with gamma_ij = delta_ij + 2H l_i l_j
// and |l| = 1, the rank-1 determinant is det(gamma) = 1 + 2H while the lapse is
// alpha = 1/sqrt(1 + 2H), so
//
//   sqrt(-g) = alpha sqrt(det gamma) = 1
//
// identically, at every point and every spin. that identity is what makes a cartesian kerr-schild
// chart the easy case for a densitized conservation law d_t U + d_j F^j = S: the measure the state
// and the flux carry is 1, so the pressure block of the connection source,
// (1/2) p g^{ab} d_i g_ab = p d_i ln sqrt(-g), vanishes and the geometry contributes no spurious
// force anywhere.
//
// the identity is a consequence of |l| = 1, so it holds only where the null condition does. a
// radius floor that clamps r while keeping l^i = x^i / r leaves |l| = |x| / r < 1, and the
// determinant and the lapse then come from different assumptions: the four-volume drifts away from
// 1 and the pressure block turns on inside the floor. this test states the identity over the whole
// chart, floor included, because the densitized scheme reads sqrt(-g) there too.
//
// run: cargo test -p symbi-geometry --test kerr_schild_unit_determinant
// =============================================================================

use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCartesian, Metric, SchwarzschildKSCartesian};

const M: f64 = 1.0;

/// sample points from far outside the horizon down through the radius floor toward the origin.
fn points() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for &s in &[8.0_f64, 3.0, 1.4, 0.9, 0.6, 0.45, 0.3, 0.15, 0.05] {
        v.push([s, 0.0, 0.0]);
        v.push([0.0, s, 0.0]);
        v.push([0.0, 0.0, s]);
        let t = s / 3.0_f64.sqrt();
        v.push([t, t, t]);
        v.push([-t, t, -t]);
    }
    v
}

#[test]
fn schwarzschild_kerr_schild_cartesian_has_unit_four_volume() {
    let bh = SchwarzschildKSCartesian { mass: M };
    for p in points() {
        let x = Tensor::<f64, 3>::new(p);
        let alpha = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse(&bh, x);
        let vol = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::volume_factor(&bh, x);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!(
            (alpha * vol - 1.0).abs() < 1e-12,
            "sqrt(-g) = {:.17e} at r = {r:.4} {p:?}, not 1: the densitized measure and the \
             pressure block of the connection source both read this",
            alpha * vol
        );
    }
}

/// the closed-form measure must equal the determinant of the metric matrix the source contraction
/// actually differentiates. `sqrt_det_gamma` uses the rank-1 shortcut det = 1 + 2H, which assumes
/// |l| = 1; the matrix is delta_ij + 2H l_i l_j, whose determinant is 1 + 2H |l|^2. where a radius
/// floor clamps r but leaves l^i = x^i / r, those two part company, and the connection source's
/// pressure block (1/2) p g^{ab} d_i g_ab = p d_i ln sqrt(-g) — analytically zero on this chart —
/// turns into a spurious force wherever they do.
#[test]
fn the_closed_form_measure_equals_the_matrix_determinant() {
    for p in points() {
        let x = Tensor::<f64, 3>::new(p);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let bh = SchwarzschildKSCartesian { mass: M };
        let gm = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&bh, x);
        let closed = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&bh, x);
        assert!(
            (gm.det() - closed * closed).abs() < 1e-11 * gm.det().abs().max(1.0),
            "a = 0: det(gamma) = {:.17e} but sqrt_det_gamma^2 = {:.17e} at r = {r:.4} {p:?}",
            gm.det(),
            closed * closed
        );
        for &a in &[0.0_f64, 0.5, 0.9] {
            let kerr = KerrKSCartesian { mass: M, spin: a };
            let gk = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&kerr, x);
            let ck = <KerrKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&kerr, x);
            assert!(
                (gk.det() - ck * ck).abs() < 1e-11 * gk.det().abs().max(1.0),
                "spin {a}: det(gamma) = {:.17e} but sqrt_det_gamma^2 = {:.17e} at r = {r:.4} {p:?}",
                gk.det(),
                ck * ck
            );
        }
    }
}

/// the inverse inverts the matrix as stored, |l| included.
/// sherman-morrison on delta + 2H l l^T takes the measured |l|^2 in its coefficient; substituting 1
/// leaves a residual wherever the radius clamp has driven |l| below 1, and the recovered
/// contravariant velocity v^i = gamma^{ij} S_j then departs from the inverse of the lowering the
/// conserved momentum used.
#[test]
fn the_spatial_metric_inverse_inverts_the_spatial_metric() {
    let bh = SchwarzschildKSCartesian { mass: M };
    for p in points() {
        let x = Tensor::<f64, 3>::new(p);
        let gm = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&bh, x);
        let gi = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric_inv(&bh, x);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        for ii in 0..3 {
            for jj in 0..3 {
                let acc: f64 = (0..3).map(|kk| gm[(ii, kk)] * gi[(kk, jj)]).sum();
                let want = if ii == jj { 1.0 } else { 0.0 };
                assert!(
                    (acc - want).abs() < 1e-12,
                    "gamma gamma^-1 [{ii}{jj}] = {acc:.17e}, want {want} at r = {r:.4} {p:?}"
                );
            }
        }
    }
}

#[test]
fn spinning_kerr_schild_cartesian_has_unit_four_volume() {
    for &a in &[0.0_f64, 0.5, 0.9] {
        let bh = KerrKSCartesian { mass: M, spin: a };
        for p in points() {
            let x = Tensor::<f64, 3>::new(p);
            let alpha = <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse(&bh, x);
            let vol = <KerrKSCartesian<f64> as Metric<f64, 3>>::volume_factor(&bh, x);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (alpha * vol - 1.0).abs() < 1e-12,
                "sqrt(-g) = {:.17e} at spin {a}, r = {r:.4} {p:?}, not 1",
                alpha * vol
            );
        }
    }
}
