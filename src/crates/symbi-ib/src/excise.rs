// =============================================================================
// excise.rs
//
// the horizon-excision predicate: the sublevel set of the kerr-schild radius,
// r_ks(x; a) < r_exc — the sphere |x| < r_exc at a = 0, and the oblate spheroid
// (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1 at spin a about z (the r = const
// surfaces of the cartesian kerr-schild chart). the mask is a carrier-generic
// select, compared in the square so the radius enters only through r_ks^2.
//
// the state an excised cell carries is chosen by the fill that consumes this
// mask: a cold vacuum floor, so the exterior gas rarefies into the excised
// region and stays there — the absorbing boundary a horizon is. a zero-gradient
// outward copy would instead impose a transmissive outflow, and on the
// staircased cartesian surface the per-axis sweep speeds carry mixed signs, so
// a transmissive interior leaks back into the exterior flux.
//
// usage:
//   let excised = ks_excised(&x_c, spin, r_exc);
// =============================================================================

use symbi_ir::algebra::Scalar;

/// the excision predicate: kerr-schild radius r_ks(x; a) < r_exc, as a select
/// mask. r_ks solves the oblate-spheroidal quartic,
///   r_ks^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 + a^2 z^2),  R^2 = |x|^2,
/// with the missing axes of a D < 3 position reading z = 0 (the equatorial
/// slice, where the excised disc has coordinate radius sqrt(r_exc^2 + a^2)).
/// compared in the square (both sides non-negative), so the radius enters the
/// mask only through r_ks^2. a = 0 reduces to |x|^2 < r_exc^2 exactly.
pub fn ks_excised<S: Scalar, const D: usize>(x_c: &[S; D], spin: S, r_exc: S) -> S::Mask {
    let z = if D > 2 { x_c[2] } else { S::ZERO };
    let mut rr2 = S::ZERO;
    for kk in 0..D {
        rr2 = rr2 + x_c[kk] * x_c[kk];
    }
    let half = S::HALF;
    let d = half * (rr2 - spin * spin);
    let az = spin * z;
    let r_ks2 = d + (d * d + az * az).sqrt();
    r_ks2.cmp_lt(r_exc * r_exc)
}

#[cfg(test)]
mod predicate_tests {
    use super::*;

    // the f64 scalar's mask type is bool.
    fn is_excised_3d(x: [f64; 3], a: f64, r: f64) -> bool {
        ks_excised(&x, a, r)
    }

    #[test]
    fn zero_spin_is_the_sphere() {
        let r = 1.4;
        for &(x, y, z) in &[
            (0.5_f64, 0.5, 0.5),
            (1.0, 0.9, 0.1),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.39),
        ] {
            let want = (x * x + y * y + z * z).sqrt() < r;
            assert_eq!(is_excised_3d([x, y, z], 0.0, r), want, "at ({x},{y},{z})");
        }
    }

    #[test]
    fn spinning_region_is_the_oblate_spheroid() {
        // r_ks < r_exc is exactly (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1.
        let (a, r) = (0.9, 1.2);
        for &(x, y, z) in &[
            (1.3_f64, 0.0, 0.0),
            (0.0, 1.45, 0.0),
            (0.0, 0.0, 1.1),
            (0.0, 0.0, 1.3),
            (1.0, 1.0, 0.3),
            (0.7, 0.7, 0.8),
        ] {
            let want = (x * x + y * y) / (r * r + a * a) + (z * z) / (r * r) < 1.0;
            assert_eq!(is_excised_3d([x, y, z], a, r), want, "at ({x},{y},{z})");
        }
    }

    #[test]
    fn equatorial_slice_is_the_widened_disc() {
        // the D = 2 predicate (z = 0) excises the coordinate disc R < sqrt(r_exc^2 + a^2).
        let (a, r) = (0.9_f64, 1.2_f64);
        let disc = (r * r + a * a).sqrt();
        for &rr in &[0.5, disc - 1e-9, disc + 1e-9, 2.0] {
            let x = [rr / 2.0_f64.sqrt(), rr / 2.0_f64.sqrt()];
            let got = ks_excised(&x, a, r);
            assert_eq!(got, rr < disc, "coordinate radius {rr}");
        }
    }

    #[test]
    fn outward_diagonal_step_leaves_the_region_monotonically() {
        // the excised region is star-shaped under axis-outward steps: moving by
        // (sign x, sign y, sign z) * dx strictly increases r_ks, so a walk that has left the
        // spheroid stays outside it.
        let (a, r, dx) = (0.9, 1.2, 0.05);
        let mut x = [0.02_f64, -0.03, 0.01];
        let mut inside = true;
        for _ in 0..200 {
            let step: [f64; 3] = std::array::from_fn(|kk| x[kk].signum() * dx);
            let next: [f64; 3] = std::array::from_fn(|kk| x[kk] + step[kk]);
            let was = is_excised_3d(x, a, r);
            let now = is_excised_3d(next, a, r);
            assert!(
                !(now && !was),
                "outward step re-entered the excised region at {next:?}"
            );
            inside = now;
            x = next;
        }
        assert!(!inside, "200 outward steps never left the spheroid");
    }
}
