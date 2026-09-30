// =============================================================================
// meridional_metric.rs
//
// schwarzschild in cylindrical kerr-schild coordinates on the meridional (R, z)
// plane, against the closed forms at off-equatorial points: r = sqrt(R^2 + z^2),
// 2H = 2M/r, alpha = 1/sqrt(1 + 2H), beta = (2H/(1 + 2H)) (R, z)/r,
// gamma_ab = delta_ab + (2H/r^2) x_a x_b over (R, z), sqrt(gamma) = R sqrt(1 + 2H)
// with the azimuthal scale R inside it. the (R, z) block of the inverse is the
// inverse of the block (phi decouples). M = 0 is the flat (r, z) chart. the
// equatorial (R, phi) chart differs from it off the midplane, so the two planes
// are distinct types.
// =============================================================================
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Metric, SchwarzschildKSCylindrical, SchwarzschildKSCylindricalRz};

const M: f64 = 0.8;
const POINTS: [[f64; 2]; 4] = [[3.0, 0.0], [2.5, 1.5], [1.2, -4.0], [6.0, 2.0]];

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 4.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn every_quantity_matches_the_closed_form_off_the_equator() {
    let g = SchwarzschildKSCylindricalRz { mass: M };
    for [big_r, z] in POINTS {
        let x = Tensor::new([big_r, z]);
        let r = (big_r * big_r + z * z).sqrt();
        let two_h = 2.0 * M / r;
        assert!(close(g.lapse(x), 1.0 / (1.0 + two_h).sqrt()), "lapse at ({big_r}, {z})");
        let beta = g.shift(x);
        assert!(close(beta[0], two_h / (1.0 + two_h) * big_r / r), "beta^R at ({big_r}, {z})");
        assert!(close(beta[1], two_h / (1.0 + two_h) * z / r), "beta^z at ({big_r}, {z})");
        let gm = g.spatial_metric(x);
        let l = [big_r, z];
        for a in 0..2 {
            for b in 0..2 {
                let want = f64::from(u8::from(a == b)) + two_h / (r * r) * l[a] * l[b];
                assert!(close(gm[(a, b)], want), "gamma_{a}{b} at ({big_r}, {z})");
            }
        }
        // the block inverse: gamma^{-1} gamma = identity on (R, z)
        let gi = g.spatial_metric_inv(x);
        for a in 0..2 {
            for b in 0..2 {
                let prod: f64 = (0..2).map(|c| gi[(a, c)] * gm[(c, b)]).sum();
                assert!((prod - f64::from(u8::from(a == b))).abs() <= 1e-14, "(gamma^-1 gamma)_{a}{b} at ({big_r}, {z})");
            }
        }
        assert!(close(g.volume_factor(x), big_r * (1.0 + two_h).sqrt()), "sqrt(gamma) at ({big_r}, {z})");
    }
}

#[test]
fn a_vanishing_mass_is_the_flat_meridional_chart() {
    let g = SchwarzschildKSCylindricalRz { mass: 0.0 };
    for [big_r, z] in POINTS {
        let x = Tensor::new([big_r, z]);
        assert_eq!(g.lapse(x), 1.0);
        assert_eq!(g.volume_factor(x), <Cylindrical as Metric<f64, 2>>::volume_factor(&Cylindrical, x));
    }
}

#[test]
fn the_equatorial_chart_differs_off_the_midplane() {
    let meridional = SchwarzschildKSCylindricalRz { mass: M };
    let equatorial = SchwarzschildKSCylindrical { mass: M };
    let x = Tensor::new([2.5, 1.5]);
    let a = meridional.volume_factor(x);
    let b = <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::volume_factor(&equatorial, x);
    assert!((a - b).abs() > 1e-3 * a, "the two planes agree off the midplane: {a} vs {b}");
}
