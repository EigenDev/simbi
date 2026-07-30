// =============================================================================
// kerr_ks_coordinate_map.rs
//
// verifies the oblate spherical-to-cartesian coordinate map used by the
// spinning ingoing kerr-schild metric.
//
// usage:
//  cargo test -p symbi-geometry --test kerr_ks_coordinate_map
// =============================================================================

use std::f64::consts::{FRAC_PI_2, PI};

use symbi_algebra::Tensor;
use symbi_geometry::{KerrKS, Metric};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12 * left.abs().max(right.abs()).max(1.0)
}

fn angle_distance(left: f64, right: f64) -> f64 {
    let delta = (left - right).rem_euclid(2.0 * PI);
    delta.min(2.0 * PI - delta)
}

#[test]
fn equatorial_radius_is_oblate() {
    let metric = KerrKS {
        mass: 1.0,
        spin: 0.8,
    };
    let r = 3.0;
    let cart = metric.to_cartesian(Tensor::new([r, FRAC_PI_2, 0.0]));
    let cylindrical_radius = (cart[0] * cart[0] + cart[1] * cart[1]).sqrt();

    assert!(close(cylindrical_radius, (r * r + 0.8_f64.powi(2)).sqrt()));
    assert!(close(cart[2], 0.0));
}

#[test]
fn oblate_coordinate_map_round_trips() {
    let metric = KerrKS {
        mass: 1.0,
        spin: -0.7,
    };

    for point in [
        Tensor::new([0.8, 0.4, -2.1]),
        Tensor::new([2.0, 1.1, 0.3]),
        Tensor::new([7.0, 2.4, 2.8]),
    ] {
        let recovered = metric.from_cartesian(metric.to_cartesian(point));
        assert!(close(recovered[0], point[0]));
        assert!(close(recovered[1], point[1]));
        assert!(angle_distance(recovered[2], point[2]) <= 1.0e-12);
    }
}

#[test]
fn zero_spin_map_is_spherical() {
    let metric = KerrKS {
        mass: 1.0,
        spin: 0.0,
    };
    let point = Tensor::new([2.5, 1.2, -0.9]);
    let cart = metric.to_cartesian(point);

    assert!(close(cart[0], point[0] * point[1].sin() * point[2].cos()));
    assert!(close(cart[1], point[0] * point[1].sin() * point[2].sin()));
    assert!(close(cart[2], point[0] * point[1].cos()));
}
