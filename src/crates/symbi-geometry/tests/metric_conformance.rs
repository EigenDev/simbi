// =============================================================================
// metric_conformance.rs
//
// a trait-wide self-consistency gate on the ADM surface of every Metric impl. the trait
// carries flat defaults for the spacetime-background methods (spacetime = Minkowski, lapse = 1,
// shift = 0, spacetime_scalars = empty) so a genuinely flat metric stays ergonomic. that default is
// a footgun for a curved metric: an unoverridden `lapse` silently bakes flat, gravity-free
// physics and reports success. rust leaves a supertrait's defaulted method inheritable by any
// subtrait, so this test is the guardrail: it asserts, for the whole realized metric set,
// that the background methods are mutually consistent —
//   - flat metrics: spacetime == Minkowski, lapse == 1, shift == 0, no spacetime scalars;
//   - curved metrics: spacetime != Minkowski, lapse != 1 (gravity wired), lapse_sq == lapse^2,
//     spacetime scalars present, and the radial shift matches the chart (0 for a static chart,
//     nonzero for a horizon-penetrating one).
// a curved metric that inherits any flat default trips the matching assertion here.
//
// (`volume_factor` is enforced at compile time — a required trait method carrying no default, so a
// metric that omits the proper reduced-dimension measure fails to build; the reduced-D semantic
// itself is spot-checked here.)
// =============================================================================

use symbi_algebra::Tensor;
use symbi_geometry::{
    Cartesian, Cylindrical, CylindricalRPhi, KerrKS, Metric, SchwarzschildKS,
    SchwarzschildKSCartesian, SchwarzschildKSCylindrical, Spacetime, Spherical,
};

const MASS: f64 = 1.0;
const SPIN: f64 = 0.5;

/// a flat metric: the background methods must all read their flat values, and `lapse_sq == lapse^2`.
fn assert_flat<const D: usize, M: Metric<f64, D>>(m: &M, x: Tensor<f64, D>, name: &str) {
    assert_eq!(
        m.spacetime(),
        Spacetime::Minkowski,
        "{name}: flat metric must report Minkowski"
    );
    assert_eq!(m.lapse(x), 1.0, "{name}: flat lapse must be 1");
    let beta = m.shift(x);
    for i in 0..D {
        assert_eq!(beta[i], 0.0, "{name}: flat shift component {i} must be 0");
    }
    assert!(
        m.spacetime_scalars().is_empty(),
        "{name}: a flat metric exposes no spacetime scalars"
    );
    assert!(
        (m.lapse_sq(x) - m.lapse(x) * m.lapse(x)).abs() < 1e-12,
        "{name}: lapse_sq must equal lapse^2"
    );
}

/// a curved metric: the background must be internally consistent — non-Minkowski tag, a lapse that
/// actually differs from 1 (proof the gravity is wired), `lapse_sq == lapse^2`,
/// scalar params present, and a radial shift matching the chart (`static_chart` = zero shift).
fn assert_curved<const D: usize, M: Metric<f64, D>>(
    m: &M,
    x: Tensor<f64, D>,
    name: &str,
    static_chart: bool,
) {
    assert_ne!(
        m.spacetime(),
        Spacetime::Minkowski,
        "{name}: curved metric must not report Minkowski"
    );
    assert!(
        !m.spacetime_scalars().is_empty(),
        "{name}: curved metric must expose its scalar params"
    );
    assert!(
        (m.lapse(x) - 1.0).abs() > 1e-6,
        "{name}: curved lapse must differ from 1 (else the gravity default leaked): got {}",
        m.lapse(x)
    );
    assert!(
        (m.lapse_sq(x) - m.lapse(x) * m.lapse(x)).abs() < 1e-9,
        "{name}: lapse_sq ({}) must equal lapse^2 ({})",
        m.lapse_sq(x),
        m.lapse(x) * m.lapse(x)
    );
    let beta_r = m.shift(x)[0];
    if static_chart {
        assert_eq!(
            beta_r, 0.0,
            "{name}: a static chart must have zero radial shift"
        );
    } else {
        assert!(
            beta_r.abs() > 1e-6,
            "{name}: a horizon-penetrating chart must have a nonzero radial shift, got {beta_r}"
        );
    }
}

#[test]
fn flat_metrics_report_flat_background() {
    // cartesian
    assert_flat(&Cartesian, Tensor::new([5.0]), "Cartesian 1D");
    assert_flat(&Cartesian, Tensor::new([5.0, 3.0]), "Cartesian 2D");
    assert_flat(&Cartesian, Tensor::new([5.0, 3.0, 2.0]), "Cartesian 3D");
    // spherical (flat spacetime, curved spatial geometry — still Minkowski, lapse 1)
    assert_flat(&Spherical, Tensor::new([5.0]), "Spherical 1D");
    assert_flat(&Spherical, Tensor::new([5.0, 1.0]), "Spherical 2D");
    assert_flat(&Spherical, Tensor::new([5.0, 1.0, 0.5]), "Spherical 3D");
    // cylindrical
    assert_flat(&Cylindrical, Tensor::new([5.0]), "Cylindrical 1D");
    assert_flat(&Cylindrical, Tensor::new([5.0, 2.0]), "Cylindrical 2D");
    assert_flat(&Cylindrical, Tensor::new([5.0, 0.5, 2.0]), "Cylindrical 3D");
    assert_flat(
        &CylindricalRPhi,
        Tensor::new([5.0, 0.5]),
        "CylindricalRPhi 2D",
    );
}

#[test]
fn curved_metrics_have_consistent_background() {
    // Schwarzschild-KS (ingoing, horizon-penetrating): nonzero radial shift.
    let ks = SchwarzschildKS { mass: MASS };
    assert_curved(&ks, Tensor::new([5.0]), "SchwarzschildKS 1D", false);
    assert_curved(&ks, Tensor::new([5.0, 1.0]), "SchwarzschildKS 2D", false);
    assert_curved(
        &ks,
        Tensor::new([5.0, 1.0, 0.5]),
        "SchwarzschildKS 3D",
        false,
    );

    // cartesian KS (2D / 3D real; 1D is a degenerate stub, excluded): nonzero shift along x_i.
    let ksc = SchwarzschildKSCartesian { mass: MASS };
    assert_curved(
        &ksc,
        Tensor::new([3.0, 4.0]),
        "SchwarzschildKSCartesian 2D",
        false,
    );
    assert_curved(
        &ksc,
        Tensor::new([3.0, 4.0, 0.0]),
        "SchwarzschildKSCartesian 3D",
        false,
    );

    // cylindrical KS: 2D equatorial disk (R, phi) + 3D (R, phi, z), both horizon-penetrating.
    let ksy = SchwarzschildKSCylindrical { mass: MASS };
    assert_curved(
        &ksy,
        Tensor::new([5.0, 0.5]),
        "SchwarzschildKSCylindrical 2D",
        false,
    );
    assert_curved(
        &ksy,
        Tensor::new([5.0, 0.5, 2.0]),
        "SchwarzschildKSCylindrical 3D",
        false,
    );

    // spinning Kerr-KS (2D (r, theta) grid view + 3D full; both carry real lapse/shift).
    let kerr = KerrKS {
        mass: MASS,
        spin: SPIN,
    };
    assert_curved(&kerr, Tensor::new([5.0, 1.0]), "KerrKS 2D", false);
    assert_curved(&kerr, Tensor::new([5.0, 1.0, 0.5]), "KerrKS 3D", false);
}

#[test]
fn reduced_dimension_volume_factor_is_the_proper_measure() {
    // the required `volume_factor` must be the proper measure including suppressed angular
    // directions. the naive `sqrt_det_gamma` of the reduced spatial block drops those directions
    // and is wrong. spherical 1D: r^2 (naive would give 1); spherical 2D: r^2 sin(theta) (naive: r);
    // cylindrical 1D/2D: r (naive: 1).
    let x1 = Tensor::new([5.0]);
    assert_eq!(
        <Spherical as Metric<f64, 1>>::volume_factor(&Spherical, x1),
        25.0
    );
    assert_ne!(
        <Spherical as Metric<f64, 1>>::volume_factor(&Spherical, x1),
        <Spherical as Metric<f64, 1>>::sqrt_det_gamma(&Spherical, x1),
        "reduced-D volume_factor must differ from the naive sqrt_det_gamma default"
    );
    let x2 = Tensor::new([5.0, 1.0]);
    let vf2 = <Spherical as Metric<f64, 2>>::volume_factor(&Spherical, x2);
    assert!(
        (vf2 - 25.0 * (1.0_f64).sin()).abs() < 1e-12,
        "spherical 2D volume = r^2 sin(theta)"
    );
    assert_eq!(
        <Cylindrical as Metric<f64, 1>>::volume_factor(&Cylindrical, x1),
        5.0
    );
}
