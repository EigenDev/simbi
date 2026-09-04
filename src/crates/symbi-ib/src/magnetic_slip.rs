// =============================================================================
// magnetic_slip.rs
//
// the ambipolar magnetic-slip constitutive tensor, carrier-generic:
//   A(B) = coeff (|B|^2 I - B B),   coeff = a_B chi_B >= 0,
// applied to a vector z as
//   A(B) z = coeff (|B|^2 z - (B . z) B).
// with z = J = curl B this is the magnetic-slip electric field E_B, the object a
// constrained-transport step places on edges. the tensor is the perpendicular
// projector onto the plane normal to B, scaled by coeff |B|^2, so it is symmetric,
// positive semidefinite, annihilates B (the force-free null), and dissipates at the
// rate J . A(B) J = coeff |J x B|^2.
//
// coeff folds the slip coefficient a_B and the shell mask chi_B into one nonnegative
// scalar. the closure a_B(|B|^2) and the shell mask chi_B(phi) are formed by the
// caller; this operator is that closure's convention-independent tensor action.
//
// carrier-generic: f64 for the reference and property laws, Gv for the traced edge
// kernel, Dual for sensitivities. one expression serves all three.
//
// usage:
//   let e_b = slip_apply(a_b * chi_b, &b, &j); // the slip electric field E_B
// =============================================================================

use symbi_algebra::Tensor;
use symbi_carrier::Scalar;

/// apply the magnetic-slip tensor `A(B) = coeff (|B|^2 I - B B)` to a vector `z`,
/// returning `coeff (|B|^2 z - (B . z) B)`. `coeff = a_B chi_B >= 0` is the slip
/// coefficient times the shell mask. the result is `coeff |B|^2` times the component
/// of `z` perpendicular to `B`: a current parallel to `B`, a zero coefficient, or a
/// vanishing field each map to the zero vector.
pub fn slip_apply<S: Scalar, const D: usize>(
    coeff: S,
    b: &Tensor<S, D>,
    z: &Tensor<S, D>,
) -> Tensor<S, D> {
    let b2 = b.dot(b);
    let bz = b.dot(z);
    z.scale(coeff * b2) - b.scale(coeff * bz)
}

/// the ambipolar slip coefficient in SIMBI code units,
///   a_B = ell_B^2 / ((|B|^2 + B_0^2) D_B tau_rho),
/// closing the magnetic model on the body's own drain time `tau_rho` through the magnetic
/// Damkohler number `D_B`, with transport length `ell_B` (= slip_length_ratio * mollification
/// width) and null regularizer `B_0`. code units normalize the field by `B = B_G / sqrt(4 pi)`,
/// so `J = curl B` carries the current the Gaussian closure writes as `curl B_G / (4 pi)` and
/// the magnetic energy is `B^2 / 2`. the Gaussian `4 pi` cancels between the coefficient and
/// `B_reg^2`, leaving this factor-free form. `B_0 > 0` keeps `a_B` finite and positive at
/// magnetic nulls.
pub fn slip_coefficient<S: Scalar>(ell_b: S, b2: S, b0: S, d_b: S, tau_rho: S) -> S {
    ell_b * ell_b / ((b2 + b0 * b0) * d_b * tau_rho)
}

/// the magnetic-shell mask `chi_B = 4 chi(phi_B) [1 - chi(phi_B)]`, a bump that vanishes in
/// the resolved exterior and the deep interior and peaks where unresolved flux-matter
/// decoupling occurs. `chi` is the mollified indicator of the signed distance, of width `w`,
/// shifted by the placement,
///   phi_B = phi - placement * w,   chi(phi_B) = 1/2 (1 - tanh(phi_B / w)),
/// so the shell centers at `phi = placement * w`: inside the mass surface for `placement < 0`,
/// symmetrically across it for `0`, outside for `> 0`. `w` is the mollification width alone —
/// the transport length ell_B that scales the coefficient is a separate role. formed from the
/// signed distance directly, so which side is interior is carried by `phi`, not inferred from
/// an already-sampled mask. `placement` is dimensionless (widths of `w`), so the shell is
/// resolution-independent when `w` is physical.
pub fn chi_shell<S: Scalar>(phi: S, w: S, placement: S) -> S {
    let phi_b = phi - placement * w;
    let c = crate::sdf::chi(phi_b, w);
    S::from_f64(4.0) * c * (S::ONE - c)
}

#[cfg(test)]
mod tests {
    use super::*;

    // an independent cross product, so the dissipation identity is verified against
    // the geometric |J x B|, not re-derived from the tensor under test.
    fn cross(a: &Tensor<f64, 3>, b: &Tensor<f64, 3>) -> Tensor<f64, 3> {
        Tensor::new([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }

    fn sample_fields() -> Vec<(Tensor<f64, 3>, Tensor<f64, 3>)> {
        let bs = [
            Tensor::new([1.0, 0.0, 0.0]),
            Tensor::new([0.3, -1.2, 0.7]),
            Tensor::new([-2.0, 0.5, 1.5]),
            Tensor::new([0.0, 0.0, 2.4]),
        ];
        let zs = [
            Tensor::new([0.0, 1.0, 0.0]),
            Tensor::new([0.9, 0.2, -0.4]),
            Tensor::new([-1.1, -0.6, 0.8]),
            Tensor::new([2.0, -2.0, 2.0]),
        ];
        let mut out = Vec::new();
        for b in bs {
            for z in zs {
                out.push((b, z));
            }
        }
        out
    }

    // A(B) is symmetric: the bilinear form z1 . A(B) z2 is invariant under swapping
    // the arguments, exercised over a lattice of field/vector pairs.
    #[test]
    fn slip_tensor_is_symmetric() {
        let coeff = 0.37_f64;
        let vecs = [
            Tensor::new([1.0, 0.0, 0.0]),
            Tensor::new([0.4, 0.9, -0.2]),
            Tensor::new([-0.7, 0.3, 1.1]),
        ];
        for b in [
            Tensor::new([0.5, -1.0, 0.8]),
            Tensor::new([2.0, 0.0, -0.3]),
        ] {
            for z1 in vecs {
                for z2 in vecs {
                    let a = z1.dot(&slip_apply(coeff, &b, &z2));
                    let at = z2.dot(&slip_apply(coeff, &b, &z1));
                    assert!((a - at).abs() < 1e-13, "asymmetric: {a} vs {at}");
                }
            }
        }
    }

    // A(B) is positive semidefinite for coeff >= 0: z . A(B) z = coeff(|B|^2|z|^2 -
    // (B.z)^2) >= 0 by Cauchy-Schwarz, so the slip channel never adds magnetic energy.
    #[test]
    fn slip_tensor_is_positive_semidefinite() {
        for coeff in [0.0, 0.6, 4.2] {
            for (b, z) in sample_fields() {
                let q = z.dot(&slip_apply(coeff, &b, &z));
                assert!(q >= -1e-13, "negative quadratic form {q} at coeff {coeff}");
            }
        }
    }

    // A(B) B = 0: the magnetic field is the force-free null direction, left exactly
    // untouched by the slip channel.
    #[test]
    fn slip_tensor_annihilates_the_field() {
        let coeff = 1.3;
        for (b, _) in sample_fields() {
            let ab = slip_apply(coeff, &b, &b);
            assert!(
                ab.dot(&ab).sqrt() < 1e-13,
                "A(B) B is nonzero: {:?}",
                ab
            );
        }
    }

    // a current parallel to B (J = c B) carries J x B = 0 and produces no slip field:
    // the parallel-current null of ambipolar diffusion.
    #[test]
    fn parallel_current_produces_no_field() {
        let coeff = 0.8;
        for (b, _) in sample_fields() {
            for c in [-1.7, 0.0, 2.5] {
                let j = b.scale(c);
                let e = slip_apply(coeff, &b, &j);
                assert!(e.dot(&e).sqrt() < 1e-12, "parallel current dissipated: {:?}", e);
            }
        }
    }

    // the dissipation identity J . A(B) J = coeff |J x B|^2, checked against an
    // independent cross product. this is the nonnegative local heating Q_B.
    #[test]
    fn dissipation_equals_the_perpendicular_current_norm() {
        for coeff in [0.0, 0.9, 3.1] {
            for (b, j) in sample_fields() {
                let q = j.dot(&slip_apply(coeff, &b, &j));
                let jxb = cross(&j, &b);
                let expect = coeff * jxb.dot(&jxb);
                assert!(
                    (q - expect).abs() < 1e-12,
                    "Q_B mismatch: {q} vs coeff|JxB|^2 = {expect}"
                );
                assert!(q >= -1e-13, "heating is nonnegative: {q}");
            }
        }
    }

    // zero coefficient is an exact off switch: A(B) z = 0 for every field and vector.
    #[test]
    fn zero_coefficient_is_an_exact_off_switch() {
        for (b, z) in sample_fields() {
            let e = slip_apply(0.0, &b, &z);
            assert_eq!(e[0], 0.0);
            assert_eq!(e[1], 0.0);
            assert_eq!(e[2], 0.0);
        }
    }

    // rotational covariance: A(R B)(R z) = R (A(B) z) for a rotation R, so the operator
    // is coordinate-frame independent in cartesian space (isotropic constitutive law).
    #[test]
    fn slip_tensor_is_rotationally_covariant() {
        // a rotation by theta about the axis (1,1,1)/sqrt(3) (Rodrigues), a generic
        // orientation misaligned with every sample field.
        let theta = 0.7_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let k = Tensor::new([1.0, 1.0, 1.0]).scale(1.0 / 3.0_f64.sqrt());
        let rot = |v: &Tensor<f64, 3>| -> Tensor<f64, 3> {
            // v cos + (k x v) sin + k (k.v)(1 - cos)
            let kxv = cross(&k, v);
            v.scale(c) + kxv.scale(s) + k.scale(k.dot(v) * (1.0 - c))
        };
        let coeff = 1.1;
        for (b, z) in sample_fields() {
            let lhs = slip_apply(coeff, &rot(&b), &rot(&z));
            let rhs = rot(&slip_apply(coeff, &b, &z));
            for a in 0..3 {
                assert!((lhs[a] - rhs[a]).abs() < 1e-12, "axis {a}: {} vs {}", lhs[a], rhs[a]);
            }
        }
    }

    // the tensor factor vanishes continuously as |B| -> 0 at fixed coeff: no division
    // by the field strength, so the slip field is finite (zero) at a magnetic null.
    // the coefficient's own boundedness at nulls (through B_reg) is a coefficient-closure
    // property, verified where a_B is derived.
    #[test]
    fn slip_field_vanishes_finitely_at_a_null() {
        let coeff = 2.0;
        let z = Tensor::new([0.6, -0.3, 0.9]);
        let dir = Tensor::new([0.5, 0.5, 0.5]).scale(1.0 / (0.75_f64).sqrt());
        let mut prev = f64::INFINITY;
        for scale in [1.0, 1e-2, 1e-4, 1e-8, 0.0] {
            let b = dir.scale(scale);
            let e = slip_apply(coeff, &b, &z);
            let mag = e.dot(&e).sqrt();
            assert!(mag.is_finite(), "non-finite slip field at |B| = {scale}");
            assert!(mag <= prev + 1e-15, "slip field did not shrink toward the null");
            prev = mag;
        }
        assert_eq!(prev, 0.0, "the slip field is exactly zero at |B| = 0");
    }

    // the coefficient and EMF describe the same physics whether evaluated in Gaussian units
    // (a_G, B_G, J_G with explicit 4 pi) or SIMBI code units (a_B, B, J). the coefficient is
    // invariant (the 4 pi cancels), the EMF transforms like the field (E_G = sqrt(4 pi) E),
    // and the dissipation is a physical scalar identical in both systems.
    #[test]
    fn coefficient_and_emf_match_gaussian_and_code_units() {
        let four_pi = 4.0 * std::f64::consts::PI;
        let root = four_pi.sqrt();
        // ell_b is the transport length (slip_length_ratio * w), the length that enters a_B.
        let (ell_b, b0, d_b, tau) = (0.05_f64, 0.1, 2.0, 0.3);

        // a physical state in code units: B = B_G/sqrt(4 pi), J = curl B.
        let b = Tensor::new([0.4, -0.9, 0.5]);
        let j = Tensor::new([0.7, 0.2, -0.6]);
        let a_code = slip_coefficient(ell_b, b.dot(&b), b0, d_b, tau);
        let e_code = slip_apply(a_code, &b, &j);

        // the same state in Gaussian units: B_G = sqrt(4 pi) B, J_G = J/sqrt(4 pi),
        // B_0G = sqrt(4 pi) B_0, and a_G carries the explicit 4 pi.
        let bg = b.scale(root);
        let jg = j.scale(1.0 / root);
        let b0g = b0 * root;
        let a_gauss = four_pi * ell_b * ell_b / ((bg.dot(&bg) + b0g * b0g) * d_b * tau);
        let e_gauss = slip_apply(a_gauss, &bg, &jg);

        assert!(
            (a_gauss - a_code).abs() < 1e-12 * a_code,
            "coefficient not invariant: a_G {a_gauss} vs a_B {a_code}"
        );
        for k in 0..3 {
            assert!(
                (e_gauss[k] - root * e_code[k]).abs() < 1e-12,
                "axis {k}: E_G {} vs sqrt(4pi) E {}",
                e_gauss[k],
                root * e_code[k]
            );
        }
        let (q_code, q_gauss) = (j.dot(&e_code), jg.dot(&e_gauss));
        assert!(
            (q_gauss - q_code).abs() < 1e-12,
            "dissipation not invariant: Q_G {q_gauss} vs Q {q_code}"
        );
    }

    // the shell mask peaks at phi = placement*ell_B, stays in [0, 1], and vanishes many widths
    // to either side of the shell.
    #[test]
    fn shell_mask_peaks_at_the_placement_and_stays_bounded() {
        let w = 0.1_f64; // the mollification width
        assert!((chi_shell(0.0, w, 0.0) - 1.0).abs() < 1e-12, "symmetric peak is 1 at phi = 0");
        assert!(chi_shell(-20.0 * w, w, 0.0) < 1e-6, "vanishes deep inside");
        assert!(chi_shell(20.0 * w, w, 0.0) < 1e-6, "vanishes far outside");
        for i in -60..=60 {
            let v = chi_shell(i as f64 * w * 0.2, w, 0.0);
            assert!(v >= -1e-15 && v <= 1.0 + 1e-12, "chi_B out of [0,1]: {v}");
        }
    }

    // placement shifts the shell along the signed distance and preserves orientation: the peak
    // tracks phi = placement*ell_B, and opposite placement signs center on opposite sides, so
    // the mask distinguishes inside from outside rather than collapsing to a symmetric bump.
    #[test]
    fn placement_shifts_the_shell_and_preserves_orientation() {
        let w = 0.1_f64;
        for placement in [-1.5_f64, 0.0, 1.5] {
            let peak = chi_shell(placement * w, w, placement);
            assert!((peak - 1.0).abs() < 1e-12, "placement {placement}: peak not at placement*w");
        }
        // an inside placement peaks at phi < 0; sampling that same phi under an outside placement
        // is off-peak. the sign of placement is a real orientation, not a symmetry.
        let inside_peak = chi_shell(-1.5 * w, w, -1.5);
        let outside_at_inside_phi = chi_shell(-1.5 * w, w, 1.5);
        assert!(inside_peak > 0.99, "inside placement peaks inside");
        assert!(
            outside_at_inside_phi < 0.5,
            "placement sign failed to distinguish inside from outside: {outside_at_inside_phi}"
        );
    }
}
