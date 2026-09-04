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
}
