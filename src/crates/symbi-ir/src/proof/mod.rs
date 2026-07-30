// =============================================================================
// proof/mod.rs
//
// symbolic div(curl B) = 0 checker over the traced IR DAG. it EXTRACTS the exact
// symbolic linear combination of edge-emf field reads the curl produces and PROVES
// the constraint by polynomial-coefficient cancellation to the zero polynomial.
//
// the curl-of-an-edge-emf is LINEAR in the staggered field reads: the only
// nonlinearity is multiplication by uniform scalar params (dt, inverse widths).
// so a curl node lowers to a `LinForm` = a map from a field read `(key, offset)`
// to a `Poly` (a multivariate polynomial in the scalar-param names with integer
// coefficients). the divergence stencil is applied symbolically by SHIFTING the
// offsets; the contribution must vanish as the zero LinForm — that is the proof.
//
// the module splits by responsibility:
// - poly:    the coefficient ring (Poly, RatFun, FieldTerm, shift_poly_coords).
// - linform: the telescoping space (LinForm, LinFormR) and its pure transforms.
// - extract: the IR-DAG -> symbolic-form extraction (eval/eval_rat + the
//            LinForm::extract / LinFormR::extract_rat entry points).
//
// usage:
//  let (kernel, writes) = some_cartesian_curl_builder();
//  let lf = LinForm::extract(&kernel.graph, writes[0].2, &fields, &scalars);
//  // shift / combine LinForms per the divergence stencil, then:
//  assert!(combined.is_zero());
// =============================================================================

mod extract;
mod linform;
mod poly;

pub use extract::extract_scalar;
pub use linform::{LinForm, LinFormR};
pub use poly::{FieldTerm, Poly, RatFun};

#[cfg(test)]
mod tests {
    use super::linform::LinForm;
    use super::poly::{Poly, RatFun, shift_poly_coords};
    use std::collections::BTreeMap;

    #[test]
    fn poly_arithmetic_cancels() {
        // id_p2 * 1 - id_p2 * 1 = 0.
        let a = Poly::var("id_p2");
        let mut s = a.clone();
        s.add_assign(&a.neg());
        assert!(s.is_zero());
        // (dt * id_p1) is a degree-2 monomial, nonzero.
        let prod = Poly::var("dt").mul(&Poly::var("id_p1"));
        assert!(!prod.is_zero());
    }

    #[test]
    fn ratfun_arithmetic_and_zero() {
        // 1/r - 1/r = 0; 1/r + 1/r = 2/r (nonzero numerator).
        let r = Poly::var("x_lo_0"); // a stand-in nonzero denominator poly.
        let inv = RatFun {
            num: Poly::constant(1),
            den: r.clone(),
        };
        assert!(inv.sub(&inv).is_zero());
        assert!(!inv.add(&inv).is_zero());
        // a/b == c/d cancellation by cross-multiply: (2)/(r) - (2*r)/(r*r) == 0.
        let two_over_r = RatFun {
            num: Poly::constant(2),
            den: r.clone(),
        };
        let two_over_r2 = RatFun {
            num: Poly::constant(2).mul(&r),
            den: r.mul(&r),
        };
        assert!(two_over_r.sub(&two_over_r2).is_zero());
    }

    #[test]
    fn poly_subst_shift_binomial() {
        // (c_0 + 1)^2 = c_0^2 + 2 c_0 + 1 under c_0 -> c_0 + 1.
        let c0sq = Poly::var("c_0").mul(&Poly::var("c_0"));
        let shifted = c0sq.subst_shift("c_0", 1);
        // expected explicitly.
        let mut expect = Poly::var("c_0").mul(&Poly::var("c_0"));
        let mut two_c0 = Poly::var("c_0");
        two_c0 = two_c0.mul(&Poly::constant(2));
        expect.add_assign(&two_c0);
        expect.add_assign(&Poly::constant(1));
        let mut diff = shifted.clone();
        diff.add_assign(&expect.neg());
        assert!(diff.is_zero());
    }

    #[test]
    fn shift_remaps_sin_symbol() {
        // a coefficient sin_th@1 under a theta shift +1 -> sin_th@3 (2*delta=2 half-units).
        let mut p = Poly::zero();
        let mut mono = BTreeMap::new();
        mono.insert("sin_th@1".to_string(), 1u32);
        p.terms.insert(mono, 1);
        let shifted = shift_poly_coords(&p, &[0, 1, 0]);
        assert!(
            shifted
                .terms
                .keys()
                .next()
                .unwrap()
                .contains_key("sin_th@3")
        );
    }

    #[test]
    fn linform_shift_and_cancel() {
        // a single read e@[0,0,0] minus the same read shifted by [0,0,0] = 0;
        // shifted by [1,0,0] does NOT cancel (distinct offset key).
        let lf = LinForm::from_term(("e".into(), vec![0, 0, 0]), Poly::constant(1));
        let mut same = lf.clone();
        same.add(&lf.shifted(&[0, 0, 0]).neg());
        assert!(same.is_zero());
        let mut diff = lf.clone();
        diff.add(&lf.shifted(&[1, 0, 0]).neg());
        assert!(!diff.is_zero());
        assert_eq!(diff.residual().len(), 2);
    }
}
