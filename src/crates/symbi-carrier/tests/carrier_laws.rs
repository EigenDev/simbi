// =============================================================================
// carrier_laws.rs
//
// the `algebra::laws` module documents the laws every `Scalar` carrier must satisfy
// (ring + abelian group, comparisons, select, sqrt/transcendental identities,
// hyperbolics, NaN). this test promotes that prose to executable properties: each law
// is swept over a deterministic sample grid (no rng — the determinism mandate) on both
// concrete carriers (f64 + f32, the precision axis of "carrier-generic"). the Gv carrier's
// homomorphism (f64 == traced-Gv) is covered separately in `carrier_oracle_new.rs`; the
// concrete carriers are pinned as honest models of the documented algebra.
//
// exact laws (IEEE-exact: identity, commutativity, involution, sub-via-neg, select,
// comparison structure) assert `==`; floating laws (associativity, distributivity, the
// transcendental/hyperbolic identities) assert bounded relative error. each law is
// classified by the guarantee it actually holds: claiming bit-exact associativity for
// floating-point addition would assert something the carrier cannot deliver.
// =============================================================================

macro_rules! carrier_law_suite {
    ($modname:ident, $S:ty, $tol:expr) => {
        mod $modname {
            use symbi_carrier::Scalar;

            type S = $S;
            const TOL: f64 = $tol;

            // a deterministic spread: sign, magnitude, fractions, edge-of-unit — moderate
            // enough that products/associativity don't overflow on f32.
            const RAW: &[f64] = &[
                0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0.1, -0.1, 0.7, -0.7, 7.0, -7.0,
                1e-3, 1e3,
            ];
            // bounded subset for cosh/sinh (cosh overflows ~|x|=710; keep small for f32 too).
            const SMALL: &[f64] = &[0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0];

            fn grid() -> Vec<S> {
                RAW.iter().map(|&x| x as S).collect()
            }
            fn small() -> Vec<S> {
                SMALL.iter().map(|&x| x as S).collect()
            }

            #[track_caller]
            fn approx(a: S, b: S, law: &str) {
                let (a, b) = (a as f64, b as f64);
                let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                assert!(rel < TOL, "{law}: {a} vs {b} (rel {rel:e}, tol {TOL:e})");
            }
            #[track_caller]
            fn exact(cond: bool, law: &str) {
                assert!(cond, "exact law violated: {law}");
            }

            // ---- ring + abelian group: the exact laws ----
            #[test]
            fn ring_exact_laws() {
                let zero = 0.0 as S;
                let one = 1.0 as S;
                for &x in &grid() {
                    exact(-(-x) == x, "neg involution -(-x)==x");
                    exact(x + zero == x, "additive identity x+0==x");
                    exact(x * one == x, "multiplicative identity x*1==x");
                    exact((-x).abs() == x.abs(), "abs is even: abs(-x)==abs(x)");
                    exact(x.abs() >= zero, "abs non-negative");
                }
                for &x in &grid() {
                    for &y in &grid() {
                        exact(x + y == y + x, "additive commutativity");
                        exact(x * y == y * x, "multiplicative commutativity");
                        // Sub is Add-of-Neg by definition — must be bit-exact.
                        exact(x - y == x + (-y), "sub-via-neg x-y==x+(-y)");
                        exact(x.min(y) == y.min(x), "min commutative");
                        exact(x.max(y) == y.max(x), "max commutative");
                        exact(x.min(y) <= x.max(y), "min<=max");
                    }
                }
            }

            // ---- ring: the floating laws (associativity/distributivity are not bit-exact) ----
            #[test]
            fn ring_floating_laws() {
                for &x in &grid() {
                    for &y in &grid() {
                        for &z in &grid() {
                            approx((x + y) + z, x + (y + z), "additive associativity");
                            approx((x * y) * z, x * (y * z), "multiplicative associativity");
                            approx(x * (y + z), x * y + x * z, "left distributivity");
                        }
                        // Div is Mul-of-Recip on the algebra; divide by non-zero only.
                        if y != (0.0 as S) {
                            approx(x / y, x * y.recip(), "div-via-recip x/y==x*recip(y)");
                        }
                    }
                }
            }

            // ---- comparisons: structure (reflexive/irreflexive/symmetric/transitive/trichotomy) ----
            #[test]
            fn comparison_laws() {
                for &x in &grid() {
                    exact(x.cmp_eq(x), "cmp_eq reflexive on finite");
                    exact(!x.cmp_lt(x), "cmp_lt irreflexive");
                    for &y in &grid() {
                        exact(x.cmp_eq(y) == y.cmp_eq(x), "cmp_eq symmetric");
                        exact(x.cmp_lt(y) == y.cmp_gt(x), "cmp_lt(x,y)==cmp_gt(y,x)");
                        // trichotomy on finite values: exactly one of <, ==, > holds.
                        let n = [x.cmp_lt(y), x.cmp_eq(y), x.cmp_gt(y)]
                            .iter()
                            .filter(|b| **b)
                            .count();
                        exact(n == 1, "trichotomy: exactly one of < == >");
                        // le == (lt or eq); ge == (gt or eq).
                        exact(x.cmp_le(y) == (x.cmp_lt(y) || x.cmp_eq(y)), "le == lt|eq");
                        exact(x.cmp_ge(y) == (x.cmp_gt(y) || x.cmp_eq(y)), "ge == gt|eq");
                        for &w in &grid() {
                            if x.cmp_lt(y) && y.cmp_lt(w) {
                                exact(x.cmp_lt(w), "cmp_lt transitive");
                            }
                        }
                    }
                }
            }

            // ---- select: picks the named arm, exactly; and distributes over scalar ops ----
            #[test]
            fn select_laws() {
                for &t in &grid() {
                    for &f in &grid() {
                        exact(
                            <S as Scalar>::select(true, t, f) == t,
                            "select(true,t,f)==t",
                        );
                        exact(
                            <S as Scalar>::select(false, t, f) == f,
                            "select(false,t,f)==f",
                        );
                        // select distributes over Neg: select(m,-t,-f) == -select(m,t,f).
                        for &m in &[true, false] {
                            let lhs = <S as Scalar>::select(m, -t, -f);
                            let rhs = -<S as Scalar>::select(m, t, f);
                            exact(lhs == rhs, "select distributes over neg");
                        }
                    }
                }
            }

            // ---- sqrt + exp/ln identities (domain-guarded, floating) ----
            #[test]
            fn sqrt_exp_ln_laws() {
                for &x in &grid() {
                    // Sqrt(x*x) == Abs(x): radicand is always non-negative.
                    approx((x * x).sqrt(), x.abs(), "sqrt(x*x)==abs(x)");
                    // Sqrt(|x|)^2 == |x|.
                    let s = x.abs().sqrt();
                    approx(s * s, x.abs(), "sqrt(|x|)^2==|x|");
                    if x > (0.0 as S) {
                        approx(x.ln().exp(), x, "exp(ln(x))==x for x>0");
                    }
                    // ln(exp(x)) == x where exp(x) doesn't overflow.
                    if x.abs() <= (7.0 as S) {
                        approx(x.exp().ln(), x, "ln(exp(x))==x");
                    }
                }
            }

            // ---- hyperbolic identities (bounded |x| to avoid overflow) ----
            #[test]
            fn hyperbolic_laws() {
                let one = 1.0 as S;
                for &x in &small() {
                    let (sh, ch) = (x.sinh(), x.cosh());
                    approx(ch * ch - sh * sh, one, "cosh^2 - sinh^2 == 1");
                    approx(x.tanh(), sh / ch, "tanh == sinh/cosh");
                    approx(
                        (x + x).sinh(),
                        (2.0 as S) * sh * ch,
                        "sinh(2x)==2 sinh cosh",
                    );
                    approx(x.sinh().asinh(), x, "asinh(sinh(x))==x");
                    approx(x.cosh().acosh(), x.abs(), "acosh(cosh(x))==abs(x)");
                }
            }

            // ---- NaN: the only value where cmp_eq(self) is false ----
            #[test]
            fn nan_laws() {
                let nan = <S as Scalar>::nan();
                let inf = <S as Scalar>::infinity();
                exact(!nan.cmp_eq(nan), "NaN.cmp_eq(NaN) == false");
                exact(<S as Scalar>::is_nan(nan), "is_nan(NaN)");
                exact(!<S as Scalar>::is_nan(inf), "!is_nan(+inf)");
                for &x in &grid() {
                    exact(!<S as Scalar>::is_nan(x), "!is_nan(finite)");
                    exact(x.cmp_eq(x), "finite is self-equal");
                }
            }
        }
    };
}

carrier_law_suite!(f64_laws, f64, 1e-11);
carrier_law_suite!(f32_laws, f32, 1e-4);
