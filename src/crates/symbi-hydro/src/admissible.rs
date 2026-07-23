// =============================================================================
// admissible.rs
//
// the relativistic admissible set and the closed-form projection onto its boundary.
//
// G = { U : D > 0,  f(U) = E^2 - D^2 - gamma^{ij} S_i S_j > 0 },  E = tau + D
//
// is the set of conserved states admitting a physical primitive (rho > 0, p > 0, |v| < c)
// (Wu & Tang 2015). G is CONVEX in the conserved variables, so the segment from a KNOWN-admissible
// anchor to any candidate crosses the boundary partial-G at most once. `admissible_theta` returns
// the largest blend `theta` in [0, 1] for which `anchor + theta (cand - anchor)` lies in G:
// `theta = 1` (the candidate unchanged) when the candidate is already admissible, else the exact
// crossing. because the anchor is admissible the projection ALWAYS yields an admissible state, so it
// is the provable replacement for a freeze/floor fallback — no cell is ever unrecoverable.
//
// the crossing is a CLOSED-FORM quadratic root: D, S_i and E are all linear along the segment, so
// `f(theta) = a theta^2 + b theta + c` with `c = f(anchor) > 0`. densitization scales D, S and E by
// the common positive factor sqrt(-g), which cancels in the sign of f, so this operates directly on
// the stored densitized conserveds with no undensitization.
//
// carrier-generic (S = f64 host / Gv trace); branch-free (select only, no control flow).
// =============================================================================

use symbi_algebra::{Matrix, Tensor};
use symbi_ir::algebra::Scalar;

/// the largest `theta` in [0, 1] with `anchor + theta (cand - anchor)` in the admissible set G,
/// given the anchor is admissible (`f(anchor) > 0`, `D_anchor > 0`).
///
/// `d`, `s`, `e`: the conserved rest-mass density D, the COVARIANT momentum S_i, and the eulerian
/// energy E = tau + D (the caller reconstructs E from the stored energy slot via the 3+1 block).
/// `gm_inv`: the inverse spatial metric gamma^{ij} for `|S|^2 = gamma^{ij} S_i S_j` (identity for
/// flat SR). `eps_d` / `eps_f`: strict-interior floors on D and on f, keeping the projected state
/// off the exact boundary so the downstream c2p converges.
#[allow(clippy::too_many_arguments)]
pub fn admissible_theta<S: Scalar>(
    d_c: S,
    s_c: Tensor<S, 3>,
    e_c: S,
    d_a: S,
    s_a: Tensor<S, 3>,
    e_a: S,
    gm_inv: &Matrix<S, 3>,
    eps_d: S,
    eps_f: S,
) -> S {
    let two = S::from_f64(2.0);
    // B(u, v) = gamma^{ij} u_i v_j (the inverse-metric contraction on covariant momentum).
    let bil = |u: &Tensor<S, 3>, v: &Tensor<S, 3>| -> S {
        let mut acc = S::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                acc = acc + gm_inv[(i, j)] * u[i] * v[j];
            }
        }
        acc
    };
    let d_delta = d_c - d_a;
    let e_delta = e_c - e_a;
    let s_delta = s_c - s_a;
    let snorm_a = bil(&s_a, &s_a);
    let cross = bil(&s_a, &s_delta);
    let snorm_d = bil(&s_delta, &s_delta);
    // f(theta) = E(theta)^2 - D(theta)^2 - |S(theta)|^2 = a theta^2 + b theta + c.
    let a = e_delta * e_delta - d_delta * d_delta - snorm_d;
    let b = two * (e_a * e_delta - d_a * d_delta - cross);
    let c = e_a * e_a - d_a * d_a - snorm_a; // = f(anchor) > 0 by admissibility

    // g(theta) = f(theta) - eps_f; g(0) = c - eps_f >= 0. the crossing to g < 0 — present iff the
    // candidate is inadmissible — is the citardauq root theta_f = 2 g0 / (-b + sqrt(b^2 - 4 a g0)):
    // branch-free, and division-safe because -b + sqrt(.) >= -b > 0 whenever a crossing exists (b < 0
    // there). convexity of G guarantees the two regimes cleanly: an admissible candidate has NO
    // crossing in (0, 1], so the raw root is >= 1 and the clamp returns exactly 1 (the candidate
    // passes through bit-for-bit); an inadmissible candidate has EXACTLY one crossing, and this is it.
    let g0 = (c - eps_f).max(S::ZERO);
    let disc = (b * b - S::from_f64(4.0) * a * g0).max(S::ZERO);
    let denom = ((S::ZERO - b) + disc.sqrt()).max(S::from_f64(1e-300));
    let theta_f = (two * g0 / denom).min(S::ONE).max(S::ZERO);

    // density floor D(theta) = d_a + theta d_delta >= eps_d, binding only when the density falls
    // (d_delta < 0). the guarded denominator keeps the unselected arm finite when d_delta = 0.
    let falling = d_delta.cmp_lt(S::ZERO);
    let dd_guard = S::select(falling, d_a - d_c, S::ONE); // = -d_delta > 0 when falling
    let theta_d = S::select(
        falling,
        ((d_a - eps_d) / dd_guard).min(S::ONE).max(S::ZERO),
        S::ONE,
    );

    // a non-finite candidate (NaN / inf momentum or energy) has no admissible point along its
    // segment; fall fully back to the anchor (theta = 0), the degenerate case the freeze once held.
    let finite = |v: S| (v - v).cmp_eq(S::ZERO);
    let mut ok = finite(d_c) & finite(e_c);
    for k in 0..3 {
        ok = ok & finite(s_c[k]);
    }
    let project = S::select(ok, theta_f.min(theta_d), S::ZERO);
    // EXACT passthrough: a candidate already in G (f > 0, D > 0, finite) is returned bit-for-bit
    // (theta = 1), so the projection is a no-op on every physical cell and only genuinely
    // inadmissible cells move. `f(1) = a + b + c` is the candidate's own f.
    let f_cand = a + b + c;
    let cand_ok = f_cand.cmp_gt(S::ZERO) & d_c.cmp_gt(S::ZERO) & ok;
    S::select(cand_ok, S::ONE, project)
}

/// project a candidate conserved state onto G along the segment from an admissible anchor, returning
/// `anchor + theta (cand - anchor)` component-wise in whatever variables the caller passes (the
/// stored densitized conserveds). `theta` from [`admissible_theta`]; an already-admissible candidate
/// returns unchanged (theta = 1). the momentum blend covers all `N` stored slots.
#[allow(clippy::too_many_arguments)]
pub fn admissible_project<S: Scalar, const N: usize>(
    den_c: S,
    mom_c: [S; N],
    nrg_c: S,
    den_a: S,
    mom_a: [S; N],
    nrg_a: S,
    theta: S,
) -> (S, [S; N], S) {
    let den = den_a + theta * (den_c - den_a);
    let nrg = nrg_a + theta * (nrg_c - nrg_a);
    let mom = std::array::from_fn(|k| mom_a[k] + theta * (mom_c[k] - mom_a[k]));
    (den, mom, nrg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Matrix;

    // f(U) = E^2 - D^2 - gamma^{ij} S_i S_j; U in G iff D > 0 and f > 0.
    fn f_of(d: f64, s: Tensor<f64, 3>, e: f64, gi: &Matrix<f64, 3>) -> f64 {
        let mut sn = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sn += gi[(i, j)] * s[i] * s[j];
            }
        }
        e * e - d * d - sn
    }

    // a diagonal inverse-metric (identity for flat; a Schwarzschild-like radial stretch otherwise).
    fn metric(f_rr: f64) -> Matrix<f64, 3> {
        Matrix::diag(Tensor::new([f_rr, 1.0, 1.0]))
    }

    // a deterministic admissible conserved state from a physical primitive on the given metric.
    // `v_phys` is the PHYSICAL radial velocity (|v| = sqrt(gamma_ij v^i v^j), subluminal on ANY
    // metric); the contravariant v^r = v_phys / sqrt(gamma_rr) then has gamma_rr (v^r)^2 = v_phys^2.
    fn admissible(rho: f64, v_phys: f64, p: f64, grr: f64) -> (f64, Tensor<f64, 3>, f64) {
        let gamma = 5.0 / 3.0;
        let vr = v_phys / grr.sqrt();
        let v_sq = grr * vr * vr; // = v_phys^2 < 1
        let w = 1.0 / (1.0 - v_sq).sqrt();
        let h = 1.0 + gamma / (gamma - 1.0) * p / rho;
        let d = rho * w;
        let rhw2 = rho * h * w * w;
        let s = Tensor::new([grr * vr * rhw2, 0.0, 0.0]); // covariant S_r
        let e = rhw2 - p; // E = tau + D = rho h W^2 - p
        (d, s, e)
    }

    #[test]
    fn projection_always_lands_in_the_admissible_set() {
        // over a grid of admissible anchors and WILD candidates (negative density, superluminal
        // momentum, negative energy, and finite admissible ones), the projected state is ALWAYS in
        // G, and an already-admissible candidate passes through unchanged (theta = 1).
        let (eps_d, eps_f) = (1e-12, 1e-14);
        for &grr in &[1.0_f64, 1.25, 2.0] {
            let gi = metric(1.0 / grr); // gamma^{rr} = 1/gamma_rr
            let anchor = admissible(1.0, 0.1, 0.5, grr);
            for &(rho, vr, p) in &[
                (1.2_f64, 0.2, 0.6), // admissible
                (0.9, -0.3, 0.4),    // admissible
                (1e-6, 0.99, 1e-8),  // near-vacuum ultrarelativistic (near the cusp of G)
            ] {
                let cand = admissible(rho, vr, p, grr);
                let fc = f_of(cand.0, cand.1, cand.2, &gi);
                let th = admissible_theta(cand.0, cand.1, cand.2, anchor.0, anchor.1, anchor.2, &gi, eps_d, eps_f);
                // admissible candidate -> theta = 1, exact passthrough
                assert!((th - 1.0).abs() < 1e-12,
                    "grr={grr} cand=({rho},{vr},{p}) f_cand={fc:.3e}: not passed through, theta={th}");
            }
            // WILD inadmissible candidates: the projection must still yield a state in G.
            let wild: [(f64, Tensor<f64, 3>, f64); 5] = [
                (-2.0, Tensor::new([5.0, 0.0, 0.0]), 1.0),   // negative density
                (0.5, Tensor::new([100.0, 0.0, 0.0]), 1.0),  // |S| >> E (superluminal)
                (1.0, Tensor::new([0.0, 0.0, 0.0]), -3.0),   // negative energy
                (1e-9, Tensor::new([50.0, 20.0, 0.0]), 0.1), // near-vacuum, huge momentum
                (0.0, Tensor::new([0.0, 0.0, 0.0]), 0.0),    // the zero state
            ];
            for (dc, sc, ec) in wild {
                let th = admissible_theta(dc, sc, ec, anchor.0, anchor.1, anchor.2, &gi, eps_d, eps_f);
                assert!((0.0..=1.0).contains(&th), "theta out of [0,1]: {th}");
                let (dp, sp, ep) = admissible_project::<f64, 3>(
                    dc, [sc[0], sc[1], sc[2]], ec,
                    anchor.0, [anchor.1[0], anchor.1[1], anchor.1[2]], anchor.2,
                    th,
                );
                let sp_t = Tensor::new(sp);
                assert!(dp > 0.0, "projected density non-positive: {dp} (cand D={dc})");
                let fp = f_of(dp, sp_t, ep, &gi);
                assert!(fp > 0.0, "projected state NOT in G: f={fp:.3e} (cand D={dc}, |S|={:?}, E={ec})", sc);
            }
        }
    }

    #[test]
    fn a_wrong_theta_is_caught_by_the_property() {
        // bug injection: theta = 1 (keep the candidate unconditionally) must FAIL to land a wildly
        // inadmissible candidate in G — proving the property test has teeth.
        let gi = metric(1.0);
        let anchor = admissible(1.0, 0.1, 0.5, 1.0);
        let (dc, sc, ec) = (0.5_f64, Tensor::new([100.0, 0.0, 0.0]), 1.0);
        let (dp, sp, ep) = admissible_project::<f64, 3>(
            dc, [sc[0], sc[1], sc[2]], ec,
            anchor.0, [anchor.1[0], anchor.1[1], anchor.1[2]], anchor.2,
            1.0, // the injected wrong theta
        );
        assert!(f_of(dp, Tensor::new(sp), ep, &gi) < 0.0, "the injected theta=1 should leave the state OUTSIDE G");
    }
}
