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

/// the RELATIVE margin below which an admissibility residual is treated as indistinguishable from
/// zero, measured against the cell's own energy density.
///
/// the residuals guarded by it — `q = E - sqrt(D^2 + |S|^2)` and the norm `sqrt(D^2 + |S|^2)` — both
/// carry one power of energy, so a bare absolute floor would encode an assumption about the problem's
/// energy scaling that nothing enforces: the same constant is far too loose for a near-vacuum
/// atmosphere and far too tight for a dense core. scaling by the local energy makes the criterion
/// DIMENSIONLESS, so it means the same thing at every density.
///
/// the value sits about six orders above the accumulated roundoff of a multi-operation flux
/// computation (roughly 1e-16 relative per operation), which is enough margin that a residual falling
/// under it is genuinely numerical noise rather than a resolved small number.
pub const ADMISSIBLE_REL_FLOOR: f64 = 1e-10;
use symbi_ir::algebra::Scalar;

/// one-power-of-energy scale for relativistic hydrodynamic conserved states.
/// `D`, `|S|`, and `E` have the same dimensions.
pub fn rhd_state_scale<S: Scalar>(d: S, s: &Tensor<S, 3>, e: S, gm_inv: &Matrix<S, 3>) -> S {
    let mut s2 = S::ZERO;
    for ii in 0..3 {
        for jj in 0..3 {
            s2 = s2 + gm_inv[(ii, jj)] * s[ii] * s[jj];
        }
    }
    d.abs().max(s2.max(S::ZERO).sqrt()).max(e.abs())
}

/// one-power-of-energy scale for relativistic magnetohydrodynamic conserved
/// states. `D`, `|S|`, `E`, and `|B|^2` have the same dimensions, so their
/// maximum defines a local scale without privileging a density normalization.
pub fn rmhd_state_scale<S: Scalar>(
    d: S,
    s: &Tensor<S, 3>,
    e: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
) -> S {
    let contract = |m: &Matrix<S, 3>, u: &Tensor<S, 3>| -> S {
        let mut acc = S::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + m[(ii, jj)] * u[ii] * u[jj];
            }
        }
        acc
    };
    let s_norm = contract(gm_inv, s).max(S::ZERO).sqrt();
    let magnetic_energy = contract(gm, b).max(S::ZERO);
    d.abs().max(s_norm).max(e.abs()).max(magnetic_energy)
}

/// momentum and eulerian-energy derivatives for a stationary-spacetime killing-energy update.
///
/// the code evolves `ehat = alpha E - D - beta.S`, whose geometric source vanishes. with
/// `dS/dt = alpha Smom` and static metric coefficients, `dE/dt = beta.Smom`.
pub fn stationary_killing_source_ray<S: Scalar>(
    alpha: S,
    beta: &Tensor<S, 3>,
    s_mom: Tensor<S, 3>,
) -> (Tensor<S, 3>, S) {
    let momentum_dot = Tensor::new(std::array::from_fn(|kk| alpha * s_mom[kk]));
    let energy_dot = (0..3).fold(S::ZERO, |acc, kk| acc + beta[kk] * s_mom[kk]);
    (momentum_dot, energy_dot)
}

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
    let denom_raw = (S::ZERO - b) + disc.sqrt();
    let coeff_scale = a.abs().max(b.abs()).max(g0.abs());
    let denom_floor = S::select(
        coeff_scale.cmp_gt(S::ZERO),
        S::from_f64(f64::EPSILON) * coeff_scale,
        S::ONE,
    );
    let denom = denom_raw.max(denom_floor);
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

/// the two RMHD admissibility residuals `(q, psi)` (Wu & Tang, arXiv:1709.05838, theorem 2.1). a
/// magnetized state is admissible iff `D > 0`, `q(U) > 0` AND `psi(U) > 0`:
///
/// ```text
/// q   = E - sqrt(D^2 + |S|^2)
/// Phi = sqrt( (|B|^2 - E)^2 + 3 (E^2 - D^2 - |S|^2) )
/// psi = (Phi - 2(|B|^2 - E)) sqrt(Phi + |B|^2 - E) - sqrt( (27/2)(D^2 |B|^2 + (S.B)^2) )
/// ```
///
/// `q > 0` alone is only NECESSARY (their lemma 2.1) — it is the B-free cone, blind to the magnetic
/// decomposition — whereas the pair is SUFFICIENT, and both are evaluable directly in the conserved
/// variables with no primitive recovery.
///
/// the three contractions carry the spatial metric exactly as their valence demands: `|S|^2 =
/// gamma^{ij} S_i S_j` on the inverse metric, `|B|^2 = gamma_{ij} B^i B^j` on the covariant metric,
/// and `S.B = S_i B^i` is a covector-vector pairing and hence METRIC-FREE. one triad orthonormalizes
/// all three at once, which is why the special-relativistic condition holds verbatim in a curved
/// chart. every square root is floored at zero: the radicands are non-negative on `q > 0` (which
/// forces `Phi > | |B|^2 - E |`), and the floor keeps the branch-free evaluation finite off it.
pub fn rmhd_admissible_residuals<S: Scalar>(
    d: S,
    s: &Tensor<S, 3>,
    e: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
) -> (S, S) {
    let contract = |m: &Matrix<S, 3>, u: &Tensor<S, 3>, v: &Tensor<S, 3>| -> S {
        let mut acc = S::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                acc = acc + m[(i, j)] * u[i] * v[j];
            }
        }
        acc
    };
    let s2 = contract(gm_inv, s, s);
    let b2 = contract(gm, b, b);
    let mut sb = S::ZERO;
    for i in 0..3 {
        sb = sb + s[i] * b[i];
    }

    let q = e - (d * d + s2).max(S::ZERO).sqrt();

    let w = b2 - e;
    let phi = (w * w + S::from_f64(3.0) * (e * e - d * d - s2))
        .max(S::ZERO)
        .sqrt();
    let psi = (phi - S::from_f64(2.0) * w) * (phi + w).max(S::ZERO).sqrt()
        - (S::from_f64(13.5) * (d * d * b2 + sb * sb))
            .max(S::ZERO)
            .sqrt();
    (q, psi)
}

/// raise only the eulerian energy of an analytically admissible RMHD anchor
/// until it lies inside the requested finite-precision margins.
///
/// density, momentum, and magnetic field remain fixed. the upper endpoint adds
/// one local conserved-state scale, which dominates the requested relative
/// margins while preserving homogeneity under a change of conserved units.
/// fixed-count bisection returns the least added energy resolved by `iters`.
#[allow(clippy::too_many_arguments)]
pub fn rmhd_anchor_energy_with_margin<S: Scalar>(
    d: S,
    s: &Tensor<S, 3>,
    e: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
    state_scale: S,
    eps_q: S,
    eps_psi: S,
    iters: usize,
) -> S {
    let ok = |energy: S| -> S::Mask {
        let (q, psi) = rmhd_admissible_residuals(d, s, energy, b, gm_inv, gm);
        q.cmp_gt(eps_q) & psi.cmp_gt(eps_psi)
    };
    let original_ok = ok(e);
    let mut lo = e;
    let mut hi = e + state_scale;
    for _ in 0..iters {
        let mid = S::from_f64(0.5) * (lo + hi);
        let mid_ok = ok(mid);
        hi = S::select(mid_ok, mid, hi);
        lo = S::select(mid_ok, lo, mid);
    }
    S::select(original_ok, e, hi)
}

/// the largest `theta` in [0, 1] with `anchor + theta (cand - anchor)` in the RMHD admissible set,
/// with the magnetic field held FIXED at `b` throughout.
///
/// holding `B` fixed is what keeps constrained transport intact: `B` is staggered and shared between
/// neighboring cells, so blending it per-cell would desynchronize the shared face value and break
/// `div(B) = 0`. the set is convex in the full conserved vector (their theorem 2.2), and the
/// intersection of a convex set with the affine slice `B = const` is convex, so the segment still
/// crosses the boundary at most once and the largest admissible `theta` is well defined.
///
/// unlike the hydro cone there is NO closed form. the hydro admissible set is a second-order
/// (Lorentz) cone whose defining function is concave — linear minus a norm — which is what makes its
/// crossing a quadratic root. `psi` is not concave (it carries products of square roots), so the
/// crossing is found by bisection: a FIXED iteration count, branch-free, so the trace carrier emits
/// straight-line code. convexity guarantees the admissible `theta` form the interval `[0, theta*]`,
/// which is exactly what makes bisection on the predicate correct rather than merely plausible.
///
/// an admissible candidate returns exactly `1` bit-for-bit, so the projection is a no-op on every
/// physical cell.
///
/// RETURNS 0 WHEN THE ANCHOR ITSELF IS INADMISSIBLE IN THE CANDIDATE'S MAGNETIC SLICE. this is the
/// one structural difference from the hydro projection, and it is not removable: the anchor is
/// admissible with its OWN `B`, but the projection must land in the slice of the CANDIDATE's `B`,
/// where the anchor's hydro state need not be admissible. the caller must therefore keep a fallback
/// for `theta = 0`; magnetized admissibility is not unconditionally recoverable by blending alone.
#[allow(clippy::too_many_arguments)]
pub fn rmhd_admissible_theta<S: Scalar>(
    d_c: S,
    s_c: Tensor<S, 3>,
    e_c: S,
    d_a: S,
    s_a: Tensor<S, 3>,
    e_a: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
    eps_d: S,
    eps_q: S,
    eps_psi: S,
    iters: usize,
) -> S {
    let finite = |v: S| (v - v).cmp_eq(S::ZERO);
    let mut cand_finite = finite(d_c) & finite(e_c);
    for k in 0..3 {
        cand_finite = cand_finite & finite(s_c[k]);
    }

    // the admissibility predicate along the segment, as a mask.
    let ok_at = |t: S| -> S::Mask {
        let d = d_a + t * (d_c - d_a);
        let e = e_a + t * (e_c - e_a);
        let s = Tensor::new(std::array::from_fn(|k| s_a[k] + t * (s_c[k] - s_a[k])));
        let (q, psi) = rmhd_admissible_residuals(d, &s, e, b, gm_inv, gm);
        d.cmp_gt(eps_d) & q.cmp_gt(eps_q) & psi.cmp_gt(eps_psi)
    };

    // bisection for theta*, the boundary of the admissible interval [0, theta*]; `iters` halvings
    // resolve theta to 2^-iters. `lo` always holds a KNOWN-ADMISSIBLE theta, so the result is
    // admissible by construction rather than by convergence — truncating the bisection is SAFE, it
    // only returns a smaller (more diffusive) blend. that is what lets a traced carrier, which
    // unrolls every iteration into the expression graph, choose a low count without risking
    // correctness.
    let mut lo = S::ZERO;
    let mut hi = S::ONE;
    for _ in 0..iters {
        let mid = S::from_f64(0.5) * (lo + hi);
        let ok = ok_at(mid);
        lo = S::select(ok, mid, lo);
        hi = S::select(ok, hi, mid);
    }

    // exact passthrough for an already-admissible candidate; a non-finite candidate takes theta = 0.
    let cand_ok = ok_at(S::ONE) & cand_finite;
    S::select(cand_ok, S::ONE, S::select(cand_finite, lo, S::ZERO))
}

/// how many trial times out the "the ray never leaves G" probe is evaluated. the magnetized
/// residual `psi` carries the FIXED `|B|`, which becomes negligible against `t |Sdot|` as `t`
/// grows; by this multiple the set along the ray is the B-free cone to well past f64 precision,
/// so an admissible endpoint here is the cone membership the certificate below needs.
const SOURCE_RAY_FAR: f64 = 1.0e6;

/// a source-ray timestep that remains inside the RMHD admissible set.
///
/// the source changes only momentum and energy: `U(t) = (D, S + t Sdot, E + t Edot, B)`.
/// `source_scale` has units of energy density per time, so `state_scale/source_scale` is the
/// dimensionally natural trial time. when that endpoint is admissible, convexity makes every
/// earlier point admissible. otherwise fixed-count bisection returns the largest known-admissible
/// fraction of the trial time. a zero source returns an infinite timestep.
///
/// AN INWARD SOURCE IMPOSES NO TIMESTEP AT ALL (Wu, arXiv:1610.06274, theorem 1: `lambda_S = 0`
/// whenever `q(S) >= 0`). the unmagnetized set is a convex cone, so `U + t Sdot` provably never
/// leaves it and the bound does not exist to be computed. the MAGNETIZED set is not a cone —
/// `psi` carries `|B|`, which does not scale with `(D, S, E)` — so cone membership of the source
/// alone does not certify the ray. two facts do: the set at fixed `B` is convex (Wu & Tang,
/// arXiv:1709.05838, theorem 2.2), which covers `[0, T]` once the endpoint at `T` is admissible;
/// and the ray's own magnetic terms vanish relative to `t |Sdot|`, so beyond `T` the set is the
/// B-free cone and `q(Sdot) > 0` covers `[T, inf)`. together they certify the whole ray.
///
/// this clause is what keeps a near-vacuum cell from throttling the global step: an evacuated
/// atmosphere cell has `state_scale -> 0`, so its trial time collapses and the bisection returns
/// an arbitrarily small "safe" time — for a source that was never driving it out of G.
#[allow(clippy::too_many_arguments)]
pub fn rmhd_source_admissible_time<S: Scalar>(
    d: S,
    s: Tensor<S, 3>,
    e: S,
    s_dot: Tensor<S, 3>,
    e_dot: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
    state_scale: S,
    source_scale: S,
    eps_d: S,
    eps_q: S,
    eps_psi: S,
    iters: usize,
) -> S {
    let active = source_scale.cmp_gt(S::ZERO);
    let guarded_source = S::select(active, source_scale, S::ONE);
    let trial_time = state_scale / guarded_source;
    let ok_at = |fraction: S| -> S::Mask {
        let time = fraction * trial_time;
        let source_state = Tensor::new(std::array::from_fn(|kk| s[kk] + time * s_dot[kk]));
        let source_energy = e + time * e_dot;
        let (q, psi) = rmhd_admissible_residuals(d, &source_state, source_energy, b, gm_inv, gm);
        d.cmp_gt(eps_d) & q.cmp_gt(eps_q) & psi.cmp_gt(eps_psi)
    };

    // the inward-source certificate: `q` of the SOURCE VECTOR (no baryon component, no magnetic
    // component — the geometric source touches neither) together with an admissible far endpoint.
    let (q_source, _) =
        rmhd_admissible_residuals(S::ZERO, &s_dot, e_dot, &Tensor::zeros(), gm_inv, gm);
    let unbounded = q_source.cmp_gt(S::ZERO) & ok_at(S::from_f64(SOURCE_RAY_FAR));

    let mut lo = S::ZERO;
    let mut hi = S::ONE;
    for _ in 0..iters {
        let mid = S::from_f64(0.5) * (lo + hi);
        let ok = ok_at(mid);
        lo = S::select(ok, mid, lo);
        hi = S::select(ok, hi, mid);
    }
    let fraction = S::select(ok_at(S::ONE), S::ONE, lo);
    let fraction_floor = S::from_f64(2.0_f64.powi(-(iters as i32)));
    let finite_time = trial_time * fraction.max(fraction_floor);
    let bounded = S::select(unbounded, S::from_f64(f64::INFINITY), finite_time);
    S::select(active, bounded, S::from_f64(f64::INFINITY))
}

/// limit one already-integrated geometric-source increment along the affine source ray.
///
/// the flux and constrained-transport candidate `(d, s, e, b)` is the anchor. the geometric source
/// proposes `(0, delta_s, delta_e, 0)`. only that local, physically nonconservative source increment
/// is scaled; density, every shared magnetic face value, and every conservative face flux remain
/// untouched. an admissible full increment passes through exactly with `theta = 1`.
///
/// the caller must provide an admissible flux anchor. this is the low-order physical-constraint-
/// preserving invariant: a first-order flux + constrained-transport update under the CFL bound
/// lands in the admissible set. an anchor that does not is a statement about the TIMESTEP, not
/// about the source — no source fraction can repair the flux operator, so the caller's only
/// recourse is to reject the step and replay it at a smaller dt.
#[allow(clippy::too_many_arguments)]
pub fn rmhd_limit_source_increment<S: Scalar>(
    d: S,
    s: Tensor<S, 3>,
    e: S,
    delta_s: Tensor<S, 3>,
    delta_e: S,
    b: &Tensor<S, 3>,
    gm_inv: &Matrix<S, 3>,
    gm: &Matrix<S, 3>,
    eps_d: S,
    eps_q: S,
    eps_psi: S,
    iters: usize,
) -> (S, Tensor<S, 3>, S) {
    let candidate_s = Tensor::new(std::array::from_fn(|kk| s[kk] + delta_s[kk]));
    let candidate_e = e + delta_e;
    let theta = rmhd_admissible_theta(
        d,
        candidate_s,
        candidate_e,
        d,
        s,
        e,
        b,
        gm_inv,
        gm,
        eps_d,
        eps_q,
        eps_psi,
        iters,
    );
    let limited_s = Tensor::new(std::array::from_fn(|kk| s[kk] + theta * delta_s[kk]));
    let limited_e = e + theta * delta_e;
    (theta, limited_s, limited_e)
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

    #[test]
    fn rmhd_source_time_follows_the_directional_admissible_ray() {
        let gm = Matrix::identity();
        let d = 1.0;
        let s = Tensor::zeros();
        let e = 2.0;
        let b = Tensor::new([0.1, 0.0, 0.0]);
        let s_dot = Tensor::new([0.2, 0.0, 0.0]);
        let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
        let eps_d = 1e-12 * scale;
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);

        // q(source) = e_dot - |s_dot| = 1.0 - 0.2 > 0: the source vector lies in the cone, so the
        // ray U + t Sdot never leaves G and NO timestep bound exists (Wu, arXiv:1610.06274,
        // theorem 1: lambda_S = 0). returning the finite trial timescale here would throttle the
        // global step for a cell the source is not endangering.
        let inward = rmhd_source_admissible_time(
            d, s, e, s_dot, 1.0, &b, &gm, &gm, scale, 1.0, eps_d, eps_q, eps_psi, 24,
        );
        assert_eq!(
            inward,
            f64::INFINITY,
            "an inward source imposes no timestep bound"
        );

        let outward = rmhd_source_admissible_time(
            d, s, e, s_dot, -1.0, &b, &gm, &gm, scale, 1.0, eps_d, eps_q, eps_psi, 24,
        );
        assert!(outward > 0.0 && outward < scale);
        let source_state = s + s_dot * outward;
        let source_energy = e - outward;
        let (q, psi) = rmhd_admissible_residuals(d, &source_state, source_energy, &b, &gm, &gm);
        assert!(q > eps_q && psi > eps_psi);
    }

    #[test]
    fn an_evacuated_cell_with_an_inward_source_does_not_throttle_the_step() {
        // the funnel pathology: an atmosphere cell evacuated by ten orders in pressure has
        // state_scale -> 0, so the dimensional trial time state_scale/source_scale collapses and a
        // bisection over it returns an arbitrarily small "safe" time. one such cell then sets dt
        // for the whole grid. it must not, when the source is not driving the cell out of G.
        let gm = Matrix::identity();
        let d = 6.3e-6;
        let s = Tensor::zeros();
        let e = 1.2 * d; // cold: total energy just above rest mass
        let b = Tensor::new([1.0e-5, 0.0, 0.0]);
        let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
        let eps_d = 1e-12 * scale;
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);

        // an inward source: energy gain dominates the momentum push, so q(Sdot) > 0.
        let s_dot = Tensor::new([1.0e-7, 0.0, 0.0]);
        let e_dot: f64 = 1.0e-6;
        let source_scale: f64 = e_dot.abs().max(1.0e-7);
        let time = rmhd_source_admissible_time(
            d,
            s,
            e,
            s_dot,
            e_dot,
            &b,
            &gm,
            &gm,
            scale,
            source_scale,
            eps_d,
            eps_q,
            eps_psi,
            16,
        );
        assert_eq!(
            time,
            f64::INFINITY,
            "an evacuated cell with an inward source must impose no bound; a finite time here \
             is the near-vacuum dt collapse"
        );

        // PREMISE: the same cell with an OUTWARD source must still be bounded, else this gate
        // would pass on a function that returns infinity unconditionally.
        let outward = rmhd_source_admissible_time(
            d,
            s,
            e,
            Tensor::new([1.0e-6, 0.0, 0.0]),
            -1.0e-6,
            &b,
            &gm,
            &gm,
            scale,
            source_scale,
            eps_d,
            eps_q,
            eps_psi,
            16,
        );
        assert!(
            outward.is_finite() && outward > 0.0,
            "an outward source must still bound the step, got {outward:e}"
        );
    }

    #[test]
    fn rmhd_source_limiter_preserves_the_flux_anchor_and_clips_only_the_source() {
        let gm = Matrix::identity();
        let d = 1.0;
        let s = Tensor::zeros();
        let e = 2.0;
        let b = Tensor::new([0.1, 0.0, 0.0]);
        let delta_s = Tensor::new([10.0, 0.0, 0.0]);
        let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
        let eps_d = 1e-12 * scale;
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);

        let (theta, limited_s, limited_e) = rmhd_limit_source_increment(
            d, s, e, delta_s, 0.0, &b, &gm, &gm, eps_d, eps_q, eps_psi, 24,
        );
        assert!(theta > 0.0 && theta < 1.0);
        let (q, psi) = rmhd_admissible_residuals(d, &limited_s, limited_e, &b, &gm, &gm);
        assert!(q > eps_q);
        assert!(psi > eps_psi);
        assert_eq!(limited_e, e);
        assert_eq!(limited_s[1], s[1]);
        assert_eq!(limited_s[2], s[2]);
    }

    #[test]
    fn rmhd_source_limiter_is_exact_for_an_admissible_full_increment() {
        let gm = Matrix::identity();
        let d = 1.0;
        let s = Tensor::new([0.1, 0.0, 0.0]);
        let e = 2.0;
        let b = Tensor::new([0.1, 0.0, 0.0]);
        let delta_s = Tensor::new([0.01, -0.02, 0.03]);
        let delta_e = 0.05;
        let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
        let eps_d = 1e-12 * scale;
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);

        let (theta, limited_s, limited_e) = rmhd_limit_source_increment(
            d, s, e, delta_s, delta_e, &b, &gm, &gm, eps_d, eps_q, eps_psi, 24,
        );
        assert_eq!(theta, 1.0);
        assert_eq!(limited_s, s + delta_s);
        assert_eq!(limited_e, e + delta_e);
    }

    #[test]
    fn rmhd_source_limiter_is_invariant_under_conserved_unit_scaling() {
        let gm = Matrix::identity();
        let solve = |unit_scale: f64| {
            let d = unit_scale;
            let s = Tensor::zeros();
            let e = 2.0 * unit_scale;
            let b = Tensor::new([0.1 * unit_scale.sqrt(), 0.0, 0.0]);
            let delta_s = Tensor::new([10.0 * unit_scale, 0.0, 0.0]);
            let state_scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
            rmhd_limit_source_increment(
                d,
                s,
                e,
                delta_s,
                0.0,
                &b,
                &gm,
                &gm,
                1e-12 * state_scale,
                ADMISSIBLE_REL_FLOOR * state_scale,
                ADMISSIBLE_REL_FLOOR * state_scale.powf(1.5),
                24,
            )
            .0
        };
        let reference = solve(1.0);
        for unit_scale in [1e-12, 1e-6, 1e6, 1e12] {
            assert!((solve(unit_scale) - reference).abs() < 2e-15);
        }
    }

    #[test]
    fn rmhd_anchor_energy_establishes_the_bisection_lower_invariant() {
        let gm = Matrix::identity();
        let d = 1.0e-8;
        let s = Tensor::zeros();
        let b = Tensor::new([1.0, 0.0, 0.0]);
        let e = 0.5 + d;
        let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);
        let (q_before, psi_before) = rmhd_admissible_residuals(d, &s, e, &b, &gm, &gm);
        assert!(
            q_before <= eps_q || psi_before <= eps_psi,
            "the setup must exercise an anchor below the numerical margin"
        );

        let safe =
            rmhd_anchor_energy_with_margin(d, &s, e, &b, &gm, &gm, scale, eps_q, eps_psi, 48);
        let (q_after, psi_after) = rmhd_admissible_residuals(d, &s, safe, &b, &gm, &gm);
        assert!(q_after > eps_q);
        assert!(psi_after > eps_psi);
        assert!(safe > e);
    }

    #[test]
    fn rmhd_anchor_energy_is_invariant_under_conserved_unit_scaling() {
        let gm = Matrix::identity();
        let solve = |units: f64| {
            let d = 1.0e-8 * units;
            let s = Tensor::zeros();
            let b = Tensor::new([units.sqrt(), 0.0, 0.0]);
            let e = (0.5 + 1.0e-8) * units;
            let scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
            let safe = rmhd_anchor_energy_with_margin(
                d,
                &s,
                e,
                &b,
                &gm,
                &gm,
                scale,
                ADMISSIBLE_REL_FLOOR * scale,
                ADMISSIBLE_REL_FLOOR * scale.powf(1.5),
                48,
            );
            safe / units
        };
        let reference = solve(1.0);
        for units in [1.0e-12, 1.0e-6, 1.0e6, 1.0e12] {
            assert!((solve(units) - reference).abs() <= 1.0e-12 * reference);
        }
    }

    #[test]
    fn rmhd_projection_starts_from_a_margin_admissible_anchor() {
        let gm = Matrix::identity();
        let d = 1.0e-8;
        let s_anchor = Tensor::zeros();
        let b = Tensor::new([1.0, 0.0, 0.0]);
        let e_anchor = 0.5 + d;
        let scale = rmhd_state_scale(d, &s_anchor, e_anchor, &b, &gm, &gm);
        let eps_d = 1.0e-12 * scale;
        let eps_q = ADMISSIBLE_REL_FLOOR * scale;
        let eps_psi = ADMISSIBLE_REL_FLOOR * scale.powf(1.5);
        let safe_energy = rmhd_anchor_energy_with_margin(
            d, &s_anchor, e_anchor, &b, &gm, &gm, scale, eps_q, eps_psi, 48,
        );
        let candidate_momentum = Tensor::new([10.0, 0.0, 0.0]);
        let theta = rmhd_admissible_theta(
            d,
            candidate_momentum,
            e_anchor,
            d,
            s_anchor,
            safe_energy,
            &b,
            &gm,
            &gm,
            eps_d,
            eps_q,
            eps_psi,
            48,
        );
        let projected_momentum = s_anchor + theta * (candidate_momentum - s_anchor);
        let projected_energy = safe_energy + theta * (e_anchor - safe_energy);
        let (q, psi) =
            rmhd_admissible_residuals(d, &projected_momentum, projected_energy, &b, &gm, &gm);
        assert!(theta < 1.0);
        assert!(q > eps_q);
        assert!(psi > eps_psi);
    }

    #[test]
    fn rmhd_source_time_is_invariant_under_conserved_unit_scaling() {
        let gm = Matrix::identity();
        let solve = |scale_units: f64| {
            let d = scale_units;
            let s = Tensor::new([0.1 * scale_units, 0.0, 0.0]);
            let e = 2.0 * scale_units;
            let b = Tensor::new([0.2 * scale_units.sqrt(), 0.0, 0.0]);
            let s_dot = Tensor::new([0.3 * scale_units, 0.0, 0.0]);
            let e_dot = -0.8 * scale_units;
            let state_scale = rmhd_state_scale(d, &s, e, &b, &gm, &gm);
            let source_scale = e_dot.abs().max(0.3 * scale_units);
            rmhd_source_admissible_time(
                d,
                s,
                e,
                s_dot,
                e_dot,
                &b,
                &gm,
                &gm,
                state_scale,
                source_scale,
                1e-12 * state_scale,
                ADMISSIBLE_REL_FLOOR * state_scale,
                ADMISSIBLE_REL_FLOOR * state_scale.powf(1.5),
                24,
            )
        };
        let reference = solve(1.0);
        for units in [1e-12, 1e-6, 1e6, 1e12] {
            let scaled = solve(units);
            assert!((scaled - reference).abs() <= 32.0 * f64::EPSILON * reference);
        }
    }

    #[test]
    fn stationary_killing_source_ray_preserves_the_evolved_energy() {
        let alpha = 0.37_f64;
        let beta = Tensor::new([0.2, -0.1, 0.05]);
        let source = Tensor::new([1.2, -0.7, 0.3]);
        let (momentum_dot, energy_dot) = stationary_killing_source_ray(alpha, &beta, source);
        let d = 0.8;
        let momentum = Tensor::new([0.4, -0.2, 0.1]);
        let energy = 1.7;
        let dt = 1e-4;
        let ehat = alpha * energy - d - (0..3).fold(0.0, |acc, kk| acc + beta[kk] * momentum[kk]);
        let momentum_next: Tensor<f64, 3> = Tensor::new(std::array::from_fn(|kk| {
            momentum[kk] + dt * momentum_dot[kk]
        }));
        let energy_next = energy + dt * energy_dot;
        let ehat_next = alpha * energy_next
            - d
            - (0..3).fold(0.0, |acc, kk| acc + beta[kk] * momentum_next[kk]);
        assert!((ehat_next - ehat).abs() <= 8.0 * f64::EPSILON * ehat.abs());
    }

    // a diagonal inverse-metric (identity for flat; a Schwarzschild-like radial stretch otherwise).
    fn metric(f_rr: f64) -> Matrix<f64, 3> {
        Matrix::diag(Tensor::new([f_rr, 1.0, 1.0]))
    }

    #[test]
    fn rhd_state_scale_tracks_conserved_rescaling() {
        let gm_inv = metric(0.5);
        let d = 0.4;
        let s = Tensor::new([0.3, -0.2, 0.1]);
        let e = 0.9;
        let reference = rhd_state_scale(d, &s, e, &gm_inv);

        for factor in [1.0e-12_f64, 1.0e-4, 1.0e4, 1.0e12] {
            let scaled = rhd_state_scale(d * factor, &(s * factor), e * factor, &gm_inv);
            assert!(
                (scaled / (reference * factor) - 1.0).abs() < 1.0e-14,
                "state scale is not homogeneous at factor {factor:e}"
            );
        }
    }

    #[test]
    fn rmhd_state_scale_tracks_conserved_rescaling() {
        let gm = metric(2.0);
        let gm_inv = metric(0.5);
        let d = 0.4;
        let s = Tensor::new([0.3, -0.2, 0.1]);
        let e = 0.9;
        let b = Tensor::new([0.5, 0.25, -0.1]);
        let reference = rmhd_state_scale(d, &s, e, &b, &gm_inv, &gm);

        for factor in [1.0e-12_f64, 1.0e-4, 1.0e4, 1.0e12] {
            let scaled_s = s * factor;
            let scaled_b = b * factor.sqrt();
            let scaled =
                rmhd_state_scale(d * factor, &scaled_s, e * factor, &scaled_b, &gm_inv, &gm);
            assert!(
                (scaled / (reference * factor) - 1.0).abs() < 1.0e-14,
                "state scale is not homogeneous at factor {factor:e}"
            );
        }
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
                let th = admissible_theta(
                    cand.0, cand.1, cand.2, anchor.0, anchor.1, anchor.2, &gi, eps_d, eps_f,
                );
                // admissible candidate -> theta = 1, exact passthrough
                assert!(
                    (th - 1.0).abs() < 1e-12,
                    "grr={grr} cand=({rho},{vr},{p}) f_cand={fc:.3e}: not passed through, theta={th}"
                );
            }
            // WILD inadmissible candidates: the projection must still yield a state in G.
            let wild: [(f64, Tensor<f64, 3>, f64); 5] = [
                (-2.0, Tensor::new([5.0, 0.0, 0.0]), 1.0), // negative density
                (0.5, Tensor::new([100.0, 0.0, 0.0]), 1.0), // |S| >> E (superluminal)
                (1.0, Tensor::new([0.0, 0.0, 0.0]), -3.0), // negative energy
                (1e-9, Tensor::new([50.0, 20.0, 0.0]), 0.1), // near-vacuum, huge momentum
                (0.0, Tensor::new([0.0, 0.0, 0.0]), 0.0),  // the zero state
            ];
            for (dc, sc, ec) in wild {
                let th =
                    admissible_theta(dc, sc, ec, anchor.0, anchor.1, anchor.2, &gi, eps_d, eps_f);
                assert!((0.0..=1.0).contains(&th), "theta out of [0,1]: {th}");
                let (dp, sp, ep) = admissible_project::<f64, 3>(
                    dc,
                    [sc[0], sc[1], sc[2]],
                    ec,
                    anchor.0,
                    [anchor.1[0], anchor.1[1], anchor.1[2]],
                    anchor.2,
                    th,
                );
                let sp_t = Tensor::new(sp);
                assert!(
                    dp > 0.0,
                    "projected density non-positive: {dp} (cand D={dc})"
                );
                let fp = f_of(dp, sp_t, ep, &gi);
                assert!(
                    fp > 0.0,
                    "projected state NOT in G: f={fp:.3e} (cand D={dc}, |S|={:?}, E={ec})",
                    sc
                );
            }
        }
    }

    // --- the exact (magnetized) admissible set -------------------------------------------------

    #[test]
    fn psi_reduces_to_the_hydro_cone_when_the_field_vanishes() {
        // at B = 0 the magnetic condition must collapse onto the hydrodynamic one. algebraically
        // Phi = sqrt(4E^2 - 3D^2 - 3|S|^2) and psi = (Phi + 2E) sqrt(Phi - E), so psi > 0 iff
        // Phi > E iff E^2 > D^2 + |S|^2 iff q > 0. this pins the formula against a known limit: a
        // transcription error in Phi or in the 27/2 term breaks the equivalence.
        let (gi, gc) = (metric(1.0), metric(1.0));
        let zero_b = Tensor::new([0.0, 0.0, 0.0]);
        let (mut n_in, mut n_out) = (0, 0);
        for &(d, sx, e) in &[
            (1.0_f64, 0.0, 2.0), // deep interior
            (1.0, 0.5, 1.2),     // admissible, moving
            (1.0, 0.0, 0.99),    // E < D: outside
            (1.0, 3.0, 2.0),     // |S| > E: outside
            (0.3, 0.2, 0.4),     // light and slow
            (2.0, 1.0, 2.30),    // just inside
            (2.0, 1.0, 2.23),    // just outside
        ] {
            let s = Tensor::new([sx, 0.0, 0.0]);
            let (q, psi) = rmhd_admissible_residuals(d, &s, e, &zero_b, &gi, &gc);
            assert_eq!(
                q > 0.0,
                psi > 0.0,
                "B=0 disagreement at (D={d}, S={sx}, E={e}): q={q:.6e} psi={psi:.6e}"
            );
            // and q must agree with the hydro cone's own residual f = E^2 - D^2 - |S|^2
            assert_eq!(
                q > 0.0,
                f_of(d, s, e, &gi) > 0.0,
                "q disagrees with f at (D={d}, S={sx}, E={e})"
            );
            if q > 0.0 { n_in += 1 } else { n_out += 1 }
        }
        // the equivalence is only meaningful if the sample straddles the boundary; an all-inside or
        // all-outside sample would satisfy it trivially.
        assert!(
            n_in > 0 && n_out > 0,
            "sample no longer straddles partial-G: {n_in} in, {n_out} out"
        );
    }

    #[test]
    fn psi_is_strictly_stronger_than_the_b_free_cone() {
        // THE JUSTIFICATION FOR THE WHOLE CONSTRUCTION: a strongly magnetized state can satisfy the
        // necessary cone q > 0 and still be inadmissible, because the magnetic energy exceeds what
        // the total energy can carry and the recovered gas pressure is non-positive. such a state is
        // exactly what the B-free projection waves through and the freeze tier then has to catch.
        let (gi, gc) = (metric(1.0), metric(1.0));
        let (d, s, e) = (1.0_f64, Tensor::new([0.0, 0.0, 0.0]), 2.0);
        let b = Tensor::new([10.0, 0.0, 0.0]); // |B|^2 = 100 >> E
        let (q, psi) = rmhd_admissible_residuals(d, &s, e, &b, &gi, &gc);
        assert!(
            q > 0.0,
            "the necessary cone should PASS this state: q={q:.6e}"
        );
        assert!(
            psi < 0.0,
            "the sufficient condition should REJECT it: psi={psi:.6e}"
        );
    }

    #[test]
    fn magnetized_projection_always_lands_in_the_admissible_set() {
        let (eps_d, eps_q, eps_psi) = (1e-12, 1e-14, 1e-14);
        for &grr in &[1.0_f64, 1.25, 2.0] {
            let gi = metric(1.0 / grr); // gamma^{rr}
            let gc = metric(grr); // gamma_rr
            let (d_a, s_a, e_a) = admissible(1.0, 0.1, 0.5, grr);
            // a weak field, so the anchor stays admissible inside the candidate's magnetic slice.
            let b = Tensor::new([0.05, 0.02, 0.0]);
            let wild: [(f64, Tensor<f64, 3>, f64); 5] = [
                (-2.0, Tensor::new([5.0, 0.0, 0.0]), 1.0),
                (0.5, Tensor::new([100.0, 0.0, 0.0]), 1.0),
                (1.0, Tensor::new([0.0, 0.0, 0.0]), -3.0),
                (1e-9, Tensor::new([50.0, 20.0, 0.0]), 0.1),
                (0.0, Tensor::new([0.0, 0.0, 0.0]), 0.0),
            ];
            let mut n_moved = 0;
            for (dc, sc, ec) in wild {
                let th = rmhd_admissible_theta(
                    dc, sc, ec, d_a, s_a, e_a, &b, &gi, &gc, eps_d, eps_q, eps_psi, 40,
                );
                assert!((0.0..=1.0).contains(&th), "theta out of [0,1]: {th}");
                if th < 1.0 {
                    n_moved += 1
                }
                let (dp, sp, ep) = admissible_project::<f64, 3>(
                    dc,
                    [sc[0], sc[1], sc[2]],
                    ec,
                    d_a,
                    [s_a[0], s_a[1], s_a[2]],
                    e_a,
                    th,
                );
                let (q, psi) = rmhd_admissible_residuals(dp, &Tensor::new(sp), ep, &b, &gi, &gc);
                assert!(
                    dp > 0.0 && q > 0.0 && psi > 0.0,
                    "projected state NOT admissible (grr={grr}, cand D={dc}, E={ec}): \
                     D={dp:.3e} q={q:.3e} psi={psi:.3e} theta={th}"
                );
            }
            // if every candidate passed through untouched the projection was never exercised and the
            // landing assertion above proved nothing.
            assert_eq!(
                n_moved,
                wild.len(),
                "grr={grr}: only {n_moved} of {} candidates were projected",
                wild.len()
            );
        }
    }

    #[test]
    fn an_admissible_magnetized_candidate_passes_through_bit_for_bit() {
        let (gi, gc) = (metric(1.0), metric(1.0));
        let (d_a, s_a, e_a) = admissible(1.0, 0.1, 0.5, 1.0);
        let b = Tensor::new([0.05, 0.0, 0.0]);
        for &(rho, v, p) in &[(1.2_f64, 0.2, 0.6), (0.9, -0.3, 0.4)] {
            let (dc, sc, ec) = admissible(rho, v, p, 1.0);
            let th = rmhd_admissible_theta(
                dc, sc, ec, d_a, s_a, e_a, &b, &gi, &gc, 1e-12, 1e-14, 1e-14, 40,
            );
            assert_eq!(
                th, 1.0,
                "admissible candidate ({rho},{v},{p}) not passed through: {th}"
            );
        }
    }

    #[test]
    fn the_magnetized_projection_reports_an_unrecoverable_anchor() {
        // the structural gap versus hydro, asserted rather than left implicit: when the anchor's
        // hydro state is NOT admissible inside the candidate's magnetic slice, no blend along the
        // segment can succeed and theta = 0 is returned, which is the caller's signal to fall back.
        // a field this strong swamps the anchor's energy, so every point of the segment fails psi.
        let (gi, gc) = (metric(1.0), metric(1.0));
        let (d_a, s_a, e_a) = admissible(1.0, 0.1, 0.5, 1.0);
        let b = Tensor::new([50.0, 0.0, 0.0]); // |B|^2 = 2500 >> E_anchor
        let (q_a, psi_a) = rmhd_admissible_residuals(d_a, &s_a, e_a, &b, &gi, &gc);
        assert!(
            q_a > 0.0 && psi_a < 0.0,
            "setup must make the ANCHOR inadmissible in this slice"
        );
        let th = rmhd_admissible_theta(
            2.0,
            Tensor::new([1.0, 0.0, 0.0]),
            3.0,
            d_a,
            s_a,
            e_a,
            &b,
            &gi,
            &gc,
            1e-12,
            1e-14,
            1e-14,
            40,
        );
        assert_eq!(
            th, 0.0,
            "an unrecoverable anchor must report theta = 0, got {th}"
        );
    }

    #[test]
    fn a_wrong_psi_is_caught_by_the_b_zero_limit() {
        // bug injection: the 27/2 coefficient is the most error-prone constant in theorem 2.1.
        // perturbing it must break the B = 0 equivalence that `psi_reduces_to_the_hydro_cone`
        // asserts, proving that test constrains the coefficient rather than passing vacuously.
        let injected = |d: f64, s: Tensor<f64, 3>, e: f64, b: Tensor<f64, 3>| -> f64 {
            let s2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            let b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
            let sb = s[0] * b[0] + s[1] * b[1] + s[2] * b[2];
            let w = b2 - e;
            let phi = (w * w + 3.0 * (e * e - d * d - s2)).max(0.0).sqrt();
            // WRONG: 27/2 -> 27
            (phi - 2.0 * w) * (phi + w).max(0.0).sqrt()
                - (27.0 * (d * d * b2 + sb * sb)).max(0.0).sqrt()
        };
        // with B = 0 the injected term vanishes too, so the equivalence survives -> the B=0 test
        // alone does NOT pin the coefficient. it is pinned by a magnetized state instead.
        let (gi, gc) = (metric(1.0), metric(1.0));
        let (d, s, e) = (1.0_f64, Tensor::new([0.3, 0.0, 0.0]), 1.6);
        let b = Tensor::new([1.0, 0.0, 0.0]);
        let (_, psi) = rmhd_admissible_residuals(d, &s, e, &b, &gi, &gc);
        let bad = injected(d, s, e, b);
        assert!(
            (psi - bad).abs() > 1e-6,
            "the injected 27/2 -> 27 must change psi on a magnetized state: psi={psi:.6e} bad={bad:.6e}"
        );
    }

    #[test]
    fn a_wrong_theta_is_caught_by_the_property() {
        // bug injection: theta = 1 (keep the candidate unconditionally) must FAIL to land a wildly
        // inadmissible candidate in G — proving the property test has teeth.
        let gi = metric(1.0);
        let anchor = admissible(1.0, 0.1, 0.5, 1.0);
        let (dc, sc, ec) = (0.5_f64, Tensor::new([100.0, 0.0, 0.0]), 1.0);
        let (dp, sp, ep) = admissible_project::<f64, 3>(
            dc,
            [sc[0], sc[1], sc[2]],
            ec,
            anchor.0,
            [anchor.1[0], anchor.1[1], anchor.1[2]],
            anchor.2,
            1.0, // the injected wrong theta
        );
        assert!(
            f_of(dp, Tensor::new(sp), ep, &gi) < 0.0,
            "the injected theta=1 should leave the state OUTSIDE G"
        );
    }
}
