// =============================================================================
// ibm.rs
//
// the carrier-generic core of the immersed-boundary source, split from the
// Gv-traced kernel builders so the same expressions serve two roles:
//   - traced at S = Gv (the `body_source` / fused-godunov kernels), and
//   - evaluated at S = f64 or S = Dual<f64> (the well-posedness suite: the conservative-gravity
//     proof differentiates the potential by forward-mode autodiff; the drain-contraction and
//     softened-gravity bounds sample the f64 forms).
// every function is a pure map over one Scalar carrier, free of field reads and trace state, so the
// property a proof establishes on the f64/Dual carrier is a property of the traced kernel too.
//
// the physics:
//   - softened newtonian gravity  g = -M r / (r^2 + eps^2)^{3/2}, the negative gradient of the
//     softened potential  phi = -M / sqrt(r^2 + eps^2). the softening eps regularizes the bare
//     1/r^2 singularity, making g bounded and lipschitz.
//   - the well-posed accretion drain: an exact-exponential relaxation  U -> U exp(-rate dt), with
//     rate = chi min(sink, cs/dx) and the mollified sink mask  chi = (1 - tanh((r - r_mask)/w))/2.
//     f = exp(-rate dt) in (0, 1] for any dt, so the drain is positivity-preserving and
//     non-expansive with no CFL condition on the sink (the property that retired the KMK04 min-gate).
// =============================================================================

use symbi_ir::algebra::Scalar;

#[inline]
fn sq<S: Scalar>(a: S) -> S {
    a * a
}
#[inline]
fn cube<S: Scalar>(a: S) -> S {
    a * a * a
}

/// softened newtonian gravity at a cell whose cartesian displacement from the body is `rvec`:
/// `g = -mass * rvec / r_eff^3`, `r_eff = sqrt(|rvec|^2 + soft^2)`. this is exactly `-grad` of
/// [`softened_potential`] (the conservative-field proof differentiates that potential and recovers
/// this), so the immersed-body force does no spurious work around a closed loop. the ops match the
/// traced body kernel bit-for-bit.
#[inline]
pub fn softened_gravity<S: Scalar>(rvec: [S; 3], mass: S, soft: S) -> [S; 3] {
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let r_eff = (r_dist2 + sq(soft)).sqrt();
    let grav_fac = -mass / cube(r_eff);
    std::array::from_fn(|i| rvec[i] * grav_fac)
}

/// the softened gravitational potential `phi = -mass / sqrt(|rvec|^2 + soft^2)`. its negative
/// gradient is [`softened_gravity`]; carried here so the conservative-field proof can autodiff it.
#[inline]
pub fn softened_potential<S: Scalar>(rvec: [S; 3], mass: S, soft: S) -> S {
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let r_eff = (r_dist2 + sq(soft)).sqrt();
    -mass / r_eff
}

/// compact newtonian gravity: the field of the density profile `rho ~ 1 - (r/h)^2` truncated at
/// `h`, which is **exactly** `-mass * rvec / |rvec|^3` for `|rvec| >= h` and regular within.
///
/// ```text
///   g = -mass rvec / h^3 * [5/2 - (3/2)(r/h)^2]     r < h
///     = -mass rvec / r^3                            r >= h
/// ```
///
/// what separates this from [`softened_gravity`] is compact support: this field is the bare point
/// mass beyond `h`, where a Plummer sphere's reach is infinite. `g_plummer / g_newton =
/// [1 + (h/r)^2]^{-3/2}` stays below unity at every radius, so a Plummer
/// softening length chosen for regularity near the body silently weakens gravity through the whole
/// domain — at `r = h` it is already down to 0.354, and it needs `r > 5h` to reach 0.99. when the
/// quantity being measured is a power law in radius, that is a systematic bias across the entire
/// fitting range, growing toward small `r` where the range is most valuable.
///
/// here the truncation is exact: outside `h` the field is the bare point mass to the last bit,
/// because the enclosed mass is complete. so `h` may be set to the largest radius the flow is
/// asked to leave unresolved — an accretion radius, say — and the measurement outside it is
/// untouched by the choice.
///
/// `rho(h) = 0` makes `dg/dr` continuous at the match (`phi` is `C^2`), where the uniform sphere's
/// density jumps there. the peak field is `1.242 mass / h^2` at `r = sqrt(5/9) h`, bounded
/// and independent of resolution — where a Plummer accurate enough at `h` peaks at `~ mass / eps^2`
/// with `eps << h`, which is what makes the timestep collapse.
#[inline]
pub fn compact_gravity<S: Scalar>(rvec: [S; 3], mass: S, h: S) -> [S; 3] {
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let r = r_dist2.sqrt();
    let inner = -mass / cube(h) * (S::from_f64(2.5) - S::from_f64(1.5) * r_dist2 / sq(h));
    let outer = -mass / cube(r);
    let fac = S::cond(r.cmp_lt(h), || inner, || outer);
    std::array::from_fn(|i| rvec[i] * fac)
}

/// the potential whose negative gradient is [`compact_gravity`]:
///
/// ```text
///   phi = (mass/h) [ (5/4)(r/h)^2 - (3/8)(r/h)^4 - 15/8 ]   r < h
///       = -mass / r                                          r >= h
/// ```
///
/// continuous with continuous first and second derivatives at `r = h`, and exactly the point-mass
/// potential outside. carried here so the conservative-field proof can autodiff it.
#[inline]
pub fn compact_potential<S: Scalar>(rvec: [S; 3], mass: S, h: S) -> S {
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let r = r_dist2.sqrt();
    let u2 = r_dist2 / sq(h);
    let inner =
        mass / h * (S::from_f64(1.25) * u2 - S::from_f64(0.375) * sq(u2) - S::from_f64(1.875));
    let outer = -mass / r;
    S::cond(r.cmp_lt(h), || inner, || outer)
}

/// the accretion drain rate for one body: `chi * min(sink, cs/dx)`, with the mollified mask
/// `chi = (1 - tanh((r - r_mask)/w))/2 in (0, 1)` and the sound-crossing cap `cs/dx`. nonnegative
/// whenever `sink >= 0`, `cs >= 0`, `w > 0` (the sign lemma the contraction proof rests on). the ops
/// match the traced body kernel bit-for-bit.
#[inline]
pub fn drain_rate<S: Scalar>(r_mag: S, r_mask: S, min_w: S, sink: S, cs: S) -> S {
    let z = (r_mag - r_mask) / min_w;
    let chi = S::HALF * (S::ONE - z.tanh());
    let sound_rate = cs / min_w;
    chi * sink.min(sound_rate)
}

/// the exact support of the drain mask, in mask widths: `tanh(z)` returns exactly 1.0 in
/// f64 for `z >= 19.1` (the taylor tail `2 exp(-2z)` falls below half an ulp of 1), so
/// `chi = (1 - tanh(z))/2` — and with it the drain rate — is exactly zero for
/// `r >= r_mask + DRAIN_SUPPORT_WIDTHS * w`. a spatial gate at this radius skips the
/// tanh/exp evaluation on the far field while every field keeps its exact bits: outside
/// the support the ungated kernel computes rate = 0 and factor = exp(0) = 1 exactly.
pub const DRAIN_SUPPORT_WIDTHS: f64 = 20.0;

/// the exact-exponential drain factor `f = exp(-total_rate * dt)`. for `total_rate >= 0`, `dt >= 0`
/// this lies in `(0, 1]`: `f > 0` (positivity-preserving) and `f <= 1` (non-expansive on the
/// intensive state) at every dt, so the sink imposes no CFL condition.
#[inline]
pub fn drain_factor<S: Scalar>(total_rate: S, dt: S) -> S {
    (S::ZERO - total_rate * dt).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // the spatial gate's bit-exactness rests on tanh saturation: at and beyond
    // the support radius the ungated rate is exactly zero and the ungated
    // factor exactly one, so a branch that skips them reproduces those values.
    #[test]
    fn drain_is_exactly_zero_beyond_the_support_radius() {
        let (r_mask, w, sink, cs) = (2.0_f64, 0.1, 1e6, 0.7);
        for extra in [0.0, 1.0, 100.0] {
            let r = r_mask + (DRAIN_SUPPORT_WIDTHS + extra) * w;
            let rate = drain_rate(r, r_mask, w, sink, cs);
            assert_eq!(rate, 0.0, "rate must saturate to exact zero at r = {r}");
            assert_eq!(drain_factor(rate, 1e3), 1.0, "factor must be exact one");
        }
        // just inside the support the mask is alive, so the gate sits at the true edge.
        let r_in = r_mask + (DRAIN_SUPPORT_WIDTHS - 2.0) * w;
        assert!(drain_rate(r_in, r_mask, w, sink, cs) > 0.0);
    }
}

/// the body's gravitational field, selecting the family by the wire scalar `kind`
/// (`0` = Plummer, `1` = compact; see `symbi_ib::SofteningKind`).
///
/// branch-free at the trace: `cond` lowers to a select, so a single baked kernel serves both
/// families with the choice carried as a runtime scalar. an inactive or non-gravitating
/// body slot carries `mass = 0`, which zeroes both arms identically, so the family it nominally
/// names is immaterial there.
#[inline]
pub fn body_gravity<S: Scalar>(rvec: [S; 3], mass: S, len: S, kind: S) -> [S; 3] {
    let compact = compact_gravity(rvec, mass, len);
    let plummer = softened_gravity(rvec, mass, len);
    let is_compact = kind.cmp_gt(S::HALF);
    std::array::from_fn(|i| S::cond(is_compact, || compact[i], || plummer[i]))
}

/// the potential paired with [`body_gravity`] under the same `kind` selector; carried so the
/// conservative-field proof can autodiff whichever family a body declares.
#[inline]
pub fn body_potential<S: Scalar>(rvec: [S; 3], mass: S, len: S, kind: S) -> S {
    // both profiles share one far field. writing s for the softening switch,
    //
    //   plummer : phi = -mass / sqrt(r^2 + h^2)     s = 1
    //   compact : phi = -mass / sqrt(r^2)           s = 0   (bare point mass beyond h)
    //
    // so `-mass / sqrt(r^2 + s h^2)` is both, and only the compact profile's interior --
    // a polynomial in (r/h)^2 that needs no root of its own -- stands outside it. selecting
    // the switch rather than the potential leaves one square root and one reciprocal where
    // evaluating both profiles took two of each. the fold is exact: `r^2 + 0` and `1 * h^2`
    // are both identities in ieee, so every carrier reproduces its previous value bit for bit.
    let is_compact = kind.cmp_gt(S::HALF);
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let h2 = sq(len);
    let r_eff = (r_dist2 + S::select(is_compact, S::ZERO, S::ONE) * h2).sqrt();
    // on the compact branch `r_eff` is the bare distance, so the interior test is the same
    // `r < h` comparison the profile has always made; off that branch the mask discards it.
    let u2 = r_dist2 / h2;
    let inner =
        mass / len * (S::from_f64(1.25) * u2 - S::from_f64(0.375) * sq(u2) - S::from_f64(1.875));
    S::select(is_compact & r_eff.cmp_lt(len), inner, -mass / r_eff)
}
