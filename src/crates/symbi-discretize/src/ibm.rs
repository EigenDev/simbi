// =============================================================================
// ibm.rs
//
// the carrier-generic core of the immersed-boundary source, split from the
// Gv-traced kernel builders so the SAME expressions serve two roles:
//   - traced at S = Gv (the `body_source` / fused-godunov kernels), and
//   - evaluated at S = f64 or S = Dual<f64> (the well-posedness suite: the conservative-gravity
//     proof differentiates the potential by forward-mode autodiff; the drain-contraction and
//     softened-gravity bounds sample the f64 forms).
// every function is a PURE map over one Scalar carrier — no field reads, no trace state — so the
// property a proof establishes on the f64/Dual carrier is a property of the traced kernel too.
//
// the physics:
//   - softened Newtonian gravity  g = -M r / (r^2 + eps^2)^{3/2}, the negative gradient of the
//     softened potential  phi = -M / sqrt(r^2 + eps^2). the softening eps regularizes the bare
//     1/r^2 singularity, making g bounded and Lipschitz.
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

/// softened Newtonian gravity at a cell whose CARTESIAN displacement from the body is `rvec`:
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

/// the accretion DRAIN RATE for one body: `chi * min(sink, cs/dx)`, with the mollified mask
/// `chi = (1 - tanh((r - r_mask)/w))/2 in (0, 1)` and the sound-crossing cap `cs/dx`. nonnegative
/// whenever `sink >= 0`, `cs >= 0`, `w > 0` (the sign lemma the contraction proof rests on). the ops
/// match the traced body kernel bit-for-bit.
#[inline]
pub fn drain_rate<S: Scalar>(r_mag: S, r_mask: S, min_w: S, sink: S, cs: S) -> S {
    let z = (r_mag - r_mask) / min_w;
    let chi = S::from_f64(0.5) * (S::ONE - z.tanh());
    let sound_rate = cs / min_w;
    chi * sink.min(sound_rate)
}

/// the EXACT support of the drain mask, in mask widths: `tanh(z)` returns exactly 1.0 in
/// f64 for `z >= 19.1` (the taylor tail `2 exp(-2z)` falls below half an ulp of 1), so
/// `chi = (1 - tanh(z))/2` — and with it the drain rate — is exactly zero for
/// `r >= r_mask + DRAIN_SUPPORT_WIDTHS * w`. a spatial gate at this radius skips the
/// tanh/exp evaluation on the far field WITHOUT changing any bit of any field: outside
/// the support the ungated kernel computes rate = 0 and factor = exp(0) = 1 exactly.
pub const DRAIN_SUPPORT_WIDTHS: f64 = 20.0;

/// the exact-exponential drain factor `f = exp(-total_rate * dt)`. for `total_rate >= 0`, `dt >= 0`
/// this lies in `(0, 1]`: `f > 0` (positivity-preserving) and `f <= 1` (non-expansive on the
/// intensive state), for ANY dt — no CFL condition on the sink.
#[inline]
pub fn drain_factor<S: Scalar>(total_rate: S, dt: S) -> S {
    (S::ZERO - total_rate * dt).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // the spatial gate's bit-exactness rests on tanh saturation: at and beyond
    // the support radius the UNGATED rate is exactly zero and the ungated
    // factor exactly one, so a branch that skips them changes nothing.
    #[test]
    fn drain_is_exactly_zero_beyond_the_support_radius() {
        let (r_mask, w, sink, cs) = (2.0_f64, 0.1, 1e6, 0.7);
        for extra in [0.0, 1.0, 100.0] {
            let r = r_mask + (DRAIN_SUPPORT_WIDTHS + extra) * w;
            let rate = drain_rate(r, r_mask, w, sink, cs);
            assert_eq!(rate, 0.0, "rate must saturate to exact zero at r = {r}");
            assert_eq!(drain_factor(rate, 1e3), 1.0, "factor must be exact one");
        }
        // just INSIDE the support the mask is alive — the gate is not oversized.
        let r_in = r_mask + (DRAIN_SUPPORT_WIDTHS - 2.0) * w;
        assert!(drain_rate(r_in, r_mask, w, sink, cs) > 0.0);
    }
}
