// =============================================================================
// hydrostatic.rs
//
// hydrostatic reconstruction for the euler equations with a gravitational source:
// reconstruct the pressure's deviation from the local mechanical equilibrium through
// each cell rather than the pressure itself, so a discretely balanced atmosphere
// presents no face jump at all.
//
// the residual this removes is the reason a low-dissipation riemann solver cannot
// be used on a stagnant stratified column. a hydrostatic state solves the continuum
// equations, not the discrete ones: a limited reconstruction leaves an O(dx^2) face
// jump on a curved profile, the numerical flux at rest is the upwind dissipation
// acting on that jump, and the deposit is one-signed every step. classical dissipation
// damps the resulting ring; a scheme that reduces acoustic dissipation at low mach
// number removes exactly the damping and the ring survives as an entropy error. the
// cure adopted here removes the jump instead of restoring the damping, so the low-mach
// reduction and the entropy floor stop competing.
//
// the local profile is the mechanical equilibrium through the cell state
// (Kaeppeli & Mishra, A&A 587, A94, 2016): integrate dp = -rho dphi with the density
// held piecewise constant, cell by cell, along the path from the anchor,
//
//   p_eq(x) = p_i - int_{x_i}^{x} rho_pc dphi,     rho_pc = rho of the cell x sits in.
//
// the equilibrium density is the piecewise-constant distribution itself, so density
// and velocity take the plain reconstruction and the pressure alone carries the
// hydrostatic correction. the scheme commits to no thermal structure — an equilibrium
// with arbitrary entropy stratification is a fixed point of the discrete balance,
// which is what a column whose stratification is selected by its own dynamics needs.
// every evaluation is a multiply and an add on quantities the stencil already reads.
//
// at phi = phi_i the integral is empty and p_eq = p_i exactly, so the profile
// reproduces its own cell bit-for-bit; with a uniform potential every integral is
// exactly zero, the departures are the plain differences, and the gravity-free
// reduction is bit-identical to plain reconstruction.
//
// usage:
//   let eq = LocalEquilibrium::through(rho_i, pre_i, phi_i);
//   let p_face = eq.pressure_at(phi_face) + limited_slope_of_deviation;
// =============================================================================

use symbi_ir::algebra::Scalar;

/// the reconstruction footprint in cell widths: the farthest offset from its anchor cell
/// that the widest stencil in the scheme (the parabola's six-point window) evaluates the
/// local equilibrium at. every operator that extrapolates along the segment weighs its
/// validity over this same span, so one cell carries one weight whichever operator reads
/// it — the flux, the body source, the ghost fill and the coarse-fine transfer follow one
/// profile, which is what keeps their telescoping exact at every weight.
pub const BALANCE_STENCIL_REACH: f64 = 3.0;

/// the share of the segment's positive domain an extrapolation may spend with the
/// balancing at full strength.
pub const BALANCE_FADE_ONSET: f64 = 0.8;

/// the share at which the balancing is fully faded out. the linear segment crosses zero
/// once the potential climbs by `p/rho` above the anchor, so `1.0` is the profile's own
/// domain boundary.
pub const BALANCE_FADE_FULL: f64 = 1.0;

/// the fraction of the potential variation the local segment carries over an
/// extrapolation that climbs `dphi_up` above the reference point.
///
/// `drop = rho dphi_up / p` is the share of the segment's positive domain the
/// extrapolation spends; the segment reaches zero at `drop = 1`. scaling the potential
/// variation by the returned weight holds the spent share at or below the onset, so the
/// weighted equilibrium stays at or above `1 - BALANCE_FADE_ONSET = 0.2` of the anchor
/// pressure and the departure a limiter carries is still a correction to the profile.
/// a footprint the segment comfortably supports gets weight one exactly, and a factor of
/// exactly one leaves an IEEE product unchanged — the healthy column keeps its bitwise
/// arithmetic. at zero weight the profile collapses to the constant state, whose
/// departures are the plain differences, so a draining near-vacuum core is reconstructed
/// by the plain scheme rather than by a segment extrapolated past the gas it describes.
#[inline]
pub fn balance_weight<S: Scalar>(rho: S, pre: S, dphi_up: S) -> S {
    let onset = S::from_f64(BALANCE_FADE_ONSET);
    let inv_width = S::from_f64(1.0 / (BALANCE_FADE_FULL - BALANCE_FADE_ONSET));
    let drop = rho * dphi_up / pre;
    S::ONE - ((drop - onset) * inv_width).max(S::ZERO).min(S::ONE)
}

/// the potential rise a footprint anchored at `phi_ref` climbs, over its two endpoints.
///
/// the segment thins going up the potential and thickens going down, so its zero crossing
/// lies in the rising direction and a footprint that only descends spends none of the
/// positive domain. a reference point at the top of its own footprint returns zero.
#[inline]
pub fn potential_rise<S: Scalar>(phi_ref: S, phi_a: S, phi_b: S) -> S {
    (phi_a - phi_ref).max(phi_b - phi_ref).max(S::ZERO)
}

/// the mechanical hydrostatic profile passing through one cell's state.
///
/// carries the cell's own state, so every evaluation is anchored on it and the point
/// `phi = phi_ref` is reproduced exactly. the single-segment form: the anchor's own
/// density carries the whole potential difference, which is the piecewise-constant
/// integral for any target inside the anchor cell (a face, a ghost the cell extends
/// into). a target beyond a neighbor's face belongs to the chain form
/// ([`equilibrium_pressure_profile`]), whose segments switch density at each face.
#[derive(Clone, Copy, Debug)]
pub struct LocalEquilibrium<S> {
    rho_ref: S,
    pre_ref: S,
    phi_ref: S,
    /// the share of the potential variation the profile follows, from [`balance_weight`].
    /// `1` is the full segment; `0` is a constant state, whose departures are the plain
    /// differences of the state itself.
    weight: S,
}

impl<S: Scalar> LocalEquilibrium<S> {
    /// the profile through `(rho, pre)` at potential `phi`, following the whole
    /// potential variation.
    #[inline]
    pub fn through(rho: S, pre: S, phi: S) -> Self {
        Self {
            rho_ref: rho,
            pre_ref: pre,
            phi_ref: phi,
            weight: S::ONE,
        }
    }

    /// the profile through `(rho, pre)` at potential `phi`, following the share of the
    /// potential variation that [`balance_weight`] supports for a footprint climbing
    /// `dphi_up` above `phi`. at full weight this is [`through`](Self::through) on the
    /// same arithmetic; at zero weight the profile collapses to the constant state.
    #[inline]
    pub fn faded(rho: S, pre: S, phi: S, dphi_up: S) -> Self {
        let mut eq = Self::through(rho, pre, phi);
        eq.weight = balance_weight(rho, pre, dphi_up);
        eq
    }

    /// the equilibrium density at any potential: the piecewise-constant distribution
    /// is the anchor's own density everywhere the anchor's segment reaches.
    #[inline]
    pub fn density_at(&self, _phi: S) -> S {
        self.rho_ref
    }

    /// the equilibrium pressure at potential `phi`, one segment from the anchor:
    /// `p_ref + rho_ref (phi_ref - phi)`, floored positive. returns `pre_ref` bit-exactly
    /// at `phi = phi_ref`, since the correction is a product with an exact zero.
    ///
    /// the floor guards the one hazard a linear profile has: an extrapolation climbing
    /// far enough up the potential drives the line negative. that is past the domain
    /// where the segment describes a state, and clamping there leaves the deviation to
    /// carry the difference rather than producing a negative pressure. on the
    /// equilibrium itself the raw value is the cell's own positive pressure and the
    /// floor never engages, so exactness is untouched.
    #[inline]
    pub fn pressure_at(&self, phi: S) -> S {
        let raw = self.pre_ref + self.rho_ref * (self.weight * (self.phi_ref - phi));
        raw.max(S::from_f64(f64::MIN_POSITIVE))
    }

    /// both components at once.
    #[inline]
    pub fn state_at(&self, phi: S) -> (S, S) {
        (self.rho_ref, self.pressure_at(phi))
    }
}

/// the equilibrium pressure at every stencil center, integrated outward from the anchor
/// with the density piecewise constant: within cell `k` the segment carries `rho[k]`,
/// and the segments meet at the interior faces.
///
/// `phi_c[k]` is the potential at center `k`, `phi_f[k]` at the face between centers
/// `k` and `k + 1` (so `phi_f.len() == phi_c.len() - 1`). the returned profile equals
/// `pre_anchor` at the anchor exactly — the integral there is empty — and the class it
/// holds is the discrete mechanical equilibrium: any `(rho, p, phi)` whose neighboring
/// pressures satisfy the same segment sums, with no thermal structure implied.
pub fn equilibrium_pressure_profile<S: Scalar>(
    anchor: usize,
    pre_anchor: S,
    rho: &[S],
    phi_c: &[S],
    phi_f: &[S],
    weight: S,
) -> Vec<S> {
    let n = rho.len();
    debug_assert_eq!(phi_c.len(), n);
    debug_assert_eq!(phi_f.len(), n - 1);
    // the unweighted correction accumulates outward from the anchor, and the weight
    // scales it once at the end: `p_anchor + 1.0 * corr` is bitwise `p_anchor + corr`,
    // so a footprint the segment supports keeps the exact class arithmetic, and a
    // faded footprint degrades every point of the chain together.
    let mut corr = vec![S::ZERO; n];
    for k in anchor..n - 1 {
        corr[k + 1] = corr[k]
            + rho[k] * (phi_c[k] - phi_f[k])
            + rho[k + 1] * (phi_f[k] - phi_c[k + 1]);
    }
    for k in (0..anchor).rev() {
        corr[k] = corr[k + 1]
            + rho[k + 1] * (phi_c[k + 1] - phi_f[k])
            + rho[k] * (phi_f[k] - phi_c[k]);
    }
    corr.into_iter().map(|c| pre_anchor + weight * c).collect()
}

/// the stencil's pressure departures from the mechanical equilibrium through the anchor.
/// this is the whole well-balancing transform, and it is independent of which
/// reconstruction consumes it: feed the departures to any operator that reproduces
/// constants, add the profile back at the face, and the scheme is well-balanced. PLM and
/// PPM therefore need no separate derivation — only the same wrapper around their own
/// limiter. density and velocity take the plain reconstruction untouched, because the
/// equilibrium density is the piecewise-constant distribution itself.
///
/// the departure at the anchor is exactly zero — the profile passes through it — so the
/// operator's one-sided differences about it reduce to `0 - d` and `d - 0`, the plain
/// differences of the departures exactly rather than to rounding. that is what carries
/// the gravity-free bit-identity through the transform, and it is why a face's two sides
/// are built from two separate anchors instead of one shared pass.
pub fn hydrostatic_departures<S: Scalar>(
    anchor: usize,
    pre: &[S],
    rho: &[S],
    phi_c: &[S],
    phi_f: &[S],
    weight: S,
) -> Vec<S> {
    let p0 = equilibrium_pressure_profile(anchor, pre[anchor], rho, phi_c, phi_f, weight);
    pre.iter().zip(&p0).map(|(&p, &p0k)| p - p0k).collect()
}

/// the van leer harmonic slope, byte-matching the kernel limiter's theta < 0 arm
/// (`gv/mod.rs::van_leer`): `2 dl dr / (dl + dr)` for same-signed one-sided slopes, zero
/// otherwise, with the denominator selected to one on the zero branch so no division by a
/// vanishing sum is ever formed.
#[inline]
fn van_leer<S: Scalar>(dl: S, dr: S) -> S {
    let prod = dl * dr;
    let pos = prod.cmp_gt(S::ZERO);
    let denom = S::select(pos, dl + dr, S::ONE);
    let two = S::ONE + S::ONE;
    S::select(pos, two * prod / denom, S::ZERO)
}

/// the 3-way minmod for the theta-MC limiter, carrier-generic and branchless: the
/// common-signed minimum-magnitude argument iff x, y, z share a strict sign, else 0.
/// identical in form to the substrate limiter the flux kernels emit, so the reconstruction
/// under test differs from the plain one only in what is limited, never in how.
#[inline]
fn minmod3<S: Scalar>(x: S, y: S, z: S) -> S {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    let all_pos = mn.cmp_gt(S::ZERO);
    let all_neg = mx.cmp_lt(S::ZERO);
    S::select(all_pos, mn, S::select(all_neg, mx, S::ZERO))
}

/// one cell's face pressure from a three-point stencil, reconstructing the deviation
/// from the cell's own mechanical equilibrium. density and velocity take
/// [`plain_face`]; only the pressure carries the hydrostatic correction.
///
/// `pre[1]` is the cell being reconstructed; `pre[0]` and `pre[2]` its neighbors along
/// the direction. `phi_c` carries the potential at the same three centers, `phi_f` at
/// the two interior faces, `phi_face` at the target face, and `sign` selects which face
/// (`+1` the upper, `-1` the lower).
///
/// the deviation at the cell itself is identically zero — the profile passes through
/// it — so the slope is built from the two neighbors' departures. on a discretely
/// balanced column of any stratification every departure vanishes, the slope is zero,
/// and the face value is the profile evaluated there: the two sides of the face agree
/// and the flux is exact.
#[inline]
pub fn hydrostatic_face<S: Scalar>(
    rho: [S; 3],
    pre: [S; 3],
    phi_c: [S; 3],
    phi_f: [S; 2],
    phi_face: S,
    theta: S,
    sign: S,
) -> S {
    let eq = LocalEquilibrium::through(rho[1], pre[1], phi_c[1]);
    let half = S::HALF;

    // departures via the one transform text the kernel path also compiles, at full
    // weight: this reference is the definition the theorem battery compares against,
    // and the fade is the kernel's own off-domain guard.
    let d = hydrostatic_departures(1, &pre, &rho, &phi_c, &phi_f, S::ONE);
    // the centre departure is exactly zero, so the one-sided differences are the departures.
    // the limiter selection mirrors the kernel's `plm_theta_from_stencil` exactly: theta-MC
    // minmod for theta >= 0, the smooth van leer harmonic for theta < 0. a reference that
    // hard-wired minmod fed a negative theta straight into `minmod3(a*theta, ...)` -- a
    // sign-flipped slope -- and the theorem battery, running only positive theta, was blind
    // to it. T1/T2 now run both signs.
    let mm = minmod3(-d[0] * theta, (d[2] - d[0]) * half, d[2] * theta);
    let vl = van_leer(-d[0], d[2]);
    let s_pre = S::select(theta.cmp_lt(S::ZERO), vl, mm);

    eq.pressure_at(phi_face) + sign * half * s_pre
}

/// the same face value with no gravitational structure: plain theta-limited reconstruction
/// of the state. present so the equivalence in `hydrostatic_reconstruction.rs` compares
/// against a definition rather than against a second copy of the scheme under test.
#[inline]
pub fn plain_face<S: Scalar>(q: [S; 3], theta: S, sign: S) -> S {
    let half = S::HALF;
    let a = q[1] - q[0];
    let b = q[2] - q[1];
    // the same runtime limiter selection the kernel's plm carries: theta-MC minmod for
    // theta >= 0, van leer for theta < 0. the definition the equivalence theorems compare
    // against must span both arms, or the negative-theta arm is compared against nothing.
    let mm = minmod3(a * theta, (a + b) * half, b * theta);
    let vl = van_leer(a, b);
    q[1] + sign * half * S::select(theta.cmp_lt(S::ZERO), vl, mm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_profile_reproduces_its_own_cell_bit_exactly() {
        // the gravity-free equivalence rests on this: the correction is a product with an
        // exact zero at the anchor, so the anchor state returns bit-for-bit.
        for (rho, pre, phi) in [(1.0, 0.6, -3.0), (1e-4, 2e-7, -1e4), (817.3, 1.1e3, 55.0)] {
            let eq = LocalEquilibrium::through(rho, pre, phi);
            assert_eq!(eq.density_at(phi), rho, "density at its own potential");
            assert_eq!(eq.pressure_at(phi), pre, "pressure at its own potential");
        }
    }

    #[test]
    fn the_profile_is_in_hydrostatic_balance() {
        // dp/dphi = -rho is the defining property, and the linear segment satisfies it
        // exactly rather than to a difference tolerance.
        // the sweep stays inside the segment's positive domain (the raw pressure crosses
        // zero at phi = phi_ref + p/rho = -3.25), so the floor sleeps and the slope is the
        // pure line. dyadic values keep every operation exact.
        let eq = LocalEquilibrium::through(2.0, 1.5, -4.0);
        let h = 0.25;
        for phi in [-4.5, -4.25, -4.0, -3.75] {
            let dpdphi = (eq.pressure_at(phi + h) - eq.pressure_at(phi - h)) / (2.0 * h);
            assert_eq!(dpdphi, -2.0, "the segment's slope is the anchor density");
        }
    }

    #[test]
    fn an_arbitrarily_stratified_equilibrium_is_a_fixed_point() {
        // the class the scheme preserves is mechanical: pressures related by the
        // piecewise-constant-density segment sums, with the density free at every cell.
        // an entropy-stratified column — the state the isentropic profile could only
        // approximate — sits inside it exactly. build one from deliberately unstructured
        // densities, then check every departure is exactly zero and the two sides of each
        // interior face reconstruct one pressure.
        let rho = [3.1_f64, 0.7, 1.9, 0.42, 2.6];
        let phi_c = [-5.0_f64, -3.8, -2.9, -2.3, -1.9];
        let phi_f = [-4.3_f64, -3.3, -2.55, -2.05];
        // deep enough that the segment sums stay positive over the whole climb: a
        // column that walks into the floor is a clamp rather than a class member.
        let mut pre = [12.0_f64, 0.0, 0.0, 0.0, 0.0];
        for k in 0..4 {
            pre[k + 1] = pre[k]
                + rho[k] * (phi_c[k] - phi_f[k])
                + rho[k + 1] * (phi_f[k] - phi_c[k + 1]);
        }
        assert!(
            pre.iter().all(|&p| p > 1.0e-3),
            "the class column left the physical regime; the fixed point is vacuous"
        );
        for anchor in 0..5 {
            let d = hydrostatic_departures(anchor, &pre, &rho, &phi_c, &phi_f, 1.0);
            // the anchor's own departure is a subtraction of a value from itself: exact.
            assert_eq!(d[anchor], 0.0, "anchor {anchor} departure");
            // every other departure re-walks segment sums the column accumulated in a
            // different association order, so the vanishing is to roundoff of the local
            // pressure rather than bitwise — the statement the face-jump theorems pin at
            // the integration level (T1, 1e-14 relative).
            for (k, dk) in d.iter().enumerate() {
                assert!(
                    dk.abs() <= 8.0 * f64::EPSILON * pre[k],
                    "departure at {k} from anchor {anchor}: {dk:e}"
                );
            }
        }
        // both sides of the face between cells 1 and 2 land on the same pressure, so the
        // riemann solver sees a stationary contact and the flux is exact.
        let left = hydrostatic_face(
            [rho[0], rho[1], rho[2]],
            [pre[0], pre[1], pre[2]],
            [phi_c[0], phi_c[1], phi_c[2]],
            [phi_f[0], phi_f[1]],
            phi_f[1],
            1.5,
            1.0,
        );
        let right = hydrostatic_face(
            [rho[1], rho[2], rho[3]],
            [pre[1], pre[2], pre[3]],
            [phi_c[1], phi_c[2], phi_c[3]],
            [phi_f[1], phi_f[2]],
            phi_f[1],
            1.5,
            -1.0,
        );
        assert!(
            (left - right).abs() <= 8.0 * f64::EPSILON * pre[2],
            "the balanced face is single-valued to roundoff: {left:e} vs {right:e}"
        );
    }

    #[test]
    fn a_uniform_potential_reduces_to_the_plain_reconstruction_bit_for_bit() {
        // with phi constant every segment integral is a product with an exact zero, so the
        // departures are the plain pressure differences and the balanced face is the plain
        // face, bitwise.
        let rho = [1.3_f64, 2.0, 0.8];
        let pre = [0.9_f64, 1.4, 1.1];
        let (phi_c, phi_f) = ([-2.0_f64; 3], [-2.0_f64; 2]);
        for theta in [1.5, 2.0, -1.0] {
            for sign in [1.0, -1.0] {
                let balanced =
                    hydrostatic_face(rho, pre, phi_c, phi_f, -2.0, theta, sign);
                let plain = plain_face(pre, theta, sign);
                assert_eq!(balanced, plain, "theta {theta}, sign {sign}");
            }
        }
    }

    #[test]
    fn the_floor_keeps_an_overreaching_extrapolation_positive() {
        // a segment climbing far above the anchor drives the line negative; past that
        // point the value is a floor rather than a state, and the deviation carries the
        // difference.
        let eq = LocalEquilibrium::through(1.0, 0.5, 0.0);
        assert!(eq.pressure_at(10.0) > 0.0);
        assert_eq!(eq.pressure_at(10.0), f64::MIN_POSITIVE);
    }
}
