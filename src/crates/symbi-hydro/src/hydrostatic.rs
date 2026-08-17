// =============================================================================
// hydrostatic.rs
//
// hydrostatic reconstruction for the euler equations with a gravitational source:
// reconstruct the deviation from the local hydrostatic profile through each cell
// rather than the state itself, so a discretely balanced atmosphere presents no
// face jump at all.
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
// the local profile is the isentrope through the cell state that satisfies
// dp/dx = -rho dphi/dx exactly (Kaeppeli & Mishra, J. Comput. Phys. 259:199, 2014):
//
//   rho_eq(phi) = rho_i [1 + (gamma - 1)(phi_i - phi) / cs_i^2]^(1/(gamma - 1)),
//   p_eq(phi)   = p_i   [rho_eq(phi) / rho_i]^gamma,     cs_i^2 = gamma p_i / rho_i.
//
// written as a ratio against the cell's own state rather than through an absolute
// enthalpy constant. that is not cosmetic: at phi = phi_i the bracket is exactly 1.0
// and 1.0^x is exact in IEEE arithmetic, so the profile reproduces the cell state
// bit-for-bit, which is what makes the gravity-free reduction bit-identical to plain
// reconstruction. an absolute-constant form would also lose precision to cancellation
// in a deep potential, where the enthalpy and the potential are large and nearly equal.
//
// usage:
//   let eq = LocalEquilibrium::through(rho_i, pre_i, phi_i, gamma);
//   let rho_face = eq.density_at(phi_face) + limited_slope_of_deviation;
// =============================================================================

use symbi_ir::algebra::Scalar;

/// the reconstruction footprint in cell widths: the farthest offset from its anchor cell that
/// the widest stencil in the scheme (the parabola's six-point window) evaluates the local
/// isentrope at. every operator that extrapolates along the isentrope weighs its validity over
/// this same span, so one cell carries one weight whichever operator reads it.
pub const BALANCE_STENCIL_REACH: f64 = 3.0;

/// the enthalpy fraction an extrapolation may spend with the balancing at full strength.
pub const BALANCE_FADE_ONSET: f64 = 0.8;

/// the enthalpy fraction at which the balancing is fully faded out. the isentrope reaches
/// vacuum when the extrapolation spends the whole enthalpy, so `1.0` is the profile's own
/// domain boundary.
pub const BALANCE_FADE_FULL: f64 = 1.0;

/// the fraction of the potential variation the local isentrope carries over an extrapolation
/// that climbs `dphi_up` above the reference point.
///
/// `drop = (gamma - 1) dphi_up / cs^2` is the share of the cell's enthalpy the extrapolation
/// spends; the profile's enthalpy ratio there is `1 - drop`, and the isentrope terminates at
/// vacuum at `drop = 1`. scaling the potential variation by the returned weight `w` scales the
/// drop to `w * drop`, which this ramp holds at or below `BALANCE_FADE_ONSET` for every input
/// (the product `drop * (BALANCE_FADE_FULL - drop) / (BALANCE_FADE_FULL - BALANCE_FADE_ONSET)`
/// peaks at the onset, since the onset sits above half the full threshold). the faded enthalpy
/// ratio therefore stays at or above `1 - BALANCE_FADE_ONSET = 0.2` and the equilibrium state
/// stays within `0.2^(1/(gamma - 1))` of the cell's own — a factor 8.9 in density at
/// gamma = 5/3, where the departure a limiter carries is still a correction to the profile.
///
/// measured drops over a three-cell footprint, all at gamma = 5/3: a stratified sealed column
/// at 128 cells over a plummer-softened GM = 100 body reaches 0.224; a gravitating gas around a
/// four-cell sealed accretor reaches 0.68 at the mask edge and 2.19 in the two cells at the
/// body's own position; a bare-newtonian drain of GM = 1 with its accretion radius at four fine
/// cells sits at 0.43 on the ambient isentrope, 0.86 at half that isentrope's sound speed and
/// 1.71 at a quarter of it — the last being the draining state whose extrapolation the ramp
/// switches off.
#[inline]
pub fn balance_weight<S: Scalar>(cs2: S, gamma: S, dphi_up: S) -> S {
    let onset = S::from_f64(BALANCE_FADE_ONSET);
    let inv_width = S::from_f64(1.0 / (BALANCE_FADE_FULL - BALANCE_FADE_ONSET));
    let drop = (gamma - S::ONE) * dphi_up / cs2;
    S::ONE - ((drop - onset) * inv_width).max(S::ZERO).min(S::ONE)
}

/// the potential rise a footprint anchored at `phi_ref` climbs, over its two endpoints.
///
/// the isentrope thins going up the potential and thickens going down, so its vacuum boundary
/// lies in the rising direction and a footprint that only descends spends none of the cell's
/// enthalpy. a reference point that sits at the top of its own footprint returns zero.
#[inline]
pub fn potential_rise<S: Scalar>(phi_ref: S, phi_a: S, phi_b: S) -> S {
    (phi_a - phi_ref).max(phi_b - phi_ref).max(S::ZERO)
}

/// the isentropic hydrostatic profile passing through one cell's state.
///
/// carries the cell's own state rather than a derived constant, so every evaluation is a
/// ratio against it and the point `phi = phi_ref` is reproduced exactly.
///
/// gamma-law only. the profile solves `dp/dphi = -rho` along `p = K rho^gamma`, and the
/// `cs2_ref = gamma p / rho` it derives is the ideal-gas sound speed spelled out -- on the
/// isothermal or taub-mathews closures this curve is not the eos's isentrope, so a transform
/// built from it reintroduces the face jump it exists to remove. the dispatch refuses the
/// pairing; a non-ideal balanced reconstruction needs its own profile through the eos's
/// actual `sound_speed_sq` and isentrope integral.
#[derive(Clone, Copy, Debug)]
pub struct LocalEquilibrium<S> {
    rho_ref: S,
    pre_ref: S,
    phi_ref: S,
    /// `cs^2 = gamma p / rho` at the reference point.
    cs2_ref: S,
    gamma: S,
    /// the share of the potential variation the profile follows, from [`balance_weight`].
    /// `1` is the full isentrope; `0` is a constant state, whose departures are the plain
    /// differences of the state itself.
    weight: S,
}

impl<S: Scalar> LocalEquilibrium<S> {
    /// the profile through `(rho, pre)` at potential `phi`, following the whole potential
    /// variation.
    #[inline]
    pub fn through(rho: S, pre: S, phi: S, gamma: S) -> Self {
        Self {
            rho_ref: rho,
            pre_ref: pre,
            phi_ref: phi,
            cs2_ref: gamma * pre / rho,
            gamma,
            weight: S::ONE,
        }
    }

    /// the profile through `(rho, pre)` at potential `phi`, following the share of the
    /// potential variation that [`balance_weight`] supports for a footprint climbing
    /// `dphi_up` above `phi`.
    ///
    /// at full weight this is [`through`](Self::through) evaluated on the same arithmetic:
    /// the weight enters as a factor on the potential difference, and a factor of exactly one
    /// leaves an IEEE product unchanged. at zero weight the profile collapses to the constant
    /// state `(rho, pre)`, so departures taken against it are the plain differences and the
    /// reconstruction that consumes them is the plain one.
    #[inline]
    pub fn faded(rho: S, pre: S, phi: S, gamma: S, dphi_up: S) -> Self {
        let mut eq = Self::through(rho, pre, phi, gamma);
        eq.weight = balance_weight(eq.cs2_ref, gamma, dphi_up);
        eq
    }

    /// `[1 + w (gamma - 1)(phi_ref - phi)/cs_ref^2]`, the dimensionless enthalpy ratio the
    /// profile is built from, over the weighted potential variation. exactly `1` at the
    /// reference point at any weight.
    ///
    /// the floor keeps the bracket positive where the extrapolation runs past the profile's
    /// own vacuum boundary — a potential rise steeper than the cell's enthalpy supports, which
    /// the weight holds off within the reconstruction footprint and which an evaluation beyond
    /// that footprint can still reach. that is outside the domain of the isentrope, not a
    /// state, and clamping there leaves the deviation to carry the difference rather than
    /// producing a negative density.
    #[inline]
    pub fn enthalpy_ratio(&self, phi: S) -> S {
        let one = S::ONE;
        let ratio = one + (self.gamma - one) * (self.weight * (self.phi_ref - phi)) / self.cs2_ref;
        ratio.max(S::from_f64(f64::MIN_POSITIVE))
    }

    /// the equilibrium density at potential `phi`. returns `rho_ref` bit-exactly at
    /// `phi = phi_ref`.
    #[inline]
    pub fn density_at(&self, phi: S) -> S {
        let one = S::ONE;
        self.rho_ref * self.enthalpy_ratio(phi).powf(one / (self.gamma - one))
    }

    /// the equilibrium pressure at potential `phi`. returns `pre_ref` bit-exactly at
    /// `phi = phi_ref`.
    #[inline]
    pub fn pressure_at(&self, phi: S) -> S {
        let one = S::ONE;
        self.pre_ref * self.enthalpy_ratio(phi).powf(self.gamma / (self.gamma - one))
    }

    /// both components at once, sharing one `powf`.
    ///
    /// the pressure exponent exceeds the density exponent by exactly one —
    /// `gamma/(gamma-1) - 1/(gamma-1) = 1` — so `ratio^(gamma/(gamma-1))` is
    /// `ratio^(1/(gamma-1)) * ratio` and the second transcendental is a multiply. an exact
    /// identity, not an approximation, so this is the form to call on any path that needs the
    /// pair: a reconstruction evaluates the profile at every stencil offset for both anchors,
    /// where halving the transcendental count is the difference between a usable kernel and a
    /// curiosity.
    #[inline]
    pub fn state_at(&self, phi: S) -> (S, S) {
        let one = S::ONE;
        let ratio = self.enthalpy_ratio(phi);
        let rho_factor = ratio.powf(one / (self.gamma - one));
        (self.rho_ref * rho_factor, self.pre_ref * rho_factor * ratio)
    }
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

/// the stencil's departures from a hydrostatic profile, each evaluated at its own potential.
/// this is the whole well-balancing transform, and it is independent of which reconstruction
/// consumes it: feed the departures to any operator that reproduces constants, add the profile
/// back at the face, and the scheme is well-balanced. PLM and PPM therefore need no separate
/// derivation — only the same wrapper around their own limiter.
///
/// `eq` passes through the cell being reconstructed, always. the departure at that cell is then
/// exactly zero, so the operator's one-sided differences about it reduce to `0 - d` and `d - 0`,
/// which are the plain differences exactly rather than to rounding. that is what carries the
/// gravity-free bit-identity through the transform, and it is why a face's two sides must be
/// built from two separate anchors instead of one shared pass: an anchor on the far cell would
/// leave both differences as `(q_j - c) - (q_k - c)`, equal to `q_j - q_k` only to roundoff.
pub fn hydrostatic_departures<S: Scalar>(
    eq: &LocalEquilibrium<S>,
    rho: &[S],
    pre: &[S],
    phi: &[S],
) -> (Vec<S>, Vec<S>) {
    let mut d_rho = Vec::with_capacity(rho.len());
    let mut d_pre = Vec::with_capacity(rho.len());
    for k in 0..rho.len() {
        let (r_eq, p_eq) = eq.state_at(phi[k]);
        d_rho.push(rho[k] - r_eq);
        d_pre.push(pre[k] - p_eq);
    }
    (d_rho, d_pre)
}


/// one cell's face pair `(rho, pre)` from a three-point stencil, reconstructing the
/// deviation from the cell's own hydrostatic profile.
///
/// `q[1]` is the cell being reconstructed; `q[0]` and `q[2]` are its neighbors along the
/// direction. `phi` carries the potential at the same three points, `phi_face` at the target
/// face, and `sign` selects which face (`+1` the upper, `-1` the lower).
///
/// the deviation at the cell itself is identically zero — the profile passes through it — so
/// the slope is built from the two neighbors' departures from that profile. on a discretely
/// balanced isentrope every departure vanishes, the slope is zero, and the face value is the
/// profile evaluated there: the two sides of the face agree and the flux is exact.
#[inline]
pub fn hydrostatic_face<S: Scalar>(
    rho: [S; 3],
    pre: [S; 3],
    phi: [S; 3],
    phi_face: S,
    gamma: S,
    theta: S,
    sign: S,
) -> (S, S) {
    let eq = LocalEquilibrium::through(rho[1], pre[1], phi[1], gamma);
    let half = S::from_f64(0.5);

    // departures via the one transform text the kernel path also compiles.
    let (d_rho, d_pre) = hydrostatic_departures(&eq, &rho, &pre, &phi);
    // the centre departure is exactly zero, so the one-sided differences are the departures.
    // the limiter selection mirrors the kernel's `plm_theta_from_stencil` exactly: theta-MC
    // minmod for theta >= 0, the smooth van leer harmonic for theta < 0. a reference that
    // hard-wired minmod fed a negative theta straight into `minmod3(a*theta, ...)` -- a
    // sign-flipped slope -- and the theorem battery, running only positive theta, was blind
    // to it. T1/T2 now run both signs.
    let slope = |dm: S, dp: S| -> S {
        let mm = minmod3(-dm * theta, (dp - dm) * half, dp * theta);
        let vl = van_leer(-dm, dp);
        S::select(theta.cmp_lt(S::ZERO), vl, mm)
    };
    let s_rho = slope(d_rho[0], d_rho[2]);
    let s_pre = slope(d_pre[0], d_pre[2]);

    (
        eq.density_at(phi_face) + sign * half * s_rho,
        eq.pressure_at(phi_face) + sign * half * s_pre,
    )
}

/// the same face pair with no gravitational structure: plain theta-limited reconstruction of
/// the state. present so the equivalence in `hydrostatic_reconstruction.rs` compares against a
/// definition rather than against a second copy of the scheme under test.
#[inline]
pub fn plain_face<S: Scalar>(q: [S; 3], theta: S, sign: S) -> S {
    let half = S::from_f64(0.5);
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

    const GAMMA: f64 = 5.0 / 3.0;

    #[test]
    fn the_profile_reproduces_its_own_cell_bit_exactly() {
        // the whole gravity-free equivalence rests on this: written as a ratio the bracket is
        // exactly 1 at the reference point, and 1.0^x is exact. an absolute-enthalpy form
        // would return the cell state only to roundoff and the equivalence would degrade to
        // an approximate one.
        for (rho, pre, phi) in [(1.0, 0.6, -3.0), (1e-4, 2e-7, -1e4), (817.3, 1.1e3, 55.0)] {
            let eq = LocalEquilibrium::through(rho, pre, phi, GAMMA);
            assert_eq!(eq.density_at(phi), rho, "density at its own potential");
            assert_eq!(eq.pressure_at(phi), pre, "pressure at its own potential");
        }
    }

    #[test]
    fn full_weight_leaves_the_profile_bit_for_bit() {
        // the weight enters as a factor on the potential difference and nothing else, so a
        // footprint the isentrope comfortably supports reproduces the unweighted profile
        // exactly -- the property that keeps a healthy stratified atmosphere on the arithmetic
        // it was measured on.
        let (rho, pre, phi) = (2.0, 1.5, -4.0);
        let plain = LocalEquilibrium::through(rho, pre, phi, GAMMA);
        let faded = LocalEquilibrium::faded(rho, pre, phi, GAMMA, 0.1);
        assert_eq!(balance_weight(GAMMA * pre / rho, GAMMA, 0.1), 1.0);
        for target in [-4.5, -4.0, -3.5, -3.0, 0.0, 12.0] {
            assert_eq!(plain.density_at(target), faded.density_at(target));
            assert_eq!(plain.pressure_at(target), faded.pressure_at(target));
        }
    }

    #[test]
    fn a_vanishing_weight_collapses_the_profile_to_the_cell_state() {
        // zero weight is a constant profile, so departures taken against it are the plain
        // differences of the state and the reconstruction that consumes them is the plain one.
        let (rho, pre, phi) = (2.0, 1.5, -4.0);
        let cs2 = GAMMA * pre / rho;
        // a rise well past the isentrope's vacuum boundary: (gamma-1) dphi / cs^2 = 4.
        let dphi_up = 4.0 * cs2 / (GAMMA - 1.0);
        assert_eq!(balance_weight(cs2, GAMMA, dphi_up), 0.0);
        let eq = LocalEquilibrium::faded(rho, pre, phi, GAMMA, dphi_up);
        for target in [-40.0, -4.0, 0.0, 40.0] {
            assert_eq!(eq.density_at(target), rho);
            assert_eq!(eq.pressure_at(target), pre);
        }
    }

    #[test]
    fn the_weighted_enthalpy_ratio_stays_above_the_declared_floor() {
        // the ramp's guarantee: whatever rise the footprint climbs, the weighted drop peaks at
        // the onset, so the enthalpy ratio evaluated at the footprint edge never falls below
        // 1 - BALANCE_FADE_ONSET and the equilibrium stays a finite factor from the cell state.
        let (rho, pre, phi) = (2.0, 1.5, -4.0);
        let cs2 = GAMMA * pre / rho;
        let floor = 1.0 - BALANCE_FADE_ONSET;
        for step in 0..=400 {
            let drop = step as f64 * 0.01;
            let dphi_up = drop * cs2 / (GAMMA - 1.0);
            let eq = LocalEquilibrium::faded(rho, pre, phi, GAMMA, dphi_up);
            let ratio = eq.enthalpy_ratio(phi + dphi_up);
            assert!(
                ratio >= floor - 1e-12,
                "a footprint spending {drop} of the enthalpy left the ratio at {ratio}, \
                 under the {floor} the ramp guarantees"
            );
        }
    }

    #[test]
    fn a_descending_footprint_spends_no_enthalpy() {
        // the isentrope thickens going down the potential, so only the rising side approaches
        // vacuum and a footprint that only descends keeps the balancing at full strength.
        assert_eq!(potential_rise(0.0_f64, -3.0, -7.0), 0.0);
        assert_eq!(potential_rise(0.0_f64, -3.0, 5.0), 5.0);
        assert_eq!(potential_rise(-2.0_f64, 4.0, 1.0), 6.0);
    }

    #[test]
    fn the_profile_is_in_hydrostatic_balance() {
        // dp/dphi = -rho is the defining property; check it as a centred difference against
        // the analytic density, which is what "follows the equilibrium exactly" means.
        let eq = LocalEquilibrium::through(2.0, 1.5, -4.0, GAMMA);
        let h = 1e-6;
        for phi in [-4.5, -4.0, -3.5, -3.0] {
            let dpdphi = (eq.pressure_at(phi + h) - eq.pressure_at(phi - h)) / (2.0 * h);
            let rho = eq.density_at(phi);
            assert!(
                (dpdphi + rho).abs() < 1e-6 * rho,
                "dp/dphi = {dpdphi} against -rho = {} at phi = {phi}",
                -rho
            );
        }
    }
}
