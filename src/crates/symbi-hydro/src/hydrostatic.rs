// =============================================================================
// hydrostatic.rs
//
// hydrostatic reconstruction for the euler equations with a gravitational source:
// reconstruct the DEVIATION from the local hydrostatic profile through each cell
// rather than the state itself, so a discretely balanced atmosphere presents no
// face jump at all.
//
// the residual this removes is the reason a low-dissipation riemann solver cannot
// be used on a stagnant stratified column. a hydrostatic state solves the CONTINUUM
// equations, not the discrete ones: a limited reconstruction leaves an O(dx^2) face
// jump on a curved profile, the numerical flux at rest is the upwind dissipation
// acting on that jump, and the deposit is one-signed every step. classical dissipation
// damps the resulting ring; a scheme that reduces acoustic dissipation at low mach
// number removes exactly the damping and the ring survives as an entropy error. the
// cure adopted here removes the JUMP instead of restoring the damping, so the low-mach
// reduction and the entropy floor stop competing.
//
// the local profile is the isentrope through the cell state that satisfies
// dp/dx = -rho dphi/dx exactly (Kaeppeli & Mishra, J. Comput. Phys. 259:199, 2014):
//
//   rho_eq(phi) = rho_i [1 + (gamma - 1)(phi_i - phi) / cs_i^2]^(1/(gamma - 1)),
//   p_eq(phi)   = p_i   [rho_eq(phi) / rho_i]^gamma,     cs_i^2 = gamma p_i / rho_i.
//
// written as a RATIO against the cell's own state rather than through an absolute
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

/// the isentropic hydrostatic profile passing through one cell's state.
///
/// carries the cell's own state rather than a derived constant, so every evaluation is a
/// ratio against it and the point `phi = phi_ref` is reproduced exactly.
///
/// GAMMA-LAW ONLY. the profile solves `dp/dphi = -rho` along `p = K rho^gamma`, and the
/// `cs2_ref = gamma p / rho` it derives is the ideal-gas sound speed spelled out -- on the
/// isothermal or taub-mathews closures this curve is not the eos's isentrope, so a transform
/// built from it REINTRODUCES the face jump it exists to remove. the dispatch refuses the
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
}

impl<S: Scalar> LocalEquilibrium<S> {
    /// the profile through `(rho, pre)` at potential `phi`.
    #[inline]
    pub fn through(rho: S, pre: S, phi: S, gamma: S) -> Self {
        Self {
            rho_ref: rho,
            pre_ref: pre,
            phi_ref: phi,
            cs2_ref: gamma * pre / rho,
            gamma,
        }
    }

    /// `[1 + (gamma - 1)(phi_ref - phi)/cs_ref^2]`, the dimensionless enthalpy ratio the
    /// profile is built from. exactly `1` at the reference point.
    ///
    /// the floor keeps the bracket positive where the extrapolation would run past the
    /// profile's own vacuum boundary — a potential drop deeper than the cell's enthalpy can
    /// support. that is outside the domain of the isentrope, not a state, and clamping there
    /// leaves the deviation to carry the difference rather than producing a negative density.
    #[inline]
    pub fn enthalpy_ratio(&self, phi: S) -> S {
        let one = S::ONE;
        let ratio = one + (self.gamma - one) * (self.phi_ref - phi) / self.cs2_ref;
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

    /// both components at once, sharing ONE `powf`.
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

/// the van leer harmonic slope, BYTE-matching the kernel limiter's theta < 0 arm
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
/// under test differs from the plain one only in WHAT is limited, never in HOW.
#[inline]
fn minmod3<S: Scalar>(x: S, y: S, z: S) -> S {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    let all_pos = mn.cmp_gt(S::ZERO);
    let all_neg = mx.cmp_lt(S::ZERO);
    S::select(all_pos, mn, S::select(all_neg, mx, S::ZERO))
}

/// the stencil's departures from the hydrostatic profile through the cell at `anchor`, each
/// evaluated at its own potential. THIS is the whole well-balancing transform, and it is
/// independent of which reconstruction consumes it: feed the departures to any operator that
/// reproduces constants, add the profile back at the face, and the scheme is well-balanced.
/// PLM and PPM therefore need no separate derivation — only the same wrapper around their own
/// limiter.
///
/// ANCHOR AT THE CELL BEING RECONSTRUCTED, always. the departure at the anchor is then exactly
/// zero, so the operator's one-sided differences about it reduce to `0 - d` and `d - 0`, which
/// are the plain differences EXACTLY rather than to rounding. that is what carries the
/// gravity-free bit-identity through the transform, and it is why a face's two sides must be
/// built from two separate anchors instead of one shared pass: an anchor on the far cell would
/// leave both differences as `(q_j - c) - (q_k - c)`, equal to `q_j - q_k` only to roundoff.
pub fn hydrostatic_departures<S: Scalar>(
    rho: &[S],
    pre: &[S],
    phi: &[S],
    anchor: usize,
    gamma: S,
) -> (Vec<S>, Vec<S>) {
    let eq = LocalEquilibrium::through(rho[anchor], pre[anchor], phi[anchor], gamma);
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

    // departures via the ONE transform text the kernel path also compiles.
    let (d_rho, d_pre) = hydrostatic_departures(&rho, &pre, &phi, 1, gamma);
    // the centre departure is exactly zero, so the one-sided differences ARE the departures.
    // the LIMITER SELECTION mirrors the kernel's `plm_theta_from_stencil` exactly: theta-MC
    // minmod for theta >= 0, the smooth van leer harmonic for theta < 0. a reference that
    // hard-wired minmod fed a NEGATIVE theta straight into `minmod3(a*theta, ...)` -- a
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
