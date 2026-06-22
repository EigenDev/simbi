// =============================================================================
// source_term.rs
//
// the carrier-generic CONSERVATION LIFT + its built-in acceleration fields. the
// `+ Σ S(U)` half of the conservation form, written ONCE over `S: Scalar` —
// COMPUTED at S=f64 (the analytical reference) and TRACED at S=Gv (the rendered
// kernel) from the SAME definition. this is the carrier discipline the physics
// (flux, c2p) already follows, applied to the source half.
//
// the lift is the load-bearing idea: a source is a FREE FIELD (an acceleration
// `a`, a cooling rate `Lambda`, a relaxation `(kappa, v_ref)`) wrapped in the
// conservation law. the lift functions (`force_momentum` / `force_energy` /
// `cooling` / `relax_*`) ARE that wrapping. because the wrapping is one carrier-
// generic definition:
//   - f64 == Gv by construction for the LAW itself, not just the field — the
//     graph-divergence bug class (a hand-built `Op` graph that traces a different
//     computation than the f64 reference) cannot occur.
//   - the energy source is DERIVED from the same field the momentum source uses
//     (`S_nrg = rho*(a.v)`), so a user cannot desync energy from the force.
//   - built-in sources and USER sources share ONE lift: a built-in supplies its
//     field in Rust here (`UniformAccel`, `PointMassGravity`), a user supplies it
//     as a runtime DAG lowered via `expr_bridge`; both ride the identical lift
//     (`source_spec::user_force_*` / `user_relax_*` / `user_cooling_source` trace
//     these same functions at S=Gv).
//
// params are carrier-typed: at S=f64 they are numbers (the analytical reference);
// at S=Gv they are `Gv::scalar(name)` / spliced-DAG leaves the runtime fills per
// step. the caller owns the names (the lift never invents them).
// =============================================================================

use crate::Scalar;

// =============================================================================
// the conservation lift — the source-half analogue of the flux. ONE definition,
// instantiated at S=f64 (reference) and S=Gv (kernel). runs at BUILD/TRACE time
// (building IR or evaluating the analytical reference), never per-cell in the hot
// path — the rendered kernel / scalarized graph carries the per-cell work — so the
// `Vec` returns are free (they allocate once per kernel build, not per cell).
// =============================================================================

/// force-MOMENTUM lift: `S_mom_k = rho * a_k`, for an acceleration field `a` (D
/// components). the field's origin — constant gravity, a point mass, a user DAG —
/// is irrelevant to the lift; only `a` differs.
pub fn force_momentum<S: Scalar>(rho: S, accel: &[S]) -> Vec<S> {
    accel.iter().map(|&a| rho * a).collect()
}

/// force-ENERGY lift: `S_nrg = rho * (a . v)` — the work the force does, DERIVED
/// from the SAME acceleration field `a` the momentum lift uses. pairing the two
/// over one `a` is the structural reason the energy source cannot desync from the
/// momentum source.
pub fn force_energy<S: Scalar>(rho: S, vel: &[S], accel: &[S]) -> S {
    rho * dot(accel, vel)
}

/// cooling lift: `S_nrg = -Lambda`, for a cooling-rate field `Lambda`. an energy
/// sink; mass + momentum are untouched (a cooling kind reaches only the nrg slot).
pub fn cooling<S: Scalar>(rate: S) -> S {
    S::ZERO - rate
}

/// velocity-relaxation MOMENTUM lift (a sponge / buffer zone): `S_mom_k =
/// max(kappa, 0) * rho * (v_ref_k - vel_k)`, the linear drag toward a reference
/// velocity. `kappa` is clamped non-negative ([`clamp_rate`]) so the relaxation can
/// only DAMP — the unstable (anti-damping) form is unexpressible.
pub fn relax_momentum<S: Scalar>(rho: S, vel: &[S], kappa: S, v_ref: &[S]) -> Vec<S> {
    let k = clamp_rate(kappa);
    vel.iter()
        .zip(v_ref.iter())
        .map(|(&v, &vr)| k * rho * (vr - v))
        .collect()
}

/// velocity-relaxation ENERGY lift: `S_nrg = sum_k vel_k * S_mom_k` — the work the
/// drag does, DERIVED from the SAME `(kappa, v_ref)` the momentum lift uses. with
/// `kappa >= 0` it removes kinetic energy when `vel` overshoots `v_ref`, never adds it.
pub fn relax_energy<S: Scalar>(rho: S, vel: &[S], kappa: S, v_ref: &[S]) -> S {
    dot(vel, &relax_momentum(rho, vel, kappa, v_ref))
}

/// the `kappa >= 0` stability clamp. a relaxation adds `kappa*(U_ref - U)`; a
/// negative rate would anti-damp (inject energy / destabilize). clamping in the
/// lift makes the unstable form UNEXPRESSIBLE — the stability invariant enforced
/// by construction, not by a runtime check. carrier-safe (`max`, no branch).
fn clamp_rate<S: Scalar>(kappa: S) -> S {
    kappa.max(S::ZERO)
}

/// carrier-generic dot product of two equal-length slices.
fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    let mut s = S::ZERO;
    for k in 0..a.len() {
        s = s + a[k] * b[k];
    }
    s
}

// =============================================================================
// built-in acceleration fields — the Rust-authored half of "the field". each
// produces an `a` the lift wraps; momentum/energy delegate to the shared lift so a
// built-in source and a user source are, past `accel()`, the SAME computation.
// =============================================================================

/// uniform external acceleration `g_ext` (the constant-gravity user source).
/// the field is constant: `a_k = g_ext_k`.
pub struct UniformAccel<S, const D: usize> {
    pub g_ext: [S; D],
}

impl<S: Scalar, const D: usize> UniformAccel<S, D> {
    /// the acceleration field `a_k = g_ext_k`.
    pub fn accel(&self) -> [S; D] {
        self.g_ext
    }

    /// momentum source `S_mom_k = rho * g_ext_k` (via the shared lift).
    pub fn momentum(&self, rho: S) -> Vec<S> {
        force_momentum(rho, &self.g_ext)
    }

    /// energy source `S_nrg = rho * (vel . g_ext)` (via the shared lift).
    pub fn energy(&self, rho: S, vel: &[S; D]) -> S {
        force_energy(rho, vel, &self.g_ext)
    }
}

/// point-mass (Plummer-softened) Newtonian gravity: a gravitating mass `G*M` at
/// position `xm`, softened by length `eps`. the acceleration field is
///
///   a_k = -GM (x-xm)_k / (|x-xm|^2 + eps^2)^{3/2}.
///
/// softening (`eps > 0`) keeps the field finite AT the mass position — without it
/// `1/|x-xm|^3` produces Inf in the cell containing the mass and traces straight
/// into the kernel. the shared `1/(...)^{3/2}` scaffolding hash-conses across the
/// momentum + energy lifts (one `sqrt`/division per kernel). a moving mass updates
/// `xm` per step; at S=Gv `gm`/`xm`/`eps` are `Gv::scalar` leaves the runtime fills.
pub struct PointMassGravity<S, const D: usize> {
    /// the product `G * M` (one scalar; repulsive forces flip its sign).
    pub gm: S,
    /// the gravitating mass position `xm_k`.
    pub xm: [S; D],
    /// the Plummer softening length. `eps > 0` regularizes the `r -> 0` singularity.
    pub eps: S,
}

impl<S: Scalar, const D: usize> PointMassGravity<S, D> {
    /// the softened acceleration field `a_k = -GM (x-xm)_k / (|x-xm|^2 + eps^2)^{3/2}`.
    /// `rho` is NOT folded in — the lift multiplies it. the radicand `|x-xm|^2 + eps^2`
    /// is strictly positive for `eps > 0`, so the `sqrt` + division never hit zero.
    pub fn accel(&self, x: &[S; D]) -> [S; D] {
        let dx: [S; D] = std::array::from_fn(|k| x[k] - self.xm[k]);
        let r_sq = dot(&dx, &dx) + self.eps * self.eps;
        // GM / (r_sq + eps^2)^{3/2} = GM / (r_sq * sqrt(r_sq)).
        let inv_r3 = self.gm / (r_sq * r_sq.sqrt());
        std::array::from_fn(|k| S::ZERO - inv_r3 * dx[k])
    }

    /// momentum source `S_mom_k = rho * a_k` (via the shared lift).
    pub fn momentum(&self, rho: S, x: &[S; D]) -> Vec<S> {
        force_momentum(rho, &self.accel(x))
    }

    /// energy source `S_nrg = rho * (vel . a)` (via the shared lift).
    pub fn energy(&self, rho: S, vel: &[S; D], x: &[S; D]) -> S {
        force_energy(rho, vel, &self.accel(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_accel_f64_matches_analytical() {
        // at S=f64 the source IS the analytical reference — no graph, no eval_source.
        let src = UniformAccel::<f64, 3> { g_ext: [-0.1, -0.2, -9.81] };
        let rho = 1.5_f64;
        let vel = [0.3_f64, -0.2, 0.4];

        let mom = src.momentum(rho);
        for k in 0..3 {
            assert!((mom[k] - rho * src.g_ext[k]).abs() < 1e-15, "S_mom_{k}");
        }

        let nrg = src.energy(rho, &vel);
        let expect = rho * (vel[0] * -0.1 + vel[1] * -0.2 + vel[2] * -9.81);
        assert!((nrg - expect).abs() < 1e-15, "S_nrg");
    }

    #[test]
    fn point_mass_gravity_f64_matches_analytical() {
        // at S=f64 the source IS the softened analytical reference:
        // S = -rho*GM (x-xm) / (|x-xm|^2 + eps^2)^{3/2}.
        let src = PointMassGravity::<f64, 3> { gm: 2.0, xm: [0.1, -0.3, 0.2], eps: 0.05 };
        let rho = 1.5_f64;
        let vel = [0.3_f64, -0.2, 0.4];
        let x = [1.0_f64, 0.5, -0.4];

        let dx = [x[0] - 0.1, x[1] + 0.3, x[2] - 0.2];
        let r_sq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + 0.05 * 0.05;
        let inv_r3 = 2.0 / (r_sq * r_sq.sqrt());
        let f = rho * inv_r3;

        let mom = src.momentum(rho, &x);
        for k in 0..3 {
            assert!((mom[k] - (-f * dx[k])).abs() < 1e-13, "S_mom_{k}");
        }

        let nrg = src.energy(rho, &vel, &x);
        let v_dot_dx = vel[0] * dx[0] + vel[1] * dx[1] + vel[2] * dx[2];
        assert!((nrg - (-f * v_dot_dx)).abs() < 1e-13, "S_nrg");
    }

    #[test]
    fn softening_keeps_acceleration_finite_at_the_mass() {
        // the bug softening fixes: WITHOUT eps, x == xm gives 1/0 = Inf. WITH eps > 0
        // the field is finite everywhere (= 0 exactly at the mass, where dx = 0).
        let src = PointMassGravity::<f64, 2> { gm: 1.0, xm: [0.3, -0.7], eps: 0.1 };
        let a = src.accel(&[0.3, -0.7]);
        for k in 0..2 {
            assert!(a[k].is_finite() && a[k] == 0.0, "a_{k} at the mass must be finite (0): {}", a[k]);
        }
    }

    #[test]
    fn relax_clamps_negative_rate_and_only_damps() {
        // kappa < 0 (anti-damping) clamps to a no-op; kappa > 0 removes kinetic energy.
        let rho = 1.0_f64;
        let vel = [3.0_f64, 0.0];
        let v_ref = [0.0_f64, 0.0];

        let no_op = relax_momentum(rho, &vel, -5.0, &v_ref);
        assert!(no_op.iter().all(|&s| s.abs() < 1e-15), "negative kappa must clamp to a no-op");

        let drag = relax_momentum(rho, &vel, 2.0, &v_ref);
        assert!((drag[0] - (-6.0)).abs() < 1e-13, "drag opposes velocity: {}", drag[0]);
        let work = relax_energy(rho, &vel, 2.0, &v_ref);
        assert!(work < 0.0, "relaxation must remove kinetic energy, got {work}");
    }
}
