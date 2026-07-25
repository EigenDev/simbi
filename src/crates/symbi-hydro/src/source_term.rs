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
//   - f64 == Gv by construction across the whole LAW computation, field values included — the
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
// `Vec` returns are free (they allocate once per kernel build).
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

/// acceleration in a frame rotating counterclockwise about the z axis at constant
/// angular speed `omega`. coordinates and velocities are measured in that frame.
/// the result includes `-2 omega cross v` and `-omega cross (omega cross r)`;
/// the z component is zero when a three-component state is supplied.
pub fn rotating_frame_acceleration<S: Scalar>(
    position: &[S],
    vel: &[S],
    omega: S,
    origin_x: S,
    origin_y: S,
) -> Vec<S> {
    assert!(position.len() >= 2);
    assert_eq!(position.len(), vel.len());
    let dx = position[0] - origin_x;
    let dy = position[1] - origin_y;
    let omega_sq = omega * omega;
    let mut accel = vec![S::ZERO; position.len()];
    accel[0] = S::from_f64(2.0) * omega * vel[1] + omega_sq * dx;
    accel[1] = S::from_f64(-2.0) * omega * vel[0] + omega_sq * dy;
    accel
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

// ── full conserved-state relaxation (the buffer zone) ────────────────────────
//
// where `relax_*` relaxes the intensive VELOCITY toward `v_ref` at fixed density,
// the `sponge_*` family relaxes EVERY conserved component toward a reference state
// `U_ref = (den_ref, mom_ref, nrg_ref)`: `S_U = max(kappa,0) * (U_ref - U)`. this is
// the well-posed sink of a buffer/damping zone that holds the flow at a known (e.g.
// analytic ambient) solution — density included, so the zone cannot let the boundary
// density drift. the three channels share the ONE `kappa`, so masking the rate masks
// the whole relaxation, and `clamp_rate` keeps it a damping-only sink.

/// full-state relaxation MASS lift: `S_den = max(kappa,0) * (den_ref - rho)`, the
/// linear drag of density toward the reference `den_ref`.
pub fn sponge_density<S: Scalar>(rho: S, kappa: S, den_ref: S) -> S {
    clamp_rate(kappa) * (den_ref - rho)
}

/// full-state relaxation MOMENTUM lift: `S_mom_k = max(kappa,0) * (mom_ref_k - rho*vel_k)`,
/// relaxing the CONSERVED momentum `rho*vel_k` toward a reference momentum `mom_ref_k`.
/// distinct from `relax_momentum` (intensive velocity at fixed rho) — this composes with
/// a simultaneous density relaxation without the two channels fighting.
pub fn sponge_momentum<S: Scalar>(rho: S, vel: &[S], kappa: S, mom_ref: &[S]) -> Vec<S> {
    let k = clamp_rate(kappa);
    vel.iter()
        .zip(mom_ref.iter())
        .map(|(&v, &mr)| k * (mr - rho * v))
        .collect()
}

/// full-state relaxation ENERGY lift: `S_nrg = max(kappa,0) * (nrg_ref - E)`, relaxing the
/// CONSERVED total energy `E = pre*inv_gm1 + (1/2)*rho*|v|^2` toward `nrg_ref`. `inv_gm1 =
/// 1/(gamma-1)` is the ideal-gas internal-energy coefficient — a build-time constant since
/// gamma is known when the source is lowered, so the lift needs no runtime gamma binding.
pub fn sponge_energy<S: Scalar>(rho: S, vel: &[S], pre: S, kappa: S, nrg_ref: S, inv_gm1: S) -> S {
    let e = pre * inv_gm1 + S::from_f64(0.5) * rho * dot(vel, vel);
    clamp_rate(kappa) * (nrg_ref - e)
}

/// the `kappa >= 0` stability clamp. a relaxation adds `kappa*(U_ref - U)`; a
/// negative rate would anti-damp (inject energy / destabilize). clamping in the
/// lift makes the unstable form UNEXPRESSIBLE — the stability invariant enforced
/// by construction. carrier-safe (`max`, no branch).
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
    /// `rho` is applied by the caller's lift, so this acceleration field excludes it. the radicand `|x-xm|^2 + eps^2`
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

/// ONE immersed body's source contribution in **3D Cartesian** (coord-free),
/// carrier-generic: softened gravity + Bondi-Hoyle accretion. this is the SINGLE
/// source of truth for the body forcing — instantiated at `S = Gv` to build the
/// fused kernel, at `S = f64` for the analytic reference test, and shared by the
/// backward feedback. the physics is done in CARTESIAN (frame-independent); the
/// composing layer supplies the cell's Cartesian position + gas velocity (via the
/// metric's to_cartesian) and projects the Cartesian momentum source back onto the
/// physical coordinate basis (vector_from_cartesian) — so it is correct in
/// Cartesian, cylindrical, AND spherical. the per-body MAX_SOURCE_BODIES sum lives one
/// layer up. future media (porous drag, deformable stress) extend THIS struct's
/// methods, inheriting the frame handling + fused machinery for free.
/// the EXACT support of the gaussian sink kernel, in accretion radii: the weight
/// `exp(-(r/(racc/2))^2)` underflows to exactly +0 in f64 once the exponent argument
/// passes ~-745, i.e. `r/(racc/2) >= 27.3`. gating at `r >= SINK_SUPPORT_RADII * racc`
/// (= 28 half-radii, with margin) therefore skips the exp / roots / divisions on the
/// far field WITHOUT changing any bit of any field: outside the support the ungated
/// kernel computes den_dot = 0 exactly.
pub const SINK_SUPPORT_RADII: f64 = 14.0;

pub struct BodySource<S> {
    /// gravitating mass (0 for an inactive body — a branch-free no-op).
    pub mass: S,
    /// body position (Cartesian, 3D embedding).
    pub xm: [S; 3],
    /// body velocity (Cartesian, 3D) — the sink frame.
    pub vm: [S; 3],
    /// Plummer softening length.
    pub soft: S,
    /// accretion radius (the Gaussian sink kernel scale).
    pub racc: S,
    /// sink rate cap (0 disables accretion).
    pub sink: S,
    /// angular-momentum retention of accreted gas (0 = radial sink, 1 = full).
    pub delta: S,
}

impl<S: Scalar> BodySource<S> {
    /// softened gravity accel `a = -mass (x-xm) / (|x-xm|^2 + soft^2)^{3/2}`
    /// (Cartesian; identical form to `PointMassGravity`).
    pub fn accel(&self, x: &[S; 3]) -> [S; 3] {
        let dx: [S; 3] = std::array::from_fn(|k| x[k] - self.xm[k]);
        let r_sq = dot(&dx, &dx) + self.soft * self.soft;
        let inv_r3 = self.mass / (r_sq * r_sq.sqrt());
        std::array::from_fn(|k| S::ZERO - inv_r3 * dx[k])
    }

    /// Bondi-Hoyle accretion rate `den_dot = rho * min(sink, 1/t_nat, 1/dt) * w(r)`,
    /// `w = exp(-(r/(racc/2))^2)`, `t_nat = min(min_w/cs, sqrt(r^3/(2 mass)))`. reads
    /// the LOCAL `rho` + `cs` — the state dependence the fused source must carry.
    ///
    /// spatially gated at the kernel's EXACT support ([`SINK_SUPPORT_RADII`]): beyond it
    /// the gaussian weight underflows to exactly +0 in f64, so the ungated rate is
    /// exactly zero and the lazy branch skips the exp + roots on the far field —
    /// ~all cells for a sink of a few cell widths — without changing any bit.
    pub fn accretion_rate(&self, rho: S, cs: S, x: &[S; 3], min_w: S, inv_dt: S) -> S {
        let dx: [S; 3] = std::array::from_fn(|k| x[k] - self.xm[k]);
        let r_mag = dot(&dx, &dx).sqrt();
        let r_cut = S::from_f64(SINK_SUPPORT_RADII) * self.racc;
        S::cond(
            r_mag.cmp_lt(r_cut),
            || {
                let tiny = S::from_f64(1e-30);
                let r_norm = r_mag / (S::from_f64(0.5) * self.racc);
                let weight = (S::ZERO - r_norm * r_norm).exp();
                let sound_crossing = min_w / cs;
                let t_ff =
                    (r_mag * r_mag * r_mag / (S::from_f64(2.0) * self.mass + tiny)).sqrt();
                let nat_rate = S::ONE / (sound_crossing.min(t_ff) + tiny);
                let sr = self.sink.min(nat_rate).min(inv_dt);
                rho * sr * weight
            },
            || S::ZERO,
        )
    }

    /// the sink velocity `v_star` (Cartesian): radial + `delta`*angular, in the body
    /// frame. the accreted momentum sink is `-v_star * den_dot`.
    pub fn sink_velocity(&self, vel: &[S; 3], x: &[S; 3]) -> [S; 3] {
        let eps_r = S::from_f64(1e-24);
        let dx: [S; 3] = std::array::from_fn(|k| x[k] - self.xm[k]);
        let inv_safe = S::ONE / (dot(&dx, &dx) + eps_r).sqrt();
        let rhat: [S; 3] = std::array::from_fn(|k| dx[k] * inv_safe);
        let vrel: [S; 3] = std::array::from_fn(|k| vel[k] - self.vm[k]);
        let vrad_comp = dot(&vrel, &rhat);
        std::array::from_fn(|k| {
            let vrad = vrad_comp * rhat[k];
            let vang = vrel[k] - vrad;
            vrad + self.delta * vang + self.vm[k]
        })
    }

    /// the Cartesian momentum source `S = rho*a - v_star*den_dot` (gravity + sink).
    /// the composer projects this onto the physical coordinate basis.
    ///
    /// gravity acts everywhere; the SINK term carries the same exact-support gate as
    /// [`Self::accretion_rate`] so the far field skips `sink_velocity`'s root and
    /// divisions too (they would be multiplied by an exact zero).
    pub fn momentum_cartesian(&self, rho: S, vel: &[S; 3], x: &[S; 3], cs: S, min_w: S, inv_dt: S) -> [S; 3] {
        let a = self.accel(x);
        let dx: [S; 3] = std::array::from_fn(|k| x[k] - self.xm[k]);
        let r_mag = dot(&dx, &dx).sqrt();
        let r_cut = S::from_f64(SINK_SUPPORT_RADII) * self.racc;
        let sink_mom: [S; 3] = S::cond_vec(
            r_mag.cmp_lt(r_cut),
            || {
                let den_dot = self.accretion_rate(rho, cs, x, min_w, inv_dt);
                let vstar = self.sink_velocity(vel, x);
                std::array::from_fn(|k| vstar[k] * den_dot)
            },
            || [S::ZERO; 3],
        );
        std::array::from_fn(|k| rho * a[k] - sink_mom[k])
    }

    /// the density source `S_den = -den_dot` (mass removed by accretion).
    pub fn density(&self, rho: S, cs: S, x: &[S; 3], min_w: S, inv_dt: S) -> S {
        S::ZERO - self.accretion_rate(rho, cs, x, min_w, inv_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the sink gate's bit-exactness rests on gaussian underflow: at and beyond
    // the support radius the UNGATED weight is exactly +0 in f64, so the lazy
    // branch that skips the exp / roots changes nothing. gravity is unaffected.
    #[test]
    fn sink_is_exactly_zero_beyond_the_support_radius() {
        let body = BodySource::<f64> {
            mass: 1.0,
            xm: [0.0; 3],
            vm: [0.0; 3],
            soft: 0.05,
            racc: 0.4,
            sink: 1e6,
            delta: 0.0,
        };
        let (rho, cs, min_w, inv_dt) = (1.3_f64, 0.7, 0.05, 1e3);
        for scale in [1.0_f64, 2.0, 50.0] {
            let r = SINK_SUPPORT_RADII * body.racc * scale;
            let x = [r, 0.0, 0.0];
            assert_eq!(body.accretion_rate(rho, cs, &x, min_w, inv_dt), 0.0);
            // momentum reduces to pure gravity, bit-for-bit
            let vel = [0.3, -0.2, 0.1];
            let m = body.momentum_cartesian(rho, &vel, &x, cs, min_w, inv_dt);
            let a = body.accel(&x);
            for k in 0..3 {
                assert_eq!(m[k], rho * a[k], "far-field momentum must be pure gravity");
            }
        }
        // just inside the support the sink is alive
        let x_in = [0.5 * body.racc, 0.0, 0.0];
        assert!(body.accretion_rate(rho, cs, &x_in, min_w, inv_dt) > 0.0);
    }

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
    fn body_source_f64_matches_immersed_spec() {
        // at S=f64 BodySource IS the analytic reference (the same spec the
        // gv_immersed kernel + immersed_iso test validate): softened gravity +
        // Bondi accretion + sink-velocity momentum loss. Cartesian 3D (z=0 plane).
        let b = BodySource::<f64> {
            mass: 1.2, xm: [0.1, -0.2, 0.0], vm: [0.0, 0.0, 0.0],
            soft: 0.1, racc: 0.6, sink: 5.0, delta: 0.3,
        };
        let (rho, cs) = (1.5_f64, 0.5_f64);
        let vel = [0.4 / 1.5_f64, -0.3 / 1.5, 0.0];
        let x = [0.37_f64, 0.31, 0.0];
        let (min_w, inv_dt) = (0.18_f64, 1.0 / 0.01);

        let dx = [x[0] - b.xm[0], x[1] - b.xm[1], 0.0];
        let r_sq = dx[0] * dx[0] + dx[1] * dx[1] + b.soft * b.soft;
        let inv_r3 = b.mass / (r_sq * r_sq.sqrt());
        let g = [-inv_r3 * dx[0], -inv_r3 * dx[1]];

        let r_mag = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt();
        let weight = (-(r_mag / (0.5 * b.racc)).powi(2)).exp();
        let t_ff = (r_mag.powi(3) / (2.0 * b.mass + 1e-30)).sqrt();
        let nat = 1.0 / ((min_w / cs).min(t_ff) + 1e-30);
        let den_dot = rho * b.sink.min(nat).min(inv_dt) * weight;

        let inv_safe = 1.0 / (r_mag * r_mag + 1e-24).sqrt();
        let rhat = [dx[0] * inv_safe, dx[1] * inv_safe];
        let vrad_c = vel[0] * rhat[0] + vel[1] * rhat[1];
        let vstar: Vec<f64> = (0..2)
            .map(|k| {
                let vrad = vrad_c * rhat[k];
                vrad + b.delta * (vel[k] - vrad)
            })
            .collect();

        let mom = b.momentum_cartesian(rho, &vel, &x, cs, min_w, inv_dt);
        for k in 0..2 {
            let want = rho * g[k] - vstar[k] * den_dot;
            assert!((mom[k] - want).abs() < 1e-12, "S_mom_{k}: {} vs {want}", mom[k]);
        }
        assert!((b.density(rho, cs, &x, min_w, inv_dt) - (-den_dot)).abs() < 1e-12, "S_den");
        // an inactive body (mass=0, sink=0) contributes exactly nothing.
        let off = BodySource::<f64> { mass: 0.0, sink: 0.0, ..b };
        let m0 = off.momentum_cartesian(rho, &vel, &x, cs, min_w, inv_dt);
        assert!(m0[0] == 0.0 && m0[1] == 0.0 && off.density(rho, cs, &x, min_w, inv_dt) == 0.0);
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

    #[test]
    fn rotating_frame_acceleration_has_coriolis_and_centrifugal_signs() {
        let accel =
            rotating_frame_acceleration(&[3.0, 4.0, 7.0], &[5.0, 6.0, 8.0], 2.0, 1.0, 1.0);
        assert_eq!(accel, vec![32.0, -8.0, 0.0]);
    }

    #[test]
    fn rotating_frame_matches_an_inertially_stationary_particle() {
        let omega = 2.0;
        let accel = rotating_frame_acceleration(&[1.0, 0.0], &[0.0, -omega], omega, 0.0, 0.0);
        assert_eq!(accel, vec![-omega * omega, 0.0]);
    }
}
