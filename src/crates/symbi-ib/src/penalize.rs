// =============================================================================
// penalize.rs
//
// the composable immersed-boundary property algebra:
// properties contribute a relaxation rate and a target into a `Relax`
// accumulator; the engine integrates the accumulated system once, as exact
// frozen-coefficient exponentials on three disjoint primitive channels —
//   rho:   den' = den * exp(-lambda_rho dt)          (the uniform drain)
//   u:     u'   = u_solid + du_n e^-l_n + du_t e^-l_t (normal/tangential wall)
//   e_int: e'   = e_wall + (e + Q - e_wall) e^-l_e    (thermal wall; Q = drag heat)
// — where Q is the frictional dissipation of the velocity channel, so an isothermal
// wall absorbs the drag heat and an adiabatic wall (l_e = 0) keeps it —
// unconditionally stable for any dt, and commuting exactly under property stack
// reordering: the energy reconstruction reads the velocity relaxation summed
// across all properties (for the wall-work / dissipation term below), so
// the accumulated Relax is order-independent.
//
// the conserved reconstruction is spelled as a conserved scaling plus
// corrections that vanish exactly at zero rates:
//   den' = den f_rho
//   mom' = mom f_rho + den' (u' - u)
//   nrg' = nrg f_rho + den' [(e' - e) + (u' - u).u_solid]
// where the energy correction is the work done on the wall (the momentum transfer
// times the local wall velocity `u_solid`), so the gas total energy the wall
// exchanges equals the body's mechanical work and gas+body energy is conserved; the
// frictional dissipation (gas KE change minus that work) stays in the gas as
// internal energy (heat). the energy channel therefore reads the velocity channel's
// `u' - u` (it is the only cross-channel coupling; density + velocity stay disjoint).
// with u' - u = -(du_n g_n + du_t g_t), e' - e = (e_target - e) g_e, and
// g = 1 - exp(-lambda dt). when only the drain is active g_n = g_t = g_e = 0
// exactly, every correction is an exact zero, and the update reduces bit-for-
// bit to `drain_cell`'s uniform scaling (up to the sign of exactly-zero
// momentum components: x + 0.0 flips -0 to +0). that reduction is the p = 1
// anchor the whole algebra rests on.
//
// carrier-generic: f64 for the reference and unit laws, Gv for the traced
// penalization kernel, Dual for sensitivities. the isothermal regime's energy
// slot discards the e-channel corrections by construction (EnergySlot).
//
// usage:
//   let mut acc = Relax::none();
//   for p in &stack { p.contribute(chi, &kin, &mut acc); }
//   let (u_new, delta) = penalize_cell(&cons, &acc, normal, dt, volume, idx);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::energy::{DyeModel, EnergyModel, EnergySlot};
use symbi_hydro::state::ConsG;
use symbi_ir::algebra::Scalar;

use crate::body_delta::BodyDelta;

/// the body kinematics a property may target: the rigid-motion velocity field
/// and the wall internal-energy target (from the wall temperature through the
/// EOS; read once a thermal property is stacked). `u_solid` is the value at
/// the cell — for a spinning body, evaluate it per cell with `at()`.
#[derive(Clone, Copy, Debug)]
pub struct BodyKin<S: Scalar, const D: usize> {
    pub u_solid: Tensor<S, D>,
    /// rigid angular velocity about the body center. 2D: the z component acts;
    /// 1D: rotation requires a plane, so the field stays inert.
    pub omega: Tensor<S, 3>,
    pub e_wall: S,
}

/// `omega x r` restricted to the D in-plane components: the rigid-rotation
/// velocity at offset `r` from the rotation center. 1D: exactly zero (rotation
/// requires a plane); 2D: the z-spin acting in the plane,
/// `(-w_z r_1, w_z r_0)`; 3D: the full cross product.
pub fn omega_cross<S: Scalar, const D: usize>(
    omega: &Tensor<S, 3>,
    r: &Tensor<S, D>,
) -> Tensor<S, D> {
    let mut out = Tensor::<S, D>::zeros();
    if D == 2 {
        out[0] = -(omega[2] * r[1]);
        out[1] = omega[2] * r[0];
    } else if D == 3 {
        out[0] = omega[1] * r[2] - omega[2] * r[1];
        out[1] = omega[2] * r[0] - omega[0] * r[2];
        out[2] = omega[0] * r[1] - omega[1] * r[0];
    }
    out
}

/// the moment `r x f` of a D-dimensional vector about the origin, embedded in
/// 3-space: the torque booking of a force applied at offset `r`. 1D: zero;
/// 2D: only the z component, `r_0 f_1 - r_1 f_0`; 3D: the full cross product.
pub fn moment<S: Scalar, const D: usize>(r: &Tensor<S, D>, f: &Tensor<S, D>) -> Tensor<S, 3> {
    let mut out = Tensor::<S, 3>::zeros();
    if D == 2 {
        out[2] = r[0] * f[1] - r[1] * f[0];
    } else if D == 3 {
        out[0] = r[1] * f[2] - r[2] * f[1];
        out[1] = r[2] * f[0] - r[0] * f[2];
        out[2] = r[0] * f[1] - r[1] * f[0];
    }
    out
}

impl<S: Scalar, const D: usize> BodyKin<S, D> {
    /// the kinematics evaluated at a cell offset `x_rel` from the body center:
    /// the returned kin's `u_solid` is the local rigid-motion target
    /// `u_translation + omega x x_rel`, ready for `Property::contribute`. the
    /// added terms are exactly `+-0` at zero spin.
    pub fn at(&self, x_rel: &Tensor<S, D>) -> BodyKin<S, D> {
        BodyKin {
            u_solid: self.u_solid + omega_cross(&self.omega, x_rel),
            omega: self.omega,
            e_wall: self.e_wall,
        }
    }
}

/// the retention floor for the torque-free tangential channel. the retention is a
/// growing exponential (`lambda_t < 0`); left unbounded it boosts the retained
/// angular momentum of a vanishing remnant to an infinite velocity, and forms
/// `0 * inf = NaN` in the conserved momentum once the density underflows. the
/// retention floor caps the tangential growth factor at `1 / f_floor`: torque-
/// free is exact while a cell keeps a fraction `>= f_floor` of its mass over a
/// step, and degrades to a bounded standard drain below it (a cell drained past
/// `f_floor` in one step is being annihilated — far outside any physical steady
/// state, where drain balances inflow). analogous to a positivity floor.
pub const TORQUE_FREE_RETENTION_FLOOR: f64 = 1e-4;

/// the accumulated relaxation system: rates and targets per primitive channel.
/// rates add across properties; targets are assigned (single-body stack — the
/// multi-body nearest-wins policy composes above this).
#[derive(Clone, Copy, Debug)]
pub struct Relax<S: Scalar, const D: usize> {
    pub lambda_rho: S,
    pub lambda_un: S,
    pub lambda_ut: S,
    pub u_solid: Tensor<S, D>,
    pub lambda_e: S,
    pub e_target: S,
    /// cap on the tangential velocity growth factor `exp(-lambda_ut dt)`.
    /// `INFINITY` (the default) is inert — every decaying wall has a factor
    /// `<= 1`. only the torque-free channel (`lambda_ut < 0`) grows; the cap
    /// bounds its retained-momentum velocity boost at the evacuation limit.
    pub ut_growth_cap: S,
}

impl<S: Scalar, const D: usize> Relax<S, D> {
    /// no relaxation on any channel: `penalize_cell` on this is an exact no-op.
    pub fn none() -> Self {
        Self {
            lambda_rho: S::ZERO,
            lambda_un: S::ZERO,
            lambda_ut: S::ZERO,
            u_solid: Tensor::zeros(),
            lambda_e: S::ZERO,
            e_target: S::ZERO,
            ut_growth_cap: S::INFINITY,
        }
    }
}

/// one boundary property: what it contributes to the relaxation system.
/// coefficients are carrier values (f64 on the host, scalar params in a
/// trace) as inverse timescales — the stiff limit is a large `inv_*`, and
/// zero is an exact off switch.
#[derive(Clone, Copy, Debug)]
pub enum Property<S: Scalar> {
    /// the uniform-scaling drain (the validated accretor at p = 1): acts on
    /// the rho channel only.
    Drain { inv_tau: S },
    /// velocity relaxation toward the solid: no-slip with both rates finite,
    /// free-slip / no-penetration with `inv_eta_t = 0`.
    Wall { inv_eta_n: S, inv_eta_t: S },
    /// internal-energy relaxation toward the wall value (dirichlet
    /// temperature through the EOS). adiabatic wall = omit this property.
    IsothermalWall { inv_eta: S },
    /// the porosity dial: p = 1 pure drain, p = 0 pure wall. the wall channels
    /// carry independent rates — free-slip (no-penetration only) with
    /// `inv_eta_t = 0` (an exact off switch: the tangential velocity is
    /// bit-untouched), no-slip with both finite.
    PorousAccretor {
        p: S,
        inv_tau: S,
        inv_eta_n: S,
        inv_eta_t: S,
    },
    /// the torque-free accretor: the drain plus a tangential
    /// anti-relaxation locked to the drain rate, `lambda_t = -xi lambda_rho`.
    /// the radial-relative momentum drains with the mass (accreted, zero center
    /// torque on a spherical mask where the normal is radial); the tangential
    /// (angular) momentum is retained by the factor `f_rho^{1-xi}`, so the
    /// removed tangential momentum is `rho (1 - f_rho^{1-xi}) du_t`:
    /// `xi = 0` is the standard sink, `xi = 1` retains the angular momentum in full
    /// (torque-free). `xi in [0, 1]`. `lambda_t < 0` is a growing exponential —
    /// bounded in momentum, divergent in velocity as the mask evacuates (the
    /// well-posedness limit as the mask evacuates).
    TorqueFreeAccretor { inv_tau: S, xi: S },
}

impl<S: Scalar> Property<S> {
    pub fn contribute<const D: usize>(&self, chi: S, kin: &BodyKin<S, D>, acc: &mut Relax<S, D>) {
        match *self {
            Property::Drain { inv_tau } => {
                acc.lambda_rho = acc.lambda_rho + chi * inv_tau;
            }
            Property::Wall {
                inv_eta_n,
                inv_eta_t,
            } => {
                acc.lambda_un = acc.lambda_un + chi * inv_eta_n;
                acc.lambda_ut = acc.lambda_ut + chi * inv_eta_t;
                acc.u_solid = kin.u_solid;
            }
            Property::IsothermalWall { inv_eta } => {
                acc.lambda_e = acc.lambda_e + chi * inv_eta;
                acc.e_target = kin.e_wall;
            }
            Property::PorousAccretor {
                p,
                inv_tau,
                inv_eta_n,
                inv_eta_t,
            } => {
                acc.lambda_rho = acc.lambda_rho + p * chi * inv_tau;
                let solidity = (S::ONE - p) * chi;
                acc.lambda_un = acc.lambda_un + solidity * inv_eta_n;
                acc.lambda_ut = acc.lambda_ut + solidity * inv_eta_t;
                acc.u_solid = kin.u_solid;
            }
            Property::TorqueFreeAccretor { inv_tau, xi } => {
                let lambda_rho = chi * inv_tau;
                acc.lambda_rho = acc.lambda_rho + lambda_rho;
                // negative tangential rate: the tangential (angular) momentum is
                // retained as mass drains, so the accreted material carries zero
                // net moment about the sink. lambda_un stays 0 — the radial
                // relative momentum drains with the mass (accreted, radial force
                // has zero moment on a spherical mask).
                acc.lambda_ut = acc.lambda_ut - xi * lambda_rho;
                // the retention floor: bound the growing tangential factor so a
                // fully draining cell holds the retained momentum at a finite
                // velocity.
                acc.ut_growth_cap = S::from_f64(1.0 / TORQUE_FREE_RETENTION_FLOOR);
                acc.u_solid = kin.u_solid;
            }
        }
    }
}

/// integrate the accumulated relaxation system on one cell over `dt`: the
/// three exact exponentials, reconstructed to conserved variables in the
/// scaling-plus-vanishing-corrections form (header). returns the updated
/// conserved state and the cell's exact contribution to the body
/// (`U_old - U_new` integrated over the cell) — the single rule that makes
/// gas+body conservation of mass, momentum, and energy machine-exact for
/// every property stack, including drag work landing on the body.
pub fn penalize_cell<S: Scalar, const D: usize, E: EnergyModel, X: DyeModel>(
    cons: &ConsG<S, D, E, X>,
    relax: &Relax<S, D>,
    normal: Tensor<S, D>,
    dt: S,
    volume: S,
    idx: usize,
) -> (ConsG<S, D, E, X>, BodyDelta<S, D>) {
    let half = S::from_f64(0.5);
    let f_rho = (-(relax.lambda_rho * dt)).exp();
    let g_n = S::ONE - (-(relax.lambda_un * dt)).exp();
    // tangential growth factor, capped. for a decaying wall (lambda_ut >= 0) the
    // factor is <= 1 and the cap (INFINITY by default) is inert; for the
    // torque-free channel (lambda_ut < 0) it grows, and the cap bounds the
    // retained-momentum velocity boost so a vanishing remnant stays finite
    // (the retention floor).
    let b_t = (-(relax.lambda_ut * dt)).exp().min(relax.ut_growth_cap);
    let g_t = S::ONE - b_t;
    let g_e = S::ONE - (-(relax.lambda_e * dt)).exp();

    let inv_den = S::ONE / cons.den;
    let u = cons.mom.scale(inv_den);
    let mom_sq = cons.mom.dot(&cons.mom);
    // the isothermal slot values to zero here; every consumer of e_int below
    // is scaled into the energy slot and discarded again, so the junk stays
    // confined to that dead slot (and the trace DCEs it).
    let e_int = (cons.nrg.value() - half * mom_sq * inv_den) * inv_den;

    // velocity channel: du split along the body normal; each component decays
    // at its own rate. u' - u = -(du_n g_n + du_t g_t) is exactly +-0 when
    // both walls are off.
    let du = u - relax.u_solid;
    let du_n = normal.scale(du.dot(&normal));
    let du_t = du - du_n;
    let u_delta = -(du_n.scale(g_n) + du_t.scale(g_t));

    let den_new = cons.den * f_rho;
    let mom_new = cons.mom.scale(f_rho) + u_delta.scale(den_new);
    // energy conservation. the gas total energy the wall exchanges with the body is the work done on
    // the wall — the transferred momentum times the local wall velocity `u_solid` (= v_com + omega x
    // r, baked per cell) — so summed, `energy_delta` equals the body's mechanical work
    // `F.v_com + torque.omega` and gas+body mechanical energy conserves. the drag's frictional
    // dissipation is the gas kinetic-energy change minus that work, `Q = wall_work - ke_delta`
    // (>= 0): mechanical energy the relaxation turns to heat.
    let ke_delta = half * (u.scale(S::from_f64(2.0)) + u_delta).dot(&u_delta);
    let wall_work = u_delta.dot(&relax.u_solid);
    let friction_heat = wall_work - ke_delta;
    // thermal channel: the wall carries heat toward its temperature target, relaxing the
    // friction-heated internal energy `e_int + Q`. an isothermal wall (stiff g_e) absorbs the drag
    // dissipation — the gas stays at `e_target`, the heat flowing to the wall reservoir (booked in
    // energy_delta); an adiabatic wall (thermal property omitted, g_e = 0) keeps `Q` in the gas as
    // heat.
    // exactly +-0 when the thermal channel is off.
    let e_delta = (relax.e_target - (e_int + friction_heat)) * g_e;
    let nrg_new = cons
        .nrg
        .scale(f_rho)
        .add(E::Slot::from_scalar(den_new * (e_delta + wall_work)));

    let updated = ConsG {
        den: den_new,
        mom: mom_new,
        nrg: nrg_new,
        // the sink removes gas together with the dye dissolved in it, so the conserved dye takes
        // the same factor the density does and the concentration of the surviving gas is
        // unchanged. the wall channels move momentum and heat alone, leaving the dye to ride with
        // the mass, so `absorbed` books the accreted dye alongside the accreted mass.
        chi: cons.chi.scale(f_rho),
    };
    let absorbed = *cons - updated;
    let mut delta = BodyDelta::new(idx);
    delta.mass_delta = absorbed.den * volume;
    delta.force_delta = absorbed.mom.scale(volume / dt);
    delta.energy_delta = absorbed.nrg.value() * volume;
    (updated, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// an adiabatic wall drives the gas entropy upward.
    ///
    /// `K = p / rho^gamma` is one-way for a gas whose only thermal input is dissipation: drag
    /// turns mechanical energy into heat and the gas keeps it. the wall's three channels each
    /// respect that separately, which makes the law an exact assertion:
    ///
    /// - the mass channel is a uniform scaling `f_rho` of the whole conserved vector, which
    ///   leaves the velocity untouched and sends `K -> f_rho^(1-gamma) K`. for `f_rho <= 1` and
    ///   `gamma > 1` that is a rise.
    /// - the velocity channels move momentum toward the wall and book the difference between the
    ///   exact kinetic-energy change and the work done on the wall as friction heat `Q >= 0`.
    /// - the thermal channel carries heat toward `e_target`, and is a reservoir: an isothermal
    ///   wall absorbing drag dissipation legitimately lowers the gas entropy. this sweep sets
    ///   `lambda_e = 0`, scoping the law to the adiabatic wall.
    ///
    /// swept over a lattice so the claim — the sign of `Q` — is exercised where `u`, `u_solid`
    /// and the normal are misaligned; a single sample covers one alignment.
    #[test]
    fn an_adiabatic_wall_does_not_destroy_entropy() {
        const GAMMA: f64 = 5.0 / 3.0;
        let entropy = |c: &Cons3| {
            let e_int = (c.nrg.value() - 0.5 * c.mom.dot(&c.mom) / c.den) / c.den;
            (GAMMA - 1.0) * e_int * c.den / c.den.powf(GAMMA)
        };

        let mut checked = 0usize;
        let mut moved = 0usize;
        for den in [0.5, 2.5, 40.0] {
            for e_int in [0.05, 1.7, 30.0] {
                for u in [
                    Tensor::new([0.0, 0.0, 0.0]),
                    Tensor::new([0.3, -0.2, 0.1]),
                    Tensor::new([-2.0, 1.5, 0.4]),
                ] {
                    for u_solid in [
                        Tensor::new([0.0, 0.0, 0.0]),
                        Tensor::new([0.5, 0.5, -0.5]),
                        Tensor::new([-1.0, 0.0, 2.0]),
                    ] {
                        for (l_rho, l_un, l_ut) in [
                            (0.0, 10.0, 0.0),
                            (0.0, 0.0, 10.0),
                            (3.0, 10.0, 10.0),
                            (3.0, 1.0e3, 1.0e3),
                        ] {
                            for dt in [1.0e-4, 1.0e-2] {
                                let cons = Cons3 {
                                    chi: Default::default(),
                                    den,
                                    mom: u.scale(den),
                                    nrg: (den * (e_int + 0.5 * u.dot(&u))).into(),
                                };
                                let relax = Relax {
                                    lambda_rho: l_rho,
                                    lambda_un: l_un,
                                    lambda_ut: l_ut,
                                    u_solid,
                                    // the adiabatic wall: no thermal reservoir to lose heat to.
                                    lambda_e: 0.0,
                                    e_target: 0.0,
                                    ut_growth_cap: f64::INFINITY,
                                };
                                // a normal deliberately misaligned with every velocity in the
                                // sweep, so the normal/tangential split is exercised on both
                                // components at once.
                                let n = Tensor::new([0.6, 0.8, 0.0]);
                                let (out, _) = penalize_cell(&cons, &relax, n, dt, 1.0, 0);

                                let (k_in, k_out) = (entropy(&cons), entropy(&out));
                                assert!(
                                    k_out.is_finite() && out.den > 0.0,
                                    "penalization produced a non-finite state: den {} nrg {}",
                                    out.den,
                                    out.nrg.value()
                                );
                                assert!(
                                    k_out >= k_in * (1.0 - 1.0e-12),
                                    "an adiabatic wall destroyed entropy: K {k_in:.12e} -> \
                                     {k_out:.12e} at den {den}, e_int {e_int}, u {u:?}, \
                                     u_solid {u_solid:?}, lambdas ({l_rho}, {l_un}, {l_ut}), \
                                     dt {dt}. drag dissipates and the mass channel scales \
                                     uniformly; neither can take heat out of the gas"
                                );
                                checked += 1;
                                if (k_out - k_in).abs() > 1.0e-9 * k_in {
                                    moved += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // non-vacuity: the wall has to have moved the entropy on a substantial fraction of the
        // lattice. a `Relax` that reduced to a no-op everywhere would satisfy the inequality
        // trivially.
        println!("adiabatic wall: {moved}/{checked} lattice points moved the entropy");
        assert!(
            moved * 4 > checked,
            "only {moved} of {checked} lattice points changed K at all; the wall is close to a \
             no-op over this lattice and the law is nearly vacuous"
        );
    }
    use crate::drain::drain_cell;
    use symbi_hydro::energy::{Adiabatic, IsoModel};

    type Cons3 = ConsG<f64, 3, Adiabatic>;

    fn sample_cons() -> Cons3 {
        let den = 2.5;
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = 1.7;
        let nrg = den * (e_int + 0.5 * v.dot(&v));
        ConsG {
            chi: Default::default(),
            den,
            mom: v.scale(den),
            nrg,
        }
    }

    fn kin() -> BodyKin<f64, 3> {
        BodyKin {
            u_solid: Tensor::new([0.05, 0.0, -0.02]),
            omega: Tensor::zeros(),
            e_wall: 0.9,
        }
    }

    fn normal() -> Tensor<f64, 3> {
        let n = Tensor::new([0.6, 0.8, 0.0]);
        n.scale(1.0 / f64::sqrt(n.dot(&n)))
    }

    fn primitives(c: &Cons3) -> (f64, Tensor<f64, 3>, f64) {
        let u = c.mom.scale(1.0 / c.den);
        let e = (c.nrg - 0.5 * c.mom.dot(&c.mom) / c.den) / c.den;
        (c.den, u, e)
    }

    // the single-property (p = 1) case: the [Drain]-only stack reduces bit-for-bit
    // to drain_cell's uniform scaling — every correction term is an exact zero.
    #[test]
    fn drain_only_stack_is_bit_identical_to_drain_cell() {
        let cons = sample_cons();
        let (chi, tau, dt, vol) = (0.73, 0.04, 0.011, 1.5e-3);
        // contribute adds chi * inv_tau; inv_tau = 1/tau makes lambda = chi/tau,
        // matching drain_cell's exponent exactly.
        let mut acc = Relax::none();
        Property::Drain { inv_tau: 1.0 / tau }.contribute(chi, &kin(), &mut acc);
        let (pen, pen_delta) = penalize_cell(&cons, &acc, normal(), dt, vol, 0);
        let (dr, dr_delta) = drain_cell(&cons, chi, tau, dt, vol, 0);
        assert_eq!(pen.den.to_bits(), dr.den.to_bits());
        for a in 0..3 {
            assert_eq!(pen.mom[a].to_bits(), dr.mom[a].to_bits());
        }
        assert_eq!(pen.nrg.to_bits(), dr.nrg.to_bits());
        assert_eq!(
            pen_delta.mass_delta.to_bits(),
            dr_delta.mass_delta.to_bits()
        );
        assert_eq!(
            pen_delta.energy_delta.to_bits(),
            dr_delta.energy_delta.to_bits()
        );
        for a in 0..3 {
            assert_eq!(
                pen_delta.force_delta[a].to_bits(),
                dr_delta.force_delta[a].to_bits()
            );
        }
    }

    // each channel's exact exponential against the analytic frozen-
    // coefficient solution, all three active at once.
    #[test]
    fn channels_match_the_analytic_exponentials() {
        let cons = sample_cons();
        let (d0, u0, e0) = primitives(&cons);
        let k = kin();
        let n = normal();
        let dt = 0.37;
        let mut acc = Relax::none();
        Property::Drain { inv_tau: 2.0 }.contribute(0.8, &k, &mut acc);
        Property::Wall {
            inv_eta_n: 30.0,
            inv_eta_t: 5.0,
        }
        .contribute(0.8, &k, &mut acc);
        Property::IsothermalWall { inv_eta: 11.0 }.contribute(0.8, &k, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, dt, 1.0, 0);
        let (d1, u1, e1) = primitives(&out);

        assert!((d1 - d0 * (-acc.lambda_rho * dt).exp()).abs() < 1e-14);
        let du = u0 - k.u_solid;
        let du_n = n.scale(du.dot(&n));
        let du_t = du - du_n;
        let expect_u = k.u_solid
            + du_n.scale((-acc.lambda_un * dt).exp())
            + du_t.scale((-acc.lambda_ut * dt).exp());
        for a in 0..3 {
            assert!(
                (u1[a] - expect_u[a]).abs() < 1e-13,
                "axis {a}: {} vs {}",
                u1[a],
                expect_u[a]
            );
        }
        // the velocity relaxation dissipates the frictional heat Q = |du_n|^2 g_n(1 - g_n/2) +
        // |du_t|^2 g_t(1 - g_t/2); the thermal channel then carries off a fraction g_e of it to the
        // wall reservoir, so the gas retains (1 - g_e) = exp(-lambda_e dt) of Q on top of the pure
        // thermal relaxation. e1 = thermal-relaxation + Q*(1 - g_e).
        let expect_e_thermal = k.e_wall + (e0 - k.e_wall) * (-acc.lambda_e * dt).exp();
        let g_n = 1.0 - (-acc.lambda_un * dt).exp();
        let g_t = 1.0 - (-acc.lambda_ut * dt).exp();
        let heat =
            du_n.dot(&du_n) * g_n * (1.0 - 0.5 * g_n) + du_t.dot(&du_t) * g_t * (1.0 - 0.5 * g_t);
        let retained = heat * (-acc.lambda_e * dt).exp();
        assert!(
            (e1 - (expect_e_thermal + retained)).abs() < 1e-13,
            "internal energy off: e1 {e1} vs thermal {expect_e_thermal} + retained heat {retained}"
        );
    }

    // the properties act on disjoint channels, so every stack
    // ordering accumulates the same Relax and produces the same bits.
    #[test]
    fn stack_order_does_not_change_a_single_bit() {
        let cons = sample_cons();
        let k = kin();
        let n = normal();
        let props = [
            Property::Drain { inv_tau: 3.0 },
            Property::Wall {
                inv_eta_n: 20.0,
                inv_eta_t: 7.0,
            },
            Property::IsothermalWall { inv_eta: 9.0 },
        ];
        let orders: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let run = |order: &[usize; 3]| {
            let mut acc = Relax::none();
            for &i in order {
                props[i].contribute(0.6, &k, &mut acc);
            }
            penalize_cell(&cons, &acc, n, 0.02, 1.0, 0).0
        };
        let first = run(&orders[0]);
        for order in &orders[1..] {
            let out = run(order);
            assert_eq!(out.den.to_bits(), first.den.to_bits());
            for a in 0..3 {
                assert_eq!(out.mom[a].to_bits(), first.mom[a].to_bits());
            }
            assert_eq!(out.nrg.to_bits(), first.nrg.to_bits());
        }
    }

    // unconditional stability: a rate stiff beyond any explicit scheme's CFL
    // drives the state onto the target, finitely, for any dt.
    #[test]
    fn stiff_rates_saturate_to_the_targets() {
        let cons = sample_cons();
        let k = kin();
        let n = normal();
        let mut acc = Relax::none();
        Property::Wall {
            inv_eta_n: 1e12,
            inv_eta_t: 1e12,
        }
        .contribute(1.0, &k, &mut acc);
        Property::IsothermalWall { inv_eta: 1e12 }.contribute(1.0, &k, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, 10.0, 1.0, 0);
        let (_, u1, e1) = primitives(&out);
        for a in 0..3 {
            assert!(u1[a].is_finite());
            assert!((u1[a] - k.u_solid[a]).abs() < 1e-12);
        }
        // the stiff velocity relaxation dumps its full relative KE as frictional heat, but the stiff
        // isothermal channel absorbs all of it (the heat flows to the wall reservoir), so the gas
        // saturates back to the wall temperature e_wall.
        assert!((e1 - k.e_wall).abs() < 1e-12);
        // the drain channel was off: density untouched, bit-exact.
        assert_eq!(out.den.to_bits(), cons.den.to_bits());
    }

    // the returned body delta is the gas's loss — conservation is a
    // subtraction, exact by construction, pinned here against regressions.
    #[test]
    fn gas_loss_equals_body_gain_exactly() {
        let cons = sample_cons();
        let mut acc = Relax::none();
        Property::PorousAccretor {
            p: 0.4,
            inv_tau: 8.0,
            inv_eta_n: 15.0,
            inv_eta_t: 15.0,
        }
        .contribute(0.9, &kin(), &mut acc);
        let (vol, dt) = (2.5e-3, 0.013);
        let (out, delta) = penalize_cell(&cons, &acc, normal(), dt, vol, 0);
        assert_eq!(
            delta.mass_delta.to_bits(),
            ((cons.den - out.den) * vol).to_bits()
        );
        for a in 0..3 {
            assert_eq!(
                delta.force_delta[a].to_bits(),
                ((cons.mom[a] - out.mom[a]) * (vol / dt)).to_bits(),
            );
        }
        assert_eq!(
            delta.energy_delta.to_bits(),
            ((cons.nrg - out.nrg) * vol).to_bits()
        );
    }

    // what the penalization sink does to a 2D orbiting flow: the pure drain
    // (p = 1) removes mass at the local gas velocity, so the body absorbs the
    // gas's full momentum -- force = Mdot u -- and the accretion torque is the
    // moment of that, Mdot (r x u)_z = Mdot |r| v_orb for azimuthal motion. this
    // is a standard (angular-momentum-absorbing) sink: the booked torque is the
    // physical angular-momentum flux of the accreted gas, exact by conservation
    // (gas loss = body gain). a
    // purely radial inflow (u parallel r) carries zero angular momentum and books
    // exactly zero torque -- the torque tracks the flow it is given.
    #[test]
    fn drain_books_the_physical_accretion_torque_in_2d() {
        type Cons2 = ConsG<f64, 2, Adiabatic>;
        let r = Tensor::new([1.5, 0.0]); // gas-cell offset from the body center
        let v_orb = 0.8;
        let u = Tensor::new([0.0, v_orb]); // azimuthal: u perpendicular to r
        let den = 3.0;
        let e_int = 1.1;
        let cons: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let (tau, dt, vol) = (0.05, 0.01, 2.0);
        let (_, delta) = drain_cell(&cons, 1.0, tau, dt, vol, 0); // chi = 1: inside the mask

        let mdot = delta.mass_delta / dt;
        // force = Mdot u: momentum absorbed at the local velocity.
        for a in 0..2 {
            assert!((delta.force_delta[a] - mdot * u[a]).abs() < 1e-12);
        }
        // accretion torque = moment(r, force) = Mdot (r x u)_z = Mdot |r| v_orb.
        let tau_z = moment(&r, &delta.force_delta)[2];
        assert!((tau_z - mdot * (r[0] * u[1] - r[1] * u[0])).abs() < 1e-12);
        assert!(
            tau_z > 0.0,
            "an orbiting standard sink absorbs angular momentum"
        );

        // purely radial inflow (u along -r) carries no angular momentum: zero
        // torque even for the standard sink -- the booked torque is physical.
        let u_rad = Tensor::new([-0.5, 0.0]);
        let cons_r: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u_rad.scale(den),
            nrg: den * (e_int + 0.5 * u_rad.dot(&u_rad)),
        };
        let (_, delta_r) = drain_cell(&cons_r, 1.0, tau, dt, vol, 0);
        assert_eq!(moment(&r, &delta_r.force_delta)[2], 0.0);
    }

    // the torque-free dial books the analytic accretion
    // torque about the sink center on a spherical mask (normal = radial),
    //   tau_z = |r| rho (1 - f_rho^{1-xi}) (n_hat x du_t) * (vol / dt),
    // the radial-relative momentum contributing zero moment. xi = 0 recovers the
    // standard sink (full angular momentum), xi = 1 is exactly torque-free.
    #[test]
    fn torque_free_dial_books_the_analytic_torque_in_2d() {
        type Cons2 = ConsG<f64, 2, Adiabatic>;
        let r = Tensor::new([1.5, 0.0]); // gas-cell offset from the sink center
        let n = Tensor::new([1.0, 0.0]); // spherical-mask normal = r_hat
        let u = Tensor::new([0.3, 0.8]); // mixed radial + azimuthal flow
        let den = 3.0;
        let e_int = 1.1;
        let cons: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let (inv_tau, dt, vol, chi) = (5.0, 0.02, 2.0, 1.0);
        // u_solid = 0, so du = u; du_t is the component perpendicular to n.
        let du_t = u - n.scale(u.dot(&n));
        let f_rho = (-(chi * inv_tau * dt)).exp();
        let kin0 = BodyKin::<f64, 2> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: 0.0,
        };

        for &xi in &[0.0, 0.5, 1.0] {
            let mut acc = Relax::none();
            Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin0, &mut acc);
            let (_out, delta) = penalize_cell(&cons, &acc, n, dt, vol, 0);
            let coeff = 1.0 - f_rho.powf(1.0 - xi); // retained-tangential complement
            let expect =
                (r[0] * (den * coeff * du_t[1]) - r[1] * (den * coeff * du_t[0])) * (vol / dt);
            let got = moment(&r, &delta.force_delta)[2];
            assert!((got - expect).abs() < 1e-11, "xi={xi}: {got} vs {expect}");
        }

        // the torque-free endpoint is exactly zero for this orbiting cell.
        let mut acc = Relax::none();
        Property::TorqueFreeAccretor { inv_tau, xi: 1.0 }.contribute(chi, &kin0, &mut acc);
        let (_o, d) = penalize_cell(&cons, &acc, n, dt, vol, 0);
        assert!(moment(&r, &d.force_delta)[2].abs() < 1e-12);
    }

    // the raw (uncapped) coupling, which is what the retention floor exists to
    // bound. built by hand so the tangential growth cap is INFINITY (the
    // Property sets a finite cap). two regimes:
    //   (1) strong-but-finite drain: tangential momentum retained (bounded) while
    //       the primitive velocity u'_t = du_t / f_rho diverges.
    //   (2) full evacuation (f_rho underflows to 0): `0 * inf = NaN` — the
    //       conserved momentum itself goes non-finite. the cliff the floor averts.
    #[test]
    fn torque_free_raw_coupling_is_ill_posed_at_evacuation() {
        type Cons2 = ConsG<f64, 2, Adiabatic>;
        let n = Tensor::new([1.0, 0.0]);
        let u = Tensor::new([0.3, 0.7]); // radial 0.3 (drains) + azimuthal 0.7 (retained)
        let den = 4.0;
        let e_int = 1.0;
        let cons: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };

        // (1) lambda_rho dt = 30 -> f_rho ~ 9e-14, finite. cap = INFINITY (raw).
        let mut acc = Relax::<f64, 2>::none();
        acc.lambda_rho = 30.0;
        acc.lambda_ut = -30.0; // xi = 1, uncapped
        let (out, _) = penalize_cell(&cons, &acc, n, 1.0, 1.0, 0);
        assert!(out.den < 1e-10 * den, "mass nearly fully drained");
        assert!(
            (out.mom[1] - cons.mom[1]).abs() < 1e-6,
            "tangential momentum retained (bounded)"
        );
        assert!(out.mom[1] / out.den > 1e10, "primitive velocity diverges");

        // (2) lambda_rho dt = 1000 -> f_rho underflows to 0 -> 0 * inf = NaN.
        let mut acc = Relax::<f64, 2>::none();
        acc.lambda_rho = 1e3;
        acc.lambda_ut = -1e3;
        let (out, _) = penalize_cell(&cons, &acc, n, 1.0, 1.0, 0);
        assert!(
            !out.mom[1].is_finite(),
            "uncapped conserved momentum is NaN at underflow"
        );
    }

    // the retention floor keeps the conserved state finite at full evacuation
    // and holds torque-free exact in the physical regime (f_rho >> f_floor).
    // the Property sets the cap.
    #[test]
    fn torque_free_saturation_bounds_the_evacuation_limit() {
        type Cons2 = ConsG<f64, 2, Adiabatic>;
        let n = Tensor::new([1.0, 0.0]);
        let r = Tensor::new([1.5, 0.0]);
        let (den, e_int) = (4.0, 1.0);
        let kin0 = BodyKin::<f64, 2> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: 0.0,
        };

        // (1) full evacuation: the conserved state stays finite. the cap
        // bounds the tangential factor at 1/f_floor, so `den' (du_t g_t)` is
        // `0 * finite = 0`, avoiding the `0 * inf = NaN` pathology.
        let u = Tensor::new([0.3, 0.7]);
        let cons: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let mut acc = Relax::none();
        Property::TorqueFreeAccretor {
            inv_tau: 1e3,
            xi: 1.0,
        }
        .contribute(1.0, &kin0, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, 1.0, 1.0, 0);
        assert!(
            out.den.is_finite()
                && out.mom[0].is_finite()
                && out.mom[1].is_finite()
                && out.nrg.value().is_finite(),
            "saturated conserved state is finite at underflow"
        );

        // (2) physical regime: lambda_rho dt = 0.5 -> f_rho = 0.607, growth factor
        // 1.65 << the 1e4 cap (inert) -> torque-free is exact.
        let u2 = Tensor::new([0.3, 0.8]);
        let cons2: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u2.scale(den),
            nrg: den * (e_int + 0.5 * u2.dot(&u2)),
        };
        let mut acc2 = Relax::none();
        Property::TorqueFreeAccretor {
            inv_tau: 0.5,
            xi: 1.0,
        }
        .contribute(1.0, &kin0, &mut acc2);
        let (_o2, d2) = penalize_cell(&cons2, &acc2, n, 1.0, 1.0, 0);
        assert!(
            moment(&r, &d2.force_delta)[2].abs() < 1e-12,
            "torque-free exact above the retention floor"
        );
    }

    // the saturated torque-free sink holds torque == 0 across the
    // whole physical range of per-step drain fractions (f_rho down to the floor)
    // — the cap is inert there — and reintroduces a bounded torque once a
    // cell is drained below the floor in a single step (`inv_tau dt >> 1`, an
    // over-aggressive/pathological rate carrying negligible mass). this is the
    // evidence that saturation preserves torque-free where the physics lives.
    #[test]
    fn saturated_torque_free_holds_across_the_physical_range() {
        type Cons2 = ConsG<f64, 2, Adiabatic>;
        let n = Tensor::new([1.0, 0.0]);
        let r = Tensor::new([2.0, 0.0]);
        let u = Tensor::new([0.1, 0.6]); // radial + azimuthal
        let (den, e_int, dt) = (3.0, 1.0, 1.0);
        let cons: Cons2 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let kin0 = BodyKin::<f64, 2> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: 0.0,
        };
        // physical per-step drain fractions, down to the floor (1e-4): torque-free.
        for &f_rho in &[0.9_f64, 0.5, 0.1, 1e-2, 1e-3, 1e-4] {
            let inv_tau = -f_rho.ln() / dt;
            let mut acc = Relax::none();
            Property::TorqueFreeAccretor { inv_tau, xi: 1.0 }.contribute(1.0, &kin0, &mut acc);
            let (_o, d) = penalize_cell(&cons, &acc, n, dt, 1.0, 0);
            assert!(
                moment(&r, &d.force_delta)[2].abs() < 1e-9,
                "torque-free at f_rho = {f_rho}"
            );
        }
        // below the floor (f_rho = 1e-6): the cap fires -> a bounded standard torque.
        let inv_tau = -(1e-6_f64).ln() / dt;
        let mut acc = Relax::none();
        Property::TorqueFreeAccretor { inv_tau, xi: 1.0 }.contribute(1.0, &kin0, &mut acc);
        let (_o, d) = penalize_cell(&cons, &acc, n, dt, 1.0, 0);
        let t = moment(&r, &d.force_delta)[2];
        assert!(
            t.is_finite() && t.abs() > 1e-3,
            "below the floor: bounded standard torque, not NaN"
        );
    }

    // the moment/cross helpers agree across dimensions: the 2D forms are the
    // z-slice of the 3D embedding, and rotation has no 1D meaning.
    #[test]
    fn moment_and_cross_embed_consistently() {
        let w3: Tensor<f64, 3> = Tensor::new([0.0, 0.0, 0.7]);
        let r2: Tensor<f64, 2> = Tensor::new([0.3, -0.4]);
        let r3: Tensor<f64, 3> = Tensor::new([0.3, -0.4, 0.0]);
        let u2 = omega_cross(&w3, &r2);
        let u3 = omega_cross(&w3, &r3);
        assert_eq!(u2[0].to_bits(), u3[0].to_bits());
        assert_eq!(u2[1].to_bits(), u3[1].to_bits());
        assert_eq!(u3[2], 0.0);

        let f2: Tensor<f64, 2> = Tensor::new([0.9, 0.2]);
        let f3: Tensor<f64, 3> = Tensor::new([0.9, 0.2, 0.0]);
        let t2 = moment(&r2, &f2);
        let t3 = moment(&r3, &f3);
        assert_eq!(t2[2].to_bits(), t3[2].to_bits());
        assert_eq!((t3[0], t3[1]), (0.0, 0.0));

        let w1: Tensor<f64, 3> = Tensor::new([0.0, 0.0, 5.0]);
        let r1: Tensor<f64, 1> = Tensor::new([2.0]);
        assert_eq!(omega_cross(&w1, &r1)[0], 0.0);
        assert_eq!(moment(&r1, &r1), Tensor::zeros());
    }

    // the rotating target rides the same exponentials —
    // u relaxes toward u_translation + omega x x_rel, analytic per channel.
    #[test]
    fn rotating_target_matches_the_analytic_exponential() {
        let cons = sample_cons();
        let (_, u0, _) = primitives(&cons);
        let base = BodyKin::<f64, 3> {
            u_solid: Tensor::new([0.05, 0.0, -0.02]),
            omega: Tensor::new([0.1, -0.3, 0.8]),
            e_wall: 0.9,
        };
        let x_rel = Tensor::new([0.21, -0.13, 0.34]);
        let k = base.at(&x_rel);
        let expect_target = base.u_solid + omega_cross(&base.omega, &x_rel);
        for a in 0..3 {
            assert_eq!(k.u_solid[a].to_bits(), expect_target[a].to_bits());
        }

        let n = normal();
        let dt = 0.23;
        let mut acc = Relax::none();
        Property::Wall {
            inv_eta_n: 12.0,
            inv_eta_t: 4.0,
        }
        .contribute(0.7, &k, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, dt, 1.0, 0);
        let (_, u1, _) = primitives(&out);
        let du = u0 - k.u_solid;
        let du_n = n.scale(du.dot(&n));
        let du_t = du - du_n;
        let expect_u = k.u_solid
            + du_n.scale((-acc.lambda_un * dt).exp())
            + du_t.scale((-acc.lambda_ut * dt).exp());
        for a in 0..3 {
            assert!((u1[a] - expect_u[a]).abs() < 1e-13);
        }
    }

    // zero spin through `at()` is bit-identical to the
    // constant-target path — the omega terms are exactly +-0.
    #[test]
    fn zero_omega_at_reduces_bit_exactly() {
        let cons = sample_cons();
        let k = kin();
        let x_rel = Tensor::new([0.4, -0.2, 0.9]);
        let n = normal();
        let dt = 0.05;
        let run = |kin_used: &BodyKin<f64, 3>| {
            let mut acc = Relax::none();
            Property::Wall {
                inv_eta_n: 9.0,
                inv_eta_t: 3.0,
            }
            .contribute(0.8, kin_used, &mut acc);
            Property::Drain { inv_tau: 2.0 }.contribute(0.8, kin_used, &mut acc);
            penalize_cell(&cons, &acc, n, dt, 1.0, 0).0
        };
        let direct = run(&k);
        let via_at = run(&k.at(&x_rel));
        assert_eq!(direct.den.to_bits(), via_at.den.to_bits());
        assert_eq!(direct.nrg.to_bits(), via_at.nrg.to_bits());
        for a in 0..3 {
            assert_eq!(direct.mom[a].to_bits(), via_at.mom[a].to_bits());
        }
    }

    // gas already in rigid co-rotation with the target
    // gets exactly +-0 corrections — a stiff spinning wall is a no-op on it.
    #[test]
    fn co_rotation_is_an_exact_no_op() {
        let base = BodyKin::<f64, 3> {
            u_solid: Tensor::new([0.02, -0.07, 0.11]),
            omega: Tensor::new([0.5, -1.1, 2.3]),
            e_wall: 0.0,
        };
        let x_rel = Tensor::new([0.17, 0.29, -0.23]);
        let k = base.at(&x_rel);
        // seed the gas exactly co-moving: same helper, same bits. den is a
        // power of two so the kernel's u = mom * (1/den) round trip is exact
        // and du is a true +-0.
        let den = 2.0;
        let u = base.u_solid + omega_cross(&base.omega, &x_rel);
        let e_int = 1.3;
        let cons: Cons3 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let mut acc = Relax::none();
        Property::Wall {
            inv_eta_n: 1e12,
            inv_eta_t: 1e12,
        }
        .contribute(1.0, &k, &mut acc);
        let (out, delta) = penalize_cell(&cons, &acc, normal(), 10.0, 1.0, 0);
        assert_eq!(out.den.to_bits(), cons.den.to_bits());
        assert_eq!(out.nrg.to_bits(), cons.nrg.to_bits());
        for a in 0..3 {
            assert_eq!(out.mom[a].to_bits(), cons.mom[a].to_bits());
            assert_eq!(delta.force_delta[a], 0.0);
        }
    }

    // free-slip porous surface (the porosity-slip dial):
    // inv_eta_t = 0 is an exact off switch — the tangential velocity is
    // bit-untouched while the normal channel relaxes.
    #[test]
    fn free_slip_leaves_the_tangential_velocity_bit_untouched() {
        // den a power of two: u = mom * (1/den) round-trips exactly, so the
        // tangential projection compares bit-for-bit (the same precondition
        // as the co-rotation test).
        let den = 4.0;
        let u = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = 1.7;
        let cons: Cons3 = ConsG {
            chi: Default::default(),
            den,
            mom: u.scale(den),
            nrg: den * (e_int + 0.5 * u.dot(&u)),
        };
        let n = Tensor::new([1.0, 0.0, 0.0]);
        let k = BodyKin::<f64, 3> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: 0.0,
        };
        let mut acc = Relax::none();
        Property::PorousAccretor {
            p: 0.3,
            inv_tau: 0.0,
            inv_eta_n: 1e12,
            inv_eta_t: 0.0,
        }
        .contribute(1.0, &k, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, 10.0, 1.0, 0);
        let u1 = out.mom.scale(1.0 / out.den);
        // normal component driven to the (zero) target, tangential exact.
        assert!(u1[0].abs() < 1e-12);
        assert_eq!(u1[1].to_bits(), u[1].to_bits());
        assert_eq!(u1[2].to_bits(), u[2].to_bits());
    }

    // the porosity endpoints: p = 1 accumulates exactly the [Drain] relax
    // (the wall term carries an exact (1 - p) = 0 factor), and p = 0
    // accumulates a zero drain rate, leaving the mass bit-exact.
    #[test]
    fn porosity_endpoints_reduce_exactly() {
        let cons = sample_cons();
        let k = kin();
        let n = normal();
        let (chi, dt) = (0.8, 0.02);

        let mut porous = Relax::none();
        Property::PorousAccretor {
            p: 1.0,
            inv_tau: 6.0,
            inv_eta_n: 11.0,
            inv_eta_t: 7.0,
        }
        .contribute(chi, &k, &mut porous);
        let mut drain = Relax::none();
        Property::Drain { inv_tau: 6.0 }.contribute(chi, &k, &mut drain);
        let (a, _) = penalize_cell(&cons, &porous, n, dt, 1.0, 0);
        let (b, _) = penalize_cell(&cons, &drain, n, dt, 1.0, 0);
        assert_eq!(a.den.to_bits(), b.den.to_bits());
        assert_eq!(a.nrg.to_bits(), b.nrg.to_bits());

        let mut sealed = Relax::none();
        Property::PorousAccretor {
            p: 0.0,
            inv_tau: 6.0,
            inv_eta_n: 11.0,
            inv_eta_t: 7.0,
        }
        .contribute(chi, &k, &mut sealed);
        let (c, delta) = penalize_cell(&cons, &sealed, n, dt, 1.0, 0);
        assert_eq!(c.den.to_bits(), cons.den.to_bits());
        assert_eq!(delta.mass_delta, 0.0);
    }

    // the isothermal regime: no energy channel — the drain and wall still act,
    // the thermal property is structurally inert (the slot discards it).
    #[test]
    fn iso_regime_has_no_energy_channel() {
        let cons: ConsG<f64, 2, IsoModel> = ConsG {
            chi: Default::default(),
            den: 1.4,
            mom: Tensor::new([0.7, -0.3]),
            nrg: Default::default(),
        };
        let k = BodyKin {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: 5.0,
        };
        let n = Tensor::new([1.0, 0.0]);
        let mut acc = Relax::none();
        Property::Drain { inv_tau: 4.0 }.contribute(0.5, &k, &mut acc);
        Property::IsothermalWall { inv_eta: 1e9 }.contribute(0.5, &k, &mut acc);
        let (out, delta) = penalize_cell(&cons, &acc, n, 0.1, 1.0, 0);
        let f = (-(acc.lambda_rho) * 0.1f64).exp();
        assert!((out.den - 1.4 * f).abs() < 1e-15);
        assert_eq!(delta.energy_delta, 0.0);
        assert!(out.mom[0].is_finite() && out.mom[1].is_finite());
    }
}
