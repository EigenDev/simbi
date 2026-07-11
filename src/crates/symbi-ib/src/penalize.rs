// =============================================================================
// penalize.rs
//
// the composable immersed-boundary property algebra (docs/design/50 layer 2):
// properties contribute relaxation RATES and TARGETS into a `Relax`
// accumulator; the engine integrates the accumulated system once, as exact
// frozen-coefficient exponentials on three DISJOINT primitive channels —
//   rho:   den' = den * exp(-lambda_rho dt)          (the uniform drain)
//   u:     u'   = u_solid + du_n e^-l_n + du_t e^-l_t (normal/tangential wall)
//   e_int: e'   = e_wall + (e - e_wall) e^-l_e        (thermal wall)
// — unconditionally stable for any dt, and commuting exactly because no
// property touches another's channel.
//
// the conserved reconstruction is spelled as CONSERVED SCALING PLUS
// CORRECTIONS THAT VANISH EXACTLY AT ZERO RATES:
//   den' = den f_rho
//   mom' = mom f_rho + den' (u' - u)
//   nrg' = nrg f_rho + den' [(e' - e) + (u + u'/2 + u/2)-form kinetic term]
// with u' - u = -(du_n g_n + du_t g_t), e' - e = (e_target - e) g_e, and
// g = 1 - exp(-lambda dt). when only the drain is active g_n = g_t = g_e = 0
// EXACTLY, every correction is an exact zero, and the update reduces bit-for-
// bit to `drain_cell`'s uniform scaling (up to the sign of exactly-zero
// momentum components: x + 0.0 flips -0 to +0). that reduction is the p = 1
// anchor the whole algebra gates on.
//
// carrier-generic: f64 for the oracle and unit laws, Gv for the traced
// penalization kernel, Dual for sensitivities. the isothermal regime's energy
// slot discards the e-channel corrections by construction (EnergySlot).
//
// usage:
//   let mut acc = Relax::none();
//   for p in &stack { p.contribute(chi, &kin, &mut acc); }
//   let (u_new, delta) = penalize_cell(&cons, &acc, normal, dt, volume, idx);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::energy::{EnergyModel, EnergySlot};
use symbi_hydro::state::ConsG;
use symbi_ir::algebra::Scalar;

use crate::body_delta::BodyDelta;

/// the body kinematics a property may target: the solid velocity field value
/// at the cell and the wall internal-energy target (from the wall temperature
/// through the EOS; unused unless a thermal property is stacked).
#[derive(Clone, Copy, Debug)]
pub struct BodyKin<S: Scalar, const D: usize> {
    pub u_solid: Tensor<S, D>,
    pub e_wall: S,
}

/// the accumulated relaxation system: rates and targets per primitive channel.
/// rates ADD across properties; targets are SET (single-body stack — the
/// multi-body nearest-wins policy composes above this).
#[derive(Clone, Copy, Debug)]
pub struct Relax<S: Scalar, const D: usize> {
    pub lambda_rho: S,
    pub lambda_un: S,
    pub lambda_ut: S,
    pub u_solid: Tensor<S, D>,
    pub lambda_e: S,
    pub e_target: S,
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
        }
    }
}

/// one boundary property: what it contributes to the relaxation system.
/// coefficients are carrier values (f64 on the host, scalar params in a
/// trace) as inverse timescales — the stiff limit is a LARGE `inv_*`, and
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
    /// the porosity dial: p = 1 pure drain, p = 0 pure no-slip wall.
    PorousAccretor { p: S, inv_tau: S, inv_eta: S },
}

impl<S: Scalar> Property<S> {
    pub fn contribute<const D: usize>(
        &self,
        chi: S,
        kin: &BodyKin<S, D>,
        acc: &mut Relax<S, D>,
    ) {
        match *self {
            Property::Drain { inv_tau } => {
                acc.lambda_rho = acc.lambda_rho + chi * inv_tau;
            }
            Property::Wall { inv_eta_n, inv_eta_t } => {
                acc.lambda_un = acc.lambda_un + chi * inv_eta_n;
                acc.lambda_ut = acc.lambda_ut + chi * inv_eta_t;
                acc.u_solid = kin.u_solid;
            }
            Property::IsothermalWall { inv_eta } => {
                acc.lambda_e = acc.lambda_e + chi * inv_eta;
                acc.e_target = kin.e_wall;
            }
            Property::PorousAccretor { p, inv_tau, inv_eta } => {
                acc.lambda_rho = acc.lambda_rho + p * chi * inv_tau;
                let wall = (S::ONE - p) * chi * inv_eta;
                acc.lambda_un = acc.lambda_un + wall;
                acc.lambda_ut = acc.lambda_ut + wall;
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
pub fn penalize_cell<S: Scalar, const D: usize, E: EnergyModel>(
    cons: &ConsG<S, D, E>,
    relax: &Relax<S, D>,
    normal: Tensor<S, D>,
    dt: S,
    volume: S,
    idx: usize,
) -> (ConsG<S, D, E>, BodyDelta<S, D>) {
    let half = S::from_f64(0.5);
    let f_rho = (-(relax.lambda_rho * dt)).exp();
    let g_n = S::ONE - (-(relax.lambda_un * dt)).exp();
    let g_t = S::ONE - (-(relax.lambda_ut * dt)).exp();
    let g_e = S::ONE - (-(relax.lambda_e * dt)).exp();

    let inv_den = S::ONE / cons.den;
    let u = cons.mom.scale(inv_den);
    let mom_sq = cons.mom.dot(&cons.mom);
    // the isothermal slot values to zero here; every consumer of e_int below
    // is scaled into the energy slot and discarded again, so the junk never
    // reaches a live channel (and the trace DCEs it).
    let e_int = (cons.nrg.value() - half * mom_sq * inv_den) * inv_den;

    // velocity channel: du split along the body normal; each component decays
    // at its own rate. u' - u = -(du_n g_n + du_t g_t) is EXACTLY +-0 when
    // both walls are off.
    let du = u - relax.u_solid;
    let du_n = normal.scale(du.dot(&normal));
    let du_t = du - du_n;
    let u_delta = -(du_n.scale(g_n) + du_t.scale(g_t));
    // thermal channel: e' - e = (e_target - e) g_e, exactly +-0 when off.
    let e_delta = (relax.e_target - e_int) * g_e;

    let den_new = cons.den * f_rho;
    let mom_new = cons.mom.scale(f_rho) + u_delta.scale(den_new);
    // |u'|^2 - |u|^2 = (u' + u).(u' - u) = (2u + u_delta).u_delta — zero
    // exactly with u_delta.
    let ke_delta = half * (u.scale(S::from_f64(2.0)) + u_delta).dot(&u_delta);
    let nrg_new = cons
        .nrg
        .scale(f_rho)
        .add(E::Slot::from_scalar(den_new * (e_delta + ke_delta)));

    let updated = ConsG { den: den_new, mom: mom_new, nrg: nrg_new };
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
    use crate::drain::drain_cell;
    use symbi_hydro::energy::{Adiabatic, IsoModel};

    type Cons3 = ConsG<f64, 3, Adiabatic>;

    fn sample_cons() -> Cons3 {
        let den = 2.5;
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = 1.7;
        let nrg = den * (e_int + 0.5 * v.dot(&v));
        ConsG { den, mom: v.scale(den), nrg }
    }

    fn kin() -> BodyKin<f64, 3> {
        BodyKin { u_solid: Tensor::new([0.05, 0.0, -0.02]), e_wall: 0.9 }
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

    // gate 4a (the p = 1 anchor): the [Drain]-only stack reduces BIT-FOR-BIT
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
        assert_eq!(pen_delta.mass_delta.to_bits(), dr_delta.mass_delta.to_bits());
        assert_eq!(pen_delta.energy_delta.to_bits(), dr_delta.energy_delta.to_bits());
        for a in 0..3 {
            assert_eq!(pen_delta.force_delta[a].to_bits(), dr_delta.force_delta[a].to_bits());
        }
    }

    // gate 2: each channel's exact exponential against the analytic frozen-
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
        Property::Wall { inv_eta_n: 30.0, inv_eta_t: 5.0 }.contribute(0.8, &k, &mut acc);
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
            assert!((u1[a] - expect_u[a]).abs() < 1e-13, "axis {a}: {} vs {}", u1[a], expect_u[a]);
        }
        let expect_e = k.e_wall + (e0 - k.e_wall) * (-acc.lambda_e * dt).exp();
        assert!((e1 - expect_e).abs() < 1e-13);
    }

    // gate 3: the properties act on disjoint channels, so every stack
    // ordering accumulates the same Relax and produces the same bits.
    #[test]
    fn stack_order_does_not_change_a_single_bit() {
        let cons = sample_cons();
        let k = kin();
        let n = normal();
        let props = [
            Property::Drain { inv_tau: 3.0 },
            Property::Wall { inv_eta_n: 20.0, inv_eta_t: 7.0 },
            Property::IsothermalWall { inv_eta: 9.0 },
        ];
        let orders: [[usize; 3]; 6] =
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
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
    // drives the state TO the target, finitely, for any dt.
    #[test]
    fn stiff_rates_saturate_to_the_targets() {
        let cons = sample_cons();
        let k = kin();
        let n = normal();
        let mut acc = Relax::none();
        Property::Wall { inv_eta_n: 1e12, inv_eta_t: 1e12 }.contribute(1.0, &k, &mut acc);
        Property::IsothermalWall { inv_eta: 1e12 }.contribute(1.0, &k, &mut acc);
        let (out, _) = penalize_cell(&cons, &acc, n, 10.0, 1.0, 0);
        let (_, u1, e1) = primitives(&out);
        for a in 0..3 {
            assert!(u1[a].is_finite());
            assert!((u1[a] - k.u_solid[a]).abs() < 1e-12);
        }
        assert!((e1 - k.e_wall).abs() < 1e-12);
        // the drain channel was off: density untouched, bit-exact.
        assert_eq!(out.den.to_bits(), cons.den.to_bits());
    }

    // gate 5: the returned body delta IS the gas's loss — conservation is a
    // subtraction, exact by construction, pinned here against regressions.
    #[test]
    fn gas_loss_equals_body_gain_exactly() {
        let cons = sample_cons();
        let mut acc = Relax::none();
        Property::PorousAccretor { p: 0.4, inv_tau: 8.0, inv_eta: 15.0 }
            .contribute(0.9, &kin(), &mut acc);
        let (vol, dt) = (2.5e-3, 0.013);
        let (out, delta) = penalize_cell(&cons, &acc, normal(), dt, vol, 0);
        assert_eq!(delta.mass_delta.to_bits(), ((cons.den - out.den) * vol).to_bits());
        for a in 0..3 {
            assert_eq!(
                delta.force_delta[a].to_bits(),
                ((cons.mom[a] - out.mom[a]) * (vol / dt)).to_bits(),
            );
        }
        assert_eq!(delta.energy_delta.to_bits(), ((cons.nrg - out.nrg) * vol).to_bits());
    }

    // the isothermal regime: no energy channel — the drain and wall still act,
    // the thermal property is structurally inert (the slot discards it).
    #[test]
    fn iso_regime_has_no_energy_channel() {
        let cons: ConsG<f64, 2, IsoModel> = ConsG {
            den: 1.4,
            mom: Tensor::new([0.7, -0.3]),
            nrg: Default::default(),
        };
        let k = BodyKin { u_solid: Tensor::zeros(), e_wall: 5.0 };
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
