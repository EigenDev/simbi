// =============================================================================
// eos.rs
//
// equation of state trait and implementations. the eos provides the
// thermodynamic closure relating density, pressure, and internal energy.
//
// usage:
//   let eos = IdealGas { gamma: 1.4 };
//   let iso = Isothermal { cs: 1.0 };
//   let a = eos.sound_speed(rho, pre);
// =============================================================================

use symbi_ir::algebra::Scalar;

/// equation of state: thermodynamic closure for the euler equations.
/// maps between (rho, p) and (rho, e_int) and provides the sound speed.
///
/// the `conserved_energy` / `recover_pressure` pair controls what the `nrg`
/// slot of `Cons` means. for ideal gas it is total energy (ke + rho*e_int).
/// for isothermal it stores cs^2 — the energy equation is not evolved.
pub trait Eos<S: Scalar>: Copy {
    /// adiabatic sound speed: a = sqrt(dp/drho|_s)
    fn sound_speed(&self, rho: S, pre: S) -> S;

    /// adiabatic sound speed SQUARED: a^2 = dp/drho|_s. the default squares
    /// `sound_speed`, but EOS impls override to skip the sqrt-then-square — the
    /// relativistic c2p Newton + wave speeds need cs^2, never cs, and a stray
    /// `sqrt(x).powi(2)` is a redundant transcendental per cell (and per Newton
    /// step) on the GPU.
    fn sound_speed_sq(&self, rho: S, pre: S) -> S {
        let a = self.sound_speed(rho, pre);
        a * a
    }

    /// specific internal energy from density and pressure
    fn internal_energy(&self, rho: S, pre: S) -> S;

    /// pressure from density and specific internal energy
    fn pressure(&self, rho: S, e_int: S) -> S;

    /// produce the nrg component of conservative state from primitives.
    /// default: total energy = 0.5 * rho * v^2 + rho * e_int.
    /// isothermal overrides to store cs^2.
    fn conserved_energy(&self, rho: S, v_sq: S, pre: S) -> S {
        S::from_f64(0.5) * rho * v_sq + rho * self.internal_energy(rho, pre)
    }

    /// recover pressure from conservative state.
    /// default: invert total energy to get e_int, then call pressure().
    /// isothermal overrides to read cs^2 from nrg.
    fn recover_pressure(&self, rho: S, v_sq: S, nrg: S) -> S {
        let ke = S::from_f64(0.5) * rho * v_sq;
        let e_int = (nrg - ke) / rho;
        self.pressure(rho, e_int)
    }

    /// adiabatic index as generic scalar.
    /// ideal gas: returns gamma. isothermal: returns 1.0 (unused).
    fn gamma(&self) -> S {
        S::ONE
    }

    /// extract EOS parameter for GPU kernel dispatch.
    /// ideal gas: returns gamma (adiabatic index).
    /// isothermal: returns cs^2 (kernels use it for c2p, flux, wave speed).
    fn gamma_for_ops(&self) -> f64 {
        1.0
    }

    /// the scalar a substrate `KernelSet` constructor takes — gamma for an ideal-gas regime,
    /// the sound speed `cs` for an isothermal one. DISTINCT from `gamma_for_ops` (which returns
    /// cs^2 for isothermal — the in-kernel form): the constructor wants `cs` itself. used by
    /// `SimState::substrate()` to build the matched KernelSet from the sim's EOS.
    fn substrate_param(&self) -> f64 {
        self.gamma_for_ops() // ideal gas: gamma; isothermal overrides to cs (below)
    }
}

/// ideal gas with constant adiabatic index gamma.
/// p = (gamma - 1) * rho * e_int
#[derive(Clone, Copy, Debug)]
pub struct IdealGas<S: Scalar> {
    pub gamma: S,
}

impl<S: Scalar> Eos<S> for IdealGas<S> {
    #[inline]
    fn sound_speed(&self, rho: S, pre: S) -> S {
        (self.gamma * pre / rho).sqrt()
    }

    #[inline]
    fn sound_speed_sq(&self, rho: S, pre: S) -> S {
        // a^2 = gamma * p / rho, directly (no sqrt-then-square).
        self.gamma * pre / rho
    }

    #[inline]
    fn internal_energy(&self, rho: S, pre: S) -> S {
        pre / ((self.gamma - S::ONE) * rho)
    }

    #[inline]
    fn pressure(&self, rho: S, e_int: S) -> S {
        (self.gamma - S::ONE) * rho * e_int
    }

    fn gamma(&self) -> S {
        self.gamma
    }

    fn gamma_for_ops(&self) -> f64 {
        self.gamma.to_f64()
    }
}

/// isothermal gas with constant sound speed.
/// p = cs^2 * rho. no energy equation — internal energy is a dummy.
#[derive(Clone, Copy, Debug)]
pub struct Isothermal<S: Scalar> {
    pub cs: S,
}

impl<S: Scalar> Eos<S> for Isothermal<S> {
    #[inline]
    fn sound_speed(&self, _rho: S, _pre: S) -> S {
        self.cs
    }

    #[inline]
    fn sound_speed_sq(&self, _rho: S, _pre: S) -> S {
        self.cs * self.cs
    }

    #[inline]
    fn internal_energy(&self, _rho: S, _pre: S) -> S {
        // isothermal: no thermodynamic internal energy.
        // return cs^2 so that pressure(rho, e_int) = rho * cs^2 roundtrips.
        self.cs * self.cs
    }

    #[inline]
    fn pressure(&self, rho: S, _e_int: S) -> S {
        self.cs * self.cs * rho
    }

    /// nrg slot stores cs^2. for globally isothermal this is self.cs^2.
    /// for locally isothermal, the user sets nrg = cs^2(x) per cell.
    #[inline]
    fn conserved_energy(&self, _rho: S, _v_sq: S, _pre: S) -> S {
        self.cs * self.cs
    }

    /// pressure from nrg = cs^2: p = nrg * rho.
    /// works for both globally and locally isothermal since it reads cs^2
    /// from the stored nrg.
    #[inline]
    fn recover_pressure(&self, rho: S, _v_sq: S, nrg: S) -> S {
        nrg * rho
    }

    /// isothermal kernels use gamma as cs^2 for c2p (pre = cs^2 * rho),
    /// flux (HLLE wave speeds), and max_wave_speed (cs = sqrt(gamma)).
    fn gamma_for_ops(&self) -> f64 {
        (self.cs * self.cs).to_f64()
    }
    /// the isothermal KernelSet constructors take the sound speed `cs` directly (cs^2 is what `gamma_for_ops` supplies).
    fn substrate_param(&self) -> f64 {
        self.cs.to_f64()
    }
}

/// the taub-mathews approximation to the synge (relativistic perfect) gas
/// (Mignone, Plewa & Bodo 2005 eq 5; Mignone & McKinney 2007 sec 2): the specific
/// enthalpy `h(theta) = 2.5 theta + sqrt(2.25 theta^2 + 1)` with `theta = p/rho`
/// in c = 1 units, parameter-free. the effective adiabatic index walks from 5/3
/// (theta -> 0, non-relativistic temperatures) to 4/3 (theta -> infinity,
/// ultra-relativistic), which is what lets a single run carry a relativistic
/// blast wave through its deceleration into the non-relativistic phase — a
/// constant-gamma law is wrong on one side of theta ~ 1 or the other.
///
/// the form saturates the taub inequality as an identity, `(h - theta)(h - 4 theta) = 1`
/// exactly, and inverts in closed form: `e = h - 1 - theta` gives
/// `p = rho e (e + 2) / (3 (e + 1))`, so the cons->prim newton needs no inner
/// root-find. `sound_speed_sq` returns the NEWTONIAN-FORM value
/// `theta (5h - 8 theta) / (3 (h - theta))` — the quantity whose division by `h`
/// in `rhd::sound_speed_sq` yields the exact relativistic cs^2 of this gas
/// (1/3 in the hot limit, (5/3) theta cold). relativistic regimes only: the
/// newtonian conserved-energy identities assume e is the thermal energy of a
/// gamma-law gas, and the substrate refuses the pairing.
#[derive(Clone, Copy, Debug)]
pub struct TaubMathews;

impl<S: Scalar> Eos<S> for TaubMathews {
    #[inline]
    fn sound_speed(&self, rho: S, pre: S) -> S {
        self.sound_speed_sq(rho, pre).sqrt()
    }

    #[inline]
    fn sound_speed_sq(&self, rho: S, pre: S) -> S {
        let theta = pre / rho;
        let h = S::from_f64(2.5) * theta
            + (S::from_f64(2.25) * theta * theta + S::ONE).sqrt();
        theta * (S::from_f64(5.0) * h - S::from_f64(8.0) * theta)
            / (S::from_f64(3.0) * (h - theta))
    }

    #[inline]
    fn internal_energy(&self, rho: S, pre: S) -> S {
        // e = h - 1 - theta = 1.5 theta + (sqrt(2.25 theta^2 + 1) - 1), with the
        // sqrt-minus-one in conjugate form: the direct subtraction cancels ~11
        // digits in the cold limit (the correction is theta^2-small against 1)
        // and the lost digits surface as a relative error ~ ulp / theta in e.
        let theta = pre / rho;
        let x_sq = S::from_f64(2.25) * theta * theta;
        S::from_f64(1.5) * theta + x_sq / ((x_sq + S::ONE).sqrt() + S::ONE)
    }

    #[inline]
    fn pressure(&self, rho: S, e_int: S) -> S {
        // the closed-form inverse of `internal_energy`: p = rho e (e + 2) / (3 (e + 1)).
        rho * e_int * (e_int + S::from_f64(2.0))
            / (S::from_f64(3.0) * (e_int + S::ONE))
    }

    /// no free adiabatic index exists for this gas; the value marks the kernel
    /// scalar slot as inert (the `_tm` kernel twins never read it).
    fn gamma_for_ops(&self) -> f64 {
        0.0
    }

    /// the COLD-limit index. the effective adiabatic index of this gas is set by the
    /// temperature — it walks from 5/3 at theta = p/rho -> 0 to 4/3 at theta -> infinity —
    /// so no constant describes it. the cold end is what the single-number reporting slot
    /// (the checkpoint's `gamma` attribute, which post-processing divides by `gamma - 1`)
    /// carries; the trait default of 1 is the isothermal marker and is degenerate there.
    fn gamma(&self) -> S {
        S::from_f64(5.0 / 3.0)
    }
}

/// a closure selected from a value rather than a type — the kernel builders pick
/// gamma-law or taub-mathews from a bake-time tag, and a single generic call site
/// (the traced physics) receives ONE concrete `Eos` impl. the match resolves when
/// each method runs, which for a traced carrier is trace time: the emitted graph
/// carries only the selected closure's operations, never a runtime branch.
///
/// on a concrete scalar (`f64`) the same type is the HOST-side closure a `SimState`
/// carries, and it must name the same gas as the compiled kernels. the two are read at
/// different moments and cannot be allowed to disagree: seeding converts the initial
/// primitives to conserved variables through the state's EOS, while every subsequent
/// cons->prim recovery runs the kernel arm. seeding a relativistic gas through a gamma
/// law and recovering it through taub-mathews conserves D = rho W (which is EOS-free)
/// and splits it wrongly between rho and W — a gamma_gas = 20 blast seeded that way
/// comes back at W = 28.6 with a third of its pressure.
#[derive(Clone, Copy, Debug)]
pub enum EosSelect<S: Scalar> {
    Ideal(IdealGas<S>),
    Tm(TaubMathews),
}

impl<S: Scalar> Eos<S> for EosSelect<S> {
    // every method is delegated explicitly, including the ones the trait defaults: a default
    // left in place here would evaluate the gamma-law form for BOTH arms.
    #[inline]
    fn sound_speed(&self, rho: S, pre: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.sound_speed(rho, pre),
            EosSelect::Tm(e) => e.sound_speed(rho, pre),
        }
    }
    #[inline]
    fn sound_speed_sq(&self, rho: S, pre: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.sound_speed_sq(rho, pre),
            EosSelect::Tm(e) => e.sound_speed_sq(rho, pre),
        }
    }
    #[inline]
    fn internal_energy(&self, rho: S, pre: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.internal_energy(rho, pre),
            EosSelect::Tm(e) => e.internal_energy(rho, pre),
        }
    }
    #[inline]
    fn pressure(&self, rho: S, e_int: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.pressure(rho, e_int),
            EosSelect::Tm(e) => e.pressure(rho, e_int),
        }
    }
    #[inline]
    fn conserved_energy(&self, rho: S, v_sq: S, pre: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.conserved_energy(rho, v_sq, pre),
            EosSelect::Tm(e) => e.conserved_energy(rho, v_sq, pre),
        }
    }
    #[inline]
    fn recover_pressure(&self, rho: S, v_sq: S, nrg: S) -> S {
        match self {
            EosSelect::Ideal(e) => e.recover_pressure(rho, v_sq, nrg),
            EosSelect::Tm(e) => e.recover_pressure(rho, v_sq, nrg),
        }
    }
    fn gamma(&self) -> S {
        match self {
            EosSelect::Ideal(e) => Eos::<S>::gamma(e),
            EosSelect::Tm(e) => Eos::<S>::gamma(e),
        }
    }
    fn gamma_for_ops(&self) -> f64 {
        match self {
            EosSelect::Ideal(e) => Eos::<S>::gamma_for_ops(e),
            EosSelect::Tm(e) => Eos::<S>::gamma_for_ops(e),
        }
    }
    fn substrate_param(&self) -> f64 {
        match self {
            EosSelect::Ideal(e) => Eos::<S>::substrate_param(e),
            EosSelect::Tm(e) => Eos::<S>::substrate_param(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-14 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn ideal_gas_sound_speed() {
        let eos = IdealGas { gamma: 1.4 };
        // a = sqrt(gamma * p / rho) = sqrt(1.4 * 1.0 / 1.0) = sqrt(1.4)
        let a = eos.sound_speed(1.0, 1.0);
        assert!(approx(a, 1.4_f64.sqrt()));

        // rho=0.125, p=0.1: a = sqrt(1.4 * 0.1 / 0.125) = sqrt(1.12)
        let a = eos.sound_speed(0.125, 0.1);
        assert!(approx(a, (1.4 * 0.1 / 0.125_f64).sqrt()));
    }

    #[test]
    fn ideal_gas_energy_roundtrip() {
        let eos = IdealGas { gamma: 1.4 };
        let rho = 2.5;
        let pre = 3.7;
        let e_int = eos.internal_energy(rho, pre);
        let pre_back = eos.pressure(rho, e_int);
        assert!(approx(pre, pre_back));
    }

    #[test]
    fn ideal_gas_internal_energy() {
        let eos = IdealGas { gamma: 1.4 };
        // e_int = p / ((gamma-1) * rho) = 1.0 / (0.4 * 1.0) = 2.5
        let e = eos.internal_energy(1.0, 1.0);
        assert!(approx(e, 2.5));
    }

    #[test]
    fn ideal_gas_f32() {
        let eos = IdealGas { gamma: 1.4_f32 };
        let a = eos.sound_speed(1.0_f32, 1.0_f32);
        assert!((a - 1.4_f32.sqrt()).abs() < 1e-6);
    }

    // ---- isothermal ----

    #[test]
    fn isothermal_sound_speed() {
        let eos = Isothermal { cs: 1.0 };
        // sound speed is constant, independent of state
        assert_eq!(eos.sound_speed(1.0, 1.0), 1.0);
        assert_eq!(eos.sound_speed(0.001, 999.0), 1.0);
        assert_eq!(eos.sound_speed(100.0, 0.0), 1.0);
    }

    #[test]
    fn isothermal_pressure() {
        let eos = Isothermal { cs: 2.0 };
        // p = cs^2 * rho = 4 * rho
        assert!(approx(eos.pressure(1.0, 0.0), 4.0));
        assert!(approx(eos.pressure(0.5, 0.0), 2.0));
        assert!(approx(eos.pressure(3.0, 0.0), 12.0));
    }

    #[test]
    fn isothermal_energy_roundtrip() {
        let eos = Isothermal { cs: 3.0 };
        let rho = 2.5;
        let pre = eos.pressure(rho, 0.0); // p = 9 * 2.5 = 22.5
        let e_int = eos.internal_energy(rho, pre);
        let pre_back = eos.pressure(rho, e_int);
        assert!(approx(pre, pre_back));
    }

    #[test]
    fn isothermal_f32() {
        let eos = Isothermal { cs: 1.5_f32 };
        assert_eq!(eos.sound_speed(1.0_f32, 1.0_f32), 1.5_f32);
        assert!((eos.pressure(2.0_f32, 0.0_f32) - 4.5_f32).abs() < 1e-6);
    }

    // ---- taub-mathews (synge gas approximation) ----

    fn tm_enthalpy(theta: f64) -> f64 {
        2.5 * theta + (2.25 * theta * theta + 1.0).sqrt()
    }

    /// the taub inequality saturates as an IDENTITY for this enthalpy:
    /// (h - theta)(h - 4 theta) = 1 exactly, by construction of the sqrt form.
    /// this is the sharpest possible pin on the closure — any algebraic drift in
    /// `internal_energy` breaks it at roundoff, across twelve decades of theta.
    #[test]
    fn taub_mathews_saturates_the_taub_inequality() {
        let eos = TaubMathews;
        for k in -6..=6 {
            let theta = 10.0_f64.powi(k);
            let e: f64 = eos.internal_energy(1.0, theta);
            let h = 1.0 + e + theta;
            let resid = (h - theta) * (h - 4.0 * theta) - 1.0;
            assert!(
                resid.abs() < 1e-12 * h * h,
                "taub identity broken at theta = 1e{k}: resid {resid:e}"
            );
        }
    }

    /// the effective adiabatic index gamma_eff = 1 + p/(rho e) walks from 5/3 in
    /// the cold limit to 4/3 in the ultra-relativistic limit — the property that
    /// carries a decelerating blast wave through theta ~ 1 with one closure.
    #[test]
    fn taub_mathews_effective_gamma_limits() {
        let eos = TaubMathews;
        let gamma_eff = |theta: f64| {
            let e: f64 = eos.internal_energy(1.0, theta);
            1.0 + theta / e
        };
        assert!((gamma_eff(1e-8) - 5.0 / 3.0).abs() < 1e-6);
        assert!((gamma_eff(1e8) - 4.0 / 3.0).abs() < 1e-6);
        // monotone descent through the trans-relativistic regime
        let mut prev = gamma_eff(1e-4);
        for k in [-2, -1, 0, 1, 2] {
            let g = gamma_eff(10.0_f64.powi(k));
            assert!(g < prev, "gamma_eff not monotone at theta = 1e{k}");
            prev = g;
        }
    }

    /// pressure(rho, internal_energy(rho, p)) = p, closed form both ways.
    #[test]
    fn taub_mathews_pressure_roundtrip() {
        let eos = TaubMathews;
        for k in -6..=6 {
            let pre = 10.0_f64.powi(k);
            for rho in [0.1, 1.0, 42.0] {
                let e = eos.internal_energy(rho, pre);
                let back = eos.pressure(rho, e);
                assert!(
                    (back - pre).abs() < 1e-12 * pre,
                    "roundtrip broken at rho {rho}, p = 1e{k}: {back}"
                );
            }
        }
    }

    /// the newtonian-form cs^2 composes with the shared relativistic division by h:
    /// cs_rel^2 = sound_speed_sq / h must reach 1/3 hot and (5/3) theta cold, and
    /// stay strictly subluminal everywhere.
    #[test]
    fn taub_mathews_relativistic_sound_speed_limits() {
        let eos = TaubMathews;
        let cs_rel_sq = |theta: f64| {
            let cs2: f64 = eos.sound_speed_sq(1.0, theta);
            cs2 / tm_enthalpy(theta)
        };
        assert!((cs_rel_sq(1e-10) / (5.0 / 3.0 * 1e-10) - 1.0).abs() < 1e-6);
        assert!((cs_rel_sq(1e10) - 1.0 / 3.0).abs() < 1e-6);
        for k in -8..=8 {
            let c2 = cs_rel_sq(10.0_f64.powi(k));
            assert!(c2 > 0.0 && c2 < 1.0 / 3.0 + 1e-12, "cs^2 out of range at 1e{k}");
        }
    }

    /// the rhd cons->prim newton recovers a taub-mathews state without any
    /// gamma-law assumption: seed conserved from primitives, recover, compare —
    /// across the cold, trans-relativistic and hot regimes at half-lightspeed.
    #[test]
    fn taub_mathews_c2p_roundtrip_through_rhd_recover() {
        use crate::rhd::Rhd;
        use crate::regime::Regime;
        use crate::spatial_metric::SpatialMetric;
        let eos = TaubMathews;
        for k in [-4, -1, 0, 1, 4] {
            let prim = crate::state::Prim::<f64, 1> {
                rho: 1.0,
                vel: symbi_algebra::Tensor::new([0.5]),
                pre: 10.0_f64.powi(k),
            };
            let cons = Rhd.to_conserved(&eos, &prim);
            let back = crate::rhd::rhd_recover(&eos, &cons, &SpatialMetric::flat(), crate::c2p_result::C2P_MAX_ITER);
            assert!(
                (back.pre - prim.pre).abs() < 1e-9 * prim.pre
                    && (back.rho - prim.rho).abs() < 1e-9
                    && (back.vel[0] - 0.5).abs() < 1e-9,
                "c2p roundtrip failed at theta = 1e{k}: rho {} pre {} v {}",
                back.rho,
                back.pre,
                back.vel[0]
            );
        }
    }
}
