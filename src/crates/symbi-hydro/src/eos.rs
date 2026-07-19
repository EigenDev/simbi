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
    fn gamma_for_ops(&self) -> f64 { (self.cs * self.cs).to_f64() }
    /// the isothermal KernelSet constructors take the sound speed `cs` directly (cs^2 is what `gamma_for_ops` supplies).
    fn substrate_param(&self) -> f64 { self.cs.to_f64() }
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
}
