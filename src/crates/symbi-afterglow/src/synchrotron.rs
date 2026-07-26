// =============================================================================
// synchrotron.rs
//
// per-cell synchrotron radiation primitives for relativistic-blast-wave afterglows
// (Sari, Piran & Narayan 1998/1999). all CGS:
// B in gauss, frequencies in Hz, energy density in erg/cm^3, number density in cm^-3.
// every dimensionful argument and result is a typed `Quantity` (src/units.rs), so the
// algebra here is checked at compile time — multiplying the wrong quantities together
// is a compile-time type error, caught before it becomes a silently wrong number.
//
// the model: behind the shock the field is set by equipartition (`shock_bfield`), the
// non-thermal electrons follow a power law N(gamma) ~ gamma^-p above `minimum_lorentz`,
// cooling above `critical_lorentz`. the synchrotron spectrum is a broken power law
// (`powerlaw_flux`) with breaks at nu_m (from gamma_min) and nu_c (from gamma_crit).
//
// usage:
//  let b   = shock_bfield(rho_e, eps_b);
//  let nug = gyration_frequency(b);
//  let num = nu(minimum_lorentz(eps_e, rho_e, n, p), nug);
//  let fnu = powerlaw_flux(power_max, p, nu_obs, nu_c, num);
// =============================================================================

use crate::constants::{C_LIGHT, E_CHARGE, M_E, PI, SIGMA_THOMSON};
use crate::units::{
    EnergyDensity, Frequency, MagneticField, NumberDensity, Power, SpectralEmissivity,
    SpectralPower, Time,
};

/// fluid speed in units of c from the four-velocity magnitude `gamma*beta` (dimensionless).
#[inline]
pub fn beta(gamma_beta: f64) -> f64 {
    gamma_beta / (1.0 + gamma_beta * gamma_beta).sqrt()
}

/// lorentz factor from the four-velocity magnitude `gamma*beta` (dimensionless).
#[inline]
pub fn lorentz_factor(gamma_beta: f64) -> f64 {
    (1.0 + gamma_beta * gamma_beta).sqrt()
}

/// post-shock magnetic field [gauss] from equipartition: B = sqrt(8 pi eps_b u_e), with the
/// internal energy density `rho_e` [erg/cm^3] and the magnetic-energy fraction `eps_b`. the
/// sqrt of an energy density yielding a magnetic field is exactly the half-integer dimension
/// the units encoding exists for.
#[inline]
pub fn shock_bfield(rho_e: EnergyDensity, eps_b: f64) -> MagneticField {
    (8.0 * PI * eps_b * rho_e).sqrt()
}

/// the gyration (Larmor) frequency scale [Hz] for the field `bfield` [gauss]:
/// nu_g = (3 / 4 pi) (e / m_e c) B. the synchrotron frequency of an electron at lorentz
/// factor gamma_e is nu = nu_g gamma_e^2 (see `nu`).
#[inline]
pub fn gyration_frequency(bfield: MagneticField) -> Frequency {
    (3.0 / 4.0 / PI) * (E_CHARGE / (M_E * C_LIGHT)) * bfield
}

/// bolometric synchrotron power per electron [erg/s] at lorentz factor `w` in field energy
/// density `ub` [erg/cm^3]: P = (4/3) sigma_T c beta^2 gamma^2 u_B.
#[inline]
pub fn total_synch_power(w: f64, ub: EnergyDensity, beta: f64) -> Power {
    (4.0 / 3.0) * SIGMA_THOMSON * C_LIGHT * (beta * beta) * (w * w) * ub
}

/// synchrotron frequency [Hz] of an electron at lorentz factor `gamma_e`: nu = nu_g gamma_e^2.
#[inline]
pub fn nu(gamma_e: f64, nu_g: Frequency) -> Frequency {
    nu_g * (gamma_e * gamma_e)
}

/// the cooling (critical) lorentz factor (dimensionless): the gamma above which an electron
/// radiates away its energy within the emitter-frame time `t_emitter` [s].
/// gamma_c = 6 pi m_e c / (sigma_T B^2 t).
#[inline]
pub fn critical_lorentz(bfield: MagneticField, t_emitter: Time) -> f64 {
    ((6.0 * PI * M_E * C_LIGHT) / (SIGMA_THOMSON * bfield.squared() * t_emitter)).value()
}

/// the maximum spectral power per electron [erg/(s Hz)] (Sari et al. 1999, eq. 5):
/// P_{nu,max} = (m_e c^2 sigma_T / 3 e) B.
#[inline]
pub fn max_power_per_frequency(bfield: MagneticField) -> SpectralPower {
    (M_E * C_LIGHT.squared() * SIGMA_THOMSON) / (3.0 * E_CHARGE) * bfield
}

/// the peak emissivity per unit frequency [erg/(s Hz cm^3)] of a cell with electron density `n`
/// [cm^-3] in field `bfield` [gauss] for spectral index `p` — the cell's emission normalization.
#[inline]
pub fn emissivity(bfield: MagneticField, n: NumberDensity, p: f64) -> SpectralEmissivity {
    let coeff = (9.6323 / 8.0 / PI) * (p - 1.0) / (3.0 * p - 1.0) * 3.0_f64.sqrt();
    coeff * (E_CHARGE.cubed() / (M_E * C_LIGHT.squared())) * n * bfield
}

/// the minimum lorentz factor of the shocked-electron power law (dimensionless): the fraction
/// `eps_e` of the thermal energy `e_thermal` [erg/cm^3] shared among `n` electrons [cm^-3] sets
/// the low cutoff.
#[inline]
pub fn minimum_lorentz(eps_e: f64, e_thermal: EnergyDensity, n: NumberDensity, p: f64) -> f64 {
    (eps_e * (p - 2.0) / (p - 1.0) * e_thermal / (n * M_E * C_LIGHT.squared())).value()
}

/// relativistic doppler boost factor delta = 1 / (gamma (1 - beta . nhat)) for a fluid element of
/// lorentz factor `w` moving with velocity `beta_vec` (units of c) seen along `nhat` (unit vector).
/// all arguments are dimensionless, so this stays a bare f64.
#[inline]
pub fn delta_doppler(w: f64, beta_vec: [f64; 3], nhat: [f64; 3]) -> f64 {
    let dot = beta_vec[0] * nhat[0] + beta_vec[1] * nhat[1] + beta_vec[2] * nhat[2];
    1.0 / (w * (1.0 - dot))
}

/// the DIMENSIONLESS broken-power-law synchrotron shape (Sari, Piran & Narayan 1998), normalized
/// to 1 at the spectral peak, at emitter-frame frequency `nu_prime` with breaks `nu_c`, `nu_m`.
/// this is `powerlaw_flux` factored out of its units carrier, so a per-frequency EMISSIVITY (not
/// just a per-electron power) can be scaled by the same spectrum — the deterministic deposition
/// reducer needs `emissivity * spectral_shape`. slow cooling is nu_c > nu_m.
pub fn spectral_shape(p: f64, nu_prime: Frequency, nu_c: Frequency, nu_m: Frequency) -> f64 {
    let slow_cool = nu_c > nu_m;
    if slow_cool {
        if nu_prime < nu_m {
            (nu_prime / nu_m).value().powf(1.0 / 3.0)
        } else if nu_prime < nu_c {
            (nu_prime / nu_m).value().powf(-0.5 * (p - 1.0))
        } else {
            (nu_c / nu_m).value().powf(-0.5 * (p - 1.0)) * (nu_prime / nu_c).value().powf(-0.5 * p)
        }
    } else if nu_prime < nu_c {
        (nu_prime / nu_c).value().powf(1.0 / 3.0)
    } else if nu_prime < nu_m {
        (nu_prime / nu_c).value().powf(-0.5)
    } else {
        (nu_m / nu_c).value().powf(-0.5) * (nu_prime / nu_m).value().powf(-0.5 * p)
    }
}

/// the piecewise power-law decomposition of `spectral_shape` restricted to `[nu_lo, nu_hi]`:
/// three segments split at the (band-clamped) spectral breaks, each an exact `A nu^a` with the
/// SAME normalization as `spectral_shape` (amplitudes anchored at the true, unclamped breaks, so
/// `amps[k] * nu^exps[k] == spectral_shape(nu)` inside segment k). this is the single source of
/// the spectrum's segment structure — the frequency sampler and the band-energy integral both
/// consume it, so the monte-carlo packets and the deterministic deposit share one emissivity.
/// frequencies are raw f64 [Hz] (the amplitudes carry the compensating Hz^-a).
pub struct SpectralSegments {
    pub bounds: [f64; 4],
    pub exps: [f64; 3],
    pub amps: [f64; 3],
}

pub fn spectral_segments(p: f64, nu_lo: f64, nu_hi: f64, nu_c: f64, nu_m: f64) -> SpectralSegments {
    let slow_cool = nu_c > nu_m;
    let mid = if slow_cool { -0.5 * (p - 1.0) } else { -0.5 };
    let exps = [1.0 / 3.0, mid, -0.5 * p];
    // true break locations (b1 = spectral peak where the shape is 1). an INFINITE break (no
    // cooling within the emitter time) leaves its segment empty; anchor its amplitude at the
    // band top so the algebra stays finite (the amplitude of an empty segment never enters).
    let b1 = nu_m.min(nu_c);
    let b2 = nu_m.max(nu_c);
    let b1a = if b1.is_finite() { b1 } else { nu_hi };
    let b2a = if b2.is_finite() { b2 } else { nu_hi };
    // amplitudes by continuity, anchored so shape(b1) = 1 (spectral_shape's normalization).
    let a0 = b1a.powf(-exps[0]);
    let a1 = a0 * b1a.powf(exps[0] - exps[1]);
    let a2 = a1 * b2a.powf(exps[1] - exps[2]);
    SpectralSegments {
        bounds: [
            nu_lo,
            b1a.clamp(nu_lo, nu_hi),
            b2a.clamp(nu_lo, nu_hi),
            nu_hi,
        ],
        exps,
        amps: [a0, a1, a2],
    }
}

/// the integral of `nu^a` over [lo, hi] (the a = -1 case is the logarithm).
#[inline]
pub(crate) fn power_integral(a: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        0.0
    } else if (a + 1.0).abs() < 1.0e-9 {
        (hi / lo).ln()
    } else {
        (hi.powf(a + 1.0) - lo.powf(a + 1.0)) / (a + 1.0)
    }
}

/// the band-integrated spectral shape `int spectral_shape(nu) dnu` over `[nu_lo, nu_hi]` [Hz].
/// `emissivity * band_integrated_shape` is the total per-volume synchrotron power radiated in
/// the band — the normalization that ties a monte-carlo packet budget to the per-Hz emissivity
/// the deterministic deposit uses.
pub fn band_integrated_shape(p: f64, nu_lo: f64, nu_hi: f64, nu_c: f64, nu_m: f64) -> Frequency {
    // NaN-safe rejection: a degenerate or non-finite band integrates to zero.
    if !(nu_lo.is_finite() && nu_hi.is_finite()) || nu_hi <= nu_lo || nu_lo <= 0.0 {
        return Frequency::new(0.0);
    }
    let seg = spectral_segments(p, nu_lo, nu_hi, nu_c, nu_m);
    let total: f64 = (0..3)
        .map(|k| seg.amps[k] * power_integral(seg.exps[k], seg.bounds[k], seg.bounds[k + 1]))
        .sum();
    Frequency::new(total)
}

/// the broken-power-law synchrotron spectrum (Sari, Piran & Narayan 1998): scale the peak power
/// `power_max` by the spectral shape at the (emitter-frame) frequency `nu_prime`, given the
/// cooling break `nu_c` and the injection break `nu_m`. slow cooling is nu_c > nu_m. the frequency
/// ratios are dimensionless (Frequency/Frequency), so `.value()` exits to apply the power law.
///
/// slow cooling: F ~ nu^{1/3} (nu<nu_m), nu^{-(p-1)/2} (nu_m<nu<nu_c), nu^{-p/2} (nu>nu_c).
/// fast cooling: F ~ nu^{1/3} (nu<nu_c), nu^{-1/2} (nu_c<nu<nu_m), nu^{-p/2} (nu>nu_m).
pub fn powerlaw_flux(
    power_max: SpectralPower,
    p: f64,
    nu_prime: Frequency,
    nu_c: Frequency,
    nu_m: Frequency,
) -> SpectralPower {
    let slow_cool = nu_c > nu_m;
    if slow_cool {
        if nu_prime < nu_m {
            power_max * (nu_prime / nu_m).value().powf(1.0 / 3.0)
        } else if nu_prime < nu_c {
            power_max * (nu_prime / nu_m).value().powf(-0.5 * (p - 1.0))
        } else {
            power_max
                * (nu_c / nu_m).value().powf(-0.5 * (p - 1.0))
                * (nu_prime / nu_c).value().powf(-0.5 * p)
        }
    } else if nu_prime < nu_c {
        power_max * (nu_prime / nu_c).value().powf(1.0 / 3.0)
    } else if nu_prime < nu_m {
        power_max * (nu_prime / nu_c).value().powf(-0.5)
    } else {
        power_max * (nu_m / nu_c).value().powf(-0.5) * (nu_prime / nu_m).value().powf(-0.5 * p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{EnergyDensity, Frequency, MagneticField, NumberDensity, SpectralPower};

    // beta/lorentz are the standard relativistic relations and round-trip through gamma*beta.
    #[test]
    fn beta_lorentz_round_trip() {
        for &gb in &[0.0, 0.5, 1.0, 10.0, 100.0] {
            let w = lorentz_factor(gb);
            let b = beta(gb);
            // gamma*beta reconstructs the input four-velocity.
            assert!((w * b - gb).abs() < 1e-9, "gb={gb}");
            // gamma^2 (1 - beta^2) == 1.
            assert!((w * w * (1.0 - b * b) - 1.0).abs() < 1e-9, "gb={gb}");
        }
        assert_eq!(beta(0.0), 0.0);
        // ultra-relativistic: approaches but stays below c.
        assert!(beta(1e4) < 1.0 && beta(1e4) > 0.999_999);
    }

    // equipartition: B^2 == 8 pi eps_b u_e.
    #[test]
    fn shock_bfield_is_equipartition() {
        let (rho_e, eps_b) = (EnergyDensity::new(1.0e3), 0.1);
        let b = shock_bfield(rho_e, eps_b).value();
        assert!((b * b - 8.0 * PI * eps_b * rho_e.value()).abs() / (b * b) < 1e-12);
    }

    // doppler: head-on (beta || nhat) boosts (delta>1); a static element gives delta=1.
    #[test]
    fn doppler_boosts_head_on() {
        let gb = 10.0;
        let (w, b) = (lorentz_factor(gb), beta(gb));
        let head_on = delta_doppler(w, [b, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(
            head_on > w,
            "head-on delta should exceed gamma: {head_on} vs {w}"
        );
        assert!((delta_doppler(1.0, [0.0; 3], [1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    // the synchrotron frequency scales as gamma_e^2.
    #[test]
    fn nu_scales_as_gamma_squared() {
        let nu_g = Frequency::new(1.0e6);
        assert!((nu(10.0, nu_g).value() / nu(1.0, nu_g).value() - 100.0).abs() < 1e-9);
    }

    // the broken power law has the canonical Sari et al. slopes. recover each slope from the
    // ratio of fluxes a decade apart inside a regime: slope = log10(F2/F1).
    #[test]
    fn powerlaw_slopes_match_sari() {
        let p = 2.5;
        let pm = SpectralPower::new(1.0);
        let (nu_m, nu_c) = (Frequency::new(1.0e10), Frequency::new(1.0e14)); // slow cooling
        let f = |hz: f64| powerlaw_flux(pm, p, Frequency::new(hz), nu_c, nu_m).value();
        let slope = |a: f64, b: f64| (f(b) / f(a)).log10() / (b / a).log10();
        // below nu_m: +1/3
        assert!((slope(1.0e8, 1.0e9) - 1.0 / 3.0).abs() < 1e-6);
        // between nu_m and nu_c: -(p-1)/2
        assert!((slope(1.0e11, 1.0e12) - (-0.5 * (p - 1.0))).abs() < 1e-6);
        // above nu_c: -p/2
        assert!((slope(1.0e15, 1.0e16) - (-0.5 * p)).abs() < 1e-6);
    }

    // fast cooling (nu_c < nu_m): the middle segment is the universal -1/2.
    #[test]
    fn powerlaw_fast_cool_middle_is_minus_half() {
        let p = 2.5;
        let pm = SpectralPower::new(1.0);
        let (nu_m, nu_c) = (Frequency::new(1.0e14), Frequency::new(1.0e10));
        let f = |hz: f64| powerlaw_flux(pm, p, Frequency::new(hz), nu_c, nu_m).value();
        let slope = (f(1.0e12) / f(1.0e11)).log10();
        assert!((slope - (-0.5)).abs() < 1e-6);
    }

    // the segment decomposition reproduces spectral_shape POINTWISE in every segment and for
    // both cooling orders — the single-source guarantee that lets the sampler and the band
    // integral share the deposit's spectrum.
    #[test]
    fn segments_match_spectral_shape_pointwise() {
        for &(nu_m, nu_c) in &[(1.0e10_f64, 1.0e14_f64), (1.0e14, 1.0e10)] {
            let p = 2.5;
            let (nu_lo, nu_hi) = (1.0e6, 1.0e18);
            let seg = spectral_segments(p, nu_lo, nu_hi, nu_c, nu_m);
            for k in 0..3 {
                if seg.bounds[k + 1] <= seg.bounds[k] {
                    continue;
                }
                for frac in [0.1, 0.5, 0.9] {
                    let nu = seg.bounds[k] * (seg.bounds[k + 1] / seg.bounds[k]).powf(frac);
                    let from_seg = seg.amps[k] * nu.powf(seg.exps[k]);
                    let from_shape = spectral_shape(
                        p,
                        Frequency::new(nu),
                        Frequency::new(nu_c),
                        Frequency::new(nu_m),
                    );
                    assert!(
                        (from_seg / from_shape - 1.0).abs() < 1e-9,
                        "segment {k} at nu={nu}: {from_seg} vs {from_shape}"
                    );
                }
            }
        }
    }

    // the analytic band integral agrees with brute-force log-grid quadrature of spectral_shape,
    // including a band that clips the breaks.
    #[test]
    fn band_integral_matches_quadrature() {
        let p = 2.5;
        for &(nu_m, nu_c, nu_lo, nu_hi) in &[
            (1.0e10_f64, 1.0e14_f64, 1.0e6_f64, 1.0e18_f64),
            (1.0e14, 1.0e10, 1.0e6, 1.0e18),
            (1.0e10, 1.0e14, 1.0e12, 1.0e13), // interior band, no breaks inside
        ] {
            let analytic = band_integrated_shape(p, nu_lo, nu_hi, nu_c, nu_m).value();
            let n = 200_000;
            let lg_lo = nu_lo.ln();
            let dlg = (nu_hi / nu_lo).ln() / n as f64;
            let mut quad = 0.0;
            for i in 0..n {
                let nu = (lg_lo + (i as f64 + 0.5) * dlg).exp();
                quad += spectral_shape(
                    p,
                    Frequency::new(nu),
                    Frequency::new(nu_c),
                    Frequency::new(nu_m),
                ) * nu
                    * dlg; // dnu = nu dlg
            }
            assert!(
                (analytic / quad - 1.0).abs() < 5.0e-3,
                "band [{nu_lo},{nu_hi}]: analytic {analytic} vs quadrature {quad}"
            );
        }
    }

    // a smoke test that the typed primitives compose end-to-end with realistic CGS magnitudes.
    #[test]
    fn typed_primitives_compose() {
        let rho_e = EnergyDensity::new(1.0e-3);
        let n = NumberDensity::new(1.0);
        let b: MagneticField = shock_bfield(rho_e, 0.01);
        let nu_g: Frequency = gyration_frequency(b);
        let gamma_m = minimum_lorentz(0.1, rho_e, n, 2.5);
        let nu_m: Frequency = nu(gamma_m, nu_g);
        assert!(b.value() > 0.0 && nu_g.value() > 0.0 && gamma_m > 0.0 && nu_m.value() > 0.0);
    }
}
