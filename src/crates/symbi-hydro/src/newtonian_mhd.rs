// =============================================================================
// newtonian_mhd.rs
//
// newtonian (non-relativistic) ideal-MHD regime. implements the Regime trait
// over the shared MhdPrim/MhdCons states with closed-form wave speeds and an
// ALGEBRAIC conserved-to-primitive (no iteration -> cannot fail the way RMHD's
// iterative inversion does in current sheets).
//
// physics (carrier-generic over S: Scalar, valid at S = f64 and S = Gv):
//   to_conserved:  nrg = p/(gamma-1) + 1/2 rho |v|^2 + 1/2 |B|^2
//   to_primitive:  p   = (gamma-1) (nrg - 1/2 rho |v|^2 - 1/2 |B|^2)   (algebraic)
//   to_flux (nhat, vn = v.nhat, bn = B.nhat, p_tot = p + 1/2 |B|^2):
//     F(den) = rho vn
//     F(mom) = rho v vn + p_tot nhat - bn B
//     F(nrg) = (nrg + p_tot) vn - bn (v.B)
//     F(mag) = vn B - bn v                            (induction; same as RMHD)
//   wave_speeds (fast magnetosonic):
//     a^2 = gamma p / rho,  cA^2 = |B|^2/rho,  cAn^2 = bn^2/rho
//     cf^2 = 1/2 [ (a^2 + cA^2) + sqrt( (a^2 + cA^2)^2 - 4 a^2 cAn^2 ) ]
//     return (vn - cf, vn + cf)
//
// only safe_sqrt guards (no carrier branching) -> traces cleanly at S = Gv.
//
// usage:
//   let regime = NewtonianMhd;
//   let nhat = Tensor::unit(0);
//   let flux = regime.to_flux(&prim, &nhat, &eos);
//   let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::Cons;
use crate::mhd_state::{MhdPrim, MhdCons};
use crate::regime::Regime;
use crate::c2p_result::{C2pResult, ErrorCode};

/// newtonian (non-relativistic) ideal magnetohydrodynamics.
#[derive(Clone, Copy, Debug)]
pub struct NewtonianMhd;

/// algebraic conserved-to-primitive recovery — the carrier-safe PURE math (no
/// comparisons, no error codes -> traces at S = Gv). strip the magnetic energy
/// 1/2|B|^2 from the total, then invert the hydro energy. mirrors `rmhd_recover`
/// (the iterative analogue) as the single source the Gv c2p builder traces; the
/// host-side `to_primitive` wraps this with the diagnostic ErrorCode.
#[inline]
pub fn nmhd_recover<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    cons: &MhdCons<S, D>,
) -> MhdPrim<S, D> {
    let half = S::from_f64(0.5);
    let bsq = cons.mag.dot(&cons.mag);
    let hydro_cons = Cons { den: cons.den, mom: cons.mom, nrg: cons.nrg - half * bsq };
    MhdPrim { hydro: hydro_cons.to_primitive(eos), mag: cons.mag }
}

/// fast magnetosonic speed along nhat. closed form, single physical sqrt.
/// the discriminant is >= 0 for physical inputs; safe_sqrt guards anyway so the
/// radicand cannot trace a NaN into the kernel (carrier gate, CLAUDE.md 4.3).
#[inline]
fn fast_magnetosonic<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> S {
    let half = S::from_f64(0.5);
    let four = S::from_f64(4.0);
    let a_sq = eos.sound_speed_sq(prim.rho, prim.pre);
    let bsq = prim.mag.dot(&prim.mag);
    let bn = prim.mag.dot(nhat);
    let ca_sq = bsq / prim.rho;        // total alfven speed squared
    let can_sq = (bn * bn) / prim.rho; // normal alfven speed squared
    let sum = a_sq + ca_sq;
    let disc = (sum * sum - four * a_sq * can_sq).safe_sqrt();
    (half * (sum + disc)).safe_sqrt()
}

impl<S: Scalar, const D: usize> Regime<S, D> for NewtonianMhd {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::NEWTONIAN_MHD_SPEC;
    type Prim = MhdPrim<S, D>;
    type Cons = MhdCons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        // hydro conserved (nrg = p/(g-1) + 1/2 rho v^2) plus magnetic energy 1/2 |B|^2.
        let half = S::from_f64(0.5);
        let bsq = prim.mag.dot(&prim.mag);
        let hydro = prim.hydro.to_conserved(eos);
        MhdCons {
            hydro: Cons { den: hydro.den, mom: hydro.mom, nrg: hydro.nrg + half * bsq },
            mag: prim.mag,
        }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where S: OrderedNumeric
    {
        // algebraic, no iteration: strip magnetic energy, then invert hydro.
        // raw IEEE math; no silent floors (feedback_no_silent_floors). the
        // ErrorCode is an explicit diagnostic, the value is the raw unfloored
        // computation so downstream NaN propagation stays visible at the dt
        // reduction. the math is `nmhd_recover` (the carrier-safe single source);
        // only the comparisons below are host-side.
        let prim = nmhd_recover(eos, cons);
        let mut code = ErrorCode::NONE;
        if prim.rho <= S::ZERO { code = code.merge(ErrorCode::NEGATIVE_DENSITY); }
        if prim.pre <= S::ZERO { code = code.merge(ErrorCode::NEGATIVE_PRESSURE); }
        if !(prim.rho == prim.rho) || !(prim.pre == prim.pre) {
            code = code.merge(ErrorCode::NON_FINITE);
        }
        if code.is_ok() { C2pResult::ok(prim) } else { C2pResult::err(prim, code) }
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        let half = S::from_f64(0.5);
        let vn = prim.vel.dot(nhat);
        let bn = prim.mag.dot(nhat);
        let bsq = prim.mag.dot(&prim.mag);
        let vdotb = prim.vel.dot(&prim.mag);
        let p_tot = prim.pre + half * bsq; // gas + magnetic pressure
        let cons = self.to_conserved(eos, prim);
        MhdCons {
            hydro: Cons {
                den: cons.den * vn,
                mom: cons.mom.scale(vn) + nhat.scale(p_tot) - prim.mag.scale(bn),
                nrg: (cons.nrg + p_tot) * vn - bn * vdotb,
            },
            mag: prim.mag.scale(vn) - prim.vel.scale(bn), // induction
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        let vn = prim.vel.dot(nhat);
        let cf = fast_magnetosonic(eos, prim, nhat);
        (vn - cf, vn + cf)
    }

    #[inline]
    fn effective_inertia(&self, _eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        prim.rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;
    use crate::eos::IdealGas;
    use crate::state::Prim;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn prim3(rho: f64, vel: [f64; 3], pre: f64, mag: [f64; 3]) -> MhdPrim<f64, 3> {
        MhdPrim { hydro: Prim { rho, vel: Tensor::new(vel), pre }, mag: Tensor::new(mag) }
    }

    #[test]
    fn roundtrip_with_bfield() {
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = prim3(0.7, [1.0, -0.3, 0.5], 0.9, [0.4, 0.2, -0.6]);
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 { assert!(approx(prim.vel[dd], prim2.vel[dd])); }
        assert!(approx(prim.pre, prim2.pre));
        for dd in 0..3 { assert!(approx(prim.mag[dd], prim2.mag[dd])); }
    }

    #[test]
    fn conserved_energy_includes_magnetic() {
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.4 };
        let prim = prim3(1.0, [0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 2.0]);
        let cons = regime.to_conserved(&eos, &prim);
        // nrg = p/(g-1) + 1/2 rho v^2 + 1/2 |B|^2 = 1/0.4 + 0 + 1/2*4 = 2.5 + 2.0
        assert!(approx(cons.nrg, 1.0 / 0.4 + 2.0));
    }

    #[test]
    fn flux_reduces_to_hydro_when_b_zero() {
        // with B = 0 the MHD flux must equal the newtonian hydro flux.
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.4 };
        let prim = prim3(1.0, [2.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        let nhat = Tensor::unit(0);
        let flux = regime.to_flux(&prim, &nhat, &eos);
        // f_den = rho vn = 2.0
        assert!(approx(flux.den, 2.0));
        // f_mom_x = rho vx vn + p = 1*2*2 + 1 = 5.0
        assert!(approx(flux.mom[0], 5.0));
        assert!(approx(flux.mom[1], 0.0));
        // mag flux is zero when B = 0
        for dd in 0..3 { assert!(approx(flux.mag[dd], 0.0)); }
    }

    #[test]
    fn flux_magnetic_pressure_and_tension() {
        // B = (Bx, 0, 0) along nhat = x: bn = Bx, p_tot = p + 1/2 Bx^2,
        // f_mom_x = rho vx vn + p_tot - bn Bx = rho vx vn + p - 1/2 Bx^2.
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.4 };
        let prim = prim3(1.0, [3.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0]);
        let nhat = Tensor::unit(0);
        let flux = regime.to_flux(&prim, &nhat, &eos);
        // rho vx vn + p - 1/2 Bx^2 = 1*3*3 + 1 - 0.5*4 = 9 + 1 - 2 = 8.0
        assert!(approx(flux.mom[0], 8.0));
    }

    #[test]
    fn wave_speed_hydro_limit() {
        // B = 0 -> cf = sound speed a.
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.4 };
        let prim = prim3(1.0, [0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        let nhat = Tensor::unit(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        let a = (1.4f64).sqrt();
        assert!(approx(sl, -a));
        assert!(approx(sr, a));
    }

    #[test]
    fn wave_speed_parallel_field_is_max_of_sound_and_alfven() {
        // B parallel to nhat (cAn = cA): cf^2 = 1/2[(a^2+cA^2)+|a^2-cA^2|] = max(a^2,cA^2).
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.0 }; // a^2 = p/rho = 1.0
        // rho=1, p=1 -> a^2 = 1.0; Bx=3 -> cA^2 = 9.0 -> cf = 3.0 (alfven dominates).
        let prim = prim3(1.0, [0.0, 0.0, 0.0], 1.0, [3.0, 0.0, 0.0]);
        let nhat = Tensor::unit(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        assert!(approx(sr, 3.0));
        assert!(approx(sl, -3.0));
    }

    #[test]
    fn wave_speed_perpendicular_field_combines_in_quadrature() {
        // B perpendicular to nhat (cAn = 0): cf^2 = a^2 + cA^2.
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.0 }; // a^2 = p/rho
        // rho=1, p=1 -> a^2=1; B=(0,2,0), nhat=x -> cAn=0, cA^2 = 4 -> cf = sqrt(5).
        let prim = prim3(1.0, [0.0, 0.0, 0.0], 1.0, [0.0, 2.0, 0.0]);
        let nhat = Tensor::unit(0);
        let (_sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        assert!(approx(sr, 5.0f64.sqrt()));
    }

    #[test]
    fn negative_pressure_flagged_with_raw_value() {
        // strong field + low total energy -> stripping 1/2|B|^2 drives p < 0.
        let regime = NewtonianMhd;
        let eos = IdealGas { gamma: 1.4 };
        let cons = MhdCons {
            hydro: Cons { den: 1.0, mom: Tensor::new([0.0, 0.0, 0.0]), nrg: 1.0 },
            mag: Tensor::new([3.0, 0.0, 0.0]), // 1/2|B|^2 = 4.5 > nrg
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.error.contains(ErrorCode::NEGATIVE_PRESSURE));
        assert!(result.value.pre < 0.0); // raw, unfloored
    }
}
