// =============================================================================
// isothermal_mhd.rs
//
// isothermal ideal-MHD regime (Mignone 2007). the energy equation is dropped and
// the gas is closed by p = a^2 rho (a = constant isothermal sound speed). built on
// the energy-model-generic MhdConsG/MhdPrimG at E = IsoModel, so the energy/pressure
// slots are Zero<S> (zst) — no energy field, no wasted memory/flops/bandwidth.
//
// physics (carrier-generic over S: Scalar, valid at S = f64 and S = Gv):
//   to_conserved:  U = (rho, rho v, B)                       (no energy)
//   to_primitive:  rho = den, v = mom/den                    (trivial, no failure)
//   to_flux (nhat, vn = v.nhat, bn = B.nhat, p_tot = a^2 rho + 1/2 |B|^2):
//     F(den) = rho vn
//     F(mom) = rho v vn + p_tot nhat - bn B
//     F(mag) = vn B - bn v                                   (induction; shared w/ RMHD)
//   wave_speeds (fast magnetosonic, a^2 = cs^2 constant):
//     cf^2 = 1/2 [ (a^2 + cA^2) + sqrt( (a^2 + cA^2)^2 - 4 a^2 cAn^2 ) ]
//     return (vn - cf, vn + cf)
//
// the contact/entropy mode is absent (no energy) — see riemann::hlld_isothermal
// (the 3-state Mignone solver). mirrors `NewtonianMhd` and `IsoNewtonian`.
//
// usage:
//   let regime = IsothermalMhd;
//   let eos = Isothermal { cs: 1.0 };
//   let flux = regime.to_flux(&prim, &nhat, &eos);
// =============================================================================

use crate::c2p_result::C2pResult;
use crate::energy::Zero;
use crate::eos::Eos;
use crate::mhd_state::{IsoMhdCons, IsoMhdPrim};
use crate::regime::Regime;
use crate::state::{ConsG, PrimG};
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

/// isothermal ideal magnetohydrodynamics. no energy equation; p = a^2 rho.
#[derive(Clone, Copy, Debug)]
pub struct IsothermalMhd;

/// conserved-to-primitive recovery — the carrier-safe pure math (no comparisons,
/// no error codes -> traces at S = Gv). trivial in isothermal MHD: rho = den,
/// v = mom/den (no pressure slot). the single source the Gv c2p builder traces.
#[inline]
pub fn imhd_recover<S: Scalar, const D: usize>(
    _eos: &impl Eos<S>,
    cons: &IsoMhdCons<S, D>,
) -> IsoMhdPrim<S, D> {
    // mul-by-reciprocal to match the kernel form bit-for-bit (CPU == GPU). IEEE
    // div-by-zero -> inf/NaN, no silent floor.
    let inv_rho = S::ONE / cons.den;
    IsoMhdPrim {
        hydro: PrimG {
            rho: cons.den,
            vel: cons.mom.scale(inv_rho),
            pre: Zero::default(),
        },
        mag: cons.mag,
    }
}

/// fast magnetosonic speed along nhat; the isothermal sound speed `a^2 = cs^2`
/// is constant (Isothermal EOS ignores p). the magnetosonic algebra is the one
/// text in `mhd_state::fast_magnetosonic_from`.
#[inline]
fn fast_magnetosonic<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &IsoMhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> S {
    let a_sq = eos.sound_speed_sq(prim.rho, S::ZERO);
    crate::mhd_state::fast_magnetosonic_from(a_sq, prim.rho, &prim.mag, nhat)
}

impl<S: Scalar, const D: usize> Regime<S, D> for IsothermalMhd {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::ISO_MHD_SPEC;
    type Prim = IsoMhdPrim<S, D>;
    type Cons = IsoMhdCons<S, D>;
    type Energy = crate::energy::IsoModel;

    #[inline]
    fn to_conserved(&self, _eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        IsoMhdCons {
            hydro: ConsG {
                chi: Default::default(),
                den: prim.rho,
                mom: prim.vel.scale(prim.rho),
                nrg: Zero::default(),
            },
            mag: prim.mag,
        }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        // trivial, no iteration. iso has no failure mode a floor meaningfully
        // recovers (matches IsoNewtonian) -> always Ok; raw IEEE math, no floor.
        C2pResult::ok(imhd_recover(eos, cons))
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        let half = S::HALF;
        let vn = prim.vel.dot(nhat);
        let bn = prim.mag.dot(nhat);
        let bsq = prim.mag.dot(&prim.mag);
        let pre = eos.pressure(prim.rho, S::ZERO); // a^2 rho
        let p_tot = pre + half * bsq; // gas + magnetic pressure
        let rho_vn = prim.rho * vn;
        IsoMhdCons {
            hydro: ConsG {
                chi: Default::default(),
                den: rho_vn,
                mom: prim.vel.scale(rho_vn) + nhat.scale(p_tot) - prim.mag.scale(bn),
                nrg: Zero::default(),
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
    use crate::eos::Isothermal;
    use symbi_algebra::Tensor;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn prim3(rho: f64, vel: [f64; 3], mag: [f64; 3]) -> IsoMhdPrim<f64, 3> {
        IsoMhdPrim {
            hydro: PrimG {
                rho,
                vel: Tensor::new(vel),
                pre: Zero::default(),
            },
            mag: Tensor::new(mag),
        }
    }

    #[test]
    fn roundtrip_c2p() {
        let regime = IsothermalMhd;
        let eos = Isothermal { cs: 1.0 };
        let prim = prim3(0.7, [1.0, -0.3, 0.5], [0.4, 0.2, -0.6]);
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 {
            assert!(approx(prim.vel[dd], prim2.vel[dd]));
        }
        for dd in 0..3 {
            assert!(approx(prim.mag[dd], prim2.mag[dd]));
        }
    }

    #[test]
    fn flux_b_zero_reduces_to_iso_hydro() {
        // with B = 0, F(mom) = rho v vn + a^2 rho nhat (pure isothermal hydro).
        let regime = IsothermalMhd;
        let cs = 1.3_f64;
        let eos = Isothermal { cs };
        let prim = prim3(2.0, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let nhat = Tensor::<f64, 3>::unit(0);
        let f = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(f.den, 2.0 * 0.5));
        assert!(approx(f.mom[0], 2.0 * 0.5 * 0.5 + cs * cs * 2.0));
        for dd in 0..3 {
            assert!(approx(f.mag[dd], 0.0));
        }
    }

    #[test]
    fn fast_speed_exceeds_sound_and_alfven() {
        let eos = Isothermal { cs: 1.0 };
        let prim = prim3(1.0, [0.0; 3], [0.6, 0.8, 0.0]);
        let nhat = Tensor::<f64, 3>::unit(0);
        let cf = fast_magnetosonic(&eos, &prim, &nhat);
        let ca = (0.6_f64 * 0.6 / 1.0).sqrt(); // |bx|/sqrt(rho)
        assert!(cf >= 1.0 - 1e-12 && cf >= ca - 1e-12);
    }
}
