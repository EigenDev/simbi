// =============================================================================
// imhd_shock_tube.rs
//
// Mignone (2007) Table-1 isothermal-MHD shock tube (test 1) — validates that
// `hlld_isothermal` (the 3-state solver) produces a correct, non-oscillatory
// wave structure. pure host physics: 1D PLM(minmod) + forward-Euler godunov over
// the carrier-generic solvers at S = f64. no CT (Bx is the normal field, F(Bx)=0
// keeps it constant in 1D). isothermal closure p = a^2 rho, a = 1.
//
// discriminator (same as the Brio-Wu test): HLLE is the most diffusive, monotone
// baseline, so TV(hlld) <~ TV(hlle) and small ||rho_hlld - rho_hlle||_1 ==> clean.
// the paper's B already absorbed sqrt(4pi), so the table's (Hx,Hy,Hz) are B.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::energy::Zero;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::{IsothermalMhd, imhd_recover};
use symbi_hydro::mhd_state::{IsoMhdCons, IsoMhdPrim};
use symbi_hydro::regime::Regime;
use symbi_hydro::riemann::{hlld_isothermal, hlle};
use symbi_hydro::state::PrimG;

const N: usize = 400;
const NG: usize = 2;
const TOT: usize = N + 2 * NG;
const CS: f64 = 1.0; // isothermal sound speed
const X0: f64 = 0.5;
const T_FINAL: f64 = 0.1; // Table 1, test 1
const CFL: f64 = 0.4;

type P = IsoMhdPrim<f64, 3>;
type C = IsoMhdCons<f64, 3>;

fn prim(rho: f64, v: [f64; 3], b: [f64; 3]) -> P {
    IsoMhdPrim {
        hydro: PrimG {
            rho,
            vel: Tensor::new(v),
            pre: Zero::default(),
        },
        mag: Tensor::new(b),
    }
}

// Mignone Table 1, test 1: L = (rho=1, By=5), R = (rho=0.1, By=2), Bx = 3.
fn ic(i: usize) -> P {
    let x = (i as f64 + 0.5 - NG as f64) / N as f64;
    if x < X0 {
        prim(1.0, [0.0; 3], [3.0, 5.0, 0.0])
    } else {
        prim(0.1, [0.0; 3], [3.0, 2.0, 0.0])
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

// PLM(minmod) reconstruction of one cell: (left-face, right-face) primitives.
fn recon(pm: &P, p0: &P, pp: &P) -> (P, P) {
    let comp = |lo: f64, mid: f64, hi: f64| -> (f64, f64) {
        let s = minmod(mid - lo, hi - mid);
        (mid - 0.5 * s, mid + 0.5 * s)
    };
    let (rl, rr) = comp(pm.rho, p0.rho, pp.rho);
    let mut vl = [0.0; 3];
    let mut vr = [0.0; 3];
    let mut bl = [0.0; 3];
    let mut br = [0.0; 3];
    for k in 0..3 {
        let (a, b) = comp(pm.vel[k], p0.vel[k], pp.vel[k]);
        vl[k] = a;
        vr[k] = b;
        let (a, b) = comp(pm.mag[k], p0.mag[k], pp.mag[k]);
        bl[k] = a;
        br[k] = b;
    }
    (prim(rl, vl, bl), prim(rr, vr, br))
}

fn run<F: Fn(&P, &P) -> C>(flux: F) -> Vec<P> {
    let eos = Isothermal { cs: CS };
    let regime = IsothermalMhd;
    let nhat = Tensor::<f64, 3>::unit(0);
    let dx = 1.0 / N as f64;

    let mut cons: Vec<C> = (0..TOT)
        .map(|i| regime.to_conserved(&eos, &ic(i)))
        .collect();
    let mut t = 0.0;
    while t < T_FINAL {
        for g in 0..NG {
            cons[g] = cons[NG];
            cons[TOT - 1 - g] = cons[TOT - 1 - NG];
        }
        let p: Vec<P> = cons.iter().map(|c| imhd_recover(&eos, c)).collect();
        let mut smax = 0.0_f64;
        for pi in &p[NG..NG + N] {
            let (sl, sr) = regime.wave_speeds(&eos, pi, &nhat);
            smax = smax.max(sl.abs().max(sr.abs()));
        }
        let dt = (CFL * dx / smax).min(T_FINAL - t);

        let mut f: Vec<C> = vec![C::zero(); TOT + 1];
        for j in NG..=(NG + N) {
            let (cl, cr) = (j - 1, j);
            let (_, pr) = recon(&p[cl - 1], &p[cl], &p[cl + 1]);
            let (pl, _) = recon(&p[cr - 1], &p[cr], &p[cr + 1]);
            f[j] = flux(&pr, &pl);
        }
        for i in NG..NG + N {
            cons[i] = cons[i] + (f[i] - f[i + 1]) * (dt / dx);
        }
        t += dt;
    }
    cons.iter().map(|c| imhd_recover(&eos, c)).collect()
}

fn total_variation(f: &[f64]) -> f64 {
    f.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}
fn l1_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

#[test]
fn imhd_hlld_is_clean_shock_capturing() {
    let eos = Isothermal { cs: CS };
    let n = Tensor::<f64, 3>::unit(0);

    let p_hlle = run(|l, r| hlle(&IsothermalMhd, &eos, l, r, &n, 0.0));
    let p_hlld = run(|l, r| hlld_isothermal(&eos, l, r, &n, 0.0));

    let interior =
        |p: &[P], f: fn(&P) -> f64| -> Vec<f64> { p[NG..NG + N].iter().map(f).collect() };
    let rho_e = interior(&p_hlle, |p| p.rho);
    let rho_d = interior(&p_hlld, |p| p.rho);
    let by_e = interior(&p_hlle, |p| p.mag[1]);
    let by_d = interior(&p_hlld, |p| p.mag[1]);

    // - physical everywhere (isothermal density = HLL average -> stays positive).
    for (i, pi) in p_hlld[NG..NG + N].iter().enumerate() {
        assert!(
            pi.rho.is_finite() && pi.rho > 0.0,
            "hlld cell {i}: rho={}",
            pi.rho
        );
    }

    // - Bx (normal field) stays exactly constant (F(Bx)=0 in 1D).
    for pi in &p_hlld[NG..NG + N] {
        assert!((pi.mag[0] - 3.0).abs() < 1e-12, "Bx drifted: {}", pi.mag[0]);
    }

    // - non-oscillatory: TV(hlld) <~ TV(hlle) (the monotone baseline).
    let (tv_re, tv_rd) = (total_variation(&rho_e), total_variation(&rho_d));
    let (tv_be, tv_bd) = (total_variation(&by_e), total_variation(&by_d));
    eprintln!("[imhd] TV(rho): hlle={tv_re:.4} hlld={tv_rd:.4}");
    eprintln!("[imhd] TV(By):  hlle={tv_be:.4} hlld={tv_bd:.4}");
    assert!(
        tv_rd < 1.3 * tv_re,
        "HLLD rho OSCILLATES: TV {tv_rd:.4} vs hlle {tv_re:.4}"
    );
    assert!(
        tv_bd < 1.3 * tv_be,
        "HLLD By OSCILLATES: TV {tv_bd:.4} vs hlle {tv_be:.4}"
    );

    // - same solution as HLLE, only sharper: small L1 distance.
    let l1 = l1_diff(&rho_d, &rho_e);
    eprintln!("[imhd] L1(rho_hlld - rho_hlle) = {l1:.4}");
    assert!(l1 < 0.05, "HLLD rho diverges from HLLE: L1 {l1}");

    // - HLLD actually sharpens the discontinuity, confirming the HLLD path is active.
    let max_grad = |f: &[f64]| {
        f.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max)
    };
    assert!(
        max_grad(&rho_d) >= max_grad(&rho_e) * 0.95,
        "HLLD not sharper than HLLE (grad {} vs {})",
        max_grad(&rho_d),
        max_grad(&rho_e),
    );

    // - the wave structure formed: density develops an intermediate state between L and R.
    let rho_mid = rho_d[N / 2];
    assert!(
        rho_mid > 0.1 && rho_mid < 1.0,
        "midpoint density {rho_mid} is a pure step — no waves"
    );
}
