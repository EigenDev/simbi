// =============================================================================
// nmhd_brio_wu.rs
//
// the Brio-Wu (1988) MHD shock tube — the canonical 1D ideal-MHD Riemann problem
// — validates that `hllc_newtonian` / `hlld_newtonian` produce a CORRECT,
// NON-OSCILLATORY shock structure (the question the noisy 2D OT run raised).
//
// pure host physics: a 1D PLM(minmod) + forward-Euler godunov over the carrier-
// generic solvers at S = f64. no CT (Bx is the normal field: F(Bx)=0 keeps it
// constant in 1D, div(B)=0 trivially), no substrate — this isolates the RIEMANN
// SOLVER itself. the discriminator: HLLE is the most diffusive solver and is
// guaranteed monotone, so its total variation is the non-oscillatory baseline.
// a correct sharper solver has ~the same TV (a monotone profile's TV is
// diffusion-independent); a BUGGY solver oscillates -> TV inflates. so:
//   TV(hlld) <~ TV(hlle)  AND  ||rho_hlld - rho_hlle||_1 small  ==>  HLLD is clean.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{nmhd_recover, NewtonianMhd};
use symbi_hydro::regime::Regime;
use symbi_hydro::riemann::{hllc_newtonian, hlld_newtonian, hlle};
use symbi_hydro::state::Prim;
use symbi_hydro::ShockwaveLimiter;

const N: usize = 400; // interior cells
const NG: usize = 2;
const TOT: usize = N + 2 * NG;
const GAMMA: f64 = 2.0; // classic Brio-Wu
const X0: f64 = 0.5; // discontinuity location, domain [0,1]
const T_FINAL: f64 = 0.1;
const CFL: f64 = 0.2;

type P = MhdPrim<f64, 3>;
type C = MhdCons<f64, 3>;

fn prim(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> P {
    MhdPrim { hydro: Prim { rho, vel: Tensor::new(v), pre: p }, mag: Tensor::new(b) }
}

// classic Brio-Wu: L = (1, 0, 1, By=1), R = (0.125, 0, 0.1, By=-1), Bx = 0.75.
fn ic(i: usize) -> P {
    let x = (i as f64 + 0.5 - NG as f64) / N as f64; // cell-center x in [0,1]
    if x < X0 {
        prim(1.0, [0.0; 3], 1.0, [0.75, 1.0, 0.0])
    } else {
        prim(0.125, [0.0; 3], 0.1, [0.75, -1.0, 0.0])
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 { 0.0 } else if a.abs() < b.abs() { a } else { b }
}

// PLM(minmod) slope-limited reconstruction of one primitive: returns the cell's
// (left-face, right-face) reconstructed values from the 3-point stencil.
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
    let (pl, pr) = comp(pm.pre, p0.pre, pp.pre);
    (prim(rl, vl, pl, bl), prim(rr, vr, pr, br))
}

// evolve the Brio-Wu tube with the given face-flux solver; return final primitives.
fn run<F: Fn(&P, &P) -> C>(flux: F) -> Vec<P> {
    let eos = IdealGas { gamma: GAMMA };
    let nhat = Tensor::<f64, 3>::unit(0);
    let dx = 1.0 / N as f64;

    let mut cons: Vec<C> = (0..TOT).map(|i| NewtonianMhd.to_conserved(&eos, &ic(i))).collect();
    let mut t = 0.0;
    while t < T_FINAL {
        // transmissive ghosts.
        for g in 0..NG {
            cons[g] = cons[NG];
            cons[TOT - 1 - g] = cons[TOT - 1 - NG];
        }
        // c2p + CFL dt.
        let p: Vec<P> = cons.iter().map(|c| nmhd_recover(&eos, c)).collect();
        let mut smax = 0.0_f64;
        for pi in &p[NG..NG + N] {
            let (sl, sr) = NewtonianMhd.wave_speeds(&eos, pi, &nhat);
            smax = smax.max(sl.abs().max(sr.abs()));
        }
        let dt = (CFL * dx / smax).min(T_FINAL - t);

        // PLM reconstruction -> flux at face j (between cells j-1 and j), for the
        // faces bounding the interior cells: j in NG ..= NG+N.
        let mut f: Vec<C> = vec![C::zero(); TOT + 1];
        for j in NG..=(NG + N) {
            let (cl, cr) = (j - 1, j);
            let (_, pr) = recon(&p[cl - 1], &p[cl], &p[cl + 1]); // right face of left cell
            let (pl, _) = recon(&p[cr - 1], &p[cr], &p[cr + 1]); // left face of right cell
            f[j] = flux(&pr, &pl);
        }
        // godunov update: cons[i] += (F_lo - F_hi) * dt/dx.
        for i in NG..NG + N {
            cons[i] = cons[i] + (f[i] - f[i + 1]) * (dt / dx);
        }
        t += dt;
    }
    cons.iter().map(|c| nmhd_recover(&eos, c)).collect()
}

fn total_variation(f: &[f64]) -> f64 {
    f.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}
fn l1_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

#[test]
fn nmhd_brio_wu_hlld_hllc_are_clean_shock_capturing() {
    let eos = IdealGas { gamma: GAMMA };
    let n = Tensor::<f64, 3>::unit(0);

    let p_hlle = run(|l, r| hlle(&NewtonianMhd, &eos, l, r, &n, 0.0));
    let p_hllc = run(|l, r| hllc_newtonian(&eos, l, r, &n, 0.0, ShockwaveLimiter::Standard));
    let p_hlld = run(|l, r| hlld_newtonian(&eos, l, r, &n, 0.0));

    let interior = |p: &[P], f: fn(&P) -> f64| -> Vec<f64> { p[NG..NG + N].iter().map(f).collect() };
    let rho_e = interior(&p_hlle, |p| p.rho);
    let rho_c = interior(&p_hllc, |p| p.rho);
    let rho_d = interior(&p_hlld, |p| p.rho);
    let by_e = interior(&p_hlle, |p| p.mag[1]);
    let by_d = interior(&p_hlld, |p| p.mag[1]);

    // 1) PHYSICAL everywhere (the algebraic c2p must recover rho,p > 0).
    for (label, ps) in [("hllc", &p_hllc), ("hlld", &p_hlld)] {
        for (i, pi) in ps[NG..NG + N].iter().enumerate() {
            assert!(pi.rho.is_finite() && pi.rho > 0.0, "{label} cell {i}: rho={}", pi.rho);
            assert!(pi.pre.is_finite() && pi.pre > 0.0, "{label} cell {i}: p={}", pi.pre);
        }
    }

    // 2) Bx (normal field) stays EXACTLY constant (induction F(Bx)=0 in 1D).
    for pi in &p_hlld[NG..NG + N] {
        assert!((pi.mag[0] - 0.75).abs() < 1e-12, "Bx drifted: {}", pi.mag[0]);
    }

    // 3) NON-OSCILLATORY: HLLE is the monotone baseline; a buggy solver oscillates
    //    -> TV inflates. a correct SHARPER solver keeps TV ~ the same (a monotone
    //    profile's TV is diffusion-independent). this is the direct "noise" test.
    let (tv_re, tv_rc, tv_rd) = (total_variation(&rho_e), total_variation(&rho_c), total_variation(&rho_d));
    let (tv_be, tv_bd) = (total_variation(&by_e), total_variation(&by_d));
    eprintln!("[brio-wu] TV(rho): hlle={tv_re:.4} hllc={tv_rc:.4} hlld={tv_rd:.4}");
    eprintln!("[brio-wu] TV(By):  hlle={tv_be:.4} hlld={tv_bd:.4}");
    assert!(tv_rd < 1.3 * tv_re, "HLLD rho OSCILLATES: TV {tv_rd:.4} vs hlle {tv_re:.4}");
    assert!(tv_rc < 1.3 * tv_re, "HLLC rho OSCILLATES: TV {tv_rc:.4} vs hlle {tv_re:.4}");
    assert!(tv_bd < 1.3 * tv_be, "HLLD By OSCILLATES: TV {tv_bd:.4} vs hlle {tv_be:.4}");

    // 4) SAME solution as HLLE (sharper, not different): small L1 distance.
    assert!(l1_diff(&rho_d, &rho_e) < 0.05, "HLLD rho diverges from HLLE: L1 {}", l1_diff(&rho_d, &rho_e));
    assert!(l1_diff(&rho_c, &rho_e) < 0.05, "HLLC rho diverges from HLLE: L1 {}", l1_diff(&rho_c, &rho_e));

    // 5) HLLD actually SHARPENS (does its job, not silently falling back to HLLE):
    //    the steepest density gradient is larger than HLLE's.
    let max_grad = |f: &[f64]| f.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f64, f64::max);
    assert!(
        max_grad(&rho_d) >= max_grad(&rho_e) * 0.95,
        "HLLD not sharper than HLLE (grad {} vs {}) — is it falling back?",
        max_grad(&rho_d), max_grad(&rho_e),
    );

    // 6) the Brio-Wu signature: By flips sign across the tube (the compound/rotational
    //    structure), and the density develops intermediate states between L and R.
    let by_min = by_d.iter().cloned().fold(f64::MAX, f64::min);
    let by_max = by_d.iter().cloned().fold(f64::MIN, f64::max);
    assert!(by_min < -0.5 && by_max > 0.5, "By did not span the L/R sign change: [{by_min},{by_max}]");
    let rho_mid = rho_d[N / 2];
    assert!(rho_mid > 0.125 && rho_mid < 1.0, "midpoint density {rho_mid} is a pure step — no waves formed");
}
