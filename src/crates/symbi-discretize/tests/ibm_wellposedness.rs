// =============================================================================
// ibm_wellposedness.rs
//
// machine-checked well-posedness proofs for the immersed-boundary source (`ibm.rs`), the same
// operators the traced body kernels run (softened gravity + the exact-exponential accretion drain).
// each theorem is stated + proved analytically here, then gated against the real carrier-generic
// code — conservativeness by forward-mode autodiff (the `Dual` carrier differentiates the actual
// potential), the sign / contraction / boundedness lemmas by dense sampling of the actual f64 forms.
//
// -----------------------------------------------------------------------------
// THEOREM 1 (conservative gravity).  g(x) = -grad phi(x), with
//     phi(x)   = -M / r_eff,     r_eff = sqrt(|r|^2 + eps^2),   r = x - x_body.
//   PROOF.  d/dx_j (1/r_eff) = -(1/2)(|r|^2+eps^2)^{-3/2} d/dx_j |r|^2, and d/dx_j |r|^2 = 2 r_j,
//   so d/dx_j (1/r_eff) = -r_j / r_eff^3.  hence
//     -d phi/dx_j = M d/dx_j (1/r_eff) = -M r_j / r_eff^3 = g_j.   QED.
//   COROLLARY.  curl g = -curl(grad phi) = 0: the work integral around any closed loop vanishes, so
//   the energy the immersed body exchanges with the gas is bookkept by phi alone.  (mixed partials
//   of the smooth phi commute — softening makes phi C^infinity everywhere, r = 0 included.)
//
// THEOREM 2 (drain is a contraction on the intensive state).  For rate >= 0, dt >= 0,
//     f = exp(-rate dt)  in  (0, 1]   (mathematically),
//   so U -> U f is positivity-preserving and non-expansive at every dt — the sink is
//   unconditionally stable, free of any CFL restriction.
//   the drain rate is nonnegative for physical inputs:
//     rate = chi min(sink, cs/w),   chi = (1 - tanh(z))/2 in (0,1),   z = (r - r_mask)/w,
//   with sink >= 0, cs >= 0, w > 0  =>  min(sink, cs/w) >= 0  =>  rate >= 0.
//   In IEEE f64 the realized operator maps  f in [0, 1]:  exp underflows to +0.0 only when rate dt
//   exceeds ~745 — deep inside a fully-evacuated mask, where den -> 0 is exactly the accretion limit.
//   so den -> den f lands in [0, den]: nonnegative straight from the arithmetic — the operator
//   itself supplies the bound a floor would otherwise impose — and non-expansive for every dt.
//   that is the well-posedness guarantee, and it is why the exact-exponential drain
//   retired the KMK04 min-gate: an operator that is a contraction by construction.
//
// THEOREM 3 (softening regularizes the singularity — bounded, lipschitz force).
//     |g| = M |r| / (|r|^2 + eps^2)^{3/2}  <=  C M / eps^2,     C = 2 / (3 sqrt 3) ~ 0.3849,
//   attained at |r| = eps / sqrt 2.  the bare 1/r^2 force diverges as r -> 0; the softened
//   force has a finite maximum set by eps, so g is globally bounded and lipschitz — the softening
//   is what makes the source well-posed at the body.
//   PROOF.  maximize h(s) = s/(s^2+eps^2)^{3/2}.  h'(s) = (eps^2 - 2 s^2)/(s^2+eps^2)^{5/2} = 0 at
//   s = eps/sqrt 2, where h = (eps/sqrt2)/((3 eps^2/2)^{3/2}) = 2/(3 sqrt 3 eps^2).   QED.
// -----------------------------------------------------------------------------
//
// run: cargo test -p symbi-discretize --test ibm_wellposedness
// =============================================================================

use symbi_discretize::ibm::{
    compact_gravity, compact_potential, drain_factor, drain_rate, softened_gravity,
    softened_potential,
};
use symbi_ir::dual::Dual;

// the analytic bound constant C = 2/(3 sqrt 3) from THEOREM 3.
fn bound_const() -> f64 {
    2.0 / (3.0 * 3.0_f64.sqrt())
}

// a deterministic parameter sweep — fixed sample points, reproducible run to run, dense enough
// to exercise the near-body
// core, the mask edge, and the far field.
fn sweep<F: FnMut(f64, f64, f64, f64)>(mut f: F) {
    for &mass in &[0.3_f64, 1.0, 7.5] {
        for &soft in &[0.02_f64, 0.1, 0.5] {
            for ri in 0..40 {
                let r = 1e-3 + ri as f64 * 0.05; // radial displacement magnitude
                for &frac in &[-0.7_f64, 0.0, 0.4, 1.0] {
                    f(mass, soft, r, frac); // frac positions r relative to the mask (used by drain)
                }
            }
        }
    }
}

// THEOREM 1 — conservative gravity: g = -grad phi, checked by autodiff on the real potential.
#[test]
fn gravity_is_conservative_grad_of_potential() {
    // put the body off-origin so every coordinate stays nonzero, and spread r over the 3 axes.
    let bpos = [0.13_f64, -0.27, 0.41];
    sweep(|mass, soft, r, frac| {
        // a position at displacement r along a fixed direction (varied a little by frac so the
        // sampled points spread over distinct directions).
        let dir = {
            let d = [0.6, -0.5 + 0.3 * frac, 0.62];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / n, d[1] / n, d[2] / n]
        };
        let pos = [
            bpos[0] + r * dir[0],
            bpos[1] + r * dir[1],
            bpos[2] + r * dir[2],
        ];

        let rvec_f = [pos[0] - bpos[0], pos[1] - bpos[1], pos[2] - bpos[2]];
        let g = softened_gravity(rvec_f, mass, soft);

        for j in 0..3 {
            // seed d/d pos_j via the Dual carrier; rvec_i = pos_i - bpos_i.
            let rvec: [Dual<f64>; 3] = std::array::from_fn(|i| {
                let p = if i == j {
                    Dual::variable(pos[i])
                } else {
                    Dual::constant(pos[i])
                };
                p - Dual::constant(bpos[i])
            });
            let phi = softened_potential(rvec, Dual::constant(mass), Dual::constant(soft));
            let dphi_dxj = phi.tangent; // exact d phi / d x_j

            // THEOREM 1: g_j = -d phi/d x_j. autodiff vs the direct formula differ only by
            // floating-point reassociation -> a tight relative tolerance.
            let scale = g[j].abs().max(dphi_dxj.abs()).max(1.0);
            assert!(
                (g[j] + dphi_dxj).abs() <= 1e-11 * scale,
                "g not conservative: g_{j} = {}, -dphi/dx_{j} = {} (mass={mass}, soft={soft}, r={r})",
                g[j],
                -dphi_dxj,
            );
        }
    });
}

// THEOREM 1 corollary — curl g = 0, checked directly by autodiff: d g_i/d x_k = d g_k/d x_i for all
// i, k (the symmetric hessian of phi), so every curl component vanishes.
#[test]
fn gravity_is_curl_free() {
    let bpos = [0.13_f64, -0.27, 0.41];
    // d g_i / d x_k via autodiff on the gravity itself.
    let dg = |pos: [f64; 3], mass: f64, soft: f64, i: usize, k: usize| -> f64 {
        let rvec: [Dual<f64>; 3] = std::array::from_fn(|a| {
            let p = if a == k {
                Dual::variable(pos[a])
            } else {
                Dual::constant(pos[a])
            };
            p - Dual::constant(bpos[a])
        });
        softened_gravity(rvec, Dual::constant(mass), Dual::constant(soft))[i].tangent
    };
    sweep(|mass, soft, r, frac| {
        let dir = [0.6, -0.5 + 0.3 * frac, 0.62];
        let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let pos = std::array::from_fn(|a| bpos[a] + r * dir[a] / n);
        for (i, k) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (dg(pos, mass, soft, i, k), dg(pos, mass, soft, k, i));
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= 1e-11 * scale,
                "curl g != 0: dg_{i}/dx_{k} = {a}, dg_{k}/dx_{i} = {b}",
            );
        }
    });
}

// THEOREM 2 — the drain is a contraction: rate >= 0 for physical inputs, and f = exp(-rate dt) in
// (0, 1] for every dt >= 0.
#[test]
fn drain_factor_is_a_contraction() {
    for &min_w in &[0.01_f64, 0.05, 0.2] {
        for &r_mask in &[0.0_f64, 0.15, 0.5] {
            for &sink in &[0.0_f64, 2.0, 1e3] {
                for &cs in &[0.0_f64, 0.4, 3.0] {
                    for ri in 0..60 {
                        let r_mag = ri as f64 * 0.02;
                        // the sign lemma: physical inputs (sink>=0, cs>=0, w>0) give rate >= 0.
                        let rate = drain_rate(r_mag, r_mask, min_w, sink, cs);
                        assert!(rate >= 0.0, "drain rate negative: {rate}");
                        // the mollified mask is bounded: rate <= min(sink, cs/w) since chi in (0,1).
                        assert!(
                            rate <= sink.min(cs / min_w) + 1e-12,
                            "rate exceeds the mask cap"
                        );
                        // THEOREM 2: den -> den f lands in [0, den] at every dt (the sink is
                        // CFL-free) — nonnegative (positivity-preserving) and non-expansive.
                        for &dt in &[0.0_f64, 1e-6, 0.03, 5.0, 1e6] {
                            let f = drain_factor(rate, dt);
                            assert!(f >= 0.0, "drain factor negative: f={f}");
                            assert!(f <= 1.0, "drain factor expands the state: f={f}");
                        }
                        // strict positivity holds below the f64 exp-underflow threshold (rate dt < 745):
                        // a physical step zeroes a cell only once the mask has fully evacuated it.
                        if rate * 0.03 < 700.0 {
                            assert!(drain_factor(rate, 0.03) > 0.0, "physical step underflowed");
                        }
                    }
                }
            }
        }
    }
}

// THEOREM 3 — softening bounds the force: |g| <= C M / eps^2, and the bound is tight (attained at
// |r| = eps/sqrt2), so it is the exact supremum.
#[test]
fn softened_gravity_is_bounded_by_softening() {
    let c = bound_const();
    for &mass in &[0.3_f64, 1.0, 7.5] {
        for &soft in &[0.02_f64, 0.1, 0.5] {
            let bound = c * mass / (soft * soft);
            // upper bound over a dense radial sweep (single axis suffices — |g| depends on |r| only).
            for ri in 0..2000 {
                let r = ri as f64 * (2.0 * soft) / 2000.0; // resolve the peak near eps/sqrt2
                let g = softened_gravity([r, 0.0, 0.0], mass, soft);
                let gmag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                assert!(
                    gmag <= bound * (1.0 + 1e-12),
                    "|g|={gmag} exceeds the softening bound {bound} (mass={mass}, soft={soft}, r={r})",
                );
            }
            // tightness: at r = eps/sqrt2 the bound is achieved to ~5 significant figures.
            let g_peak = softened_gravity([soft / 2.0_f64.sqrt(), 0.0, 0.0], mass, soft);
            let peak = (g_peak[0] * g_peak[0]).sqrt();
            assert!(
                (peak - bound).abs() <= 1e-9 * bound,
                "bound not tight: peak |g|={peak}, bound={bound}",
            );
        }
    }
}

// THEOREM 5 — the compact field is conservative: g = -grad phi, by autodiff on the real potential,
// sampled across the match radius, the transition itself included.
#[test]
fn compact_gravity_is_conservative_grad_of_potential() {
    let bpos = [0.13_f64, -0.27, 0.41];
    sweep(|mass, h, r, frac| {
        let dir = {
            let d = [0.6, -0.5 + 0.3 * frac, 0.62];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / n, d[1] / n, d[2] / n]
        };
        let pos = [
            bpos[0] + r * dir[0],
            bpos[1] + r * dir[1],
            bpos[2] + r * dir[2],
        ];
        let rvec_f = [pos[0] - bpos[0], pos[1] - bpos[1], pos[2] - bpos[2]];
        let g = compact_gravity(rvec_f, mass, h);
        for j in 0..3 {
            let rvec: [Dual<f64>; 3] = std::array::from_fn(|i| {
                let p = if i == j {
                    Dual::variable(pos[i])
                } else {
                    Dual::constant(pos[i])
                };
                p - Dual::constant(bpos[i])
            });
            let dphi = compact_potential(rvec, Dual::constant(mass), Dual::constant(h)).tangent;
            assert!(
                (g[j] + dphi).abs() < 1.0e-9 * (1.0 + g[j].abs()),
                "compact field is not -grad phi at r = {r}, h = {h}, axis {j}: \
                 g = {}, -dphi = {}",
                g[j],
                -dphi
            );
        }
    });
}

// THEOREM 6 — compact support. outside the match radius the field is the bare point mass, to the
// last bit. a Plummer sphere departs from the point mass at every radius and every softening
// length, so this property is what lets `h` be chosen for regularity near the body while a power
// law measured outside it stays unbiased.
#[test]
fn the_compact_field_is_exactly_newtonian_outside_the_match_radius() {
    let mut checked = 0usize;
    for &mass in &[0.3_f64, 1.0, 7.5] {
        for &h in &[0.02_f64, 0.1, 0.5] {
            for k in 1..60 {
                let r = h * (1.0 + 0.05 * k as f64); // strictly outside
                let rvec = [r * 0.6, r * -0.8, 0.0]; // |rvec| = r exactly
                let g = compact_gravity(rvec, mass, h);
                // the oracle is the bare point mass evaluated through the same arithmetic:
                // `softened_gravity` at zero softening recovers |rvec| by the identical
                // `sqrt(dot)` and cubes it identically, which keeps the comparison on the model;
                // a separately-formed `-mass/r^3` would measure how the norm round-trips.
                let want = softened_gravity(rvec, mass, 0.0);
                for i in 0..3 {
                    assert_eq!(
                        g[i].to_bits(),
                        want[i].to_bits(),
                        "compact field differs from the point mass at r/h = {}: {} vs {}",
                        r / h,
                        g[i],
                        want[i]
                    );
                }
                assert_eq!(
                    compact_potential(rvec, mass, h).to_bits(),
                    softened_potential(rvec, mass, 0.0).to_bits(),
                    "compact potential differs from -M/r at r/h = {}",
                    r / h
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 400, "the sweep covered only {checked} points");

    // non-vacuity: a Plummer sphere at the same length departs from the point mass at every radius
    // -- that departure is the bias the compact form removes, and it is large enough here to be
    // worth removing.
    let (mass, h) = (1.0_f64, 0.1_f64);
    for &mult in &[1.0_f64, 2.0, 5.0] {
        let r = h * mult;
        let rvec = [r, 0.0, 0.0];
        let ratio = softened_gravity(rvec, mass, h)[0] / (-mass / (r * r));
        let exact = compact_gravity(rvec, mass, h)[0] / (-mass / (r * r));
        println!("  r = {mult} h:  plummer/newton = {ratio:.4}   compact/newton = {exact:.4}");
        assert!(
            ratio < 0.995,
            "plummer at r = {mult} h is already indistinguishable from newton ({ratio}); the \
             compact form would be solving nothing"
        );
    }
}

// THEOREM 7 — the peak field is bounded by the match radius alone, at 1.242 mass/h^2 (attained at
// r = sqrt(5/9) h). resolution-independent, where a Plummer sphere accurate at `h` peaks as
// 1/eps^2 and takes the timestep with it.
#[test]
fn the_compact_field_peaks_at_a_bounded_resolution_independent_value() {
    for &mass in &[0.3_f64, 1.0, 7.5] {
        for &h in &[0.02_f64, 0.1, 0.5] {
            let mut peak = 0.0_f64;
            for k in 0..4000 {
                let r = h * 2.0 * k as f64 / 4000.0;
                let rvec = [r, 0.0, 0.0];
                peak = peak.max(compact_gravity(rvec, mass, h)[0].abs());
            }
            let want = 1.2417_f64 * mass / (h * h); // (5/2)u - (3/2)u^3 maximized at u^2 = 5/9
            assert!(
                (peak - want).abs() < 2.0e-3 * want,
                "peak |g| = {peak:.6e} at mass {mass}, h {h}; expected {want:.6e}"
            );
        }
    }
}
