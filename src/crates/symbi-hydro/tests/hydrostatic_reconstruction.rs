// =============================================================================
// hydrostatic_reconstruction.rs
//
// the properties that let a low-dissipation riemann solver run on a stratified
// atmosphere, stated as theorems about the mechanical hydrostatic reconstruction
// (Kaeppeli & Mishra, A&A 587, A94, 2016) and checked at the tolerance each proof
// allows -- roundoff where the statement is exact, a measured rate where it is
// asymptotic.
//
//   T1 well-balanced, arbitrary stratification. on a column in the scheme's own
//      discrete mechanical class -- pressures related by the piecewise-constant-
//      density segment sums, density free at every cell -- the two sides of every
//      face agree to roundoff, so the numerical flux at rest carries zero
//      dissipation and the state is a discrete fixed point. no thermal structure
//      is assumed, which is the property the isentropic predecessor lacked.
//
//   T1b continuum limit. an analytically sampled isentrope sits in the discrete
//      class only to truncation, and the residual face jump converges away at
//      second order, so the discrete equilibria approach the continuum ones.
//
//   T2 gravity-free reduction. at zero potential difference across the stencil the
//      pressure path is bit-identical to plain theta-limited reconstruction; the
//      density and velocity paths are the plain reconstruction by construction.
//
//   T3 order. on a smooth state away from equilibrium the reconstruction is still
//      second-order, so the scheme keeps its order where well-balancing has nothing
//      to gain.
//
//   T4 stratification-blind. the residual on an analytic hydrostatic column is set
//      by the grid alone: columns whose entropy variation differs by three decades
//      carry the same residual, and it converges at second order in h. the
//      isentropic predecessor degraded linearly in the entropy variation, and this
//      is the theorem that records the upgrade.
//
//   T5/T6 reconstruction-agnostic. the well-balancing lives in the departure
//      transform, upstream of the limiter that consumes it, so a parabolic
//      (ppm-shaped) operator inherits T1 directly; its gravity-free reduction holds
//      to a few ulp, where plm's is bit-exact, for the structural reason recorded
//      at T6.
//
// run: cargo test -p symbi-hydro --test hydrostatic_reconstruction -- --nocapture
// =============================================================================

use symbi_hydro::hydrostatic::{
    LocalEquilibrium, hydrostatic_departures, hydrostatic_face, plain_face,
};

const GAMMA: f64 = 5.0 / 3.0;
/// swept over both limiter arms: theta-MC at +2, van leer at -1 (the kernel selects the
/// smooth harmonic limiter on a negative theta, and a host reference that only ever ran
/// positive theta was blind to a sign-flipped slope on that arm).
const THETAS: [f64; 2] = [2.0, -1.0];
const THETA: f64 = 2.0;
/// the gravitating mass sits one domain width below x = 0, so the column covers r in
/// [1, 2], clear of the singularity, with a potential that is genuinely curved across it.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 3.0;

fn potential(x: f64) -> f64 {
    -GM / (x + G_OFFSET)
}

/// the exact isentropic hydrostatic column: `(gamma/(gamma-1)) K rho^(gamma-1) + phi = H`.
/// `k_entropy` selects which member, `h_const` fixes the normalization.
fn isentrope(x: f64, k_entropy: f64, h_const: f64) -> (f64, f64) {
    let rho = ((GAMMA - 1.0) / (GAMMA * k_entropy) * (h_const - potential(x)))
        .powf(1.0 / (GAMMA - 1.0));
    (rho, k_entropy * rho.powf(GAMMA))
}

/// a column in the scheme's own discrete mechanical class: pressures built by the
/// piecewise-constant-density segment sums over the given centers and interior faces,
/// with a deliberately unstructured positive density. class membership is the whole
/// input: no thermal relation ties the density to the pressure, which is exactly the
/// generality the mechanical reconstruction preserves.
fn class_column(xc: &[f64], xf: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let rho: Vec<f64> = xc
        .iter()
        .map(|&x| 1.4 + 0.9 * (5.0 * x).sin() * (2.3 * x).cos())
        .collect();
    let mut pre = vec![0.0; xc.len()];
    pre[0] = 9.0;
    for k in 0..xc.len() - 1 {
        pre[k + 1] = pre[k]
            + rho[k] * (potential(xc[k]) - potential(xf[k]))
            + rho[k + 1] * (potential(xf[k]) - potential(xc[k + 1]));
    }
    assert!(
        pre.iter().all(|&p| p > 1.0e-3),
        "the class column left the physical regime; the fixed-point statement is vacuous"
    );
    (rho, pre)
}

/// a hydrostatic column whose entropy varies: `K(x) = k0 (1 + eps * x)`. the state is
/// marched directly onto the stencil points with an rk4 on the exact balance, many
/// substeps between consecutive ones, so every stencil value comes straight from the
/// integrator and no interpolation error rides into the residual being measured.
fn stratified_column(xs: &[f64], eps: f64, k0: f64, rho0: f64) -> Vec<(f64, f64)> {
    let k_of = |x: f64| k0 * (1.0 + eps * x);
    // dp/dx = -rho dphi/dx with p = K(x) rho^gamma, i.e.
    // K gamma rho^(gamma-1) rho' = -rho phi' - K' rho^gamma.
    let deriv = |x: f64, rho: f64| {
        let k = k_of(x);
        let dphi = GM / (x + G_OFFSET).powi(2);
        let dk = k0 * eps;
        (-rho * dphi - dk * rho.powf(GAMMA)) / (k * GAMMA * rho.powf(GAMMA - 1.0))
    };
    const SUBSTEPS: usize = 512;
    let mut rho = rho0;
    let mut out = Vec::with_capacity(xs.len());
    out.push((rho, k_of(xs[0]) * rho.powf(GAMMA)));
    for w in xs.windows(2) {
        let h = (w[1] - w[0]) / SUBSTEPS as f64;
        let mut x = w[0];
        for _ in 0..SUBSTEPS {
            let k1 = deriv(x, rho);
            let k2 = deriv(x + 0.5 * h, rho + 0.5 * h * k1);
            let k3 = deriv(x + 0.5 * h, rho + 0.5 * h * k2);
            let k4 = deriv(x + h, rho + h * k3);
            rho += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            x += h;
        }
        out.push((rho, k_of(w[1]) * rho.powf(GAMMA)));
    }
    assert!(
        out.iter().all(|&(r, p)| r.is_finite() && p.is_finite() && r > 0.0 && p > 0.0),
        "the marched column is not a positive finite state; the residual sweep would read \
         zero because f64::max discards NaN, and the theorem would pass vacuously"
    );
    out
}

/// the pressure jump the riemann solver would see at the face between cells `ii` and
/// `ii + 1` on a lattice with centers `xc` and interior faces `xf` (`xf[k]` between
/// centers `k - 1` and `k`): cell `ii` reconstructs to its upper face, `ii + 1` to its
/// lower.
fn face_jump(
    rho: &[f64],
    pre: &[f64],
    xc: &[f64],
    xf: &[f64],
    ii: usize,
    theta: f64,
) -> f64 {
    let phi_c = |k: usize| potential(xc[k]);
    let phi_f = |k: usize| potential(xf[k]);
    let pl = hydrostatic_face(
        [rho[ii - 1], rho[ii], rho[ii + 1]],
        [pre[ii - 1], pre[ii], pre[ii + 1]],
        [phi_c(ii - 1), phi_c(ii), phi_c(ii + 1)],
        [phi_f(ii - 1), phi_f(ii)],
        phi_f(ii),
        theta,
        1.0,
    );
    let pr = hydrostatic_face(
        [rho[ii], rho[ii + 1], rho[ii + 2]],
        [pre[ii], pre[ii + 1], pre[ii + 2]],
        [phi_c(ii), phi_c(ii + 1), phi_c(ii + 2)],
        [phi_f(ii), phi_f(ii + 1)],
        phi_f(ii),
        theta,
        -1.0,
    );
    (pl - pr).abs()
}

/// centers and interior faces of a uniform lattice with `n + 4` cells of width `h`:
/// `xf[k]` sits between `xc[k]` and `xc[k + 1]`.
fn lattice(n: usize, h: f64) -> (Vec<f64>, Vec<f64>) {
    let xc: Vec<f64> = (0..n + 4).map(|ii| (ii as f64 + 0.5) * h).collect();
    let xf: Vec<f64> = (1..n + 4).map(|ii| ii as f64 * h).collect();
    (xc, xf)
}

// =============================================================================
// T1 — well-balanced on an arbitrarily stratified column in the discrete class
// =============================================================================

#[test]
fn t1_the_face_jump_vanishes_on_a_balanced_column_of_arbitrary_stratification() {
    // the defining property, at the generality the mechanical scheme claims: the column's
    // density is unstructured, so no thermal assumption could reproduce it, and every face
    // still reconstructs one pressure. the positive control shows the plain scheme leaves
    // a real jump on the same column, so a pass carries weight.
    for theta in THETAS {
        for n in [16usize, 32, 64] {
            let h = 1.0 / n as f64;
            let (xc, xf) = lattice(n, h);
            let (rho, pre) = class_column(&xc, &xf);

            let mut worst_wb: f64 = 0.0;
            let mut worst_plain: f64 = 0.0;
            for ii in 1..n {
                let dp = face_jump(&rho, &pre, &xc, &xf, ii, theta);
                worst_wb = worst_wb.max(dp / pre[ii]);
                let ppl = plain_face([pre[ii - 1], pre[ii], pre[ii + 1]], THETA, 1.0);
                let ppr = plain_face([pre[ii], pre[ii + 1], pre[ii + 2]], THETA, -1.0);
                worst_plain = worst_plain.max((ppl - ppr).abs() / pre[ii]);
            }
            println!(
                "theta = {theta:+.1}, n = {n:3}: well-balanced jump {worst_wb:.3e}, \
                 plain jump {worst_plain:.3e}"
            );
            assert!(
                worst_plain > 1.0e-6,
                "positive control failed: plain reconstruction left no jump at n = {n} \
                 ({worst_plain:.3e}), so this column does not exercise the imbalance and \
                 T1 is vacuous"
            );
            assert!(
                worst_wb < 1.0e-14,
                "theta = {theta}, n = {n}: the mechanical reconstruction left a relative \
                 face jump of {worst_wb:.3e} on a column in its own discrete class"
            );
        }
    }
}

#[test]
fn t1b_the_residual_on_an_analytic_isentrope_converges_at_second_order() {
    // an analytically sampled column satisfies the discrete class only to truncation:
    // the piecewise-constant-density segments approximate the true integral of rho dphi
    // at second order per cell. the residual jump must therefore converge away at least
    // that fast, which is what ties the discrete equilibria to the continuum ones.
    let mut prev: Option<(f64, f64)> = None;
    for n in [32usize, 64, 128, 256] {
        let h = 1.0 / n as f64;
        let (xc, xf) = lattice(n, h);
        let cols: Vec<(f64, f64)> = xc.iter().map(|&x| isentrope(x, 0.7, 4.0)).collect();
        let rho: Vec<f64> = cols.iter().map(|c| c.0).collect();
        let pre: Vec<f64> = cols.iter().map(|c| c.1).collect();
        let mut worst: f64 = 0.0;
        for ii in 1..n {
            worst = worst.max(face_jump(&rho, &pre, &xc, &xf, ii, THETA) / pre[ii]);
        }
        if let Some((hp, wp)) = prev {
            let rate = (wp / worst).ln() / (hp / h).ln();
            println!("n = {n:3}: residual {worst:.3e}, observed order {rate:.2}");
            assert!(
                rate > 1.8,
                "the residual on an analytic isentrope converges at order {rate:.2}; the \
                 discrete class is not approaching the continuum equilibrium"
            );
        }
        prev = Some((h, worst));
    }
}

// =============================================================================
// T2 — bit-identical without gravity
// =============================================================================

#[test]
fn t2_it_is_bit_identical_to_plain_reconstruction_without_gravity() {
    // with a flat potential the segment corrections are products with an exact zero, the
    // departures are the plain differences, and the limiter sees identical arguments.
    // density and velocity take the plain reconstruction by construction, so pressure is
    // the whole statement. states span smooth, extremal and sign-flipping stencils so the
    // limiter's every branch is exercised.
    let cases: [[f64; 3]; 6] = [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [1.0, 5.0, 1.0],
        [2.0, 1.0, 9.0],
        [1.0e-3, 1.0e3, 1.0e-3],
    ];
    for rho in cases {
        for pre in cases {
            for theta in THETAS {
                for sign in [1.0, -1.0] {
                    let phi_c = [-2.5, -2.5, -2.5];
                    let phi_f = [-2.5, -2.5];
                    let p_wb =
                        hydrostatic_face(rho, pre, phi_c, phi_f, -2.5, theta, sign);
                    assert_eq!(
                        p_wb,
                        plain_face(pre, theta, sign),
                        "pressure differs for pre = {pre:?}, theta = {theta}, sign = {sign}"
                    );
                }
            }
        }
    }
}

// =============================================================================
// T3 — second order on a non-equilibrium smooth state
// =============================================================================

#[test]
fn t3_it_stays_second_order_on_a_smooth_non_equilibrium_state() {
    // well-balancing must keep second-order accuracy off equilibrium while staying exact
    // on it. the state here is smooth and deliberately off hydrostatic balance, so the
    // deviation carries real structure and the limiter is doing ordinary work.
    let exact = |x: f64| (1.5 + 0.5 * (3.0 * x).sin(), 0.9 + 0.4 * (2.0 * x).cos());
    let mut prev: Option<(f64, f64)> = None;
    for n in [32usize, 64, 128, 256] {
        let h = 1.0 / n as f64;
        let (xc, xf) = lattice(n, h);
        let rho: Vec<f64> = xc.iter().map(|&x| exact(x).0).collect();
        let pre: Vec<f64> = xc.iter().map(|&x| exact(x).1).collect();
        let mut err: f64 = 0.0;
        for ii in 1..n {
            let pl = hydrostatic_face(
                [rho[ii - 1], rho[ii], rho[ii + 1]],
                [pre[ii - 1], pre[ii], pre[ii + 1]],
                [potential(xc[ii - 1]), potential(xc[ii]), potential(xc[ii + 1])],
                [potential(xf[ii - 1]), potential(xf[ii])],
                potential(xf[ii]),
                THETA,
                1.0,
            );
            err = err.max((pl - exact(xf[ii]).1).abs());
        }
        if let Some((hp, ep)) = prev {
            let rate = (ep / err).ln() / (hp / h).ln();
            println!("n = {n:3}: err {err:.3e}, observed order {rate:.2}");
            assert!(
                rate > 1.8,
                "observed order {rate:.2} at n = {n}; the deviation reconstruction has \
                 lost second-order accuracy off equilibrium"
            );
        }
        prev = Some((h, err));
    }
}

// =============================================================================
// T4 — the residual is set by the grid, whatever the stratification
// =============================================================================

#[test]
fn t4_the_residual_is_blind_to_the_entropy_variation_and_second_order_in_h() {
    // the isentropic predecessor degraded linearly in the entropy variation across a
    // cell; the mechanical class contains every stratification, so the residual on an
    // analytic column is truncation of the segment quadrature alone. two statements pin
    // that: columns whose entropy variation spans three decades carry the same residual,
    // and that residual converges at second order in h.
    let n = 64usize;
    let h = 1.0 / n as f64;
    let (xc, xf) = lattice(n, h);
    let residual = |eps: f64, xc: &[f64], xf: &[f64], n: usize| -> f64 {
        let cols = stratified_column(xc, eps, 0.7, 2.0);
        let rho: Vec<f64> = cols.iter().map(|c| c.0).collect();
        let pre: Vec<f64> = cols.iter().map(|c| c.1).collect();
        let mut worst: f64 = 0.0;
        for ii in 1..n {
            worst = worst.max(face_jump(&rho, &pre, xc, xf, ii, THETA) / pre[ii]);
        }
        worst
    };

    let strong = residual(1.0e-1, &xc, &xf, n);
    let faint = residual(1.0e-4, &xc, &xf, n);
    println!("eps 1e-1: residual {strong:.3e}; eps 1e-4: residual {faint:.3e}");
    assert!(
        strong < 10.0 * faint + 1.0e-13,
        "a thousandfold change in the entropy variation moved the residual from \
         {faint:.3e} to {strong:.3e}; the scheme is still charging for stratification"
    );

    let mut prev: Option<(f64, f64)> = None;
    for n in [32usize, 64, 128, 256] {
        let h = 1.0 / n as f64;
        let (xc, xf) = lattice(n, h);
        let worst = residual(1.0e-1, &xc, &xf, n);
        if let Some((hp, wp)) = prev {
            let rate = (wp / worst).ln() / (hp / h).ln();
            println!("n = {n:3}: residual {worst:.3e}, observed order {rate:.2}");
            assert!(
                rate > 1.8,
                "the stratified-column residual converges at order {rate:.2}; the \
                 mechanical quadrature is not second order"
            );
        }
        prev = Some((h, worst));
    }
}

// =============================================================================
// T5/T6 — the transform is reconstruction-agnostic
//
// the well-balancing lives entirely in the departure transform, upstream of the limiter
// that consumes it. these check that against a parabolic operator of ppm's shape, so ppm
// inherits the property rather than needing its own derivation.
// =============================================================================

/// the ppm interface interpolant on six cell values: the fourth-order face value between
/// the third and fourth entries, `(7/12)(q_i + q_{i+1}) - (1/12)(q_{i-1} + q_{i+2})`.
/// unlimited on purpose — the monotonicity constraint is a function of the same
/// departures and cannot reintroduce a jump the interpolant did not create.
fn ppm_face(q: [f64; 6]) -> f64 {
    (7.0 / 12.0) * (q[2] + q[3]) - (1.0 / 12.0) * (q[1] + q[4])
}

#[test]
fn t5_a_parabolic_operator_inherits_the_well_balanced_property() {
    let n = 64usize;
    let h = 1.0 / n as f64;
    let (xc, xf) = lattice(n, h);
    let (rho, pre) = class_column(&xc, &xf);
    let phi_c: Vec<f64> = xc.iter().map(|&x| potential(x)).collect();
    let phi_f: Vec<f64> = xf.iter().map(|&x| potential(x)).collect();

    let mut worst_wb: f64 = 0.0;
    let mut worst_plain: f64 = 0.0;
    for ii in 3..n {
        // the six-cell window [ii-3 .. ii+2], its five interior faces, and the target
        // face between window indices 2 and 3 (cells ii-1 and ii).
        let wc = |v: &[f64]| -> Vec<f64> { v[ii - 3..ii + 3].to_vec() };
        let wf: Vec<f64> = phi_f[ii - 3..ii + 2].to_vec();
        let phi_face = wf[2];

        let d_l = hydrostatic_departures(2, &wc(&pre), &wc(&rho), &wc(&phi_c), &wf, 1.0);
        let d_r = hydrostatic_departures(3, &wc(&pre), &wc(&rho), &wc(&phi_c), &wf, 1.0);
        let eq_l = LocalEquilibrium::through(rho[ii - 1], pre[ii - 1], phi_c[ii - 1]);
        let eq_r = LocalEquilibrium::through(rho[ii], pre[ii], phi_c[ii]);
        let face_l = eq_l.pressure_at(phi_face) + ppm_face(d_l.try_into().unwrap());
        let face_r = eq_r.pressure_at(phi_face) + ppm_face(d_r.try_into().unwrap());
        worst_wb = worst_wb.max((face_l - face_r).abs() / pre[ii]);
        // positive control: the plain parabola on the pressure itself leaves a real
        // offset from the cell value here.
        let win: [f64; 6] = wc(&pre).try_into().unwrap();
        worst_plain = worst_plain.max((ppm_face(win) - pre[ii]).abs() / pre[ii]);
    }
    println!("ppm: well-balanced jump {worst_wb:.3e}, plain offset {worst_plain:.3e}");
    assert!(
        worst_plain > 1.0e-6,
        "positive control failed: the parabola sits on the column to {worst_plain:.3e}, \
         so this setup does not exercise the imbalance and T5 is vacuous"
    );
    assert!(
        worst_wb < 1.0e-13,
        "a parabolic operator on the departures left a relative face jump of \
         {worst_wb:.3e}; the transform is not reconstruction-agnostic"
    );
}

#[test]
fn t6_the_parabolic_path_matches_plain_reconstruction_to_roundoff_without_gravity() {
    // T2's bit-identity does not carry to a parabola, and the reason is structural rather
    // than a tolerance to be tuned. plm's face value is `q_anchor + slope/2` built from
    // one-sided differences about the anchor; anchoring on the reconstructed cell makes
    // that departure exactly zero, so the differences reduce to `0 - d` and `d - 0` and
    // no rounding enters. a parabola is a weighted sum over four cells: shifting all four
    // by the anchor value and adding it back is exact in real arithmetic and lands within
    // a few ulp in floating point, because the weights sum to one only after rounding.
    // the honest statement is therefore agreement to roundoff, measured here rather than
    // asserted, with the bit-exact claim kept where it is actually true (T2, the linear
    // operator).
    let cases: [([f64; 6], [f64; 6]); 3] = [
        ([1.0, 2.5, 0.3, 7.25, 4.0, 0.125], [0.5, 1.5, 0.25, 3.0, 2.0, 0.0625]),
        ([1.0; 6], [0.6; 6]),
        ([1e-3, 1e3, 1e-3, 1e3, 1e-3, 1e3], [1e-4, 1e2, 1e-4, 1e2, 1e-4, 1e2]),
    ];
    let phi_c = [-2.5; 6];
    let phi_f = [-2.5; 5];
    let mut worst_ulp = 0.0f64;
    for (q, p) in cases {
        for anchor in [2usize, 3] {
            let d = hydrostatic_departures(anchor, &p, &q, &phi_c, &phi_f, 1.0);
            let d: [f64; 6] = d.try_into().unwrap();
            assert_eq!(
                d[anchor], 0.0,
                "the anchor departure must be exactly zero — that is what carries T2"
            );
            let eq = LocalEquilibrium::through(q[anchor], p[anchor], phi_c[anchor]);
            let got = eq.pressure_at(phi_c[anchor]) + ppm_face(d);
            let want = ppm_face(p);
            let ulp = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE) / f64::EPSILON;
            worst_ulp = worst_ulp.max(ulp);
        }
    }
    println!("ppm gravity-free agreement: worst {worst_ulp:.2} ulp");
    // 8 ulp bounds a four-term weighted sum re-centred once: each term carries at most a
    // half-ulp from the shift and a half-ulp from the sum, and the cancellation in the
    // `7/12`/`-1/12` pairing can amplify that by the ratio of the summed magnitudes to
    // the result. it is a numerical statement about this interpolant, recorded with the
    // measurement above so a regression shows as a number.
    assert!(
        worst_ulp < 8.0,
        "the parabolic path drifted {worst_ulp:.2} ulp from plain reconstruction with no \
         gravity; the departure transform is doing more than re-centring"
    );
}
