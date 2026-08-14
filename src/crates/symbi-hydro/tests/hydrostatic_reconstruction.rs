// =============================================================================
// hydrostatic_reconstruction.rs
//
// the four properties that let a low-dissipation riemann solver run on a stratified
// atmosphere. each is stated as a theorem about the reconstruction and checked at the
// tolerance the proof allows -- roundoff where the statement is exact, a measured rate
// where it is asymptotic.
//
//   T1 WELL-BALANCED. on a discretely balanced isentrope the two sides of every face
//      agree to roundoff, so the numerical flux at rest carries no dissipation and the
//      state is a discrete fixed point. this is what removes the entropy leak that
//      forces a compressibility clamp onto the low-mach scheme.
//
//   T2 GRAVITY-FREE REDUCTION. with no potential difference across the stencil the
//      scheme is BIT-IDENTICAL to plain theta-limited reconstruction, so every
//      gravity-free result in the suite is untouched by construction rather than by
//      re-validation.
//
//   T3 ORDER. on a smooth state that is NOT an equilibrium the reconstruction is still
//      second-order, so well-balancing costs no accuracy where it buys nothing.
//
//   T4 ERROR BOUND. on a hydrostatic column that is not isentropic the residual face
//      jump is first order in the entropy variation across a cell and vanishes with it,
//      so the scheme degrades continuously rather than at a threshold.
//
//   T5/T6 RECONSTRUCTION-AGNOSTIC. the well-balancing lives in the DEPARTURE TRANSFORM, not
//      in the limiter that consumes it, so a parabolic (ppm-shaped) operator inherits T1 with
//      no separate derivation. its gravity-free reduction is to ROUNDOFF rather than
//      bit-exact: plm's face value is built from one-sided differences about an anchor whose
//      departure is exactly zero, which rounds away to nothing, while a parabola is a weighted
//      sum over four cells whose re-centring lands within a few ulp. measured, not assumed.
//
// run: cargo test -p symbi-hydro --test hydrostatic_reconstruction -- --nocapture
// =============================================================================

use symbi_hydro::hydrostatic::{hydrostatic_face, plain_face};

const GAMMA: f64 = 5.0 / 3.0;
const THETA: f64 = 2.0;
/// the gravitating mass sits one domain width below x = 0, so the column covers r in
/// [1, 2] with no singularity and a potential that is genuinely curved across it.
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

/// a hydrostatic column whose entropy varies: `K(x) = k0 (1 + eps * x)`. integrating
/// `dp/dx = -rho dphi/dx` with `p = K rho^gamma` no longer has a closed form, so the state
/// is built by marching the balance on a grid far finer than the stencil and sampling it.
/// eps = 0 returns the isentrope, which is what makes T4 a continuous statement rather
/// than a comparison of two different constructions.
fn stratified_column(xs: &[f64], eps: f64, k0: f64, rho0: f64) -> Vec<(f64, f64)> {
    // the column is marched DIRECTLY onto the stencil points, with many substeps between
    // consecutive ones and no interpolation. sampling a fine grid by linear interpolation
    // instead puts an O(h_fine^2) error into the state, and at small entropy variation that
    // error exceeds the residual being measured -- the observed order in eps then bends
    // toward zero and reports a property of the harness rather than of the scheme.
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

/// the jump the riemann solver would see at the face between cells `i` and `i+1`:
/// cell `i` reconstructs to its upper face, cell `i+1` to its lower.
fn face_jump(
    rho: &[f64],
    pre: &[f64],
    phi: &[f64],
    phi_face: f64,
    ii: usize,
) -> (f64, f64) {
    let (rl, pl) = hydrostatic_face(
        [rho[ii - 1], rho[ii], rho[ii + 1]],
        [pre[ii - 1], pre[ii], pre[ii + 1]],
        [phi[ii - 1], phi[ii], phi[ii + 1]],
        phi_face,
        GAMMA,
        THETA,
        1.0,
    );
    let (rr, pr) = hydrostatic_face(
        [rho[ii], rho[ii + 1], rho[ii + 2]],
        [pre[ii], pre[ii + 1], pre[ii + 2]],
        [phi[ii], phi[ii + 1], phi[ii + 2]],
        phi_face,
        GAMMA,
        THETA,
        -1.0,
    );
    ((rl - rr).abs(), (pl - pr).abs())
}

// =============================================================================
// T1 — well-balanced on the isentrope
// =============================================================================

#[test]
fn t1_the_face_jump_vanishes_on_a_discretely_balanced_isentrope() {
    // the defining property. a limited reconstruction of the STATE leaves an O(dx^2) jump on
    // this column; reconstructing the deviation leaves nothing, because the deviation is
    // identically zero in the whole stencil. the positive control below shows the plain
    // scheme really does leave a jump here, so passing is not vacuous.
    for n in [16usize, 32, 64] {
        let h = 1.0 / n as f64;
        let xs: Vec<f64> = (0..n + 4).map(|ii| ii as f64 * h).collect();
        let (k_entropy, h_const) = (0.7, 4.0);
        let cols: Vec<(f64, f64)> = xs.iter().map(|&x| isentrope(x, k_entropy, h_const)).collect();
        let rho: Vec<f64> = cols.iter().map(|c| c.0).collect();
        let pre: Vec<f64> = cols.iter().map(|c| c.1).collect();
        let phi: Vec<f64> = xs.iter().map(|&x| potential(x)).collect();

        let mut worst_wb: f64 = 0.0;
        let mut worst_plain: f64 = 0.0;
        for ii in 1..n {
            let phi_face = potential(0.5 * (xs[ii] + xs[ii + 1]));
            let (dr, dp) = face_jump(&rho, &pre, &phi, phi_face, ii);
            worst_wb = worst_wb.max(dr / rho[ii]).max(dp / pre[ii]);

            let prl = plain_face([rho[ii - 1], rho[ii], rho[ii + 1]], THETA, 1.0);
            let prr = plain_face([rho[ii], rho[ii + 1], rho[ii + 2]], THETA, -1.0);
            worst_plain = worst_plain.max((prl - prr).abs() / rho[ii]);
        }
        println!("n = {n:3}: well-balanced jump {worst_wb:.3e}, plain jump {worst_plain:.3e}");
        assert!(
            worst_plain > 1.0e-6,
            "positive control failed: plain reconstruction left no jump at n = {n} \
             ({worst_plain:.3e}), so this column does not exercise the imbalance and T1 is vacuous"
        );
        assert!(
            worst_wb < 1.0e-14,
            "n = {n}: hydrostatic reconstruction left a relative face jump of {worst_wb:.3e} \
             on an exactly balanced isentrope; the scheme is not well-balanced"
        );
    }
}

// =============================================================================
// T2 — bit-identical without gravity
// =============================================================================

#[test]
fn t2_it_is_bit_identical_to_plain_reconstruction_without_gravity() {
    // with a flat potential the local profile is the constant state, every departure is the
    // plain difference, and the limiter sees identical arguments. BIT-identical, not close:
    // the enthalpy ratio is exactly 1 and `1.0.powf(x)` is exact, so nothing rounds.
    // states chosen to span smooth, extremal and sign-flipping stencils so the limiter's
    // every branch is exercised.
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
            for sign in [1.0, -1.0] {
                let phi = [-2.5, -2.5, -2.5];
                let (r_wb, p_wb) =
                    hydrostatic_face(rho, pre, phi, phi[1], GAMMA, THETA, sign);
                assert_eq!(
                    r_wb,
                    plain_face(rho, THETA, sign),
                    "density differs for rho = {rho:?}, sign = {sign}"
                );
                assert_eq!(
                    p_wb,
                    plain_face(pre, THETA, sign),
                    "pressure differs for pre = {pre:?}, sign = {sign}"
                );
            }
        }
    }
}

// =============================================================================
// T3 — second order on a non-equilibrium smooth state
// =============================================================================

#[test]
fn t3_it_stays_second_order_on_a_smooth_non_equilibrium_state() {
    // well-balancing must not buy its exactness on the equilibrium by losing order off it.
    // the state here is smooth and deliberately NOT hydrostatic, so the deviation carries
    // real structure and the limiter is doing ordinary work.
    let exact = |x: f64| (1.5 + 0.5 * (3.0 * x).sin(), 0.9 + 0.4 * (2.0 * x).cos());
    let mut prev: Option<(f64, f64)> = None;
    for n in [32usize, 64, 128, 256] {
        let h = 1.0 / n as f64;
        let xs: Vec<f64> = (0..n + 4).map(|ii| ii as f64 * h).collect();
        let rho: Vec<f64> = xs.iter().map(|&x| exact(x).0).collect();
        let pre: Vec<f64> = xs.iter().map(|&x| exact(x).1).collect();
        let phi: Vec<f64> = xs.iter().map(|&x| potential(x)).collect();
        let mut err: f64 = 0.0;
        for ii in 1..n {
            let xf = 0.5 * (xs[ii] + xs[ii + 1]);
            let (rl, pl) = hydrostatic_face(
                [rho[ii - 1], rho[ii], rho[ii + 1]],
                [pre[ii - 1], pre[ii], pre[ii + 1]],
                [phi[ii - 1], phi[ii], phi[ii + 1]],
                potential(xf),
                GAMMA,
                THETA,
                1.0,
            );
            err = err.max((rl - exact(xf).0).abs()).max((pl - exact(xf).1).abs());
        }
        if let Some((hp, ep)) = prev {
            let rate = (ep / err).ln() / (hp / h).ln();
            println!("n = {n:3}: err {err:.3e}, observed order {rate:.2}");
            assert!(
                rate > 1.8,
                "observed order {rate:.2} at n = {n}; the deviation reconstruction has lost \
                 second-order accuracy off equilibrium"
            );
        }
        prev = Some((h, err));
    }
}

// =============================================================================
// T4 — the residual is controlled by the entropy variation
// =============================================================================

#[test]
fn t4_the_residual_is_first_order_in_the_entropy_variation() {
    // off the isentrope the scheme is no longer exact, and the honest statement is HOW it
    // degrades: the residual face jump is proportional to the entropy variation the column
    // carries, and returns to roundoff as that variation goes to zero. a scheme whose error
    // did not vanish with eps would be balancing the wrong thing.
    let n = 64usize;
    let h = 1.0 / n as f64;
    let xs: Vec<f64> = (0..n + 4).map(|ii| ii as f64 * h).collect();
    let phi: Vec<f64> = xs.iter().map(|&x| potential(x)).collect();
    let mut prev: Option<(f64, f64)> = None;
    for eps in [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4] {
        let cols = stratified_column(&xs, eps, 0.7, 2.0);
        let rho: Vec<f64> = cols.iter().map(|c| c.0).collect();
        let pre: Vec<f64> = cols.iter().map(|c| c.1).collect();
        let mut worst: f64 = 0.0;
        for ii in 1..n {
            let phi_face = potential(0.5 * (xs[ii] + xs[ii + 1]));
            let (dr, dp) = face_jump(&rho, &pre, &phi, phi_face, ii);
            worst = worst.max(dr / rho[ii]).max(dp / pre[ii]);
        }
        if let Some((ep, wp)) = prev {
            let rate = (wp / worst).ln() / (ep / eps).ln();
            println!("eps = {eps:.0e}: residual {worst:.3e}, order in eps {rate:.2}");
            assert!(
                rate > 0.9,
                "residual scales as eps^{rate:.2}; it must be at least first order in the \
                 entropy variation or the balance is not the one being followed"
            );
        }
        prev = Some((eps, worst));
    }
}

// =============================================================================
// T5/T6 — the transform is RECONSTRUCTION-AGNOSTIC
//
// the well-balancing lives entirely in the departure transform, not in the limiter that
// consumes it. these check that against a PARABOLIC operator of ppm's shape, so ppm inherits
// the property rather than needing its own derivation:
//
//   T5 a parabolic reconstruction of the departures still cancels at the face on a balanced
//      isentrope (well-balanced), and
//   T6 with no gravity it still returns the plain parabolic result BIT-for-bit, which holds
//      only because each side is anchored on its OWN cell.
// =============================================================================

use symbi_hydro::hydrostatic::{Thermodynamic, hydrostatic_deviations};

/// the ppm interface interpolant on six cell values: the fourth-order face value between the
/// third and fourth entries, `(7/12)(q_i + q_{i+1}) - (1/12)(q_{i-1} + q_{i+2})`. unlimited on
/// purpose — the monotonicity constraint is a function of the same departures and cannot
/// reintroduce a jump the interpolant did not create.
fn ppm_face(q: [f64; 6]) -> f64 {
    (7.0 / 12.0) * (q[2] + q[3]) - (1.0 / 12.0) * (q[1] + q[4])
}

#[test]
fn t5_a_parabolic_operator_inherits_the_well_balanced_property() {
    let n = 64usize;
    let h = 1.0 / n as f64;
    let xs: Vec<f64> = (0..n + 8).map(|ii| ii as f64 * h).collect();
    let (k_entropy, h_const) = (0.7, 4.0);
    let cols: Vec<(f64, f64)> = xs.iter().map(|&x| isentrope(x, k_entropy, h_const)).collect();
    let rho: Vec<f64> = cols.iter().map(|c| c.0).collect();
    let pre: Vec<f64> = cols.iter().map(|c| c.1).collect();
    let phi: Vec<f64> = xs.iter().map(|&x| potential(x)).collect();

    let mut worst_wb: f64 = 0.0;
    let mut worst_plain: f64 = 0.0;
    for ii in 3..n {
        let win = |v: &Vec<f64>| -> [f64; 6] {
            [v[ii - 3], v[ii - 2], v[ii - 1], v[ii], v[ii + 1], v[ii + 2]]
        };
        let phi_face = potential(0.5 * (xs[ii - 1] + xs[ii]));
        // the two sides of the face between cells ii-1 and ii, each anchored on its own cell.
        let dl = hydrostatic_deviations(win(&rho), win(&phi), 2, pre[ii - 1], GAMMA, Thermodynamic::Density);
        let dr = hydrostatic_deviations(win(&rho), win(&phi), 3, pre[ii], GAMMA, Thermodynamic::Density);
        let eq_l = symbi_hydro::hydrostatic::LocalEquilibrium::through(
            rho[ii - 1], pre[ii - 1], phi[ii - 1], GAMMA);
        let eq_r = symbi_hydro::hydrostatic::LocalEquilibrium::through(
            rho[ii], pre[ii], phi[ii], GAMMA);
        let face_l = eq_l.density_at(phi_face) + ppm_face(dl);
        let face_r = eq_r.density_at(phi_face) + ppm_face(dr);
        worst_wb = worst_wb.max((face_l - face_r).abs() / rho[ii]);
        // positive control: the plain parabola on the state itself leaves a real jump here.
        worst_plain = worst_plain.max((ppm_face(win(&rho)) - rho[ii]).abs() / rho[ii]);
    }
    println!("ppm: well-balanced jump {worst_wb:.3e}, plain offset {worst_plain:.3e}");
    assert!(
        worst_plain > 1.0e-6,
        "positive control failed: the parabola sits on the column to {worst_plain:.3e}, so this \
         setup does not exercise the imbalance and T5 is vacuous"
    );
    assert!(
        worst_wb < 1.0e-14,
        "a parabolic operator on the departures left a relative face jump of {worst_wb:.3e}; \
         the transform is not reconstruction-agnostic"
    );
}

#[test]
fn t6_the_parabolic_path_matches_plain_reconstruction_to_roundoff_without_gravity() {
    // T2's BIT-identity does NOT carry to a parabola, and the reason is structural rather than
    // a tolerance to be tuned. plm's face value is `q_anchor + slope/2` built from one-sided
    // DIFFERENCES about the anchor; anchoring on the reconstructed cell makes that departure
    // exactly zero, so the differences reduce to `0 - d` and `d - 0` and no rounding enters.
    // a parabola is a WEIGHTED SUM over four cells: shifting all four by the anchor value and
    // adding it back is exact in real arithmetic and lands within a few ulp in floating point,
    // because the weights sum to one only after rounding. the honest statement is therefore
    // agreement to roundoff, MEASURED here rather than asserted, with the bit-exact claim kept
    // where it is actually true (T2, the linear operator).
    let cases: [([f64; 6], [f64; 6]); 3] = [
        ([1.0, 2.5, 0.3, 7.25, 4.0, 0.125], [0.5, 1.5, 0.25, 3.0, 2.0, 0.0625]),
        ([1.0; 6], [0.6; 6]),
        ([1e-3, 1e3, 1e-3, 1e3, 1e-3, 1e3], [1e-4, 1e2, 1e-4, 1e2, 1e-4, 1e2]),
    ];
    let phi = [-2.5; 6];
    let mut worst_ulp = 0.0f64;
    for (q, p) in cases {
        for anchor in [2usize, 3] {
            let d = hydrostatic_deviations(q, phi, anchor, p[anchor], GAMMA, Thermodynamic::Density);
            assert_eq!(
                d[anchor], 0.0,
                "the anchor departure must be exactly zero — that is what carries T2"
            );
            let eq = symbi_hydro::hydrostatic::LocalEquilibrium::through(
                q[anchor], p[anchor], phi[anchor], GAMMA);
            let got = eq.density_at(phi[anchor]) + ppm_face(d);
            let want = ppm_face(q);
            let ulp = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE) / f64::EPSILON;
            worst_ulp = worst_ulp.max(ulp);
        }
    }
    println!("ppm gravity-free agreement: worst {worst_ulp:.2} ulp");
    // 8 ulp bounds a four-term weighted sum re-centred once: each term carries at most a
    // half-ulp from the shift and a half-ulp from the sum, and the cancellation in the
    // `7/12`/`-1/12` pairing can amplify that by the ratio of the summed magnitudes to the
    // result. it is a numerical statement about this interpolant, not a physical requirement,
    // and it is recorded with the measurement above so a regression shows as a number.
    assert!(
        worst_ulp < 8.0,
        "the parabolic path drifted {worst_ulp:.2} ulp from plain reconstruction with no \
         gravity; the departure transform is doing more than re-centring"
    );
}
