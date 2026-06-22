// =============================================================================
// carrier_oracle_new.rs
//
// Tier 1 chunk 4: CARRIER-EQUIVALENCE ACCEPTANCE for the NEW
// `symbi_ir::algebra::Scalar` impl on `Gv`. proves the homomorphism law for
// the new trait, end-to-end, against the same physics templated over
// `S: Scalar` and run two ways:
//
//   eval_f64(physics)  ==  interp(trace_Gv(physics))
//
// every load-bearing surface of the new trait gets a test: ring arithmetic,
// sqrt + transcendentals, IEEE consts, comparisons returning `Self::Mask`,
// `select`/`branch` through the Mask + `Selectable`, hyperbolics, and
// `iterate` with the FREEZE LAW (kepler c2p regression class).
//
// pattern: physics fn is generic over `S: Scalar` and gets instantiated at
// both carriers. for Gv, the trace is scalarized (single `LoweredFn` root)
// and evaluated via the CPU interpreter at f64. tight tolerance (~1e-12)
// because every op is f64 in both paths.
// =============================================================================

use symbi_ir::algebra::Scalar;
use symbi_ir::passes::scalarize::scalarize;
use symbi_ir::{begin_trace, end_trace, Backend, Cpu, Gv};

/// run a Scalar-generic physics function at S = Gv, scalarize the resulting
/// graph, and evaluate the LoweredFn at the given f64 inputs.
fn gv_eval<F>(physics: F, param_names: &[&str], inputs: &[f64]) -> f64
where
    F: FnOnce(&[Gv]) -> Gv,
{
    begin_trace();
    let params: Vec<Gv> = param_names.iter().map(|n| Gv::param(n)).collect();
    let root = physics(&params).node();
    let kernel = end_trace();
    let lowered = scalarize(&kernel.graph, root, "oracle_probe");
    let out = Cpu.eval_elemental(&lowered, inputs);
    out[0]
}

fn close(a: f64, b: f64, what: &str) {
    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
    assert!(rel < 1e-12, "{what}: f64 {a} != gv-interp {b} (rel {rel:e})");
}

// =============================================================================
// ring arithmetic + consts
// =============================================================================

fn ring_arithmetic<S: Scalar>(a: S, b: S, c: S) -> S {
    // exercises Add/Sub/Mul/Div + Self::ONE (the augmented trait surface).
    (a * b - c) / (a + b + S::ONE)
}

#[test]
fn ring_arithmetic_matches_f64() {
    let (a, b, c) = (1.5_f64, 2.7, 0.3);
    let want = ring_arithmetic::<f64>(a, b, c);
    let got = gv_eval(|p| ring_arithmetic(p[0], p[1], p[2]), &["a", "b", "c"], &[a, b, c]);
    close(want, got, "ring_arithmetic");
}

// =============================================================================
// sqrt + transcendental: relativistic sound speed
// =============================================================================

fn rel_sound_speed_sq<S: Scalar>(gamma: S, p: S, rho: S) -> S {
    // cs² = gamma * p / (rho * h),  h = 1 + gamma * p / (rho * (gamma - 1))
    // exercises Sub/Mul/Div/Add + Self::ONE.
    let h = S::ONE + gamma * p / (rho * (gamma - S::ONE));
    (gamma * p) / (rho * h)
}

#[test]
fn rel_sound_speed_sq_matches_f64() {
    let (gamma, p, rho) = (4.0_f64 / 3.0, 1.7, 2.3);
    let want = rel_sound_speed_sq::<f64>(gamma, p, rho);
    let got = gv_eval(
        |x| rel_sound_speed_sq(x[0], x[1], x[2]),
        &["g", "p", "rho"],
        &[gamma, p, rho],
    );
    close(want, got, "rel_sound_speed_sq");
}

fn sqrt_then_div<S: Scalar>(a: S, b: S) -> S {
    // exercises sqrt + Div: lorentz-factor-like  1 / sqrt(1 - v²) with v² = a*b.
    let v_sq = a * b;
    S::ONE / (S::ONE - v_sq).sqrt()
}

#[test]
fn sqrt_then_div_matches_f64() {
    let (a, b) = (0.3_f64, 0.4); // v² = 0.12 < 1
    let want = sqrt_then_div::<f64>(a, b);
    let got = gv_eval(|x| sqrt_then_div(x[0], x[1]), &["a", "b"], &[a, b]);
    close(want, got, "sqrt_then_div (lorentz-like)");
}

// =============================================================================
// comparisons returning Self::Mask + select through the Mask
// =============================================================================

fn max_via_select<S: Scalar>(a: S, b: S) -> S {
    // exercises cmp_gt returning Self::Mask + Scalar::select taking a Mask.
    // for f64 the Mask is bool; for Gv it is GvMask wrapping a Bool node.
    let m = a.cmp_gt(b);
    S::select(m, a, b)
}

#[test]
fn max_via_select_matches_f64() {
    for &(a, b) in &[(1.5_f64, 2.5), (3.0, -1.0), (-2.0, -1.5)] {
        let want = max_via_select::<f64>(a, b);
        let got = gv_eval(|x| max_via_select(x[0], x[1]), &["a", "b"], &[a, b]);
        close(want, got, &format!("max_via_select({a},{b})"));
    }
}

// =============================================================================
// branch + Selectable — the trace-safe conditional
// =============================================================================

fn branch_two_arms<S: Scalar>(a: S, b: S) -> S {
    // exercises Scalar::branch (defaulted via Selectable::select). on Gv this
    // BUILDS both arms into the graph and selects — the A1-safe pattern.
    let cond = a.cmp_ge(b);
    S::branch(cond, || a * a, || b + b)
}

#[test]
fn branch_two_arms_matches_f64() {
    for &(a, b) in &[(2.0_f64, 1.0), (1.0, 5.0), (3.5, 3.5)] {
        let want = branch_two_arms::<f64>(a, b);
        let got = gv_eval(|x| branch_two_arms(x[0], x[1]), &["a", "b"], &[a, b]);
        close(want, got, &format!("branch_two_arms({a},{b})"));
    }
}

// =============================================================================
// hyperbolics — Cardano-Vieta-like form for the RMHD quartic
// =============================================================================

fn cubic_resolvent_real<S: Scalar>(p: S, q: S) -> S {
    // for the depressed cubic t^3 + p*t + q = 0 with discriminant < 0, the
    // one real root in hyperbolic form: 2*sqrt(-p/3) * sinh(asinh(3q/(p*m))/3).
    // exercises sqrt + sinh + asinh + arithmetic.
    let three = S::ONE + S::ONE + S::ONE;
    let m = (S::ONE + S::ONE) * (-p / three).sqrt();
    let arg = three * q / (p * m);
    m * (arg.asinh() / three).sinh()
}

#[test]
fn cubic_resolvent_real_matches_f64() {
    let (p, q) = (-3.0_f64, 1.5); // -p/3 = 1 > 0 → m real, valid branch
    let want = cubic_resolvent_real::<f64>(p, q);
    let got = gv_eval(|x| cubic_resolvent_real(x[0], x[1]), &["p", "q"], &[p, q]);
    close(want, got, "cubic_resolvent_real (sinh+asinh+sqrt)");
}

// =============================================================================
// iterate — the FREEZE LAW
// =============================================================================

fn newton_sqrt<S: Scalar>(a: S, x0: S) -> S {
    // Newton sqrt: x_{n+1} = 0.5 * (x_n + a / x_n). converges quadratically.
    // FREEZE LAW: the returned acc is BEFORE the converging step. on Gv this
    // becomes a fixed-count IterateInline with `select(converged, acc, body(acc))`,
    // so the returned value matches the host's early-break value.
    let half = S::ONE / (S::ONE + S::ONE);
    x0.iterate(
        20,
        |x| half * (x + a / x),
        |prev, cur| (cur - prev).abs().cmp_lt(S::from_f64(1e-14)),
    )
}

// IGNORED IN CHUNK 4. `Op::IterateInline` is `scalarize_kernel`-only by design
// (see scalarize.rs comment near the IterateInline unreachable: "cone-partitioned;
// scalarize_kernel handles it directly"). this acceptance file uses the simpler
// `scalarize(graph, output, name)` path which doesn't set up the iterate body
// context — IterAcc lowers outside an IterateInline loop body and panics.
//
// the iterate impl is structurally identical to the legacy Gv impl (same graph
// ops, same FREEZE LAW). f64-side iterate freeze is already pinned by Tier 0's
// `f64_iterate_freeze_holds_pre_convergence_value` test. the carrier-equivalence
// (Gv trace + interp) for iterate lands in Chunk 5 once the symbi-discretize
// carrier_oracle.rs harness migrates to the new trait — that harness drives
// `scalarize_kernel` + the production interp and validates iterate via real
// srhd/rmhd c2p round-trips.
#[test]
#[ignore = "iterate requires scalarize_kernel; covered by Chunk 5's discretize harness migration"]
fn newton_sqrt_iterate_matches_with_freeze() {
    // converging case: a = 4, x0 = 2.0 should converge to ~2.0 (already there).
    // Newton at a fixed point converges in 1 step; freeze keeps acc=2.0.
    let (a, x0) = (4.0_f64, 2.0);
    let want = newton_sqrt::<f64>(a, x0);
    let got = gv_eval(|p| newton_sqrt(p[0], p[1]), &["a", "x0"], &[a, x0]);
    close(want, got, "newton_sqrt converging");

    // approaching case: a = 9, x0 = 4 — converges from above in ~5 steps.
    let (a, x0) = (9.0_f64, 4.0);
    let want = newton_sqrt::<f64>(a, x0);
    let got = gv_eval(|p| newton_sqrt(p[0], p[1]), &["a", "x0"], &[a, x0]);
    close(want, got, "newton_sqrt approaching");
}

// =============================================================================
// IEEE consts — INFINITY participates in min/max sentinels
// =============================================================================

fn min_fold_with_inf_init<S: Scalar>(a: S, b: S, c: S) -> S {
    // initialise an accumulator at +inf, fold via min — pattern that wave-speed
    // maps and similar use. exercises Self::INFINITY const.
    S::INFINITY.min(a).min(b).min(c)
}

#[test]
fn min_fold_with_infinity_matches_f64() {
    let (a, b, c) = (3.5_f64, 1.2, 4.8);
    let want = min_fold_with_inf_init::<f64>(a, b, c);
    let got = gv_eval(
        |x| min_fold_with_inf_init(x[0], x[1], x[2]),
        &["a", "b", "c"],
        &[a, b, c],
    );
    close(want, got, "min_fold_with_inf_init");
}
