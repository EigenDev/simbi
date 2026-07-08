// =============================================================================
// dissipation_carrier_oracle.rs
//
// unit + carrier-equivalence coverage for the adaptive-dissipation detectors in
// `symbi_hydro::dissipation` (quirk_strong_shock, detect_shock, detect_interface,
// detect_alignment, adaptive_phi, local_mach). these are carrier-generic over
// `S: Scalar`, used in the HLLC riemann path, and the carrier gate (CLAUDE.md
// 4.3) demands a Gv-equivalence test that ALSO renders.
//
// two layers:
// - f64 unit tests that straddle every codim-1 branch-switch surface (the cmp/
//   select thresholds): strong vs weak shock, aligned vs misaligned, interface vs
//   smooth, plus the threshold-boundary flips and the division-by-zero guards.
// - a carrier oracle per function: trace at S = Gv, scalarize, CPU-interpret, and
//   assert bit/ULP agreement with the SAME function at S = f64 on NON-trivial
//   states that drive the branches. an emit step proves the graph renders to
//   CPU + CUDA source.
//
// usage:
//  cargo test -p symbi-hydro --test dissipation_carrier_oracle --release
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::Tensor;
use symbi_hydro::dissipation::{
    adaptive_phi, detect_alignment, detect_interface, detect_shock, local_mach,
    quirk_strong_shock, QUIRK_THRESHOLD,
};
use symbi_hydro::state::Prim;
use symbi_ir::algebra::Scalar;
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::passes::scalarize::scalarize;
use symbi_ir::{
    begin_trace, emit_kernel_cpu, emit_kernel_from_lowering, end_trace, Backend, Cpu, Gv,
    KernelEmitInputs,
};

// =============================================================================
// gv oracle harness — mirrors symbi-ir/tests/carrier_oracle_new.rs::gv_eval.
// trace a Scalar-generic fn at S = Gv that returns a single carrier scalar, then
// scalarize the graph and CPU-interpret it at the f64 inputs. tight tolerance: every
// op is f64 in both paths, so the carriers must agree to ULP modulo benign codegen.
// =============================================================================

/// run `physics(&[Gv]) -> Gv` at S = Gv, scalarize, and interpret at `inputs`.
fn gv_eval<F>(physics: F, param_names: &[&str], inputs: &[f64]) -> f64
where
    F: FnOnce(&[Gv]) -> Gv,
{
    begin_trace();
    let params: Vec<Gv> = param_names.iter().map(|n| Gv::param(n)).collect();
    let root = physics(&params).node();
    let kernel = end_trace();
    let lowered = scalarize(&kernel.graph, root, "dissipation_probe");
    Cpu.eval_elemental(&lowered, inputs)[0]
}

/// the LOWERABILITY half of the carrier gate: the traced single-root graph must
/// emit non-empty CPU (rust) AND CUDA source. an unlowerable op panics here.
fn assert_lowers<F>(physics: F, param_names: &[&str])
where
    F: FnOnce(&[Gv]) -> Gv,
{
    begin_trace();
    let params: Vec<Gv> = param_names.iter().map(|n| Gv::param(n)).collect();
    let _root = physics(&params).node();
    let kernel = end_trace();
    assert!(!kernel.graph.has_errors(), "graph errors: {:?}", kernel.graph.errors());
    let inputs = KernelEmitInputs {
        kernel_name: "dissipation_lower_probe",
        coalesce_layout: false,        ndim: 1,
        target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
        field_inputs: &[],
        scalar_params: &param_names.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
        field_writes: &[],
        coord_components: &[],
        device_preamble: &[],
        tile_spec: None,
    };
    let cpu = emit_kernel_cpu(&kernel.graph, &inputs);
    let cuda = emit_kernel_from_lowering(&kernel.graph, &inputs);
    assert!(!cpu.source.is_empty(), "lowerability: CPU (rust) emit produced no source");
    assert!(!cuda.source.is_empty(), "lowerability: CUDA emit produced no source");
}

const GAMMA: f64 = 1.4;

// the carrier-generic detectors take Prim<S, D> / Tensor<S, D>. these builders
// assemble those structs from a flat carrier-param slice so ONE physics body runs
// at both S = f64 and S = Gv. layout for D = 2: [rho_l, vx_l, vy_l, pre_l, rho_r,
// vx_r, vy_r, pre_r, nx, ny], gamma appended as p[10] when present.
const D: usize = 2;

fn prim_l<S: Scalar>(p: &[S]) -> Prim<S, D> {
    Prim { rho: p[0], vel: Tensor::new([p[1], p[2]]), pre: p[3] }
}
fn prim_r<S: Scalar>(p: &[S]) -> Prim<S, D> {
    Prim { rho: p[4], vel: Tensor::new([p[5], p[6]]), pre: p[7] }
}
fn nhat<S: Scalar>(p: &[S]) -> Tensor<S, D> {
    Tensor::new([p[8], p[9]])
}

const PARAM_NAMES: [&str; 10] =
    ["rho_l", "vx_l", "vy_l", "pre_l", "rho_r", "vx_r", "vy_r", "pre_r", "nx", "ny"];

// pack an f64 state (L prim, R prim, nhat) into the flat param slice.
fn pack(l: &Prim<f64, D>, r: &Prim<f64, D>, n: &Tensor<f64, D>) -> [f64; 10] {
    [l.rho, l.vel[0], l.vel[1], l.pre, r.rho, r.vel[0], r.vel[1], r.pre, n[0], n[1]]
}

// names slice including the trailing gamma param (p[10]).
fn names_with_gamma() -> Vec<&'static str> {
    PARAM_NAMES.iter().copied().chain(["gamma"]).collect()
}

fn close(a: f64, b: f64, what: &str) {
    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
    assert!(rel < 1e-12, "{what}: f64 {a} != gv-interp {b} (rel {rel:e})");
}

// =============================================================================
// quirk_strong_shock — fires when |pr - pl| / min(pl, pr) > QUIRK_THRESHOLD.
// returns S::Mask; at f64 that is bool. BOTH sides of the threshold are exercised
// (1e-4) plus the symmetry pr<->pl, the boundary, and the smooth no-fire case.
// =============================================================================

#[test]
fn quirk_strong_shock_f64_branches() {
    // strong jump: huge pressure ratio -> fires.
    assert!(quirk_strong_shock::<f64>(1.0, 100.0));
    assert!(quirk_strong_shock::<f64>(100.0, 1.0)); // symmetric: |pr-pl|/min picks the small one

    // smooth: equal pressures -> jump 0 -> does NOT fire.
    assert!(!quirk_strong_shock::<f64>(1.0, 1.0));

    // straddle the threshold. with p_min = 1, jump = |pr - pl|.
    // pr = 1 + 2e-4 -> jump/min = 2e-4 > 1e-4 -> fires.
    assert!(quirk_strong_shock::<f64>(1.0, 1.0 + 2.0 * QUIRK_THRESHOLD));
    // pr = 1 + 0.5e-4 -> jump/min = 0.5e-4 < 1e-4 -> does NOT fire.
    assert!(!quirk_strong_shock::<f64>(1.0, 1.0 + 0.5 * QUIRK_THRESHOLD));

    // the boundary itself is strict-greater: exactly == threshold does NOT fire.
    assert!(!quirk_strong_shock::<f64>(1.0, 1.0 + QUIRK_THRESHOLD));
}

#[test]
fn quirk_strong_shock_carrier_equivalence() {
    // mask -> scalar via select(mask, 1, 0) so the trace returns a single carrier node.
    let probe = |pl: f64, pr: f64| {
        let want = f64::select(quirk_strong_shock::<f64>(pl, pr), 1.0, 0.0);
        let got = gv_eval(
            |p| Gv::select(quirk_strong_shock::<Gv>(p[0], p[1]), Gv::ONE, Gv::ZERO),
            &["pl", "pr"],
            &[pl, pr],
        );
        close(want, got, &format!("quirk_strong_shock({pl},{pr})"));
    };
    // both branches of the select must be driven across calls.
    probe(1.0, 100.0); // fires
    probe(1.0, 1.0); // smooth
    probe(1.0, 1.0 + 2.0 * QUIRK_THRESHOLD); // just over
    probe(1.0, 1.0 + 0.5 * QUIRK_THRESHOLD); // just under
    probe(50.0, 2.0); // reversed strong
}

#[test]
fn quirk_strong_shock_lowers() {
    assert_lowers(
        |p| Gv::select(quirk_strong_shock::<Gv>(p[0], p[1]), Gv::ONE, Gv::ZERO),
        &["pl", "pr"],
    );
}

// =============================================================================
// local_mach — max of left/right |v| / cs, cs = sqrt(gamma * pre / rho). smooth
// (no branch besides the inner max); covered for completeness + carrier match.
// =============================================================================

#[test]
fn local_mach_f64_picks_max_side() {
    // left supersonic, right subsonic -> the left mach dominates the max.
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let fast = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([3.0 * cs, 0.0]), pre: 1.0 };
    let slow = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.1 * cs, 0.0]), pre: 1.0 };
    let ma = local_mach(&fast, &slow, GAMMA);
    assert!((ma - 3.0).abs() < 1e-12, "expected mach ~3, got {ma}");
    // symmetric: swapping sides keeps the max.
    assert!((local_mach(&slow, &fast, GAMMA) - 3.0).abs() < 1e-12);
}

#[test]
fn local_mach_carrier_equivalence() {
    let l = Prim::<f64, D> { rho: 1.2, vel: Tensor::new([0.8, -0.3]), pre: 0.9 };
    let r = Prim::<f64, D> { rho: 0.7, vel: Tensor::new([1.5, 0.4]), pre: 0.4 };
    let n = Tensor::<f64, D>::unit(0);
    let want = local_mach(&l, &r, GAMMA);
    let inputs = [pack(&l, &r, &n).to_vec(), vec![GAMMA]].concat();
    let got = gv_eval(|p| local_mach(&prim_l(p), &prim_r(p), p[10]), &names_with_gamma(), &inputs);
    close(want, got, "local_mach");
}

// =============================================================================
// detect_shock — AND of (entropy_production > 0.01) and (vn_l - vn_r > 0). returns
// 1.0 iff both hold, else 0.0. four corners of the 2x2 truth table, plus both
// threshold boundaries (the codim-1 select-flip surfaces).
// =============================================================================

// build (L, R) with prescribed entropy production and normal-velocity convergence
// along +x. entropy s = ln(pre) - gamma*ln(rho); fix rho = 1 so s = ln(pre). pick
// pre_l = 1 (s_l = 0), pre_r = exp(ds) (s_r = ds) -> entropy_production = ds.
// vn = vx along +x; vn_l - vn_r = dvx.
fn shock_state(ds: f64, dvx: f64) -> (Prim<f64, D>, Prim<f64, D>, Tensor<f64, D>) {
    let l = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([dvx, 0.0]), pre: 1.0 };
    let r = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: ds.exp() };
    (l, r, Tensor::unit(0))
}

#[test]
fn detect_shock_f64_truth_table() {
    // both conditions true: ds = 0.5 (> 0.01), dvx = 0.3 (> 0) -> 1.0
    let (l, r, n) = shock_state(0.5, 0.3);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 1.0);

    // entropy false (ds below 0.01), convergence true -> 0.0
    let (l, r, n) = shock_state(0.005, 0.3);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 0.0);

    // entropy true, convergence false (dvx <= 0, diverging) -> 0.0
    let (l, r, n) = shock_state(0.5, -0.2);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 0.0);

    // both false -> 0.0
    let (l, r, n) = shock_state(0.0, -0.2);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 0.0);
}

#[test]
fn detect_shock_f64_threshold_boundaries() {
    // entropy boundary is strict-greater at 0.01: ds just above fires, just below not.
    let (l, r, n) = shock_state(0.0101, 0.3);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 1.0);
    let (l, r, n) = shock_state(0.0099, 0.3);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 0.0);

    // convergence boundary is strict-greater at 0: tiny positive fires, zero does not.
    let (l, r, n) = shock_state(0.5, 1e-9);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 1.0);
    let (l, r, n) = shock_state(0.5, 0.0);
    assert_eq!(detect_shock(&l, &r, &n, GAMMA), 0.0);
}

#[test]
fn detect_shock_carrier_equivalence() {
    // sample all four corners so BOTH select arms of each condition are traced+driven.
    for &(ds, dvx) in &[(0.5, 0.3), (0.005, 0.3), (0.5, -0.2), (0.0, -0.2)] {
        let (l, r, n) = shock_state(ds, dvx);
        let want = detect_shock(&l, &r, &n, GAMMA);
        let inputs = [pack(&l, &r, &n).to_vec(), vec![GAMMA]].concat();
        let got = gv_eval(
            |p| detect_shock(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
            &names_with_gamma(),
            &inputs,
        );
        close(want, got, &format!("detect_shock(ds={ds}, dvx={dvx})"));
    }
}

#[test]
fn detect_shock_lowers() {
    assert_lowers(
        |p| detect_shock(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
        &names_with_gamma(),
    );
}

// =============================================================================
// detect_interface — 0.4 iff (rho_jump > 0.1) AND (pre_jump < 0.05), else 0.0.
// rho_jump = |rho_l - rho_r| / avg, pre_jump = |pre_l - pre_r| / avg. contact-
// discontinuity signature: large density contrast, near-uniform pressure.
// =============================================================================

#[test]
fn detect_interface_f64_truth_table() {
    let l = |rho: f64, pre: f64| Prim::<f64, D> { rho, vel: Tensor::zeros(), pre };

    // contact: big rho jump (1 vs 2 -> jump 0.667 > 0.1), tiny pre jump (1 vs 1.01 -> ~0.01 < 0.05)
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(2.0, 1.01)), 0.4);

    // smooth density (rho jump 0 < 0.1), small pre jump -> 0.0
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(1.0, 1.01)), 0.0);

    // big rho jump but big pre jump too (shock, not contact: pre_jump > 0.05) -> 0.0
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(2.0, 2.0)), 0.0);

    // neither -> 0.0
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(1.01, 2.0)), 0.0);
}

#[test]
fn detect_interface_f64_threshold_boundaries() {
    let l = |rho: f64, pre: f64| Prim::<f64, D> { rho, vel: Tensor::zeros(), pre };

    // rho_jump boundary at 0.1 (strict-greater). avg-normalized; pressure kept identical
    // (pre_jump = 0 < 0.05 so the pre condition always holds here, isolating the rho flip).
    // rho_r = 1.2 -> jump = 0.2 / 1.1 = 0.1818 > 0.1 -> fires.
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(1.2, 1.0)), 0.4);
    // rho_r = 1.05 -> jump = 0.05 / 1.025 = 0.0488 < 0.1 -> does NOT fire.
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(1.05, 1.0)), 0.0);

    // pre_jump boundary at 0.05 (strict-less). big rho jump fixed. pre_l = 1.
    // pre_r = 1.02 -> jump = 0.02 / 1.01 = 0.0198 < 0.05 -> fires.
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(2.0, 1.02)), 0.4);
    // pre_r = 1.1 -> jump = 0.1 / 1.05 = 0.0952 > 0.05 -> does NOT fire.
    assert_eq!(detect_interface(&l(1.0, 1.0), &l(2.0, 1.1)), 0.0);
}

#[test]
fn detect_interface_carrier_equivalence() {
    let cases: &[((f64, f64), (f64, f64))] = &[
        ((1.0, 1.0), (2.0, 1.01)), // contact -> 0.4
        ((1.0, 1.0), (1.0, 1.01)), // smooth -> 0.0
        ((1.0, 1.0), (2.0, 2.0)),  // shock -> 0.0
        ((1.0, 1.0), (1.01, 2.0)), // neither -> 0.0
    ];
    for &((rl, pl), (rr, pr)) in cases {
        let l = Prim::<f64, D> { rho: rl, vel: Tensor::zeros(), pre: pl };
        let r = Prim::<f64, D> { rho: rr, vel: Tensor::zeros(), pre: pr };
        let n = Tensor::<f64, D>::zeros();
        let want = detect_interface(&l, &r);
        let inputs = pack(&l, &r, &n).to_vec();
        let got = gv_eval(|p| detect_interface(&prim_l(p), &prim_r(p)), &PARAM_NAMES, &inputs);
        close(want, got, &format!("detect_interface(rho {rl}->{rr}, pre {pl}->{pr})"));
    }
}

#[test]
fn detect_interface_lowers() {
    assert_lowers(|p| detect_interface(&prim_l(p), &prim_r(p)), &PARAM_NAMES);
}

// =============================================================================
// detect_alignment — 1.0 iff |v_l| > eps AND |v_r| > eps AND max_align > 0.8 AND
// avg_mach > 0.5, else 0.0. max_align = max(|vn_l|/|v_l|, |vn_r|/|v_r|). exercises
// the zero-velocity division guard (the safe_v select) AND all four ANDed conds.
// =============================================================================

#[test]
fn detect_alignment_f64_truth_table() {
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    // aligned + fast: v along +x at mach ~1 -> align = 1 > 0.8, mach ~1 > 0.5 -> 1.0
    let fast_aligned = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([1.0 * cs, 0.0]), pre: 1.0 };
    assert_eq!(detect_alignment(&fast_aligned, &fast_aligned, &n, GAMMA), 1.0);

    // misaligned: v purely transverse (+y) -> align = 0 < 0.8 -> 0.0 (even if fast)
    let fast_transverse = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.0, 1.0 * cs]), pre: 1.0 };
    assert_eq!(detect_alignment(&fast_transverse, &fast_transverse, &n, GAMMA), 0.0);

    // aligned but slow: mach ~0.1 < 0.5 -> 0.0
    let slow_aligned = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.1 * cs, 0.0]), pre: 1.0 };
    assert_eq!(detect_alignment(&slow_aligned, &slow_aligned, &n, GAMMA), 0.0);
}

#[test]
fn detect_alignment_f64_zero_velocity_guard() {
    // |v| = 0 on both sides: the safe_v select replaces |v| with 1, but c_vl/c_vr gate
    // to 0 -> result MUST be 0.0 (no NaN, no spurious fire). this is the division-guard
    // failure mode the branchless guard exists to prevent.
    let n = Tensor::<f64, D>::unit(0);
    let still = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 };
    let out = detect_alignment(&still, &still, &n, GAMMA);
    assert_eq!(out, 0.0);
    assert!(out.is_finite(), "zero-velocity must not produce NaN/Inf");

    // one side still, the other fast+aligned: c_vl (still side) gates to 0 -> 0.0.
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let fast = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([cs, 0.0]), pre: 1.0 };
    assert_eq!(detect_alignment(&still, &fast, &n, GAMMA), 0.0);
}

#[test]
fn detect_alignment_f64_align_boundary() {
    // max_align boundary at 0.8 (strict-greater). build v with a known cos to nhat:
    // v = (cos*|v|, sin*|v|). align = |cos|. choose |v| = 1.5*cs so mach (~1.5) > 0.5.
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let vmag = 1.5 * cs;
    let with_cos = |c: f64| {
        let s = (1.0 - c * c).sqrt();
        Prim::<f64, D> { rho: 1.0, vel: Tensor::new([c * vmag, s * vmag]), pre: 1.0 }
    };
    // cos = 0.85 > 0.8 -> aligned -> 1.0
    let p = with_cos(0.85);
    assert_eq!(detect_alignment(&p, &p, &n, GAMMA), 1.0);
    // cos = 0.75 < 0.8 -> not aligned -> 0.0
    let p = with_cos(0.75);
    assert_eq!(detect_alignment(&p, &p, &n, GAMMA), 0.0);
}

#[test]
fn detect_alignment_carrier_equivalence() {
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let n = Tensor::<f64, D>::unit(0);
    let mk = |vx: f64, vy: f64| Prim::<f64, D> { rho: 1.0, vel: Tensor::new([vx, vy]), pre: 1.0 };
    // drive: aligned-fast (fires), transverse-fast (align gate), slow-aligned (mach gate),
    // zero-velocity (the safe_v division guard + c_vl/c_vr gates), mixed still/fast.
    let cases: &[(Prim<f64, D>, Prim<f64, D>)] = &[
        (mk(cs, 0.0), mk(cs, 0.0)),
        (mk(0.0, cs), mk(0.0, cs)),
        (mk(0.1 * cs, 0.0), mk(0.1 * cs, 0.0)),
        (mk(0.0, 0.0), mk(0.0, 0.0)),
        (mk(0.0, 0.0), mk(cs, 0.0)),
    ];
    for (l, r) in cases {
        let want = detect_alignment(l, r, &n, GAMMA);
        let inputs = [pack(l, r, &n).to_vec(), vec![GAMMA]].concat();
        let got = gv_eval(
            |p| detect_alignment(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
            &names_with_gamma(),
            &inputs,
        );
        close(want, got, &format!("detect_alignment(vl={:?}, vr={:?})", l.vel, r.vel));
    }
}

#[test]
fn detect_alignment_lowers() {
    assert_lowers(
        |p| detect_alignment(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
        &names_with_gamma(),
    );
}

// =============================================================================
// adaptive_phi — the composite: phi = max(sin(min(ma/0.1, 1) * pi/2), shock,
// interface, alignment), clamped to [0, 1]. exercises the low-mach ramp AND the
// detector-driven floors.
// =============================================================================

#[test]
fn adaptive_phi_f64_low_mach_ramp() {
    // smooth subsonic flow, no detector fires: phi follows the sin ramp.
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let slow = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([0.02 * cs, 0.0]), pre: 1.0 };
    let phi = adaptive_phi(&slow, &slow, &n, GAMMA);
    assert!(phi > 0.0 && phi < 1.0, "low-mach phi must be in (0,1), got {phi}");
    // ma = 0.02 -> ratio = 0.2 -> phi_ramp = sin(0.1*pi). detectors all 0 here.
    let want = ((0.02_f64 / 0.1).min(1.0) * std::f64::consts::FRAC_PI_2).sin();
    assert!((phi - want).abs() < 1e-12, "phi {phi} vs ramp {want}");
}

#[test]
fn adaptive_phi_f64_clamped_high_mach() {
    // supersonic: ma >> mach_lim -> ratio clamps to 1 -> sin(pi/2) = 1 -> phi = 1.
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let fast = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([5.0 * cs, 0.0]), pre: 1.0 };
    let phi = adaptive_phi(&fast, &fast, &n, GAMMA);
    assert!((phi - 1.0).abs() < 1e-12, "high-mach phi must clamp to 1, got {phi}");
}

#[test]
fn adaptive_phi_f64_detector_floor_dominates_ramp() {
    // a smooth LOW-mach flow whose interface detector fires: phi must be floored at 0.4
    // (the interface weight), ABOVE the tiny low-mach ramp value. proves the max() wiring.
    let n = Tensor::<f64, D>::unit(0);
    let l = Prim::<f64, D> { rho: 1.0, vel: Tensor::new([1e-3, 0.0]), pre: 1.0 };
    let r = Prim::<f64, D> { rho: 2.0, vel: Tensor::new([1e-3, 0.0]), pre: 1.005 };
    let phi = adaptive_phi(&l, &r, &n, GAMMA);
    assert!((phi - 0.4).abs() < 1e-12, "interface floor must give phi=0.4, got {phi}");
}

#[test]
fn adaptive_phi_carrier_equivalence() {
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let n = Tensor::<f64, D>::unit(0);
    let mk = |rho: f64, vx: f64, pre: f64| Prim::<f64, D> {
        rho,
        vel: Tensor::new([vx, 0.0]),
        pre,
    };
    // drive every contributing path: low-mach ramp, high-mach clamp, interface floor,
    // shock floor.
    let cases: &[(Prim<f64, D>, Prim<f64, D>)] = &[
        (mk(1.0, 0.02 * cs, 1.0), mk(1.0, 0.02 * cs, 1.0)), // low-mach ramp
        (mk(1.0, 5.0 * cs, 1.0), mk(1.0, 5.0 * cs, 1.0)),   // high-mach clamp
        (mk(1.0, 1e-3, 1.0), mk(2.0, 1e-3, 1.005)),         // interface floor
        (mk(1.0, 0.5 * cs, 1.0), mk(1.0, 0.0, 3.0)),        // shock-ish (entropy + converge)
    ];
    for (l, r) in cases {
        let want = adaptive_phi(l, r, &n, GAMMA);
        let inputs = [pack(l, r, &n).to_vec(), vec![GAMMA]].concat();
        let got = gv_eval(
            |p| adaptive_phi(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
            &names_with_gamma(),
            &inputs,
        );
        close(want, got, &format!("adaptive_phi(rho_l={}, rho_r={})", l.rho, r.rho));
    }
}

#[test]
fn adaptive_phi_lowers() {
    assert_lowers(
        |p| adaptive_phi(&prim_l(p), &prim_r(p), &nhat(p), p[10]),
        &names_with_gamma(),
    );
}
