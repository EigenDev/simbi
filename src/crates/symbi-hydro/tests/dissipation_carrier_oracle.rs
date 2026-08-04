// =============================================================================
// dissipation_carrier_oracle.rs
//
// unit + carrier-equivalence coverage for the adaptive-dissipation detectors in
// `symbi_hydro::dissipation` (adaptive_phi, local_mach).
// these are carrier-generic over `S: Scalar` and used in the HLLC riemann path, so each one
// needs a Gv-equivalence test that ALSO renders to device source.
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

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_hydro::dissipation::{adaptive_phi, local_mach};
use symbi_hydro::state::Prim;
use symbi_ir::algebra::Scalar;
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::passes::scalarize::scalarize;
use symbi_ir::{
    Backend, Cpu, Gv, KernelEmitInputs, begin_trace, emit_kernel_cpu, emit_kernel_from_lowering,
    end_trace,
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
    assert!(
        !kernel.graph.has_errors(),
        "graph errors: {:?}",
        kernel.graph.errors()
    );
    let inputs = KernelEmitInputs {
        kernel_name: "dissipation_lower_probe",
        coalesce_layout: false,
        ndim: 1,
        target: TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        },
        field_inputs: &[],
        scalar_params: &param_names
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        field_writes: &[],
        coord_components: &[],
        device_preamble: &[],
        tile_spec: None,
    };
    let cpu = emit_kernel_cpu(&kernel.graph, &inputs);
    let cuda = emit_kernel_from_lowering(&kernel.graph, &inputs);
    assert!(
        !cpu.source.is_empty(),
        "lowerability: CPU (rust) emit produced no source"
    );
    assert!(
        !cuda.source.is_empty(),
        "lowerability: CUDA emit produced no source"
    );
}

const GAMMA: f64 = 1.4;

// the carrier-generic detectors take Prim<S, D> / Tensor<S, D>. these builders
// assemble those structs from a flat carrier-param slice so ONE physics body runs
// at both S = f64 and S = Gv. layout for D = 2: [rho_l, vx_l, vy_l, pre_l, rho_r,
// vx_r, vy_r, pre_r, nx, ny], gamma appended as p[10] when present.
const D: usize = 2;

fn prim_l<S: Scalar>(p: &[S]) -> Prim<S, D> {
    Prim {
        rho: p[0],
        vel: Tensor::new([p[1], p[2]]),
        pre: p[3],
    }
}
fn prim_r<S: Scalar>(p: &[S]) -> Prim<S, D> {
    Prim {
        rho: p[4],
        vel: Tensor::new([p[5], p[6]]),
        pre: p[7],
    }
}
fn nhat<S: Scalar>(p: &[S]) -> Tensor<S, D> {
    Tensor::new([p[8], p[9]])
}

const PARAM_NAMES: [&str; 10] = [
    "rho_l", "vx_l", "vy_l", "pre_l", "rho_r", "vx_r", "vy_r", "pre_r", "nx", "ny",
];

// pack an f64 state (L prim, R prim, nhat) into the flat param slice.
fn pack(l: &Prim<f64, D>, r: &Prim<f64, D>, n: &Tensor<f64, D>) -> [f64; 10] {
    [
        l.rho, l.vel[0], l.vel[1], l.pre, r.rho, r.vel[0], r.vel[1], r.pre, n[0], n[1],
    ]
}

// names slice including the trailing gamma param (p[10]).
fn names_with_gamma() -> Vec<&'static str> {
    PARAM_NAMES.iter().copied().chain(["gamma"]).collect()
}

/// the solver's OWN call shape for the scaling: project both velocities onto the face normal and
/// take each side's sound speed from its regime. kept here so the tests below read in terms of
/// states while exercising the same scalar interface the riemann solvers call.
fn phi_of(l: &Prim<f64, D>, r: &Prim<f64, D>, n: &Tensor<f64, D>) -> f64 {
    let cs = |p: &Prim<f64, D>| (GAMMA * p.pre / p.rho).sqrt();
    adaptive_phi(l.vel.dot(n), r.vel.dot(n), cs(l), cs(r), l.pre, r.pre)
}

/// the same shape, carrier-generic, for the Gv lowering checks.
fn phi_of_gv<S: symbi_hydro::Scalar>(p: &[S]) -> S {
    let (pl, pr, nh) = (prim_l(p), prim_r(p), nhat(p));
    let cs = |q: &Prim<S, D>| (p[10] * q.pre / q.rho).sqrt();
    adaptive_phi(pl.vel.dot(&nh), pr.vel.dot(&nh), cs(&pl), cs(&pr), pl.pre, pr.pre)
}

fn close(a: f64, b: f64, what: &str) {
    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
    assert!(
        rel < 1e-12,
        "{what}: f64 {a} != gv-interp {b} (rel {rel:e})"
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
    let fast = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([3.0 * cs, 0.0]),
        pre: 1.0,
    };
    let slow = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([0.1 * cs, 0.0]),
        pre: 1.0,
    };
    let cs = |p: &Prim<f64, D>| (GAMMA * p.pre / p.rho).sqrt();
    let ma = local_mach(fast.vel[0], slow.vel[0], cs(&fast), cs(&slow));
    assert!((ma - 3.0).abs() < 1e-12, "expected mach ~3, got {ma}");
    // symmetric: swapping sides keeps the max.
    assert!((local_mach(slow.vel[0], fast.vel[0], cs(&slow), cs(&fast)) - 3.0).abs() < 1e-12);
}

#[test]
fn local_mach_carrier_equivalence() {
    let l = Prim::<f64, D> {
        rho: 1.2,
        vel: Tensor::new([0.8, -0.3]),
        pre: 0.9,
    };
    let r = Prim::<f64, D> {
        rho: 0.7,
        vel: Tensor::new([1.5, 0.4]),
        pre: 0.4,
    };
    let n = Tensor::<f64, D>::unit(0);
    // the solver projects the velocity onto the face normal and takes each side's own sound speed
    // from its regime before calling in; the carrier check follows the same shape.
    let cs = |p: &Prim<f64, D>| (GAMMA * p.pre / p.rho).sqrt();
    let vn = |p: &Prim<f64, D>| p.vel.dot(&n);
    let want = local_mach(vn(&l), vn(&r), cs(&l), cs(&r));
    let inputs = [pack(&l, &r, &n).to_vec(), vec![GAMMA]].concat();
    let got = gv_eval(
        |p| {
            let (pl, pr, nh) = (prim_l(p), prim_r(p), nhat(p));
            let csg = |q: &Prim<_, D>| (p[10] * q.pre / q.rho).sqrt();
            local_mach(pl.vel.dot(&nh), pr.vel.dot(&nh), csg(&pl), csg(&pr))
        },
        &names_with_gamma(),
        &inputs,
    );
    close(want, got, "local_mach");
}

// =============================================================================
// adaptive_phi — the acoustic-dissipation scaling, and nothing else:
// phi = sin(min(Ma_local / 0.1, 1) * pi/2), with Ma_local the FACE-NORMAL mach number.
// =============================================================================

#[test]
fn adaptive_phi_f64_low_mach_ramp() {
    // smooth subsonic flow, no detector fires: phi follows the sin ramp.
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let slow = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([0.02 * cs, 0.0]),
        pre: 1.0,
    };
    let phi = phi_of(&slow, &slow, &n);
    assert!(
        phi > 0.0 && phi < 1.0,
        "low-mach phi must be in (0,1), got {phi}"
    );
    // ma = 0.02 -> ratio = 0.2 -> phi_ramp = sin(0.1*pi). detectors all 0 here.
    let want = ((0.02_f64 / 0.1).min(1.0) * std::f64::consts::FRAC_PI_2).sin();
    assert!((phi - want).abs() < 1e-12, "phi {phi} vs ramp {want}");
}

#[test]
fn adaptive_phi_f64_clamped_high_mach() {
    // supersonic: ma >> mach_lim -> ratio clamps to 1 -> sin(pi/2) = 1 -> phi = 1.
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let fast = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([5.0 * cs, 0.0]),
        pre: 1.0,
    };
    let phi = phi_of(&fast, &fast, &n);
    assert!(
        (phi - 1.0).abs() < 1e-12,
        "high-mach phi must clamp to 1, got {phi}"
    );
}

#[test]
fn adaptive_phi_is_the_sine_ramp_on_pressure_uniform_faces() {
    // `phi = sin(min(1, Ma_local / Ma_limit) * pi/2)` — Fleischmann, Adami & Adams 2020 eq 24 —
    // against the closed form at several mach numbers, on faces with NO pressure jump. there the
    // compressibility-consistency clamp is an exact zero, so the ramp is the whole law and the
    // scheme's low dissipation is fully realized: equal-pressure faces are the contact
    // discontinuities and shock-transverse faces the reduction exists for, and nothing may add
    // dissipation there. (faces whose pressure jump exceeds the incompressible `dp/p ~ gamma Ma^2`
    // scale are OUTSIDE the ramp's derivation and are governed by the clamp — see the
    // stratified-face tests below and the sealed-column entropy floor law in symbi.)
    let n = Tensor::<f64, D>::unit(0);
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let at = |ma: f64| {
        let s = Prim::<f64, D> {
            rho: 1.0,
            vel: Tensor::new([ma * cs, 0.0]),
            pre: 1.0,
        };
        phi_of(&s, &s, &n)
    };
    let want = |ma: f64| ((ma / 0.1).min(1.0) * std::f64::consts::FRAC_PI_2).sin();
    for ma in [0.0, 1e-4, 0.01, 0.05, 0.099, 0.1, 0.5, 3.0] {
        let (got, expect) = (at(ma), want(ma));
        assert!(
            (got - expect).abs() < 1e-12,
            "Ma = {ma}: phi = {got}, eq 24 gives {expect}"
        );
    }

    // a density jump at fixed pressure and fixed low mach — a contact discontinuity — must NOT
    // raise phi. this is the configuration a "detect the interface and restore dissipation" term
    // fires on, and it is precisely where the scheme's low dissipation at the contact is the
    // benefit being sought.
    let l = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([1e-3, 0.0]),
        pre: 1.0,
    };
    let r = Prim::<f64, D> {
        rho: 2.0,
        vel: Tensor::new([1e-3, 0.0]),
        pre: 1.0,
    };
    let phi = phi_of(&l, &r, &n);
    // eq 25 takes the max over the two sides, each with its OWN sound speed — the denser side here
    // is the colder one, so it sets the mach number.
    let ma = (1e-3 / (GAMMA * l.pre / l.rho).sqrt()).max(1e-3 / (GAMMA * r.pre / r.rho).sqrt());
    let expect = want(ma);
    assert!(
        (phi - expect).abs() < 1e-12,
        "a contact discontinuity at Ma = {ma:.4e} gave phi = {phi}; eq 24 alone gives {expect}"
    );
}

#[test]
fn the_clamp_restores_dissipation_on_stratified_faces_and_only_there() {
    // the compressibility-consistency clamp: the ramp's derivation assumes the incompressible
    // fluctuation scaling `dp/p ~ gamma Ma^2`; a face whose pressure jump sits far above that
    // scale at low mach is a stratified balance (dp/p ~ dx/H, mach-independent), where cutting
    // the acoustic dissipation lets the hydrostatic residual ring and the entropy floor
    // `K >= K_0` fails on a sealed column. measured regimes this separates: an accretor
    // atmosphere carries dx/H = 0.075-0.25 per face at Ma ~ 0.025 (jump/scale ~ 100), while
    // mach-0.06 taylor-green turbulence carries jump/scale ~ 0.05.
    let n = Tensor::<f64, D>::unit(0);
    let face = |pre_r: f64, ma: f64| {
        let mk = |pre: f64| Prim::<f64, D> {
            rho: 1.0,
            vel: Tensor::new([ma * (GAMMA * 1.0_f64 / 1.0).sqrt(), 0.0]),
            pre,
        };
        phi_of(&mk(1.0), &mk(pre_r), &n)
    };
    // a hydrostatic-scale jump at deep low mach: full classical dissipation restored.
    assert_eq!(
        face(1.05, 0.025),
        1.0,
        "a dp/p = 0.05 face at Ma 0.025 is two orders above the incompressible scale and \
         must recover classical HLLC exactly"
    );
    // an incompressible-scale jump at the same mach: the clamp must lose to the ramp
    // bit-for-bit — inert means the same float, which is what keeps a low-mach turbulence
    // run byte-identical to the pure-ramp scheme.
    let ma = 0.06;
    let tiny_jump = 1.0 + 0.1 * GAMMA * ma * ma;
    let ramp_only = face(1.0, ma);
    assert_eq!(
        face(tiny_jump, ma),
        ramp_only,
        "an incompressible-scale pressure fluctuation engaged the clamp; smooth subsonic \
         turbulence would no longer be byte-stable"
    );
    // the clamp is dimensionless: rescaling both pressures together leaves phi unchanged —
    // exactly wherever the clamp is saturated, and to rounding (the `p_l - p_r` subtraction
    // re-rounds at each scale) on the ramp of the clamp itself.
    let scaled = |s: f64, hi: f64| {
        // the velocity rides each state's own sound speed so the mach number is held
        // fixed while the pressure scale sweeps — otherwise the probe moves along the
        // ramp and measures the mach dependence instead of the units dependence.
        let mk = |pre: f64| Prim::<f64, D> {
            rho: 1.0,
            vel: Tensor::new([0.025 * (GAMMA * pre / 1.0).sqrt(), 0.0]),
            pre,
        };
        phi_of(&mk(1.0 * s), &mk(hi * s), &n)
    };
    assert_eq!(
        scaled(1.0, 1.05),
        scaled(1e30, 1.05),
        "pressure units leaked into the saturated clamp"
    );
    let (a, b) = (scaled(1.0, 1.005), scaled(1e30, 1.005));
    assert!(
        ((a - b) / a).abs() < 1e-12,
        "pressure units leaked into the unsaturated clamp: {a} vs {b}"
    );
}

#[test]
fn the_mach_number_is_the_face_normal_component_not_the_speed() {
    // eq 25 is `Ma_local = max(|u_L/c_L|, |u_R/c_R|)`, where the paper states that `u` is "the
    // velocity component dependent on the direction of the cell-face Riemann problem".
    //
    // this is the whole mechanism. a grid-aligned shock (their fig 1) carries a large velocity along
    // its propagation direction and a vanishing component transverse to it, so the TRANSVERSE faces
    // run at a local mach number near zero — and it is the acoustic dissipation there that scales
    // wrongly and drives the instability. keyed on the SPEED instead, those faces read as
    // supersonic, phi = 1, and the correction does nothing on the only faces it was built for.
    let cs = (GAMMA * 1.0_f64 / 1.0).sqrt();
    let l = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([6.0 * cs, 1.0e-6]),
        pre: 1.0,
    };
    let r = Prim::<f64, D> {
        rho: 1.0,
        vel: Tensor::new([6.0 * cs, -1.0e-6]),
        pre: 1.0,
    };

    let along = Tensor::<f64, D>::unit(0);
    let across = Tensor::<f64, D>::unit(1);

    // along the shock normal the flow is supersonic: classical HLLC, exactly.
    let phi_along = phi_of(&l, &r, &along);
    assert!(
        (phi_along - 1.0).abs() < 1e-12,
        "the shock-normal face is at Ma = 6 and must recover classical HLLC, got phi = {phi_along}"
    );

    // across it the flow is nearly at rest, and the acoustic dissipation must collapse.
    let phi_across = phi_of(&l, &r, &across);
    assert!(
        phi_across < 1.0e-4,
        "the transverse face is at Ma ~ {:.1e} yet phi = {phi_across}; the mach number is being \
         taken from the SPEED rather than the face-normal component, so the low-mach correction is \
         inert on exactly the faces the scheme targets",
        1.0e-6 / cs
    );

    // and the two faces must genuinely disagree — a scaling that returned the same value in both
    // directions would satisfy neither the mechanism nor this test's intent.
    assert!(
        phi_along / phi_across > 1.0e3,
        "the two directions give nearly the same phi ({phi_along} vs {phi_across}); the scaling is \
         not direction-dependent at all"
    );
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
        let want = phi_of(&l, &r, &n);
        let inputs = [pack(l, r, &n).to_vec(), vec![GAMMA]].concat();
        let got = gv_eval(|p| phi_of_gv(p), &names_with_gamma(), &inputs);
        close(
            want,
            got,
            &format!("adaptive_phi(rho_l={}, rho_r={})", l.rho, r.rho),
        );
    }
}

#[test]
fn adaptive_phi_lowers() {
    assert_lowers(|p| phi_of_gv(p), &names_with_gamma());
}
