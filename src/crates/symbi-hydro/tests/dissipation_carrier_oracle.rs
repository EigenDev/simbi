// =============================================================================
// dissipation_carrier_oracle.rs
//
// unit + carrier-equivalence coverage for the adaptive-dissipation detectors in
// `symbi_hydro::dissipation` (fleischmann_phi, local_mach).
// these are carrier-generic over `S: Scalar` and used in the HLLC riemann path, so each one
// needs a Gv-equivalence test that also renders to device source.
//
// two layers:
// - f64 unit tests that straddle every codim-1 branch-switch surface (the cmp/
//   select thresholds): strong vs weak shock, aligned vs misaligned, interface vs
//   smooth, plus the threshold-boundary flips and the division-by-zero guards.
// - a carrier oracle per function: trace at S = Gv, scalarize, CPU-interpret, and
//   assert bit/ULP agreement with the same function at S = f64 on non-trivial
//   states that drive the branches. an emit step proves the graph renders to
//   CPU + CUDA source.
//
// usage:
//  cargo test -p symbi-hydro --test dissipation_carrier_oracle --release
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_hydro::dissipation::{MACH_LIMIT, fleischmann_phi, local_mach};
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

/// the lowerability half of the carrier gate: the traced single-root graph must
/// emit non-empty CPU (rust) and CUDA source. an unlowerable op panics here.
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
// assemble those structs from a flat carrier-param slice so one physics body runs
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

/// the solver's own call shape for the scaling: project both velocities onto the face normal and
/// take each side's sound speed from its regime. kept here so the tests below read in terms of
/// states while exercising the same scalar interface the riemann solvers call.
fn phi_of(l: &Prim<f64, D>, r: &Prim<f64, D>, n: &Tensor<f64, D>) -> f64 {
    let cs = |p: &Prim<f64, D>| (GAMMA * p.pre / p.rho).sqrt();
    fleischmann_phi(l.vel.dot(n), r.vel.dot(n), cs(l), cs(r), MACH_LIMIT)
}

/// the same shape, carrier-generic, for the Gv lowering checks.
fn phi_of_gv<S: symbi_hydro::Scalar>(p: &[S]) -> S {
    let (pl, pr, nh) = (prim_l(p), prim_r(p), nhat(p));
    let cs = |q: &Prim<S, D>| (p[10] * q.pre / q.rho).sqrt();
    fleischmann_phi(
        pl.vel.dot(&nh),
        pr.vel.dot(&nh),
        cs(&pl),
        cs(&pr),
        S::from_f64(MACH_LIMIT),
    )
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
// fleischmann_phi — the published acoustic-dissipation ramp, and nothing else:
// phi = sin(min(Ma_local / 0.1, 1) * pi/2), with Ma_local the face-normal mach number.
// =============================================================================

#[test]
fn ramp_f64_low_mach() {
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
fn ramp_f64_saturates_above_the_limit() {
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
fn ramp_is_the_published_sine_on_any_face() {
    // `phi = sin(min(1, Ma_local / Ma_limit) * pi/2)` — Fleischmann, Adami & Adams 2020 eq 24 —
    // against the closed form at several mach numbers, on faces with no pressure jump. there the
    // compressibility-consistency clamp is an exact zero, so the ramp is the whole law and the
    // scheme's low dissipation is fully realized: equal-pressure faces are the contact
    // discontinuities and shock-transverse faces the reduction exists for, and nothing may add
    // dissipation there. (faces whose pressure jump exceeds the incompressible `dp/p ~ gamma Ma^2`
    // scale are outside the ramp's derivation and are governed by the clamp — see the
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

    // a density jump at fixed pressure and fixed low mach — a contact discontinuity — must not
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
    // eq 25 takes the max over the two sides, each with its own sound speed — the denser side here
    // is the colder one, so it sets the mach number.
    let ma = (1e-3 / (GAMMA * l.pre / l.rho).sqrt()).max(1e-3 / (GAMMA * r.pre / r.rho).sqrt());
    let expect = want(ma);
    assert!(
        (phi - expect).abs() < 1e-12,
        "a contact discontinuity at Ma = {ma:.4e} gave phi = {phi}; eq 24 alone gives {expect}"
    );
}

#[test]
fn the_mach_number_is_the_face_normal_component_not_the_speed() {
    // eq 25 is `Ma_local = max(|u_L/c_L|, |u_R/c_R|)`, where the paper states that `u` is "the
    // velocity component dependent on the direction of the cell-face Riemann problem".
    //
    // this is the whole mechanism. a grid-aligned shock (their fig 1) carries a large velocity along
    // its propagation direction and a vanishing component transverse to it, so the transverse faces
    // run at a local mach number near zero — and it is the acoustic dissipation there that scales
    // wrongly and drives the instability. keyed on the speed instead, those faces read as
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
fn ramp_carrier_equivalence() {
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
            &format!("fleischmann_phi(rho_l={}, rho_r={})", l.rho, r.rho),
        );
    }
}

#[test]
fn ramp_lowers() {
    assert_lowers(|p| phi_of_gv(p), &names_with_gamma());
}

// =============================================================================
// acoustic_phi — the knob-free acoustic-consistency scaling.
//
// the sensor scales the acoustic dissipation by how much of the face data obeys
// the impedance relation dp = rho c du, rather than by a reference mach number.
// these check that each flow regime lands where the derivation says it must,
// with no constant available to tune.
// =============================================================================
mod acoustic_sensor {
    use symbi_hydro::dissipation::{acoustic_phi, local_mach};

    const GAMMA: f64 = 5.0 / 3.0;

    /// the relative precision with which a face's pressure jump survives being formed
    /// from two absolute pressures. a low-mach face carries `dp/p ~ Ma^2`, so the
    /// subtraction loses `eps * p / dp` of it — the low-mach roundoff problem itself,
    /// present in the test's construction rather than in the sensor.
    fn jump_conditioning(p: f64, dp: f64) -> f64 {
        8.0 * f64::EPSILON * p / dp.abs()
    }

    /// a face in smooth low-mach flow: the pressure jump is the one the momentum
    /// balance supports across the velocity jump, `dp = rho u du`. returns the sensor
    /// value and the face's own `Ma_local`, since that — not the mean mach — is the
    /// quantity the scaling is defined against.
    fn smooth_face(mach: f64, du_over_u: f64) -> (f64, f64) {
        let (rho, p) = (1.0, 1.0);
        let cs = (GAMMA * p / rho).sqrt();
        let u = mach * cs;
        let du = du_over_u * u;
        let dp = rho * u * du;
        let (vl, vr) = (u + 0.5 * du, u - 0.5 * du);
        let phi = acoustic_phi(
            vl, vr, cs, cs, p + 0.5 * dp, p - 0.5 * dp, rho, rho, 0.0,
        );
        (phi, local_mach(vl, vr, cs, cs))
    }

    /// the low-mach requirement, checked as a scaling across decades: in smooth flow the
    /// unsupported pressure jump is `O(Ma^2)` and the mach term governs, so the sensor
    /// must equal the face's own local mach number with no offset and no threshold. a
    /// reference mach number anywhere inside would show up as a departure at the low end.
    #[test]
    fn smooth_low_mach_flow_is_scaled_by_the_local_mach_number() {
        for &mach in &[1.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.3] {
            let (phi, ma_local) = smooth_face(mach, 1.0);
            let rel = (phi - ma_local).abs() / ma_local;
            assert!(
                rel < 1.0e-12,
                "at Ma = {mach:e} the sensor gave phi = {phi:e} against a local mach of \
                 {ma_local:e} (relative departure {rel:e})"
            );
        }
    }

    /// no threshold: halving the flow speed halves the scaling, all the way down. a
    /// scheme with a reference mach number flattens out below it — this is what would
    /// catch one being reintroduced.
    #[test]
    fn the_scaling_is_linear_in_speed_with_no_threshold() {
        let mut previous: Option<f64> = None;
        for &mach in &[0.2, 0.1, 0.05, 0.025, 0.0125] {
            let (phi, _) = smooth_face(mach, 1.0);
            if let Some(prev) = previous {
                let ratio = prev / phi;
                assert!(
                    (ratio - 2.0).abs() < 1.0e-9,
                    "halving the speed changed the scaling by {ratio}, not 2 — the \
                     response is not linear, so a threshold has crept in"
                );
            }
            previous = Some(phi);
        }
    }

    /// saturation happens where the acoustic and advective scales genuinely meet, at
    /// `Ma_local = 1`, and not at a chosen fraction of it.
    #[test]
    fn the_scaling_saturates_at_unit_local_mach() {
        let (phi, ma) = smooth_face(0.5, 1.0);
        assert!(ma < 1.0 && (phi - ma).abs() < 1.0e-12, "phi {phi} ma {ma}");
        let (phi, ma) = smooth_face(1.5, 1.0);
        assert!(ma > 1.0 && phi == 1.0, "phi {phi} ma {ma}");
    }

    /// a wave obeying the impedance relation `dp = rho c du` is scaled by its own
    /// amplitude in acoustic units — a weak wave needs little dissipation, which is the
    /// whole point of a low-dissipation solver, and a wave whose velocity jump reaches
    /// the sound speed gets the classical dissipation in full.
    #[test]
    fn an_impedance_consistent_wave_is_scaled_by_its_amplitude() {
        let (rho, p) = (1.0, 1.0);
        let cs = (GAMMA * p / rho).sqrt();
        for &du in &[1.0e-6, 1.0e-3, 0.1] {
            let dp = rho * cs * du;
            let phi = acoustic_phi(
                0.5 * du, -0.5 * du, cs, cs, p + 0.5 * dp, p - 0.5 * dp, rho, rho, 0.0,
            );
            let expected = du / cs; // the jump term; it dominates the mach term (du/2cs)
            let tol = expected * jump_conditioning(p, dp).max(1.0e-12);
            assert!(
                (phi - expected).abs() <= tol,
                "a wave of amplitude du/c = {expected:e} was scaled by {phi:e}"
            );
        }
        // and a wave at the sound speed saturates
        let du = 2.0 * cs;
        let phi = acoustic_phi(
            0.5 * du, -0.5 * du, cs, cs, p + rho * cs * du, p, rho, rho, 0.0,
        );
        assert_eq!(phi, 1.0);
    }

    /// a contact carries a density jump and no pressure jump. at rest it presents
    /// nothing for the acoustic dissipation to act on and the scaling vanishes; in
    /// motion it is scaled by the flow's mach number like any other face. either way
    /// the contact itself rides on the contact wave, which this scaling never touches.
    #[test]
    fn a_contact_is_scaled_by_its_motion_and_never_more() {
        let cs_l = (GAMMA * 1.0 / 1.0f64).sqrt();
        let cs_r = (GAMMA * 1.0 / 4.0f64).sqrt();
        let at_rest = acoustic_phi(0.0, 0.0, cs_l, cs_r, 1.0, 1.0, 1.0, 4.0, 0.0);
        assert_eq!(at_rest, 0.0, "a contact at rest must not be dissipated");
        let moving = acoustic_phi(0.3, 0.3, cs_l, cs_r, 1.0, 1.0, 1.0, 4.0, 0.0);
        let expected = local_mach(0.3, 0.3, cs_l, cs_r);
        assert!(
            (moving - expected).abs() < 1.0e-12,
            "a moving contact was scaled by {moving} rather than by its mach {expected}"
        );
    }

    /// the carbuncle mechanism, as the paper states it: the face transverse to a
    /// grid-aligned shock sees a smooth front — near-uniform pressure along it and a
    /// vanishing transverse velocity — so both terms are small and the acoustic
    /// dissipation is reduced. taking the larger of the two is what makes this work; a
    /// sensor built as their ratio diverges here instead, restoring the dissipation that
    /// drives the instability (measured in `odd_even_decoupling.rs`).
    #[test]
    fn a_grid_aligned_shock_reduces_transverse_but_not_normal_dissipation() {
        let (rho, p, cs) = (1.0, 1.0, (GAMMA * 1.0 / 1.0f64).sqrt());
        let transverse = acoustic_phi(
            1.0e-6, -1.0e-6, cs, cs, p, p * (1.0 + 1.0e-9), rho, rho, 0.0,
        );
        assert!(
            transverse < 1.0e-5,
            "the transverse face read phi = {transverse}; the acoustic dissipation that \
             drives the grid-aligned instability is not being reduced"
        );
        let du = 2.0 * cs;
        let normal = acoustic_phi(du, 0.0, cs, cs, p + rho * cs * du, p, rho, rho, 0.0);
        assert_eq!(normal, 1.0, "the shock-normal face must keep full dissipation");
    }

    /// force balance. an unsupported pressure jump sets a floor proportional to itself,
    /// so a stratified face is dissipated whatever its mach number — which is what damps
    /// a hydrostatic residual. supply the balance and the floor empties, leaving the
    /// low-mach scaling intact across the stratification instead of switched off
    /// throughout it; supply only part of it and the remainder sets the floor.
    #[test]
    fn a_balanced_face_is_only_dissipated_through_its_residual() {
        let (rho, p, cs) = (1.0, 1.0, (GAMMA * 1.0 / 1.0f64).sqrt());
        let dp_balance = 0.1;
        let mach = 1.0e-3;
        let u = mach * cs;
        let du = u;
        let dp_dyn = rho * u * du;
        let at = |bal: f64| {
            acoustic_phi(
                u + 0.5 * du, u - 0.5 * du, cs, cs,
                p + 0.5 * (dp_balance + dp_dyn), p - 0.5 * (dp_balance + dp_dyn),
                rho, rho, bal,
            )
        };
        let unsupplied = at(0.0);
        let floor = (dp_balance + dp_dyn) / (rho * cs * cs);
        assert!(
            (unsupplied - floor).abs() / floor < 1.0e-9,
            "an unsupported jump must set the floor {floor:e}, got {unsupplied:e}"
        );
        assert!(
            unsupplied > 20.0 * mach,
            "the floor {unsupplied:e} is not meaningfully above the mach scaling \
             {mach:e}; this face cannot show that the balance term does anything"
        );
        let supplied = at(dp_balance);
        let ma_local = local_mach(u + 0.5 * du, u - 0.5 * du, cs, cs);
        assert!(
            (supplied - ma_local).abs() / ma_local < 1.0e-6,
            "with the balance supplied the face should fall back to the mach scaling \
             {ma_local:e}, got {supplied:e}"
        );
        let half = at(0.5 * dp_balance);
        let half_floor = (0.5 * dp_balance + dp_dyn) / (rho * cs * cs);
        assert!(
            (half - half_floor).abs() / half_floor < 1.0e-9,
            "a half-supported balance must leave the remainder as the floor \
             {half_floor:e}, got {half:e}"
        );
    }

    /// the sensor compares two dimensionless quantities, so rescaling the flow's units
    /// cannot move it. this is what makes it knob-free: there is no dimensional constant
    /// inside to be calibrated to a problem.
    #[test]
    fn the_sensor_is_invariant_under_a_rescaling_of_units() {
        let (reference, _) = smooth_face(0.01, 1.0);
        for &scale in &[1.0e-6, 1.0e-3, 1.0e3, 1.0e6] {
            let (rho, p) = (1.0, 1.0 * scale * scale);
            let cs = (GAMMA * p / rho).sqrt();
            let u = 0.01 * cs;
            let du = u;
            let dp = rho * u * du;
            let phi = acoustic_phi(
                u + 0.5 * du, u - 0.5 * du, cs, cs,
                p + 0.5 * dp, p - 0.5 * dp, rho, rho, 0.0,
            );
            assert!(
                (phi - reference).abs() < 1.0e-12,
                "rescaling by {scale:e} moved phi from {reference} to {phi}"
            );
        }
    }

    /// bounded, finite, and safe on degenerate data.
    #[test]
    fn the_sensor_is_bounded_and_finite_on_degenerate_faces() {
        let cs = (GAMMA * 1.0 / 1.0f64).sqrt();
        for phi in [
            acoustic_phi(0.0, 0.0, cs, cs, 1.0, 1.0, 1.0, 1.0, 0.0),
            acoustic_phi(0.0, 0.0, cs, cs, 1.0, 2.0, 1.0, 1.0, 0.0),
            acoustic_phi(5.0, -5.0, cs, cs, 1.0, 1.0, 1.0, 1.0, 0.0),
        ] {
            assert!(phi.is_finite() && (0.0..=1.0).contains(&phi), "phi = {phi}");
        }
    }
}
