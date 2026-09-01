// =============================================================================
// dissipation_carrier_oracle.rs
//
// unit + carrier-equivalence coverage for the dissipation rescalings in
// `symbi_hydro::dissipation` (mach_scale, shear_weight).
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

use symbi_hydro::dissipation::{mach_scale, shear_weight};
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
            .map(|n| symbi_ir::ScalarParam::new(n))
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

fn close(a: f64, b: f64, what: &str) {
    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
    assert!(
        rel < 1e-12,
        "{what}: f64 {a} != gv-interp {b} (rel {rel:e})"
    );
}

// =============================================================================
// mach_scale — max of left/right |v| / cs, capped at one. the rescaling factor the
// normal-jump correction is built from; smooth apart from the inner max and the cap.
// =============================================================================

#[test]
fn mach_scale_picks_the_faster_side_and_caps_at_the_sonic_point() {
    // a max over the two sides, each against its own sound speed: a face with one stagnant
    // side takes its dissipation from the side that is moving. that is what makes an immersed
    // body's penalization mask harmless here -- the mask imposes a velocity rather than
    // evolving one, and a scaling that read the smaller side would take the boundary condition
    // for smooth subsonic flow and strip the dissipation off the mask's own boundary faces.
    assert_eq!(mach_scale(0.02, 0.4, 1.0, 1.0), 0.4);
    assert_eq!(mach_scale(0.4, 0.02, 1.0, 1.0), 0.4);
    // the cap is what makes the correction inert on a shock, with no reference value to set.
    assert_eq!(mach_scale(3.0, 0.02, 1.0, 1.0), 1.0);
    assert_eq!(mach_scale(1.0, 1.0, 1.0, 1.0), 1.0);
    // a magnitude: the flow direction across the face does not enter.
    assert_eq!(
        mach_scale(-0.4, 0.02, 1.0, 1.0),
        mach_scale(0.4, 0.02, 1.0, 1.0)
    );
    // per-side sound speeds: a colder side reaches saturation at a lower speed.
    assert_eq!(mach_scale(0.02, 0.02, 1.0, 0.02), 1.0);
}

#[test]
fn mach_scale_carrier_equivalence() {
    // straddles both branch surfaces the expression carries: which side wins the max, and
    // whether the cap is active. a carrier that reorders either would show up here.
    let names = ["speed_l", "speed_r", "cs_l", "cs_r"];
    for inputs in [
        [0.02, 0.40, 1.0, 1.0],  // right side wins, uncapped
        [0.40, 0.02, 1.0, 1.0],  // left side wins, uncapped
        [3.00, 0.02, 1.0, 1.0],  // capped
        [1.00, 1.00, 1.0, 1.0],  // exactly at the cap
        [-0.4, 0.02, 1.0, 1.0],  // negative speed, magnitude taken
        [0.02, 0.02, 1.0, 0.02], // the cold side saturates
    ] {
        let want = mach_scale(inputs[0], inputs[1], inputs[2], inputs[3]);
        let got = gv_eval(
            |p: &[Gv]| mach_scale(p[0], p[1], p[2], p[3]),
            &names,
            &inputs,
        );
        close(want, got, &format!("mach_scale{inputs:?}"));
    }
    assert_lowers(|p: &[Gv]| mach_scale(p[0], p[1], p[2], p[3]), &names);
}

// =============================================================================
// shear_weight — `1 - h^Ma` on the neighborhood pressure ratio h. the weight that
// confines the transverse viscosity to shocks; carries a pow, whose carrier
// lowering is the reason this gate exists.
// =============================================================================

#[test]
fn shear_weight_needs_a_pressure_jump_and_a_flow_at_once() {
    // a smooth interface is exempt at every speed: no pressure structure, no viscosity.
    assert_eq!(shear_weight(1.0, 1.0), 0.0);
    assert_eq!(shear_weight(1.0, 0.01), 0.0);
    // a stagnant stratified column carries the pressure ratio and no flow. the mach exponent
    // empties the weight, which is what keeps the viscosity out of a hydrostatic atmosphere.
    assert_eq!(shear_weight(0.2, 0.0), 0.0);
    assert!(shear_weight(0.2, 1.0e-3) < 1.0e-2);
    // a strong shock at the sonic point carries both and draws nearly the whole viscosity.
    assert!(shear_weight(0.02, 1.0) > 0.97);
    // monotone in the jump at fixed speed, and in the speed at fixed jump.
    assert!(shear_weight(0.02, 1.0) > shear_weight(0.5, 1.0));
    assert!(shear_weight(0.02, 1.0) > shear_weight(0.02, 0.3));
}

#[test]
fn shear_weight_carrier_equivalence() {
    // the pow is the op at risk: a carrier lowering it to exp(mach * ln(ratio)) drifts by a
    // few ulp, and one lowering it wrongly at ratio = 1 or mach = 0 returns the wrong
    // exemption entirely. both degenerate corners are in the sweep.
    let names = ["pressure_ratio", "mach"];
    for inputs in [
        [1.0, 1.0],    // smooth interface, exempt
        [1.0, 0.0],    // smooth and stagnant
        [0.2, 0.0],    // stratified and stagnant, exempt through the exponent
        [0.2, 1.0e-3], // deeply subsonic stratification
        [0.02, 1.0],   // strong shock at the sonic point
        [0.5, 0.3],    // an ordinary compressive face
    ] {
        let want = shear_weight(inputs[0], inputs[1]);
        let got = gv_eval(|p: &[Gv]| shear_weight(p[0], p[1]), &names, &inputs);
        close(want, got, &format!("shear_weight{inputs:?}"));
    }
    assert_lowers(|p: &[Gv]| shear_weight(p[0], p[1]), &names);
}
