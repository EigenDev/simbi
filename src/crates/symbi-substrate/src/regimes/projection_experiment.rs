// =============================================================================
// projection_experiment.rs
//
// the dispatch half of the projection-anchor measurement apparatus: arm
// selection and the experiment-kernel dispatch with its host receipt
// reduction. the step-scoped transaction book lives in
// `symbi_sim::projection_experiment` (re-exported here), where every runner
// reaches the commit/discard boundary and the session guards.
//
// arm selection: `SIMBI_ANCHOR_EXPERIMENT` set to exactly `stage_input` or
// `eulerian_rebuilt`. unset runs production; any other value panics — an
// experiment names its convention, and a typo must never silently choose an
// arm. no configuration path sets the variable.
// =============================================================================

use std::sync::OnceLock;

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_sim::state::FieldStore;
use symbi_xpu::MemorySpace;

pub use symbi_discretize::AnchorConvention;
pub use symbi_sim::projection_experiment::{
    ExperimentTotals, FirstFire, ProjectionReceipts, SignedAbs, experiment_first_report,
    experiment_report, experiment_reset, receipts_from_diagnostics,
};

/// the experiment arm named by the environment, `None` in production. an
/// unrecognized value panics: the experiment names its convention and a typo
/// must never silently choose an arm.
pub fn experiment_arm() -> Option<AnchorConvention> {
    static ARM: OnceLock<Option<AnchorConvention>> = OnceLock::new();
    *ARM.get_or_init(|| {
        match std::env::var(symbi_sim::projection_experiment::ANCHOR_EXPERIMENT_ENV) {
            Err(_) => None,
            Ok(v) if v == "stage_input" => Some(AnchorConvention::StageInput),
            Ok(v) if v == "eulerian_rebuilt" => Some(AnchorConvention::EulerianRebuilt),
            Ok(v) => panic!(
                "SIMBI_ANCHOR_EXPERIMENT must be 'stage_input' or 'eulerian_rebuilt', got '{v}'"
            ),
        }
    })
}

/// dispatch the experiment projection under the named convention and book the
/// pass's receipts (attempted + the pending-step bucket the driver commits or
/// discards at the step boundary). `injection_weight` is the substage's
/// downstream shu-osher propagation weight, folded into the scheme-effective
/// injected totals while the intervention totals keep the raw deltas. host
/// memory only: the receipts reduce on the host.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_projection_experiment<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eos_param: f64,
    bcell: [&Field<Sc, D, Mem>; 3],
    convention: AnchorConvention,
    injection_weight: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    assert!(
        !Mem::IS_DEVICE_ACCESSIBLE,
        "the anchor experiment records its receipts on host memory"
    );
    let geom = &sim.geom;
    let arm = match convention {
        AnchorConvention::StageInput => "stage",
        AnchorConvention::EulerianRebuilt => "rebuilt",
    };
    let chart = symbi_discretize::kernel_slug::fofc_project_chart(
        symbi_discretize::kernel_slug::ChartKeying::GridAxes,
        geom.coords,
        &geom.axes,
        DOF,
        D,
    );
    let name = symbi_discretize::kernel_slug::fofc_project_experiment_name(
        "rmhd",
        chart,
        geom.spacetime,
        D,
        arm,
    );
    let diag: [Field<Sc, D, Mem>; 4] =
        std::array::from_fn(|_| Field::zeros(&geom.allocated).expect("experiment diag field"));
    let extra: Vec<(&str, &Field<Sc, D, Mem>)> = vec![
        ("xd_theta", &diag[0]),
        ("xd_d_den", &diag[1]),
        ("xd_d_nrg_seg", &diag[2]),
        ("xd_d_nrg_raise", &diag[3]),
    ];
    super::substrate_kernels::fofc_project_named(
        sim,
        name,
        eos_param,
        sim.stage_input(),
        &sim.fields.cons,
        &sim.fields.prim,
        Some(bcell),
        &extra,
    );
    let views: [_; 4] = std::array::from_fn(|k| diag[k].view());
    let mut theta = Vec::new();
    let mut d_den = Vec::new();
    let mut d_seg = Vec::new();
    let mut d_raise = Vec::new();
    for c in geom.interior.iter() {
        theta.push((*views[0].at(c)).to_f64());
        d_den.push((*views[1].at(c)).to_f64());
        d_seg.push((*views[2].at(c)).to_f64());
        d_raise.push((*views[3].at(c)).to_f64());
    }
    let receipts = receipts_from_diagnostics(&theta, &d_den, &d_seg, &d_raise);
    symbi_sim::projection_experiment::record_pass(&receipts, injection_weight);
}
