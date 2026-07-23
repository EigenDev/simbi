// =============================================================================
// refinement_excise.rs
//
// horizon excision on a REFINED hierarchy: the excise pass runs on the ROOT
// level once per root step (the request gate rejects fine patches overlapping
// the excised region, so the root owns every excised cell), while an
// off-horizon fine patch refines the far field. gates:
// - the excision genuinely acts on the refined run (excising differs from not
//   excising — the non-vacuity that once exposed a dead phase);
// - the excision is CAUSALLY CORRECT: two surfaces strictly inside the horizon
//   produce the same root, because nothing done inside a horizon may reach the
//   exterior. demanding instead that the two radii DIFFER asserts the opposite,
//   and holds only while a defect leaks the surface position outward;
// - the fine patch genuinely evolves (its state departs the seeded IC);
// - the run stays finite and positive with both machineries active.
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_hydro::Rhd;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 48;
const L: f64 = 1.2;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 0.3; // r_+ = 0.6
const STEPS: u64 = 20;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn build_hier(
    r_exc: f64,
) -> Hierarchy<Rhd, 2, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory, Kset>
{
    let coarse = Sim::build(Rhd, IdealGas { gamma: GAMMA }, SchwarzschildKSCartesian { mass: MASS })
        .cells([N; 2])
        .origin([-L; 2])
        .spacing([DX; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 2]), pre: 0.1 })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated).with_excision(r_exc);
    // the fine patch sits in the far-field corner quadrant, clear of the excised
    // region (the request gate enforces this separation on the python path; the
    // harness honors the same contract).
    let region = RefinementRegion { x_lo: [0.4, 0.4], x_hi: [1.0, 1.0] };
    Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated).with_excision(r_exc)
    })
    .unwrap()
}

fn run(r_exc: f64) -> (Vec<f64>, Vec<f64>) {
    let mut hier = build_hier(r_exc);
    hier.seed_fine_from_coarse().unwrap();
    hier.evolve_steps(STEPS).unwrap();
    let root: Vec<f64> = hier.levels[0]
        .state
        .geom
        .interior
        .iter()
        .map(|c| *hier.levels[0].state.fields.cons.den.view().at(c))
        .collect();
    let fine: Vec<f64> = hier.levels[1]
        .state
        .geom
        .interior
        .iter()
        .map(|c| *hier.levels[1].state.fields.cons.den.view().at(c))
        .collect();
    (root, fine)
}

#[test]
fn refined_run_excises_the_root_and_evolves_the_fine_patch() {
    let (root_a, fine_a) = run(0.35);
    let (root_b, _) = run(0.5);

    assert!(root_a.iter().all(|v| v.is_finite() && *v > 0.0), "root state broke");
    assert!(fine_a.iter().all(|v| v.is_finite() && *v > 0.0), "fine state broke");

    // the excision genuinely acted: the two radii produced different interiors
    // (the non-vacuity that once exposed the excise phase being silently dead).
    let max_diff = root_a
        .iter()
        .zip(&root_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    // NON-VACUITY: the excise phase actually ran on the hierarchy. the comparison is against NO
    // excision, which is the only baseline a correct scheme is obliged to differ from — measured
    // 1.96e1 against the r_exc = 0.35 run.
    let none = run(0.0).0;
    let acted = root_a
        .iter()
        .zip(&none)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        acted > 1e-10,
        "excising and not excising produced identical roots; the pass never ran on the hierarchy"
    );

    // and the phase is CAUSALLY CORRECT: r_+ = 2M = 0.6, so both radii sit strictly inside the
    // horizon, and nothing done there may reach the exterior. the two runs must therefore agree —
    // measured bit-identical. an excision surface whose position leaks outward is the defect this
    // asserts against, and it is exactly what a core metric with an inconsistent determinant
    // produces: while `sqrt_det_gamma` disagreed with det(gamma) inside the r < M/2 radius floor,
    // the same two radii differed by 9.0 here, growing monotonically with the radius.
    assert!(
        max_diff <= 1e-10,
        "excision radii inside the horizon changed the root by {max_diff:e}: the excised region \
         is leaking into the causally disconnected exterior"
    );

    // the fine patch genuinely evolved: the infall reaches the corner quadrant
    // within 20 root steps, so the fine state departs its uniform seed.
    let fine_dev = fine_a.iter().map(|v| (v - fine_a[0]).abs()).fold(0.0_f64, f64::max);
    let seed_dev = fine_a
        .iter()
        .map(|v| (v - 1.0_f64).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        fine_dev > 1e-12 || seed_dev > 1e-6,
        "the fine patch never evolved; the refined half of the run is vacuous"
    );
}
