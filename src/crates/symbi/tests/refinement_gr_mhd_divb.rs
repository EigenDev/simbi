// =============================================================================
// refinement_gr_mhd_divb.rs
//
// the densitized divergence of a refined GR-MHD hierarchy across the
// fine-to-coarse transfer. on a curved chart the GR curl updates the face field
// by curl(E)/sqrt(gamma), so the discrete invariant is the divergence of the
// densitized face flux sqrt(gamma)|face * B. a field B^x = B0 / sqrt(gamma)
// seeded on every face of both levels of a cartesian kerr-schild box holds
// that divergence at machine zero on each level. restricting the fine faces
// into the coarse faces they cover keeps it there only when the restriction
// sums densitized flux; the same state on a flat metric (mass zero) is the
// control that separates the metric from the indexing. the curved arm is a
// diagnostic record: the transfers carry the flat flux, and run admission
// refuses magnetized refinement on a curved spacetime by name.
//
// run: cargo test -p symbi --test refinement_gr_mhd_divb
// =============================================================================
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCartesian, Metric};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;
type Hier = Hierarchy<Rmhd, 3, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 8;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const SPIN: f64 = 0.5;
const B0: f64 = 0.1;
// the box [1.2, 2.2]^3 lies outside the horizon: every cell sits at r >= sqrt(3) * 1.2
const X_LO: f64 = 1.2;
const DIVB_TOL: f64 = 1e-12;

fn kernels(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

/// B^x = B0 / sqrt(gamma) at each x face's own center: the densitized flux is B0 on every x face
fn seed(sim: &Sim, metric: KerrKSCartesian<f64>) {
    sim.seed_face_with(0, |x| B0 / metric.sqrt_det_gamma(Tensor::new(x)));
    sim.seed_face(1, 0.0);
    sim.seed_face(2, 0.0);
}

fn two_level(mass: f64) -> (Hier, KerrKSCartesian<f64>) {
    let metric = KerrKSCartesian { mass, spin: SPIN };
    let dx = 1.0 / N as f64;
    let root = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, metric)
        .cells([N; 3])
        .origin([X_LO; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root")
        .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.5)), Tensor::new([0.0; 3])))
        .seed_faces(|d, x| if d == 0 { B0 / metric.sqrt_det_gamma(Tensor::new(x)) } else { 0.0 })
        .build();
    let k = kernels(&root);
    let region = RefinementRegion { x_lo: [X_LO + 0.25; 3], x_hi: [X_LO + 0.75; 3] };
    let h = Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, kernels).expect("hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    // both levels carry the same closure on their own faces
    seed(&h.levels[1].state, metric);
    (h, metric)
}

/// the largest |sum over a cell's faces of sqrt(gamma)|face * B * area| / volume over a level's
/// interior, sqrt(gamma) at each face's own center
fn max_densitized_divergence(sim: &Sim, metric: &KerrKSCartesian<f64>) -> f64 {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let inv_d: [f64; 3] = std::array::from_fn(|a| 1.0 / sim.geom.dx[a]);
    sim.geom
        .interior
        .iter()
        .map(|c| {
            (0..3)
                .map(|a| {
                    let mut n = c;
                    n[a] += 1;
                    let flux = |f: [isize; 3]| metric.sqrt_det_gamma(Tensor::new(sim.geom.face_coord(f, a))) * *mhd.bface[a].view().at(f);
                    (flux(n) - flux(c)) * inv_d[a]
                })
                .sum::<f64>()
                .abs()
        })
        .fold(0.0, f64::max)
}

fn coarse_faces(h: &Hier) -> Vec<f64> {
    let mhd = h.levels[0].state.fields.mhd.as_ref().unwrap();
    mhd.bface[0].domain().clone().iter().map(|f| *mhd.bface[0].view().at(f)).collect()
}

/// the coarse divergence after restricting the fine faces, with its premises asserted
fn restricted_divergence(mass: f64) -> f64 {
    let (h, metric) = two_level(mass);
    for (ll, level) in h.levels.iter().enumerate() {
        let div = max_densitized_divergence(&level.state, &metric);
        assert!(div < DIVB_TOL, "mass {mass}: level {ll} starts with a densitized divergence of {div:e}");
    }
    let before = coarse_faces(&h);
    h.sync_fine_to_coarse(0);
    let after = coarse_faces(&h);
    let moved = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    if mass > 0.0 {
        assert!(moved > 1e-10, "the restriction changed no coarse face by more than {moved:e}; the gate is vacuous");
    }
    max_densitized_divergence(&h.levels[0].state, &metric)
}

#[test]
fn restricting_a_flat_metric_field_keeps_the_divergence_at_machine_zero() {
    let div = restricted_divergence(0.0);
    assert!(div < DIVB_TOL, "the coarse divergence after the restriction is {div:e}");
}

/// the refined face transfers carry the flat face flux, so this records a coarse densitized
/// divergence of ~1.1e-6 after one restriction; magnetized refinement on a curved spacetime is
/// refused by name in run admission until the transfers carry the densitized flux, and this
/// becomes the gate of that repair
#[test]
#[ignore = "diagnostic record of the refused scope; run with --ignored"]
fn restricting_a_kerr_schild_field_keeps_the_densitized_divergence_at_machine_zero() {
    let div = restricted_divergence(0.2);
    assert!(div < DIVB_TOL, "the coarse densitized divergence after the restriction is {div:e}");
}
