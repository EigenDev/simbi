// =============================================================================
// composite_slice_orient.rs
//
// the AMR composite display slice through a NON-Z orientation: the coverage
// descent resolves each root cell to the finest level covering it for every
// display plane, not just the z mid-plane. the fine level is painted with a
// sentinel value far outside the coarse range, so its appearance in the slice
// is unambiguous; the y-mid-plane picture must show the sentinel inside the
// refined region's (x, z) footprint and the coarse value outside it.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 32;
const SENTINEL: f64 = 777.0;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

#[test]
fn composite_reads_fine_data_through_the_y_orientation() {
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .unwrap()
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    let ck = Kset::new(GAMMA, 0.3, &coarse.geom.allocated);
    // the refined box covers the y mid-plane in its center octant.
    let region = RefinementRegion { x_lo: [0.25; 3], x_hi: [0.75; 3] };
    let hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, 0.3, &s.geom.allocated)
    })
    .unwrap();
    hier.seed_fine_from_coarse().unwrap();
    // paint the FINE prim rho with the sentinel (the slice reads prims).
    let fine = &hier.levels[1].state;
    for c in fine.geom.interior.iter() {
        fine.fields.prim.rho.set(c, SENTINEL);
    }
    for c in hier.levels[0].state.geom.interior.iter() {
        hier.levels[0].state.fields.prim.rho.set(c, 1.0);
    }

    let sl = hier.field_slice_composite(64, 0, 1, 0).expect("composite y-slice");
    // the y mid-plane (y = 0.5) intersects the refined box: samples with
    // (x, z) inside [0.25, 0.75]^2 must carry the FINE sentinel; the far
    // corner stays coarse.
    let sent = sl.data.iter().filter(|v| (**v as f64 - SENTINEL).abs() < 1e-3).count();
    let coarse_n = sl.data.iter().filter(|v| (**v as f64 - 1.0).abs() < 1e-3).count();
    assert!(
        sent > 0,
        "no fine-sentinel samples in the y-orientation composite: the descent \
         never reached the fine level off the z plane"
    );
    assert!(coarse_n > 0, "no coarse samples: the mask geometry is wrong");
    // roughly a quarter of the picture is refined footprint (half-extent square).
    let frac = sent as f64 / sl.data.len() as f64;
    assert!(
        (0.15..0.4).contains(&frac),
        "fine footprint fraction {frac:.2} is not ~1/4: the coverage walk is off"
    );
}
