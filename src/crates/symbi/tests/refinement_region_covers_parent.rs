// =============================================================================
// refinement_region_covers_parent.rs
//
// a refinement region that covers its parent level entirely leaves the parent no leaf cell, so
// the parent's timestep would be reduced over nothing; the hierarchy refuses such a region when
// it is built, naming the region, instead of stopping later on a zero timestep.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn root() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([32])
        .origin([0.0])
        .spacing([1.0 / 32.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .allocate()
        .expect("root")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build()
}

#[test]
fn a_region_covering_the_whole_parent_is_refused_at_construction() {
    let kset = |s: &Sim| Kset::new(1.4, 0.4, &s.geom.allocated);
    let coarse = root();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [0.0], x_hi: [1.0] }];
    let err = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).err().expect("a region covering its parent must be refused");
    assert!(err.to_string().contains("covers its parent level entirely"), "unexpected refusal: {err}");
    let coarse = root();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [0.25], x_hi: [0.75] }];
    assert!(Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).is_ok(), "a region inside its parent is accepted");
}
